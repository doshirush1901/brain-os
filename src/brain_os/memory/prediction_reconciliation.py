"""Closed-loop reconciliation: hot-lead predictions vs outcomes → procedural memory.

Scoring rules (v1):
- Still HOT_NOW or moved to WON → correct
- Pipeline replied / won / stage advanced → correct
- Demoted to STALE without WON → incorrect (false hot)
- LOST CRM or pipeline → incorrect
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from brain_os.config import get_settings
from brain_os.contracts.learning_paths import candidate_procedures_path
from brain_os.memory.lead_board_snapshots import (
    bucket_for_account,
    payload_to_board_map,
    read_board_snapshot,
    write_board_snapshot,
)
from brain_os.memory.prediction_log import (
    HOT_LEAD_BOARD_PATTERN,
    MUST_ACT_TODAY_PATTERN,
    TYCHE_DEAL_FORECAST_PATTERN,
    PredictionLog,
)
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import PredictionReflection
from brain_os.services.deal_dynamics_predictor import DEAL_DYNAMICS_OUTBOUND_PATTERN
from brain_os.services.hot_leads_board import LeadBoardBucket, clear_lead_board_cache, load_lead_board
from brain_os.services.revenue_mode_pipeline_store import (
    _company_key,
    get_pipeline_row,
    list_pipeline_rows,
)

logger = logging.getLogger(__name__)

_REFLECTION_PROMPT = load_prompt("dream_sophia_prediction_reflection")

_POSITIVE_PIPELINE = frozenset({"replied", "won", "quote_sent", "negotiation", "qualified"})
_ADVANCED_STAGES = frozenset({"NEGOTIATION", "WON", "PROPOSAL"})


@dataclass
class ReconciliationRow:
    prediction_id: str
    account: str
    predicted_bucket: str
    today_bucket: str | None
    pipeline_status: str | None
    crm_stage: str | None
    was_correct: bool
    actual_outcome: str
    signals: list[str] = field(default_factory=list)


@dataclass
class ReconciliationReport:
    reconciled: int = 0
    correct: int = 0
    incorrect: int = 0
    skipped: int = 0
    rows: list[ReconciliationRow] = field(default_factory=list)
    snapshot_written: bool = False
    predictions_recorded: int = 0
    must_act_reconciled: int = 0
    must_act_recorded: int = 0
    tyche_reconciled: int = 0
    tyche_recorded: int = 0
    procedures_merged: int = 0
    failures_recorded: int = 0
    sophia_lessons: list[str] = field(default_factory=list)
    weight_proposals_written: int = 0
    drift_alerts: list[dict[str, Any]] = field(default_factory=list)
    drift_enqueued: int = 0


def _pipeline_row_for_account(account: str) -> dict[str, Any] | None:
    row = get_pipeline_row(account)
    if row:
        return row
    key = _company_key(account)
    for pr in list_pipeline_rows():
        if _company_key(str(pr.get("company_name") or "")) == key:
            return pr
    return None


async def _crm_stage_async(crm: Any, account: str) -> str | None:
    try:
        if hasattr(crm, "search_deals"):
            found = await crm.search_deals(company_name=account, limit=1)
        elif hasattr(crm, "get_deals_by_company"):
            found = await crm.get_deals_by_company(account)
        else:
            return None
        if not found:
            return None
        deal = found[0] if isinstance(found, list) else found
        if hasattr(deal, "stage"):
            return str(getattr(deal, "stage", "") or "").upper() or None
        if isinstance(deal, dict):
            return str(deal.get("stage") or "").upper() or None
    except Exception:
        logger.debug("Async CRM stage lookup failed for %s", account, exc_info=True)
    return None


def score_hot_lead_outcome(
    *,
    account: str,
    yesterday_bucket: str,
    today_bucket: str | None,
    pipeline_status: str | None,
    crm_stage: str | None,
) -> tuple[bool, str, list[str]]:
    """Return (was_correct, actual_outcome, signals)."""
    signals: list[str] = []
    yb = (yesterday_bucket or "").upper()
    tb = (today_bucket or "").upper() if today_bucket else ""
    pst = (pipeline_status or "").strip().lower()
    cst = (crm_stage or "").upper()

    if cst == "LOST":
        signals.append("crm_lost")
        return False, f"{account}: CRM LOST", signals

    if pst == "won" or cst == "WON":
        signals.append("won")
        return True, f"{account}: won", signals

    if tb == LeadBoardBucket.WON.value:
        signals.append("board_won")
        return True, f"{account}: board WON", signals

    if pst in _POSITIVE_PIPELINE:
        signals.append(f"pipeline_{pst}")
        return True, f"{account}: pipeline {pst}", signals

    if cst in _ADVANCED_STAGES and cst != "PROPOSAL":
        signals.append(f"crm_{cst}")
        return True, f"{account}: CRM {cst}", signals

    if tb == LeadBoardBucket.HOT_NOW.value:
        signals.append("still_hot_now")
        return True, f"{account}: still HOT_NOW", signals

    if tb == LeadBoardBucket.STALE.value and yb == LeadBoardBucket.HOT_NOW.value:
        signals.append("demoted_stale")
        return False, f"{account}: HOT_NOW → STALE", signals

    if tb == LeadBoardBucket.RE_ENGAGE.value and yb == LeadBoardBucket.HOT_NOW.value:
        signals.append("demoted_re_engage")
        return False, f"{account}: HOT_NOW → RE_ENGAGE (cooled)", signals

    if not tb and yb == LeadBoardBucket.HOT_NOW.value:
        signals.append("removed_from_board")
        return False, f"{account}: removed from board", signals

    signals.append("unchanged_other")
    return True, f"{account}: bucket {tb or 'unknown'} (no strong negative)", signals


async def _has_inbound_since(
    email_processor: Any,
    *,
    account: str,
    owner_contact: str,
    since_iso: str,
) -> bool:
    """Best-effort: any non-sent Gmail hit for account/contact since prediction timestamp."""
    try:
        since_day = since_iso[:10].replace("-", "/")
        parts: list[str] = []
        contact = (owner_contact or "").strip()
        if contact and "@" in contact:
            parts.append(f"from:{contact}")
        token = account.split()[0].lower() if account else ""
        if len(token) >= 3:
            parts.append(token)
        if not parts:
            return False
        query = f"{' '.join(parts)} after:{since_day}"
        emails = await email_processor.search_emails(query=query, max_results=8)
        for em in emails or []:
            labels = [str(x).upper() for x in (getattr(em, "labels", None) or [])]
            if "SENT" not in labels:
                return True
    except Exception:
        logger.debug("Inbound mail check failed for %s", account, exc_info=True)
    return False


async def reconcile_hot_lead_predictions(
    log: PredictionLog,
    *,
    crm: Any | None = None,
    email_processor: Any | None = None,
    yesterday: date | None = None,
) -> ReconciliationReport:
    """Reconcile unreconciled ``hot_lead_board`` predictions using board + pipeline + CRM."""
    report = ReconciliationReport()
    yday = yesterday or (date.today() - timedelta(days=1))

    snap = read_board_snapshot(yday)
    yesterday_map: dict[str, Any] = {}
    if snap is None:
        logger.info(
            "No board snapshot for %s — reconciling hot_lead via pipeline/CRM fallback",
            yday.isoformat(),
        )
    else:
        yesterday_map = payload_to_board_map(snap.get("rows") or [])
    clear_lead_board_cache()
    today_map = payload_to_board_map(
        [
            {
                "account": r.account,
                "bucket": r.bucket.value,
                "owner_contact": r.owner_contact,
                "last_real_touch": r.last_real_touch,
            }
            for r in load_lead_board()
        ]
    )

    pending = await log.get_unreconciled(pattern_id=HOT_LEAD_BOARD_PATTERN, max_age_days=0)
    for pred in pending:
        account = str(pred.context.get("account") or "").strip()
        if not account:
            report.skipped += 1
            continue
        y_bucket = str(
            pred.context.get("bucket") or bucket_for_account(yesterday_map, account) or ""
        )
        t_bucket = bucket_for_account(today_map, account)
        pipe = _pipeline_row_for_account(account)
        pst = str(pipe.get("status") or "").strip().lower() if pipe else None
        cst = await _crm_stage_async(crm, account) if crm is not None else None

        ok, actual, signals = score_hot_lead_outcome(
            account=account,
            yesterday_bucket=y_bucket,
            today_bucket=t_bucket,
            pipeline_status=pst,
            crm_stage=cst,
        )
        if (
            not ok
            and email_processor is not None
            and await _has_inbound_since(
                email_processor,
                account=account,
                owner_contact=str(pred.context.get("owner_contact") or ""),
                since_iso=pred.timestamp,
            )
        ):
            ok = True
            actual = f"{account}: inbound mail since prediction"
            signals = [*signals, "gmail_inbound"]
        await log.record_outcome(pred.prediction_id, actual, ok)
        report.reconciled += 1
        if ok:
            report.correct += 1
        else:
            report.incorrect += 1
        report.rows.append(
            ReconciliationRow(
                prediction_id=pred.prediction_id,
                account=account,
                predicted_bucket=y_bucket,
                today_bucket=t_bucket,
                pipeline_status=pst,
                crm_stage=cst,
                was_correct=ok,
                actual_outcome=actual,
                signals=signals,
            )
        )

    return report


async def _count_unreconciled_hot(log: PredictionLog) -> int:
    return len(await log.get_unreconciled(pattern_id=HOT_LEAD_BOARD_PATTERN))


async def record_hot_lead_predictions(
    log: PredictionLog,
    *,
    dedupe_hours: float = 20.0,
) -> int:
    """Record predictions for every HOT_NOW account on the live board."""
    clear_lead_board_cache()
    rows = load_lead_board()
    count = 0
    for row in rows:
        if row.bucket != LeadBoardBucket.HOT_NOW:
            continue
        if await log.has_recent_unreconciled(
            HOT_LEAD_BOARD_PATTERN,
            account=row.account,
            within_hours=dedupe_hours,
        ):
            continue
        predicted = f"{row.account} will stay hot (quote dialogue + inbound within 14d)"
        await log.record_prediction(
            HOT_LEAD_BOARD_PATTERN,
            predicted,
            context={
                "account": row.account,
                "bucket": row.bucket.value,
                "owner_contact": row.owner_contact,
                "last_real_touch": row.last_real_touch,
                "snapshot_date": date.today().isoformat(),
            },
        )
        count += 1
    return count


def score_must_act_outcome(
    *,
    company: str,
    act_bucket: str,
    pipeline_status: str | None,
    still_in_must_act: bool,
) -> tuple[bool, str, list[str]]:
    """Score whether a must-act prediction was satisfied within ~24h."""
    signals: list[str] = []
    pst = (pipeline_status or "").strip().lower()
    bucket = (act_bucket or "").strip().lower()

    if pst in {"won", "lost"}:
        signals.append(f"pipeline_{pst}")
        return pst == "won", f"{company}: pipeline {pst}", signals

    if pst in _POSITIVE_PIPELINE:
        signals.append(f"pipeline_{pst}")
        return True, f"{company}: progressed to {pst}", signals

    if bucket == "reply" and not still_in_must_act:
        signals.append("reply_cleared")
        return True, f"{company}: reply queue cleared", signals

    if bucket == "rewrite_draft" and pst in {"sent", "quote_sent", "replied"}:
        signals.append("draft_progressed")
        return True, f"{company}: draft pipeline advanced", signals

    if bucket == "followup" and pst == "replied":
        signals.append("followup_got_reply")
        return True, f"{company}: inbound after follow-up window", signals

    if still_in_must_act:
        signals.append("still_must_act")
        return False, f"{company}: still on must-act list", signals

    signals.append("no_longer_must_act")
    return True, f"{company}: dropped from must-act", signals


def _must_act_company_set(followup_idle_days: int = 5) -> dict[str, dict[str, Any]]:
    from brain_os.services.revenue_mode_metrics import must_act_today_items

    return {
        str(it.get("company_name") or "").strip(): it
        for it in must_act_today_items(followup_idle_days=followup_idle_days, limit=40)
        if str(it.get("company_name") or "").strip()
    }


async def reconcile_must_act_predictions(
    log: PredictionLog,
    *,
    followup_idle_days: int = 5,
) -> ReconciliationReport:
    """Reconcile ``must_act_today`` predictions against current desk queues."""
    report = ReconciliationReport()
    must_map = _must_act_company_set(followup_idle_days=followup_idle_days)
    pending = await log.get_unreconciled(pattern_id=MUST_ACT_TODAY_PATTERN)

    for pred in pending:
        company = str(pred.context.get("account") or pred.context.get("company_name") or "").strip()
        if not company:
            report.skipped += 1
            continue
        act_bucket = str(pred.context.get("act_bucket") or pred.context.get("bucket") or "")
        still = company in must_map
        pipe = _pipeline_row_for_account(company)
        pst = str(pipe.get("status") or "").strip().lower() if pipe else None

        ok, actual, signals = score_must_act_outcome(
            company=company,
            act_bucket=act_bucket,
            pipeline_status=pst,
            still_in_must_act=still,
        )
        await log.record_outcome(pred.prediction_id, actual, ok)
        report.must_act_reconciled += 1
        report.reconciled += 1
        if ok:
            report.correct += 1
        else:
            report.incorrect += 1
        report.rows.append(
            ReconciliationRow(
                prediction_id=pred.prediction_id,
                account=company,
                predicted_bucket=act_bucket,
                today_bucket=must_map.get(company, {}).get("bucket") if still else None,
                pipeline_status=pst,
                crm_stage=None,
                was_correct=ok,
                actual_outcome=actual,
                signals=signals,
            )
        )
    return report


async def record_must_act_predictions(
    log: PredictionLog,
    *,
    followup_idle_days: int = 5,
    dedupe_hours: float = 20.0,
) -> int:
    """Record one prediction per must-act desk row (accuracy-gated + narrowable)."""
    from brain_os.memory.must_act_forecast_gate import must_act_recording_policy

    policy = await must_act_recording_policy(log)
    if not policy.get("record"):
        logger.info(
            "Must-act prediction recording paused: reason=%s accuracy=%s reconciled=%s",
            policy.get("reason"),
            policy.get("accuracy"),
            policy.get("reconciled"),
        )
        return 0

    max_per_cycle = int(policy.get("max_per_cycle") or 40)
    blocking_only = bool(policy.get("blocking_only"))
    count = 0
    for company, item in _must_act_company_set(followup_idle_days).items():
        if count >= max_per_cycle:
            break
        if blocking_only and not bool(item.get("blocking_progress")):
            continue
        if await log.has_recent_unreconciled(
            MUST_ACT_TODAY_PATTERN,
            account=company,
            within_hours=dedupe_hours,
        ):
            continue
        bucket = str(item.get("bucket") or "")
        reason = str(item.get("reason") or "")[:200]
        predicted = f"{company} must-act today ({bucket}): {reason}"
        await log.record_prediction(
            MUST_ACT_TODAY_PATTERN,
            predicted,
            context={
                "account": company,
                "act_bucket": bucket,
                "blocking_progress": bool(item.get("blocking_progress")),
                "recommended_action": str(item.get("recommended_action") or "")[:200],
                "recording_policy": str(policy.get("reason") or ""),
            },
        )
        count += 1
    return count


_TYCHE_OPEN_PIPELINE = frozenset({"qualified", "proposal", "negotiation", "quote_sent", "engaged"})


def score_tyche_deal_outcome(
    *,
    company: str,
    prior_status: str,
    pipeline_status: str | None,
) -> tuple[bool, str, list[str]]:
    """Deal was predicted to progress; win if status advanced or won."""
    signals: list[str] = []
    pst = (pipeline_status or "").strip().lower()
    prior = (prior_status or "").strip().lower()

    if pst == "won":
        signals.append("won")
        return True, f"{company}: won", signals
    if pst == "lost":
        signals.append("lost")
        return False, f"{company}: lost", signals
    if pst in _POSITIVE_PIPELINE and pst != prior:
        signals.append(f"advanced_to_{pst}")
        return True, f"{company}: advanced to {pst}", signals
    if pst in _TYCHE_OPEN_PIPELINE and pst == prior:
        signals.append("stalled")
        return False, f"{company}: still {prior}", signals
    signals.append("unchanged")
    return False, f"{company}: no progression", signals


_STAGE_RANK: dict[str, int] = {
    "NEW": 0,
    "CONTACTED": 1,
    "ENGAGED": 2,
    "QUALIFIED": 3,
    "PROPOSAL": 4,
    "NEGOTIATION": 5,
    "WON": 6,
    "LOST": -1,
    "LEAD": 0,
    "OPEN": 1,
    "QUOTE": 4,
}


def score_outbound_deal_dynamics_outcome(
    *,
    company: str,
    predicted_reply_prob: float,
    predicted_stage_delta: int,
    prior_crm_stage: str | None,
    pipeline_status: str | None,
    crm_stage: str | None,
    age_days: float,
) -> tuple[bool, str, list[str]]:
    """Score deal-dynamics forecasts once they are mature enough to judge."""
    signals: list[str] = []
    pst = (pipeline_status or "").strip().lower()
    prior = (prior_crm_stage or "").strip().upper()
    current = (crm_stage or "").strip().upper()

    if pst in {"replied", "won", "quote_sent", "negotiation", "qualified"}:
        signals.append(f"pipeline_{pst}")
        return True, f"{company}: pipeline {pst}", signals
    if pst == "lost":
        signals.append("pipeline_lost")
        return False, f"{company}: pipeline lost", signals

    if prior and current and prior in _STAGE_RANK and current in _STAGE_RANK:
        if _STAGE_RANK[current] > _STAGE_RANK[prior]:
            signals.append("crm_stage_advanced")
            return True, f"{company}: CRM {prior} → {current}", signals
        if current == "WON":
            signals.append("crm_won")
            return True, f"{company}: CRM won", signals
        if current == "LOST":
            signals.append("crm_lost")
            return False, f"{company}: CRM lost", signals

    # Mature predictions with no positive signal → incorrect (false optimism).
    if age_days >= 3.0:
        if predicted_stage_delta > 0 or predicted_reply_prob >= 0.35:
            signals.append("mature_no_progress")
            return (
                False,
                f"{company}: no reply/stage progress after {age_days:.0f}d",
                signals,
            )
        signals.append("mature_low_prob_hold")
        return True, f"{company}: low-prob hold confirmed ({age_days:.0f}d)", signals

    # Too young — caller should skip.
    signals.append("immature")
    return False, f"{company}: immature ({age_days:.1f}d)", signals


async def reconcile_outbound_deal_dynamics_predictions(
    log: PredictionLog,
    *,
    crm: Any | None = None,
    max_age_days: int = 0,
    min_age_days: float = 3.0,
) -> ReconciliationReport:
    """Reconcile ``outbound_deal_dynamics`` predictions (previously never scored)."""
    report = ReconciliationReport()
    pending = await log.get_unreconciled(
        pattern_id=DEAL_DYNAMICS_OUTBOUND_PATTERN,
        max_age_days=max_age_days,
    )
    now = datetime.now(UTC)
    for pred in pending:
        company = str(pred.context.get("account") or "").strip()
        if not company:
            report.skipped += 1
            continue
        try:
            created = datetime.fromisoformat(pred.timestamp.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            age_days = (now - created.astimezone(UTC)).total_seconds() / 86400.0
        except ValueError:
            age_days = 999.0
        if age_days < min_age_days:
            report.skipped += 1
            continue

        prior_stage = str(pred.context.get("crm_stage") or "") or None
        reply_prob = float(pred.context.get("reply_prob") or 0.0)
        stage_delta = int(pred.context.get("stage_delta") or 0)
        pipe = _pipeline_row_for_account(company)
        pst = str(pipe.get("status") or "").strip().lower() if pipe else None
        cst = await _crm_stage_async(crm, company) if crm is not None else None

        ok, actual, signals = score_outbound_deal_dynamics_outcome(
            company=company,
            predicted_reply_prob=reply_prob,
            predicted_stage_delta=stage_delta,
            prior_crm_stage=prior_stage,
            pipeline_status=pst,
            crm_stage=cst,
            age_days=age_days,
        )
        if "immature" in signals:
            report.skipped += 1
            continue
        await log.record_outcome(pred.prediction_id, actual, ok)
        report.reconciled += 1
        if ok:
            report.correct += 1
        else:
            report.incorrect += 1
        report.rows.append(
            ReconciliationRow(
                prediction_id=pred.prediction_id,
                account=company,
                predicted_bucket=f"reply_prob={reply_prob:.2f}",
                today_bucket=pst or cst,
                pipeline_status=pst,
                crm_stage=cst,
                was_correct=ok,
                actual_outcome=actual,
                signals=signals,
            )
        )
    return report


async def expire_aged_unreconciled(
    log: PredictionLog,
    *,
    older_than_days: float = 45.0,
    pattern_id: str | None = None,
) -> int:
    """Mark ancient unreconciled rows expired so the backlog cannot grow forever."""
    pending = await log.get_unreconciled(pattern_id=pattern_id, max_age_days=0)
    now = datetime.now(UTC)
    expired = 0
    for pred in pending:
        try:
            created = datetime.fromisoformat(pred.timestamp.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            age_days = (now - created.astimezone(UTC)).total_seconds() / 86400.0
        except ValueError:
            age_days = 999.0
        if age_days < older_than_days:
            continue
        await log.record_outcome(
            pred.prediction_id,
            f"expired_unresolved after {age_days:.0f}d",
            False,
        )
        expired += 1
    return expired


async def reconcile_tyche_deal_predictions(log: PredictionLog) -> ReconciliationReport:
    """Reconcile open-deal progress forecasts against pipeline today."""
    report = ReconciliationReport()
    pending = await log.get_unreconciled(pattern_id=TYCHE_DEAL_FORECAST_PATTERN)

    for pred in pending:
        company = str(pred.context.get("account") or "").strip()
        if not company:
            report.skipped += 1
            continue
        prior = str(pred.context.get("pipeline_status") or "")
        pipe = _pipeline_row_for_account(company)
        pst = str(pipe.get("status") or "").strip().lower() if pipe else None

        ok, actual, signals = score_tyche_deal_outcome(
            company=company,
            prior_status=prior,
            pipeline_status=pst,
        )
        await log.record_outcome(pred.prediction_id, actual, ok)
        report.tyche_reconciled += 1
        report.reconciled += 1
        if ok:
            report.correct += 1
        else:
            report.incorrect += 1
        report.rows.append(
            ReconciliationRow(
                prediction_id=pred.prediction_id,
                account=company,
                predicted_bucket=prior,
                today_bucket=pst,
                pipeline_status=pst,
                crm_stage=None,
                was_correct=ok,
                actual_outcome=actual,
                signals=signals,
            )
        )
    return report


async def record_tyche_deal_predictions(log: PredictionLog, *, dedupe_hours: float = 20.0) -> int:
    """Record progress forecasts for open pipeline deals (deterministic Tyche hook)."""
    count = 0
    for row in list_pipeline_rows():
        company = str(row.get("company_name") or "").strip()
        if not company:
            continue
        pst = str(row.get("status") or "").strip().lower()
        if pst not in _TYCHE_OPEN_PIPELINE:
            continue
        if await log.has_recent_unreconciled(
            TYCHE_DEAL_FORECAST_PATTERN,
            account=company,
            within_hours=dedupe_hours,
        ):
            continue
        predicted = f"{company} deal will progress from {pst} within 7d"
        await log.record_prediction(
            TYCHE_DEAL_FORECAST_PATTERN,
            predicted,
            context={"account": company, "pipeline_status": pst},
        )
        count += 1
    return count


async def record_single_deal_forecast(
    *,
    company: str,
    predicted_outcome: str,
    pipeline_status: str | None = None,
    log: PredictionLog | None = None,
) -> dict[str, Any]:
    """Log one Tyche deal forecast for later reconciliation."""
    cfg = get_settings().app
    if not cfg.prediction_record_tyche_deals:
        return {"status": "skipped", "reason": "disabled"}
    company = (company or "").strip()
    if not company:
        return {"status": "error", "reason": "missing company"}
    own_log = log is None
    if log is None:
        log = PredictionLog()
        await log.initialize()
    try:
        pst = (pipeline_status or "").strip().lower()
        if await log.has_recent_unreconciled(
            TYCHE_DEAL_FORECAST_PATTERN,
            account=company,
            within_hours=20.0,
        ):
            return {"status": "skipped", "reason": "recent_pending"}
        pred_id = await log.record_prediction(
            TYCHE_DEAL_FORECAST_PATTERN,
            predicted_outcome,
            context={"account": company, "pipeline_status": pst},
        )
        return {"status": "ok", "prediction_id": pred_id}
    finally:
        if own_log:
            await log.close()


def record_intraday_hot_predictions_sync(rows: list[dict[str, str]]) -> None:
    """Optional hook after board CSV write (sync CLI path)."""
    cfg = get_settings().app
    if not cfg.prediction_record_intraday_on_board_write:
        return
    import asyncio

    async def _run() -> None:
        log = PredictionLog()
        await log.initialize()
        try:
            n = 0
            for row in rows:
                account = str(row.get("account") or "").strip()
                bucket = str(row.get("bucket") or "").upper()
                if not account or bucket != LeadBoardBucket.HOT_NOW.value:
                    continue
                if await log.has_recent_unreconciled(
                    HOT_LEAD_BOARD_PATTERN, account=account, within_hours=20.0
                ):
                    continue
                await log.record_prediction(
                    HOT_LEAD_BOARD_PATTERN,
                    f"{account} will stay hot (intraday board write)",
                    context={
                        "account": account,
                        "bucket": bucket,
                        "intraday": True,
                        "snapshot_date": date.today().isoformat(),
                    },
                )
                n += 1
            if n:
                logger.info("Intraday hot-lead predictions recorded: %d", n)
        finally:
            await log.close()

    try:
        asyncio.run(_run())
    except Exception:
        logger.exception("Intraday hot-lead prediction recording failed")


async def sophia_reflect_on_deltas(
    report: ReconciliationReport,
    *,
    max_rows: int | None = None,
) -> PredictionReflection | None:
    """Bounded LLM reflection on reconciliation rows (no full ReAct loop)."""
    if not report.rows:
        return None
    cfg = get_settings().app
    cap = max_rows if max_rows is not None else int(cfg.prediction_reconciliation_max_accounts)
    summary = [
        {
            "account": r.account,
            "predicted_bucket": r.predicted_bucket,
            "today_bucket": r.today_bucket,
            "pipeline_status": r.pipeline_status,
            "crm_stage": r.crm_stage,
            "was_correct": r.was_correct,
            "actual_outcome": r.actual_outcome,
            "signals": r.signals,
        }
        for r in report.rows[:cap]
    ]
    user_msg = json.dumps(
        {
            "reconciled": report.reconciled,
            "correct": report.correct,
            "incorrect": report.incorrect,
            "rows": summary,
        },
        ensure_ascii=True,
        indent=2,
    )
    from brain_os.services.llm_client import get_llm_client

    llm = get_llm_client()
    try:
        return await llm.generate_structured(
            _REFLECTION_PROMPT,
            user_msg,
            PredictionReflection,
            name="dream.sophia_predictions",
        )
    except Exception:
        logger.exception("Sophia prediction reflection LLM failed")
        return None


def _failure_trigger(row: ReconciliationRow, patterns: list[str]) -> str:
    base = f"hot lead follow-up for {row.account}"
    if patterns:
        return f"{base} — {patterns[0][:120]}"
    if row.today_bucket:
        return f"{base} when demoted to {row.today_bucket}"
    return base


async def apply_procedural_updates(
    report: ReconciliationReport,
    reflection: PredictionReflection | None,
    procedural: Any,
    *,
    must_act_accuracy: float | None = None,
    must_act_reconciled: int = 0,
) -> tuple[int, int]:
    """Apply record_failure / merge_procedure_from_learning; returns (merged, failures)."""
    from brain_os.memory.must_act_forecast_gate import (
        is_must_act_reconciliation_row,
        must_act_procedural_learning_enabled,
    )

    merged = 0
    failures = 0
    fp_patterns = list(reflection.false_positive_patterns) if reflection else []
    learn_from_must_act = must_act_procedural_learning_enabled(
        accuracy=must_act_accuracy,
        reconciled=must_act_reconciled,
    )

    for row in report.rows:
        if not row.was_correct:
            if is_must_act_reconciliation_row(row) and not learn_from_must_act:
                continue
            trigger = _failure_trigger(row, fp_patterns)
            try:
                await procedural.record_failure(trigger)
                failures += 1
            except Exception:
                logger.exception("record_failure failed for %s", row.account)

    if reflection:
        for cand in reflection.procedure_candidates:
            if (cand.confidence or "").upper() != "HIGH":
                continue
            steps = [s.strip() for s in cand.steps if s and str(s).strip()]
            trigger = (cand.trigger or "").strip()
            if not trigger or not steps:
                continue
            try:
                await procedural.merge_procedure_from_learning(trigger, steps)
                merged += 1
            except Exception:
                logger.exception("merge_procedure failed for trigger %s", trigger[:80])

    return merged, failures


def append_procedure_candidates(reflection: PredictionReflection) -> int:
    """Append HIGH-confidence candidates to compiler JSON for optional stage 12d promotion."""
    added = 0
    path = candidate_procedures_path()
    existing: dict[str, Any] = {"candidates": []}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {"candidates": []}
    candidates = list(existing.get("candidates") or [])
    for cand in reflection.procedure_candidates:
        if (cand.confidence or "").upper() != "HIGH":
            continue
        steps = [str(s).strip().lower() for s in cand.steps if str(s).strip()]
        trigger = (cand.trigger or "").strip()
        if not trigger or not steps:
            continue
        candidates.append(
            {
                "trigger_pattern": trigger,
                "steps": steps,
                "source": "sophia_prediction_reflection",
                "evidence_count": 1,
            }
        )
        added += 1
    if added:
        existing["candidates"] = candidates
        existing["compiled_at"] = datetime.now(UTC).isoformat()
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return added


async def run_prediction_reconciliation_cycle(
    *,
    crm: Any | None = None,
    email_processor: Any | None = None,
    procedural: Any | None = None,
    long_term: Any | None = None,
    skip_llm: bool = False,
    followup_idle_days: int = 5,
    record_new: bool = True,
    write_snapshot: bool = True,
) -> ReconciliationReport:
    """Full closed-loop pass: reconcile → reflect → apply → snapshot → record."""
    cfg = get_settings().app
    log = PredictionLog()
    await log.initialize()
    pending_before = len(await log.get_unreconciled())
    try:
        hot_report = await reconcile_hot_lead_predictions(
            log, crm=crm, email_processor=email_processor
        )
        report = hot_report

        if cfg.prediction_record_must_act:
            ma_report = await reconcile_must_act_predictions(
                log, followup_idle_days=followup_idle_days
            )
            report.must_act_reconciled = ma_report.must_act_reconciled
            report.reconciled += ma_report.reconciled
            report.correct += ma_report.correct
            report.incorrect += ma_report.incorrect
            report.skipped += ma_report.skipped
            report.rows.extend(ma_report.rows)

        if cfg.prediction_record_tyche_deals:
            ty_report = await reconcile_tyche_deal_predictions(log)
            report.tyche_reconciled = ty_report.tyche_reconciled
            report.reconciled += ty_report.reconciled
            report.correct += ty_report.correct
            report.incorrect += ty_report.incorrect
            report.skipped += ty_report.skipped
            report.rows.extend(ty_report.rows)

        # outbound_deal_dynamics was recorded for months without a reconciler —
        # score mature rows and expire ancient leftovers so the backlog clears.
        dd_report = await reconcile_outbound_deal_dynamics_predictions(log, crm=crm)
        report.reconciled += dd_report.reconciled
        report.correct += dd_report.correct
        report.incorrect += dd_report.incorrect
        report.skipped += dd_report.skipped
        report.rows.extend(dd_report.rows)
        expired = await expire_aged_unreconciled(log, older_than_days=45.0)
        if expired:
            report.reconciled += expired
            report.incorrect += expired
            logger.info("Expired %d aged unreconciled predictions (>45d)", expired)

        reflection: PredictionReflection | None = None
        if not skip_llm and report.rows:
            reflection = await sophia_reflect_on_deltas(report)
            if reflection:
                report.sophia_lessons = list(reflection.lessons)

        must_act_accuracy: float | None = None
        must_act_reconciled = 0
        if cfg.prediction_record_must_act:
            by_pattern = await log.accuracy_by_pattern(min_predictions=1)
            must_act_accuracy = by_pattern.get(MUST_ACT_TODAY_PATTERN)
            must_act_reconciled = await log.reconciled_count(pattern_id=MUST_ACT_TODAY_PATTERN)

        if procedural is not None and (report.rows or reflection):
            merged, failures = await apply_procedural_updates(
                report,
                reflection,
                procedural,
                must_act_accuracy=must_act_accuracy,
                must_act_reconciled=must_act_reconciled,
            )
            report.procedures_merged = merged
            report.failures_recorded = failures

        if reflection:
            append_procedure_candidates(reflection)

        try:
            from brain_os.memory.prediction_weight_tuning import (
                detect_prediction_drift,
                enqueue_prediction_drift_for_morning_brain,
                process_reconciliation_weight_loop,
            )

            by_pattern = await log.accuracy_by_pattern(min_predictions=1)
            if report.sophia_lessons or reflection:
                weight_loop = await process_reconciliation_weight_loop(
                    lessons=report.sophia_lessons,
                    reflection=reflection,
                    log=log,
                    reconciliation_accuracy=by_pattern,
                )
                report.weight_proposals_written = int(
                    weight_loop.get("weight_proposals_written") or 0
                )
                report.drift_alerts = list(weight_loop.get("drift_alerts") or [])
                report.drift_enqueued = int(weight_loop.get("drift_enqueued") or 0)
            else:
                drift = await detect_prediction_drift(log)
                report.drift_alerts = drift
                report.drift_enqueued = enqueue_prediction_drift_for_morning_brain(drift)
        except Exception:
            logger.exception("Prediction weight loop failed")

        if long_term is not None and report.sophia_lessons:
            try:
                lesson_text = "; ".join(report.sophia_lessons[:5])
                await long_term.store_gated(
                    f"[sophia_prediction_reflection] {lesson_text}",
                    user_id="global",
                    metadata={
                        "type": "sophia_prediction_reflection",
                        "memory_category": "sophia_lesson",
                    },
                    source="sophia:prediction_reflection",
                    category="sophia_lesson",
                )
            except Exception:
                logger.debug("Long-term store of Sophia lessons failed", exc_info=True)

        if write_snapshot:
            write_board_snapshot()
            report.snapshot_written = True

        if record_new:
            report.predictions_recorded = await record_hot_lead_predictions(log)
            if cfg.prediction_record_must_act:
                report.must_act_recorded = await record_must_act_predictions(
                    log, followup_idle_days=followup_idle_days
                )
            if cfg.prediction_record_tyche_deals:
                report.tyche_recorded = await record_tyche_deal_predictions(log)
        try:
            from brain_os.memory.dream_reconcile_metrics import append_dream_reconcile_event

            append_dream_reconcile_event(
                source="prediction_reconciliation",
                reconciled=report.reconciled,
                correct=report.correct,
                incorrect=report.incorrect,
                skipped=report.skipped,
                pending_before=pending_before,
                sophia_lessons=len(report.sophia_lessons),
                operator_tickets=len(report.sophia_lessons),
            )
        except Exception:
            logger.debug("dream reconcile metrics event failed", exc_info=True)
    finally:
        await log.close()
    return report
