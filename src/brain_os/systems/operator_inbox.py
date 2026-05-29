"""Unified operator approval inbox — aggregate, decide, release session."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from brain_os.config import get_settings
from brain_os.data.quotes import QuoteStatus
from brain_os.schemas.operator_inbox import (
    OperatorInboxDecision,
    OperatorInboxItem,
    OperatorInboxPayload,
    OperatorInboxSummary,
    OperatorReleaseSession,
)
from brain_os.services.agency.agency_collector import collect_board_candidates
from brain_os.services.revenue_mode_drafts import find_latest_email_draft, recent_drafts_by_company
from brain_os.services.revenue_mode_pipeline_store import _company_key
from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)

_QUOTE_LIST_ERRORS = (
    SQLAlchemyError,
    ValueError,
    TypeError,
    KeyError,
    OSError,
)

_AUDIT_NAME = "operator_inbox_audit.jsonl"
_DISMISSED_NAME = "operator_inbox_dismissed.json"
_SESSION_NAME = "operator_session.json"


def _operations_dir() -> Path:
    d = get_data_dir() / "operations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _audit_path() -> Path:
    return _operations_dir() / _AUDIT_NAME


def _dismissed_path() -> Path:
    return _operations_dir() / _DISMISSED_NAME


def _session_path() -> Path:
    return _operations_dir() / _SESSION_NAME


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        return None


def _body_snippet(body: str, *, max_len: int = 320) -> str:
    text = " ".join((body or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


class OperatorInboxService:
    """Collect pending operator work, record decisions, manage release session."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    def _load_dismissed(self) -> dict[str, str]:
        path = _dismissed_path()
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_dismissed(self, data: dict[str, str]) -> None:
        _dismissed_path().write_text(
            json.dumps(data, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _is_dismissed(self, item_id: str) -> bool:
        until = self._load_dismissed().get(item_id)
        if not until:
            return False
        exp = _parse_iso(until)
        if exp is None:
            return False
        if datetime.now(UTC) >= exp:
            return False
        return True

    def _dismiss(self, item_id: str, *, days: int = 7) -> None:
        data = self._load_dismissed()
        until = datetime.now(UTC) + timedelta(days=max(1, days))
        data[item_id] = until.isoformat().replace("+00:00", "Z")
        self._save_dismissed(data)

    def _append_audit(
        self,
        *,
        item_id: str,
        decision: str,
        actor: str,
        result: dict[str, Any],
    ) -> None:
        line = {
            "item_id": item_id,
            "decision": decision,
            "actor": actor,
            "at": _utcnow_iso(),
            "result": result,
        }
        with _audit_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=True) + "\n")

    def load_session(self) -> OperatorReleaseSession:
        path = _session_path()
        if not path.is_file():
            return OperatorReleaseSession()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return OperatorReleaseSession()
        until = data.get("released_until")
        exp = _parse_iso(until)
        released = exp is not None and datetime.now(UTC) < exp
        return OperatorReleaseSession(
            released_until=until,
            released_by=data.get("released_by"),
            pending_at_release=int(data.get("pending_at_release") or 0),
            released=released,
        )

    def is_released(self) -> bool:
        return self.load_session().released

    def save_release(
        self,
        *,
        actor: str,
        ttl_hours: float,
        pending_at_release: int,
    ) -> OperatorReleaseSession:
        until = datetime.now(UTC) + timedelta(hours=max(0.25, ttl_hours))
        until_s = until.isoformat().replace("+00:00", "Z")
        payload = {
            "released_until": until_s,
            "released_by": actor.strip() or "operator",
            "pending_at_release": pending_at_release,
            "released_at": _utcnow_iso(),
        }
        _session_path().write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return OperatorReleaseSession(
            released_until=until_s,
            released_by=payload["released_by"],
            pending_at_release=pending_at_release,
            released=True,
        )

    def _collect_outbound_batch_items(
        self,
        outbound_approvals: Any,
        *,
        limit: int,
    ) -> list[OperatorInboxItem]:
        items: list[OperatorInboxItem] = []
        try:
            batches = outbound_approvals._load()
        except AttributeError:
            return items
        for batch in batches:
            if batch.get("status") != "pending_approval":
                continue
            batch_id = str(batch.get("batch_id") or "")
            campaign = str(batch.get("campaign_name") or "Campaign")
            created = batch.get("created_at")
            for idx, msg in enumerate(batch.get("messages") or []):
                if msg.get("approved_at") or msg.get("skipped_at"):
                    continue
                item_id = f"outbound:{batch_id}:{idx}"
                if self._is_dismissed(item_id):
                    continue
                to_addr = str(msg.get("to") or "")
                subj = str(msg.get("subject") or "")
                items.append(
                    OperatorInboxItem(
                        id=item_id,
                        kind="outbound_email",
                        title=f"{to_addr} — {subj[:60]}" if subj else to_addr,
                        subtitle=f"Batch: {campaign}",
                        company_name="",
                        risk="external_visible",
                        preview={
                            "to": to_addr,
                            "subject": subj,
                            "body_snippet": _body_snippet(str(msg.get("body") or "")),
                        },
                        source={
                            "type": "batch",
                            "batch_id": batch_id,
                            "index": idx,
                            "campaign_name": campaign,
                        },
                        created_at=str(created) if created else None,
                    )
                )
                if len(items) >= limit:
                    return items
        return items

    def _collect_revenue_draft_items(self, *, limit: int) -> list[OperatorInboxItem]:
        items: list[OperatorInboxItem] = []
        seen: set[str] = set()
        for row in recent_drafts_by_company(limit=60, include_body=True):
            pst = str(row.get("pilot_status") or "").strip().lower()
            if pst not in {"", "pending_review"}:
                continue
            dq = row.get("draft_quality") or {}
            if dq.get("final_status") == "needs_rewrite":
                continue
            company = str(row.get("company_name") or "").strip()
            if not company:
                continue
            ck = _company_key(company)
            if ck in seen:
                continue
            seen.add(ck)
            item_id = f"draft:{ck}"
            if self._is_dismissed(item_id):
                continue
            subj = str(row.get("subject") or "")[:72]
            body = str(row.get("email_body") or "")
            items.append(
                OperatorInboxItem(
                    id=item_id,
                    kind="outbound_email",
                    title=f"{company} — approve draft",
                    subtitle=subj or "Revenue draft pending review",
                    company_name=company,
                    risk="external_visible",
                    preview={
                        "subject": subj,
                        "body_snippet": _body_snippet(body),
                    },
                    source={"type": "revenue_draft", "company_name": company},
                    created_at=str(row.get("timestamp") or "") or None,
                )
            )
            if len(items) >= limit:
                break
        return items

    def _collect_tinder_item(self, tinder_service: Any | None) -> OperatorInboxItem | None:
        if tinder_service is None:
            return None
        st = tinder_service.status()
        pending = st.get("pending_draft")
        if not isinstance(pending, dict) or not pending.get("to"):
            return None
        item_id = "tinder:pending"
        if self._is_dismissed(item_id):
            return None
        to_addr = str(pending.get("to") or "")
        subj = str(pending.get("subject") or "")
        body = str(pending.get("body") or "")
        card = st.get("card") or {}
        company = str(card.get("company_name") or "")
        return OperatorInboxItem(
            id=item_id,
            kind="outbound_email",
            title=f"Tinder — {company or to_addr}",
            subtitle=subj[:72],
            company_name=company,
            risk="external_visible",
            preview={
                "to": to_addr,
                "subject": subj,
                "body_snippet": _body_snippet(body),
            },
            source={"type": "tinder"},
            created_at=None,
        )

    def _collect_lead_review_items(self, *, limit: int) -> list[OperatorInboxItem]:
        items: list[OperatorInboxItem] = []
        for card in collect_board_candidates(limit=limit):
            account = card.company_name.strip()
            if not account:
                continue
            ck = _company_key(account)
            item_id = f"board:{ck}"
            if self._is_dismissed(item_id):
                continue
            items.append(
                OperatorInboxItem(
                    id=item_id,
                    kind="lead_review",
                    title=card.title,
                    subtitle=card.impact_line[:120],
                    company_name=account,
                    risk="internal",
                    preview={
                        "why_now": card.why_now[:400],
                        "recommended_action": card.recommended_action[:300],
                        "source_bucket": card.source_bucket,
                    },
                    source={
                        "type": "board",
                        "company_name": account,
                        "card_id": card.id,
                    },
                    created_at=card.created_at,
                )
            )
        return items

    async def _collect_quote_items(
        self, quotes: Any | None, *, limit: int
    ) -> list[OperatorInboxItem]:
        if quotes is None:
            return []
        items: list[OperatorInboxItem] = []
        try:
            rows = await quotes.list_quotes(filters={"status": QuoteStatus.DRAFT.value})
        except _QUOTE_LIST_ERRORS as exc:
            logger.warning("operator inbox: quote list failed: %s", exc)
            return []
        rows.sort(key=lambda q: q.created_at or datetime.min.replace(tzinfo=UTC), reverse=True)
        for q in rows[:limit]:
            qid = str(q.id)
            item_id = f"quote:{qid}"
            if self._is_dismissed(item_id):
                continue
            company = str(q.company_name or "")
            model = str(q.machine_model or "")
            val = q.estimated_value
            val_s = f"{float(val):,.0f}" if val is not None else "—"
            items.append(
                OperatorInboxItem(
                    id=item_id,
                    kind="quote_draft",
                    title=f"{company or 'Quote'} — {model or 'machine TBD'}",
                    subtitle=f"DRAFT · {q.currency} {val_s}",
                    company_name=company,
                    risk="internal",
                    preview={
                        "quote_id": qid,
                        "machine_model": model,
                        "estimated_value": val_s,
                        "currency": str(q.currency or "USD"),
                    },
                    source={"type": "quote", "quote_id": qid},
                    created_at=q.created_at.isoformat() if q.created_at else None,
                )
            )
        return items

    def _dedupe_outbound_by_company(
        self, items: list[OperatorInboxItem]
    ) -> list[OperatorInboxItem]:
        """Prefer revenue draft over batch/tinder for the same company."""
        by_company: dict[str, OperatorInboxItem] = {}
        rest: list[OperatorInboxItem] = []
        for it in items:
            if it.kind != "outbound_email":
                rest.append(it)
                continue
            co = it.company_name.strip()
            if not co:
                rest.append(it)
                continue
            ck = _company_key(co)
            prev = by_company.get(ck)
            if prev is None:
                by_company[ck] = it
                continue
            if (
                prev.source.get("type") != "revenue_draft"
                and it.source.get("type") == "revenue_draft"
            ):
                by_company[ck] = it
        return rest + list(by_company.values())

    async def collect_inbox(
        self,
        *,
        outbound_approvals: Any | None = None,
        quotes: Any | None = None,
        tinder_service: Any | None = None,
        limit_per_kind: int = 40,
    ) -> OperatorInboxPayload:
        outbound: list[OperatorInboxItem] = []
        if outbound_approvals is not None:
            outbound.extend(
                self._collect_outbound_batch_items(outbound_approvals, limit=limit_per_kind)
            )
        outbound.extend(self._collect_revenue_draft_items(limit=limit_per_kind))
        tinder_item = self._collect_tinder_item(tinder_service)
        if tinder_item is not None:
            ck = _company_key(tinder_item.company_name) if tinder_item.company_name else ""
            if not ck or not any(
                i.kind == "outbound_email"
                and _company_key(i.company_name) == ck
                and i.source.get("type") == "revenue_draft"
                for i in outbound
            ):
                outbound.append(tinder_item)
        outbound = self._dedupe_outbound_by_company(outbound)

        leads = self._collect_lead_review_items(limit=limit_per_kind)
        quotes_items = await self._collect_quote_items(quotes, limit=limit_per_kind)

        all_items = outbound + leads + quotes_items
        from brain_os.services.math_mode import (
            build_math_promote_checklist,
            enrich_operator_inbox_item_math,
            math_mode_advisory_enabled,
            math_priority_hints_from_shadow,
        )

        if math_mode_advisory_enabled():
            all_items = [enrich_operator_inbox_item_math(i) for i in all_items]
        ext = sum(1 for i in all_items if i.risk == "external_visible")
        summary = OperatorInboxSummary(
            outbound_email=len(outbound),
            lead_review=len(leads),
            quote_draft=len(quotes_items),
            total=len(all_items),
            external_pending=ext,
        )
        session = self.load_session()
        hints: list[dict[str, Any]] = []
        advisory_on = math_mode_advisory_enabled()
        if advisory_on:
            hints = math_priority_hints_from_shadow(limit=10)
        promote_checklist = build_math_promote_checklist()
        return OperatorInboxPayload(
            summary=summary,
            items=all_items,
            released=session.released,
            released_until=session.released_until,
            math_advisory_enabled=advisory_on,
            math_priority_hints=hints,
            math_promote_checklist=promote_checklist,
        )

    async def decide(
        self,
        *,
        item_id: str,
        decision: OperatorInboxDecision,
        actor: str,
        snooze_days: int = 7,
        to_address: str | None = None,
        outbound_approvals: Any | None = None,
        email_processor: Any | None = None,
        tinder_service: Any | None = None,
        quotes: Any | None = None,
    ) -> dict[str, Any]:
        inbox = await self.collect_inbox(
            outbound_approvals=outbound_approvals,
            quotes=quotes,
            tinder_service=tinder_service,
        )
        item = next((i for i in inbox.items if i.id == item_id), None)
        if item is None:
            for i in inbox.items:
                if i.id.lower().startswith(item_id.lower()):
                    item = i
                    break
        if item is None:
            return {"ok": False, "error": f"Unknown item_id: {item_id}"}

        if decision == "snooze":
            self._dismiss(item.id, days=snooze_days)
            self._append_audit(
                item_id=item.id,
                decision=decision,
                actor=actor,
                result={"ok": True, "snooze_days": snooze_days},
            )
            return {"ok": True, "action": "snoozed", "item_id": item.id}

        if decision == "reject":
            result = await self._reject_item(
                item,
                actor=actor,
                outbound_approvals=outbound_approvals,
                tinder_service=tinder_service,
            )
            self._append_audit(item_id=item.id, decision=decision, actor=actor, result=result)
            return result

        result = await self._approve_item(
            item,
            actor=actor,
            to_address=to_address,
            outbound_approvals=outbound_approvals,
            email_processor=email_processor,
            tinder_service=tinder_service,
        )
        self._append_audit(item_id=item.id, decision="approve", actor=actor, result=result)
        return result

    async def _approve_item(
        self,
        item: OperatorInboxItem,
        *,
        actor: str,
        to_address: str | None = None,
        outbound_approvals: Any | None = None,
        email_processor: Any | None = None,
        tinder_service: Any | None = None,
    ) -> dict[str, Any]:
        src_type = str(item.source.get("type") or "")

        if item.kind == "quote_draft":
            self._dismiss(item.id, days=3650)
            return {
                "ok": True,
                "action": "quote_acknowledged",
                "item_id": item.id,
                "message": "Quote marked reviewed in operator queue (status remains DRAFT).",
            }

        if item.kind == "lead_review":
            self._dismiss(item.id, days=3650)
            return {
                "ok": True,
                "action": "lead_acknowledged",
                "item_id": item.id,
                "message": "Lead review acknowledged.",
            }

        if src_type == "batch":
            if outbound_approvals is None or email_processor is None:
                return {"ok": False, "error": "Outbound approvals or email processor unavailable."}
            from brain_os.interfaces.email_processor_draft_sender import GmailDraftSender

            sender = GmailDraftSender(email_processor=email_processor)
            batch_id = str(item.source.get("batch_id") or "")
            index = int(item.source.get("index") or 0)
            try:
                row = await outbound_approvals.approve_message(
                    batch_id=batch_id,
                    index=index,
                    approved_by=actor,
                    gmail_draft_sender=sender,
                )
            except (KeyError, ValueError, IndexError) as exc:
                return {"ok": False, "error": str(exc)}
            return {
                "ok": True,
                "action": "gmail_draft_staged",
                "item_id": item.id,
                "batch_id": batch_id,
                "draft": row.get("draft"),
            }

        if src_type == "revenue_draft":
            company = str(item.source.get("company_name") or item.company_name or "")
            draft = find_latest_email_draft(company) if company else None
            if not draft:
                return {"ok": False, "error": "No email draft found for company."}
            if email_processor is None:
                return {"ok": False, "error": "Email processor unavailable."}
            from brain_os.interfaces.email_processor_draft_sender import GmailDraftSender
            from brain_os.services.revenue_mode import set_pilot_decision, upsert_pipeline_status

            to_addr = (to_address or str(draft.get("to") or "")).strip()
            if not to_addr or "@" not in to_addr:
                return {
                    "ok": False,
                    "error": "Recipient email required — pass to_address on decide or run ira revenue gmail-draft.",
                    "needs_to": True,
                    "company_name": company,
                }
            sender = GmailDraftSender(email_processor=email_processor)
            gmail_draft = await sender.create_draft(
                to=to_addr,
                subject=str(draft.get("subject") or ""),
                body=str(draft.get("email_body") or ""),
            )
            set_pilot_decision(company=company, decision="approved_to_send")
            upsert_pipeline_status(company=company, status="approved_to_send")
            self._dismiss(item.id, days=3650)
            return {
                "ok": True,
                "action": "gmail_draft_staged",
                "item_id": item.id,
                "company_name": company,
                "gmail_draft": gmail_draft,
            }

        if src_type == "tinder":
            if tinder_service is None or email_processor is None:
                return {"ok": False, "error": "Tinder or email processor unavailable."}
            staged = await tinder_service.stage_pending_draft_to_gmail(
                email_processor=email_processor,
            )
            if not staged.get("ok"):
                return staged
            self._dismiss(item.id, days=3650)
            return {**staged, "action": "gmail_draft_staged", "item_id": item.id}

        return {"ok": False, "error": f"Unsupported source type: {src_type}"}

    async def _reject_item(
        self,
        item: OperatorInboxItem,
        *,
        actor: str,
        outbound_approvals: Any | None,
        tinder_service: Any | None,
    ) -> dict[str, Any]:
        src_type = str(item.source.get("type") or "")

        if item.kind == "quote_draft" or item.kind == "lead_review":
            self._dismiss(item.id, days=30)
            return {"ok": True, "action": "dismissed", "item_id": item.id}

        if src_type == "batch":
            batch_id = str(item.source.get("batch_id") or "")
            index = int(item.source.get("index") or 0)
            if outbound_approvals is not None:
                try:
                    await outbound_approvals.skip_message(
                        batch_id=batch_id,
                        index=index,
                        skipped_by=actor,
                    )
                except (KeyError, ValueError, IndexError) as exc:
                    return {"ok": False, "error": str(exc)}
            self._dismiss(item.id, days=3650)
            return {"ok": True, "action": "batch_message_skipped", "item_id": item.id}

        if src_type == "revenue_draft":
            from brain_os.services.revenue_mode import set_pilot_decision

            company = str(item.source.get("company_name") or item.company_name or "")
            if company:
                set_pilot_decision(company=company, decision="rejected", reason="operator_inbox")
            self._dismiss(item.id, days=3650)
            return {"ok": True, "action": "pilot_rejected", "item_id": item.id}

        if src_type == "tinder":
            if tinder_service is not None:
                tinder_service.left(note="operator_inbox_reject")
            self._dismiss(item.id, days=3650)
            return {"ok": True, "action": "tinder_skipped", "item_id": item.id}

        self._dismiss(item.id, days=30)
        return {"ok": True, "action": "dismissed", "item_id": item.id}


def operator_release_blocks_external_work() -> bool:
    """True when heartbeat/query jobs should wait for operator release."""
    if not get_settings().app.operator_release_required:
        return False
    return not OperatorInboxService().is_released()


_EXTERNAL_TASK_AGENTS = frozenset(
    {"calliope", "prometheus", "hermes", "argus", "na_sales"},
)
_EXTERNAL_TASK_KEYWORDS = (
    "outbound",
    "email",
    "gmail",
    "outreach",
    "follow-up",
    "follow up",
    "send ",
    "draft reply",
    "cold email",
)


def phase_requires_operator_release(phase: Any) -> bool:
    """Heuristic: phase likely produces external-visible work (email, CRM outreach)."""
    agent = str(getattr(phase, "agent", "") or "").strip().lower()
    if agent in _EXTERNAL_TASK_AGENTS:
        return True
    blob = f"{getattr(phase, 'title', '')} {getattr(phase, 'description', '')}".lower()
    return any(k in blob for k in _EXTERNAL_TASK_KEYWORDS)


def require_operator_release_for_task_phase(phase: Any) -> str | None:
    """Return skip reason when standing task phase needs operator release first."""
    if not operator_release_blocks_external_work():
        return None
    if phase_requires_operator_release(phase):
        return "operator_release_required"
    return None


def require_operator_release_for_heartbeat(*, job: dict[str, Any]) -> str | None:
    """Return skip reason if job must not run without release; else None."""
    if not operator_release_blocks_external_work():
        return None
    action = str(job.get("action") or "").strip().lower()
    if action in {"enqueue_pending_memory", "agency_morning_digest"}:
        return None
    if action:
        return None
    query = str(job.get("query") or "").strip()
    if not query:
        return None
    tags = job.get("requires_operator_release")
    if tags is False:
        return None
    return "operator_release_required"
