"""Correction ledger audit, verify, and archive (P4 coverage hygiene)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from brain_os.memory.correction_ledger_metrics import (
    _FRESHNESS_DAYS,
    _parse_corrected_at,
    append_ledger_verify_event,
    correction_ledger_path,
    load_ledger_entities,
)

logger = logging.getLogger(__name__)

_REASON_EMPTY = "empty_value"
_REASON_MALFORMED = "malformed_entry"
_REASON_MISSING_DATE = "missing_corrected_at"
_REASON_EXPIRED = "expired_30d"


@dataclass
class LedgerEntityAudit:
    entity_key: str
    reason: str
    suggested_action: str
    corrected_at: str | None
    value_preview: str
    crm_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_full_ledger() -> dict[str, Any]:
    path = correction_ledger_path()
    if not path.is_file():
        return {"entities": {}, "archived_entities": {}, "_metadata": {"last_updated": ""}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("correction ledger read failed")
        return {"entities": {}, "archived_entities": {}, "_metadata": {"last_updated": ""}}
    if not isinstance(data, dict):
        return {"entities": {}, "archived_entities": {}, "_metadata": {"last_updated": ""}}
    data.setdefault("entities", {})
    data.setdefault("archived_entities", {})
    data.setdefault("_metadata", {})
    return data


def save_full_ledger(ledger: dict[str, Any]) -> None:
    path = correction_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger.setdefault("_metadata", {})
    ledger["_metadata"]["last_updated"] = datetime.now(UTC).strftime("%Y-%m-%d")
    path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _value_preview(entry: dict[str, Any], *, limit: int = 80) -> str:
    value = str(entry.get("current_status") or entry.get("correct_value") or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def classify_entity_entry(
    entity_key: str, raw: Any, *, cutoff: datetime
) -> LedgerEntityAudit | None:
    """Return audit row when entity is stale; None when fresh within window."""
    if not isinstance(raw, dict):
        return LedgerEntityAudit(
            entity_key=entity_key,
            reason=_REASON_MALFORMED,
            suggested_action="archive",
            corrected_at=None,
            value_preview="",
        )

    value = str(raw.get("current_status") or raw.get("correct_value") or "").strip()
    if not value:
        return LedgerEntityAudit(
            entity_key=entity_key,
            reason=_REASON_EMPTY,
            suggested_action="archive",
            corrected_at=None,
            value_preview="",
        )

    corrected = _parse_corrected_at(raw.get("corrected_at") or raw.get("effective_from"))
    corrected_s = str(raw.get("corrected_at") or raw.get("effective_from") or "").strip() or None

    if corrected is None:
        return LedgerEntityAudit(
            entity_key=entity_key,
            reason=_REASON_MISSING_DATE,
            suggested_action="verify",
            corrected_at=corrected_s,
            value_preview=_value_preview(raw),
        )

    if corrected < cutoff:
        return LedgerEntityAudit(
            entity_key=entity_key,
            reason=_REASON_EXPIRED,
            suggested_action="verify",
            corrected_at=corrected_s,
            value_preview=_value_preview(raw),
        )

    return None


def audit_stale_entities(*, freshness_days: int = _FRESHNESS_DAYS) -> list[LedgerEntityAudit]:
    """List active ledger entities that fail 30-day freshness coverage."""
    cutoff = datetime.now(UTC) - timedelta(days=freshness_days)
    entities = load_ledger_entities()
    stale: list[LedgerEntityAudit] = []
    for key, raw in entities.items():
        row = classify_entity_entry(str(key), raw, cutoff=cutoff)
        if row is not None:
            stale.append(row)
    return stale


async def enrich_audit_with_crm_hints(
    rows: list[LedgerEntityAudit],
    *,
    crm: Any | None = None,
) -> list[LedgerEntityAudit]:
    """Optional CRM pass: flag entities whose deals are all LOST or missing."""
    if crm is None:
        return rows
    enriched: list[LedgerEntityAudit] = []
    for row in rows:
        hint = await _crm_hint_for_entity(row.entity_key, crm)
        if hint and hint in {"no_crm_deal", "all_deals_lost"} and row.suggested_action == "verify":
            enriched.append(
                LedgerEntityAudit(
                    entity_key=row.entity_key,
                    reason=row.reason,
                    suggested_action="archive",
                    corrected_at=row.corrected_at,
                    value_preview=row.value_preview,
                    crm_hint=hint,
                )
            )
        else:
            enriched.append(
                LedgerEntityAudit(
                    entity_key=row.entity_key,
                    reason=row.reason,
                    suggested_action=row.suggested_action,
                    corrected_at=row.corrected_at,
                    value_preview=row.value_preview,
                    crm_hint=hint,
                )
            )
    return enriched


async def _crm_hint_for_entity(entity_key: str, crm: Any) -> str | None:
    company = (entity_key or "").strip()
    if not company or "@" in company:
        return None
    try:
        if hasattr(crm, "search_deals"):
            found = await crm.search_deals(company_name=company, limit=5)
        elif hasattr(crm, "get_deals_by_company"):
            found = await crm.get_deals_by_company(company)
        else:
            return None
        if not found:
            return "no_crm_deal"
        stages: list[str] = []
        for deal in found:
            if hasattr(deal, "stage"):
                stages.append(str(getattr(deal, "stage", "") or "").upper())
            elif isinstance(deal, dict):
                stages.append(str(deal.get("stage") or "").upper())
        if stages and all(s == "LOST" for s in stages):
            return "all_deals_lost"
        if any(s == "WON" for s in stages):
            return "has_won_deal"
        return "open_pipeline"
    except Exception:
        logger.debug("CRM hint failed for %s", company, exc_info=True)
        return None


def verify_entities(
    entity_keys: list[str],
    *,
    source: str = "operator_verify",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Re-verify entities by bumping ``corrected_at`` (still-active overrides)."""
    ledger = load_full_ledger()
    entities = ledger.get("entities") or {}
    now = datetime.now(UTC).isoformat()
    touched: list[str] = []
    skipped: list[str] = []

    for key in entity_keys:
        raw = entities.get(key)
        if not isinstance(raw, dict):
            skipped.append(key)
            continue
        value = str(raw.get("current_status") or raw.get("correct_value") or "").strip()
        if not value:
            skipped.append(key)
            continue
        if not dry_run:
            raw["corrected_at"] = now
            raw["last_verified_at"] = now
            raw["verify_source"] = source[:64]
            raw["version"] = int(raw.get("version") or 1) + 1
            entities[key] = raw
        touched.append(key)

    if touched and not dry_run:
        ledger["entities"] = entities
        save_full_ledger(ledger)

    return {
        "dry_run": dry_run,
        "touched": touched,
        "skipped": skipped,
        "touched_count": len(touched),
    }


def archive_entities(
    entity_keys: list[str],
    *,
    reason: str = "operator_archive",
    source: str = "operator_archive",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Move entities from active ledger to ``archived_entities``."""
    ledger = load_full_ledger()
    entities = dict(ledger.get("entities") or {})
    archived = dict(ledger.get("archived_entities") or {})
    now = datetime.now(UTC).isoformat()
    moved: list[str] = []
    skipped: list[str] = []

    for key in entity_keys:
        raw = entities.pop(key, None)
        if not isinstance(raw, dict):
            skipped.append(key)
            continue
        if not dry_run:
            raw["archived_at"] = now
            raw["archive_reason"] = reason[:120]
            raw["archive_source"] = source[:64]
            archived[key] = raw
        moved.append(key)

    if moved and not dry_run:
        ledger["entities"] = entities
        ledger["archived_entities"] = archived
        save_full_ledger(ledger)

    return {
        "dry_run": dry_run,
        "archived": moved,
        "skipped": skipped,
        "archived_count": len(moved),
    }


async def run_ledger_hygiene_pass(
    *,
    dry_run: bool = False,
    fix_missing_dates: bool = True,
    crm: Any | None = None,
    source: str = "heartbeat",
) -> dict[str, Any]:
    """Weekly hygiene: audit stale rows; auto-verify only missing-date entities."""
    from brain_os.memory.correction_ledger_metrics import build_correction_ledger_snapshot

    before = await build_correction_ledger_snapshot()
    stale = audit_stale_entities()
    if crm is not None:
        stale = await enrich_audit_with_crm_hints(stale, crm=crm)

    auto_keys = [r.entity_key for r in stale if r.reason == _REASON_MISSING_DATE]
    verify_summary: dict[str, Any] | None = None
    if fix_missing_dates and auto_keys:
        verify_summary = verify_entities(
            auto_keys,
            source=source,
            dry_run=dry_run,
        )

    after = await build_correction_ledger_snapshot()
    append_ledger_verify_event(
        verified=int(after.get("verified_30d") or 0),
        stale=int(after.get("stale_entities") or 0),
        total=int(after.get("total_entities") or 0),
        source=source,
    )

    return {
        "dry_run": dry_run,
        "fix_missing_dates": fix_missing_dates,
        "coverage_before": before.get("coverage_30d"),
        "coverage_after": after.get("coverage_30d"),
        "stale_count": len(stale),
        "stale_entities": [r.to_dict() for r in stale[:50]],
        "auto_verify_missing_date": verify_summary,
        "snapshot_after": after,
    }
