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
from brain_os.contracts.draft_sender import DraftSender
from brain_os.data.quotes import QuoteStatus
from brain_os.schemas.operator_inbox import (
    OperatorInboxDecision,
    OperatorInboxItem,
    OperatorInboxPayload,
    OperatorInboxSummary,
    OperatorReleaseSession,
)
from brain_os.services.agency.agency_collector import collect_board_candidates
from brain_os.services.operator_approval_sqlite import (
    age_badge,
    get_pending_item,
    log_decision,
    remove_pending,
    replace_pending_items,
)
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


def _annotate_item(item: OperatorInboxItem) -> OperatorInboxItem:
    """Stamp producer + age badge for the unified surface."""
    producer = (item.producer or str(item.source.get("type") or item.kind) or "").strip()
    badge = item.age_badge or age_badge(item.created_at)
    return item.model_copy(update={"producer": producer, "age_badge": badge})


def _sort_newest_first(items: list[OperatorInboxItem]) -> list[OperatorInboxItem]:
    return sorted(items, key=lambda i: str(i.created_at or ""), reverse=True)


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
        created_at: str | None = None,
    ) -> dict[str, Any]:
        telemetry = log_decision(
            item_id=item_id,
            verdict=decision,
            created_at=created_at,
            actor=actor,
            result=result,
        )
        line = {
            "item_id": item_id,
            "decision": decision,
            "actor": actor,
            "at": telemetry.get("at") or _utcnow_iso(),
            "latency_from_created": telemetry.get("latency_from_created"),
            "result": result,
        }
        with _audit_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=True) + "\n")
        return telemetry

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

    def _collect_overnight_pack_items(self, *, limit: int) -> list[OperatorInboxItem]:
        """Recent global_pf1 overnight angle packs awaiting human review."""
        items: list[OperatorInboxItem] = []
        root = get_data_dir() / "experiments" / "global_pf1_overnight"
        if not root.is_dir():
            # Repo-relative fallback when BRAIN_DATA_DIR is the workspace data/
            alt = (
                Path(__file__).resolve().parents[3]
                / "data"
                / "experiments"
                / "global_pf1_overnight"
            )
            root = alt if alt.is_dir() else root
        if not root.is_dir():
            return items
        pack_files: list[Path] = []
        try:
            stamp_dirs = sorted(
                [p for p in root.iterdir() if p.is_dir()],
                key=lambda p: p.name,
                reverse=True,
            )[:5]
            for stamp_dir in stamp_dirs:
                packs_dir = stamp_dir / "packs"
                if not packs_dir.is_dir():
                    continue
                pack_files.extend(
                    sorted(packs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                )
        except OSError:
            return items
        for path in pack_files:
            if len(items) >= limit:
                break
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            company = str(raw.get("company") or raw.get("domain") or path.stem).strip()
            domain = str(raw.get("domain") or "").strip()
            item_id = f"overnight:{path.parent.parent.name}:{path.stem}"
            if self._is_dismissed(item_id):
                continue
            eng = raw.get("english") if isinstance(raw.get("english"), dict) else {}
            subj = str(eng.get("subject") or "")[:72]
            body = str(eng.get("body") or "")
            send_ready = bool(raw.get("send_ready"))
            items.append(
                OperatorInboxItem(
                    id=item_id,
                    kind="outbound_email",
                    title=f"{company} — overnight pack",
                    subtitle=subj or f"DEMO overnight ({domain})",
                    company_name=company,
                    risk="external_visible",
                    preview={
                        "subject": subj,
                        "body_snippet": _body_snippet(body),
                        "send_ready": send_ready,
                        "status": raw.get("status"),
                        "voice_status_line": raw.get("voice_status_line"),
                        "must_review": True,
                    },
                    source={
                        "type": "overnight_pack",
                        "path": str(path),
                        "domain": domain,
                        "stamp": path.parent.parent.name,
                    },
                    created_at=None,
                )
            )
        return items

    @staticmethod
    def _review_rank_key(item: OperatorInboxItem) -> tuple[int, int, str]:
        """Lower tuple sorts first — VIP/must_act style priority for must-review."""
        src = str(item.source.get("type") or "")
        preview = item.preview or {}
        # 0 = morning / VIP-ish, 1 = must_act reply signals, 2 = overnight send_ready,
        # 3 = other outbound, 4 = quotes/leads, 5 = rest
        if item.kind == "morning_action":
            tier = 0
        elif src == "tinder":
            tier = 1
        elif src == "overnight_pack" and preview.get("send_ready"):
            tier = 2
        elif item.kind == "outbound_email":
            tier = 3
        elif item.kind in {"quote_draft", "lead_review"}:
            tier = 4
        else:
            tier = 5
        external = 0 if item.risk == "external_visible" else 1
        return (tier, external, item.id)

    def _rank_must_review(self, items: list[OperatorInboxItem]) -> list[OperatorInboxItem]:
        ranked = sorted(items, key=self._review_rank_key)
        out: list[OperatorInboxItem] = []
        for idx, item in enumerate(ranked, start=1):
            preview = dict(item.preview or {})
            preview["review_rank"] = idx
            out.append(item.model_copy(update={"preview": preview}))
        return out

    def _collect_revenue_draft_items(self, *, limit: int) -> list[OperatorInboxItem]:
        items: list[OperatorInboxItem] = []
        seen: set[str] = set()
        for row in recent_drafts_by_company(limit=80, include_body=True):
            draft_status = str(row.get("status") or "").strip().lower()
            if draft_status in {"stale", "rejected", "sent", "approved_to_send"}:
                continue
            pst = str(row.get("pilot_status") or "").strip().lower()
            if pst in {"stale", "rejected", "approved_to_send"}:
                continue
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
            campaign = str(row.get("campaign_id") or row.get("source") or "revenue_draft")
            items.append(
                _annotate_item(
                    OperatorInboxItem(
                        id=item_id,
                        kind="outbound_email",
                        title=f"{company} — approve draft",
                        subtitle=subj or "Revenue draft pending review",
                        company_name=company,
                        risk="external_visible",
                        producer=campaign if campaign.startswith("warm") else "revenue_draft",
                        preview={
                            "subject": subj,
                            "body_snippet": _body_snippet(body),
                            "campaign_id": campaign,
                        },
                        source={
                            "type": "revenue_draft",
                            "company_name": company,
                            "campaign_id": campaign,
                        },
                        created_at=str(row.get("timestamp") or "") or None,
                    )
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
        card = st.get("card") or {}
        if isinstance(pending, dict) and pending.get("to"):
            item_id = "tinder:pending"
            if self._is_dismissed(item_id):
                return None
            to_addr = str(pending.get("to") or "")
            subj = str(pending.get("subject") or "")
            body = str(pending.get("body") or "")
            company = str(card.get("company_name") or "")
            return _annotate_item(
                OperatorInboxItem(
                    id=item_id,
                    kind="outbound_email",
                    title=f"Tinder — {company or to_addr}",
                    subtitle=subj[:72],
                    company_name=company,
                    risk="external_visible",
                    producer="tinder",
                    preview={
                        "to": to_addr,
                        "subject": subj,
                        "body_snippet": _body_snippet(body),
                    },
                    source={"type": "tinder"},
                    created_at=str(pending.get("created_at") or st.get("saved_at") or "") or None,
                )
            )
        # Surface current card when enabled but no pending draft (frozen checkpoint).
        if not st.get("enabled"):
            return None
        if not isinstance(card, dict) or card.get("empty") or not card.get("recipient"):
            return None
        item_id = "tinder:needs_draft"
        if self._is_dismissed(item_id):
            return None
        company = str(card.get("company_name") or "")
        to_addr = str(card.get("recipient") or "")
        return _annotate_item(
            OperatorInboxItem(
                id=item_id,
                kind="outbound_email",
                title=f"Tinder card — {company or to_addr}",
                subtitle="Needs draft (checkpoint unfrozen / no pending_draft)",
                company_name=company,
                risk="external_visible",
                producer="tinder",
                preview={
                    "to": to_addr,
                    "cursor": st.get("cursor"),
                    "queue_length": st.get("queue_length") or st.get("queue_len"),
                    "history_summary": str(card.get("history_summary") or "")[:240],
                },
                source={"type": "tinder_needs_draft", "recipient": to_addr},
                created_at=str(st.get("saved_at") or "") or None,
            )
        )

    def _collect_proactive_items(self, *, limit: int) -> list[OperatorInboxItem]:
        from brain_os.services.proactive_inbox_queue import read_proactive_inbox_items

        items: list[OperatorInboxItem] = []
        for raw in read_proactive_inbox_items(limit=limit * 2):
            if self._is_dismissed(raw.id):
                continue
            # Normalize orphan kinds into schema-safe values
            kind = raw.kind
            if kind not in {
                "outbound_email",
                "lead_review",
                "quote_draft",
                "aftermarket_trigger",
                "morning_action",
                "proactive_watch",
                "onshoring_pack",
                "agent_steering",
            }:
                kind = "proactive_watch"
            items.append(
                _annotate_item(
                    raw.model_copy(
                        update={
                            "kind": kind,  # type: ignore[arg-type]
                            "producer": str(raw.source.get("type") or kind),
                        }
                    )
                )
            )
            if len(items) >= limit:
                break
        return items

    def _collect_onshoring_items(self, *, limit: int) -> list[OperatorInboxItem]:
        from brain_os.services.operator_approval_hygiene import iter_onshoring_pending

        items: list[OperatorInboxItem] = []
        for row in iter_onshoring_pending(limit=limit):
            slug = str(row.get("slug") or "")
            item_id = f"onshoring:{slug}"
            if self._is_dismissed(item_id):
                continue
            company = str(row.get("company") or slug)
            items.append(
                _annotate_item(
                    OperatorInboxItem(
                        id=item_id,
                        kind="onshoring_pack",
                        title=f"Onshoring — {company}",
                        subtitle=str(row.get("subject") or row.get("review_status") or "")[:72],
                        company_name=company,
                        risk="external_visible",
                        producer="onshoring",
                        preview={
                            "subject": row.get("subject"),
                            "body_snippet": _body_snippet(str(row.get("body") or "")),
                            "contact_email": row.get("contact_email"),
                            "review_status": row.get("review_status"),
                            "campaign_id": row.get("campaign_id"),
                        },
                        source={
                            "type": "onshoring_pack",
                            "slug": slug,
                            "path": row.get("path"),
                            "contact_email": row.get("contact_email"),
                        },
                        created_at=str(row.get("generated_at") or "") or None,
                    )
                )
            )
        return items

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

    def _collect_aftermarket_trigger_items(self, *, limit: int) -> list[OperatorInboxItem]:
        from brain_os.services.installed_base.trigger_store import pending_operator_triggers

        items: list[OperatorInboxItem] = []
        for row in pending_operator_triggers()[:limit]:
            trigger_id = str(row.get("trigger_id") or "")
            if not trigger_id:
                continue
            item_id = f"ibtrigger:{trigger_id}"
            if self._is_dismissed(item_id):
                continue
            company = str(row.get("company_name") or "")
            trigger_type = str(row.get("trigger_type") or "trigger")
            model = str(row.get("model") or "")
            items.append(
                OperatorInboxItem(
                    id=item_id,
                    kind="aftermarket_trigger",
                    title=f"{company or 'Fleet'} — {trigger_type}",
                    subtitle=f"{model} · {row.get('detail', '')[:100]}",
                    company_name=company,
                    risk="internal",
                    preview={
                        "trigger_id": trigger_id,
                        "trigger_type": trigger_type,
                        "model": model,
                        "detail": str(row.get("detail") or ""),
                        "suggested_play": str(row.get("suggested_play") or ""),
                        "draft_cta": str(row.get("draft_cta") or ""),
                        "evidence": row.get("evidence") or [],
                        "campaign_id": str(row.get("campaign_id") or ""),
                    },
                    source={
                        "type": "aftermarket_trigger",
                        "trigger_id": trigger_id,
                        "asset_id": str(row.get("asset_id") or ""),
                    },
                    created_at=str(row.get("fired_at") or ""),
                )
            )
        return items

    def _collect_morning_brain_items(self, *, limit: int) -> list[OperatorInboxItem]:
        from brain_os.services.morning_brain import _inbox_snapshot_path, _local_date_iso

        path = _inbox_snapshot_path()
        if not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if str(raw.get("local_date") or "") != _local_date_iso():
            return []
        items: list[OperatorInboxItem] = []
        for row in raw.get("items") or []:
            if not isinstance(row, dict):
                continue
            item_id = str(row.get("id") or "")
            if not item_id or self._is_dismissed(item_id):
                continue
            try:
                items.append(OperatorInboxItem.model_validate(row))
            except Exception:
                continue
            if len(items) >= limit:
                break
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
        persist_sqlite: bool = True,
        include_quotes: bool = True,
    ) -> OperatorInboxPayload:
        outbound: list[OperatorInboxItem] = []
        if outbound_approvals is not None:
            outbound.extend(
                [
                    _annotate_item(i)
                    for i in self._collect_outbound_batch_items(
                        outbound_approvals, limit=limit_per_kind
                    )
                ]
            )
        outbound.extend(self._collect_revenue_draft_items(limit=limit_per_kind))
        outbound.extend(
            [_annotate_item(i) for i in self._collect_overnight_pack_items(limit=limit_per_kind)]
        )
        onshoring = self._collect_onshoring_items(limit=min(20, limit_per_kind))
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

        leads = [_annotate_item(i) for i in self._collect_lead_review_items(limit=limit_per_kind)]
        quotes_items: list[OperatorInboxItem] = []
        if include_quotes and quotes is not None:
            quotes_items = [
                _annotate_item(i)
                for i in await self._collect_quote_items(quotes, limit=limit_per_kind)
            ]
        aftermarket = [
            _annotate_item(i) for i in self._collect_aftermarket_trigger_items(limit=limit_per_kind)
        ]
        morning_items = [
            _annotate_item(i) for i in self._collect_morning_brain_items(limit=limit_per_kind)
        ]
        proactive = self._collect_proactive_items(limit=limit_per_kind)

        all_items = (
            outbound + onshoring + leads + quotes_items + aftermarket + morning_items + proactive
        )
        # Newest-first is the operator surface order; keep must-review rank in preview.
        all_items = self._rank_must_review(all_items)
        all_items = _sort_newest_first(all_items)
        from brain_os.services.math_mode import (
            build_math_promote_checklist,
            enrich_operator_inbox_item_math,
            math_mode_advisory_enabled,
            math_priority_hints_from_shadow,
        )

        if math_mode_advisory_enabled():
            all_items = [enrich_operator_inbox_item_math(i) for i in all_items]
        ext = sum(1 for i in all_items if i.risk == "external_visible")
        producers = sorted(
            {
                str(i.producer or i.source.get("type") or i.kind)
                for i in all_items
                if i.producer or i.kind
            }
        )
        summary = OperatorInboxSummary(
            outbound_email=len(outbound),
            lead_review=len(leads),
            quote_draft=len(quotes_items),
            aftermarket_trigger=len(aftermarket),
            morning_action=len(morning_items),
            proactive_watch=sum(1 for i in proactive if i.kind == "proactive_watch"),
            onshoring_pack=len(onshoring),
            agent_steering=sum(1 for i in proactive if i.kind == "agent_steering"),
            total=len(all_items),
            external_pending=ext,
            producer_types=producers,
        )
        session = self.load_session()
        hints: list[dict[str, Any]] = []
        advisory_on = math_mode_advisory_enabled()
        if advisory_on:
            hints = math_priority_hints_from_shadow(limit=10)
        promote_checklist = build_math_promote_checklist()
        if persist_sqlite:
            try:
                replace_pending_items([i.model_dump() for i in all_items])
            except OSError:
                logger.debug("operator inbox sqlite persist failed", exc_info=True)
        return OperatorInboxPayload(
            summary=summary,
            items=all_items,
            released=session.released,
            released_until=session.released_until,
            math_advisory_enabled=advisory_on,
            math_priority_hints=hints,
            math_promote_checklist=promote_checklist,
        )

    def _resolve_item_from_cache_or_list(
        self,
        item_id: str,
        items: list[OperatorInboxItem],
    ) -> OperatorInboxItem | None:
        cached = get_pending_item(item_id)
        if cached is not None:
            try:
                return OperatorInboxItem.model_validate(cached)
            except Exception:
                pass
        item = next((i for i in items if i.id == item_id), None)
        if item is None:
            for i in items:
                if i.id.lower().startswith(item_id.lower()):
                    return i
        return item

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
        draft_sender: DraftSender | None = None,
        tinder_service: Any | None = None,
        quotes: Any | None = None,
        sqlite_first: bool = True,
        reason: str | None = None,
    ) -> dict[str, Any]:
        item: OperatorInboxItem | None = None
        if sqlite_first:
            item = self._resolve_item_from_cache_or_list(item_id, [])
        if item is None:
            # PG-down safe: skip quotes when not required for this id
            need_quotes = item_id.lower().startswith("quote:")
            inbox = await self.collect_inbox(
                outbound_approvals=outbound_approvals,
                quotes=quotes if need_quotes else None,
                tinder_service=tinder_service,
                include_quotes=need_quotes,
            )
            item = self._resolve_item_from_cache_or_list(item_id, list(inbox.items))
        if item is None:
            return {"ok": False, "error": f"Unknown item_id: {item_id}"}

        if decision == "snooze":
            self._dismiss(item.id, days=snooze_days)
            result = {"ok": True, "action": "snoozed", "item_id": item.id}
            telemetry = self._append_audit(
                item_id=item.id,
                decision=decision,
                actor=actor,
                result=result,
                created_at=item.created_at,
            )
            remove_pending(item.id)
            return {**result, "telemetry": telemetry}

        if decision == "reject":
            result = await self._reject_item(
                item,
                actor=actor,
                outbound_approvals=outbound_approvals,
                tinder_service=tinder_service,
            )
            if reason:
                result["reason"] = reason
            telemetry = self._append_audit(
                item_id=item.id,
                decision=decision,
                actor=actor,
                result=result,
                created_at=item.created_at,
            )
            if result.get("ok"):
                remove_pending(item.id)
            return {**result, "telemetry": telemetry}

        result = await self._approve_item(
            item,
            actor=actor,
            to_address=to_address,
            outbound_approvals=outbound_approvals,
            email_processor=email_processor,
            draft_sender=draft_sender,
            tinder_service=tinder_service,
        )
        telemetry = self._append_audit(
            item_id=item.id,
            decision="approve",
            actor=actor,
            result=result,
            created_at=item.created_at,
        )
        if result.get("ok"):
            remove_pending(item.id)
        return {**result, "telemetry": telemetry}

    async def _approve_item(
        self,
        item: OperatorInboxItem,
        *,
        actor: str,
        to_address: str | None = None,
        outbound_approvals: Any | None = None,
        email_processor: Any | None = None,
        draft_sender: DraftSender | None = None,
        tinder_service: Any | None = None,
    ) -> dict[str, Any]:
        src_type = str(item.source.get("type") or "")
        _ = email_processor  # L3.3c: EmailProcessor DI; staging uses draft_sender only

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

        if item.kind == "aftermarket_trigger":
            from brain_os.services.installed_base.draft_chain import approve_draft_for_send

            trigger_id = str(item.source.get("trigger_id") or item.preview.get("trigger_id") or "")
            if not trigger_id:
                return {"ok": False, "error": "Missing trigger_id on aftermarket inbox item."}
            try:
                from brain_os.data.crm import CRMDatabase

                crm = CRMDatabase()
            except Exception:
                crm = None
            draft_result = await approve_draft_for_send(trigger_id, crm, draft_sender=draft_sender)
            if draft_result.get("status") not in {"ok"}:
                return {
                    "ok": False,
                    "error": draft_result.get("reason") or draft_result.get("status"),
                    "draft_result": draft_result,
                }
            self._dismiss(item.id, days=3650)
            gmail_staged = (draft_result.get("gmail_stage") or {}).get("status") == "ok"
            return {
                "ok": True,
                "action": "aftermarket_gmail_draft_staged"
                if gmail_staged
                else "aftermarket_draft_staged",
                "item_id": item.id,
                "trigger_id": trigger_id,
                "draft_result": draft_result,
                "gmail_draft_id": (draft_result.get("gmail_stage") or {}).get("gmail_draft_id"),
                "message": (
                    "Gmail draft staged — send from Gmail or Tinder after review."
                    if gmail_staged
                    else "Warm draft recorded; Gmail staging skipped (no processor or recipient)."
                ),
            }

        if src_type == "batch":
            if outbound_approvals is None or draft_sender is None:
                return {"ok": False, "error": "Outbound approvals or draft sender unavailable."}
            batch_id = str(item.source.get("batch_id") or "")
            index = int(item.source.get("index") or 0)
            try:
                row = await outbound_approvals.approve_message(
                    batch_id=batch_id,
                    index=index,
                    approved_by=actor,
                    gmail_draft_sender=draft_sender,
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
            if draft_sender is None:
                return {"ok": False, "error": "Draft sender unavailable."}
            from brain_os.services.revenue_mode import set_pilot_decision, upsert_pipeline_status

            to_addr = (to_address or str(draft.get("to") or "")).strip()
            if not to_addr or "@" not in to_addr:
                return {
                    "ok": False,
                    "error": "Recipient email required — pass to_address on decide or run brain revenue gmail-draft.",
                    "needs_to": True,
                    "company_name": company,
                }
            gmail_draft = await draft_sender.create_draft(
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
            if tinder_service is None or draft_sender is None:
                return {"ok": False, "error": "Tinder or draft sender unavailable."}
            staged = await tinder_service.stage_pending_draft_to_gmail(
                draft_sender=draft_sender,
            )
            if not staged.get("ok"):
                return staged
            self._dismiss(item.id, days=3650)
            return {**staged, "action": "gmail_draft_staged", "item_id": item.id}

        if src_type == "tinder_needs_draft":
            self._dismiss(item.id, days=1)
            return {
                "ok": True,
                "action": "tinder_draft_hint",
                "item_id": item.id,
                "message": "Run `brain tinder draft` for the current card, then `brain ok tinder:pending`.",
                "recipient": item.source.get("recipient"),
            }

        if src_type == "onshoring_pack" or item.kind == "onshoring_pack":
            from brain_os.services.operator_approval_hygiene import mark_onshoring_decision

            slug = str(item.source.get("slug") or "")
            marked = mark_onshoring_decision(slug, approved=True, actor=actor)
            if not marked.get("ok"):
                return marked
            # Stage Gmail when draft sender + contact available
            to_addr = (to_address or str(item.source.get("contact_email") or "")).strip()
            body = str((item.preview or {}).get("body_snippet") or "")
            subject = str((item.preview or {}).get("subject") or item.subtitle or "")
            gmail_draft = None
            if draft_sender is not None and to_addr and "@" in to_addr:
                from pathlib import Path

                pack_path = Path(str(item.source.get("path") or ""))
                draft_file = pack_path / "outbound_draft.txt"
                if draft_file.is_file():
                    text = draft_file.read_text(encoding="utf-8")
                    for line in text.splitlines():
                        if line.lower().startswith("subject:"):
                            subject = line.split(":", 1)[1].strip()
                    parts = text.split("\n\n", 1)
                    body = parts[1] if len(parts) > 1 else text
                gmail_draft = await draft_sender.create_draft(
                    to=to_addr, subject=subject, body=body
                )
            self._dismiss(item.id, days=3650)
            return {
                "ok": True,
                "action": "onshoring_approved",
                "item_id": item.id,
                "slug": slug,
                "gmail_draft": gmail_draft,
            }

        if src_type in {"overnight_pack", "morning_brain", "event_reactor"} or item.kind in {
            "morning_action",
            "proactive_watch",
            "agent_steering",
        }:
            self._dismiss(item.id, days=3650)
            return {
                "ok": True,
                "action": "acknowledged",
                "item_id": item.id,
                "message": f"Acknowledged {src_type or item.kind} (no auto-send).",
            }

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

        if item.kind == "aftermarket_trigger":
            from brain_os.services.installed_base.trigger_store import mark_trigger_operator_status

            trigger_id = str(item.source.get("trigger_id") or item.preview.get("trigger_id") or "")
            if trigger_id:
                mark_trigger_operator_status(trigger_id, "rejected")
            self._dismiss(item.id, days=30)
            return {"ok": True, "action": "aftermarket_trigger_rejected", "item_id": item.id}

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

        if src_type == "tinder_needs_draft":
            if tinder_service is not None:
                tinder_service.left(note="operator_inbox_reject_needs_draft")
            self._dismiss(item.id, days=3650)
            return {"ok": True, "action": "tinder_card_skipped", "item_id": item.id}

        if src_type == "onshoring_pack" or item.kind == "onshoring_pack":
            from brain_os.services.operator_approval_hygiene import mark_onshoring_decision

            slug = str(item.source.get("slug") or "")
            mark_onshoring_decision(slug, approved=False, actor=actor)
            self._dismiss(item.id, days=3650)
            return {"ok": True, "action": "onshoring_rejected", "item_id": item.id}

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
