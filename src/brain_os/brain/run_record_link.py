"""Cross-link pipeline run records to revenue drafts, email send, and feedback."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from brain_os.brain.run_record_access import fetch_run_record, run_records_enabled
from brain_os.brain.run_record_store import RunRecordStore, build_run_record_store
from brain_os.exceptions import BrainOSError
from brain_os.schemas.run_record import RunRecord

logger = logging.getLogger(__name__)


def run_record_api_path(run_id: str) -> str:
    rid = (run_id or "").strip()
    return f"/api/runs/{rid}" if rid else ""


def envelope_with_run_id(envelope: dict[str, Any], run_id: str | None) -> dict[str, Any]:
    rid = (run_id or "").strip()
    if not rid:
        return envelope
    out = dict(envelope)
    out["pipeline_run_id"] = rid
    return out


async def append_run_link(
    run_id: str | None,
    *,
    kind: str,
    detail: dict[str, Any] | None = None,
    store: RunRecordStore | None = None,
) -> None:
    """Append an artifact link on an existing run record (fail-open)."""
    rid = (run_id or "").strip()
    if not rid or not run_records_enabled():
        return
    try:
        st = store or build_run_record_store()
        record = await st.get(rid)
        if record is None:
            return
        links = list(record.artifacts.links or [])
        links.append({"kind": kind, "ts": time.time(), **(detail or {})})
        record.artifacts.links = links
        await st.save(record)
    except (BrainOSError, OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.debug("append_run_link failed run_id=%s kind=%s", rid, kind, exc_info=True)


def load_latest_draft_run_id(company: str) -> str | None:
    """Best-effort ``pipeline_run_id`` from latest revenue/persuasion draft packet."""
    name = (company or "").strip()
    if not name:
        return None
    try:
        from brain_os.services import revenue_mode as _rm

        slug_prefix = _rm.campaign_slug(name)
        for directory, pattern in (
            (_rm.draft_packets_dir(), f"draft_{slug_prefix}_*.json"),
            (_rm.REVENUE_DIR / "persuasion_packets", f"persuasion_{slug_prefix}_*.json"),
        ):
            d = Path(directory)
            if not d.exists():
                continue
            hits = sorted(d.glob(pattern), reverse=True)
            for path in hits[:5]:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                rid = str(data.get("pipeline_run_id") or "").strip()
                if rid:
                    return rid
    except (ImportError, OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.debug("load_latest_draft_run_id failed company=%s", name[:80], exc_info=True)
    return None


async def resolve_feedback_from_run(
    run_id: str | None,
    *,
    previous_query: str,
    previous_response: str,
) -> tuple[str, str, list[str]]:
    """Fill missing query/response text and agents from a persisted run record."""
    rid = (run_id or "").strip()
    if not rid:
        return previous_query, previous_response, []
    record: RunRecord | None = await fetch_run_record(rid)
    if record is None:
        return previous_query, previous_response, []
    pq = previous_query.strip() or record.input_summary
    pr = previous_response.strip() or record.response_summary
    return pq, pr, list(record.agents)
