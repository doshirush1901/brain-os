"""Read helpers for persisted pipeline run records."""

from __future__ import annotations

import asyncio
import logging
import time

from brain_os.brain.run_record_store import RunRecordStore
from brain_os.config import get_settings
from brain_os.schemas.run_record import RunRecord, RunRecordSummary

logger = logging.getLogger(__name__)

_STORE: RunRecordStore | None = None


def run_records_enabled() -> bool:
    return bool(get_settings().app.run_record_enabled)


def _get_store() -> RunRecordStore:
    global _STORE
    if _STORE is None:
        _STORE = RunRecordStore()
    return _STORE


def reset_run_record_access_store() -> None:
    """Clear lazy store singleton (tests only)."""
    global _STORE
    _STORE = None


async def fetch_run_record(run_id: str) -> RunRecord | None:
    if not run_records_enabled():
        return None
    try:
        return await _get_store().get(run_id)
    except Exception:  # intentional — SQLite read must not break API/MCP callers
        logger.exception("fetch_run_record failed")
        return None


async def fetch_run_summary(run_id: str) -> RunRecordSummary | None:
    record = await fetch_run_record(run_id)
    return record.summary() if record is not None else None


async def fetch_run_summary_best_effort(
    run_id: str,
    *,
    wait_s: float = 0.35,
    poll_interval_s: float = 0.05,
) -> RunRecordSummary | None:
    """Poll briefly so query/CLI can include summary after background persist."""
    rid = (run_id or "").strip()
    if not rid or not run_records_enabled():
        return None
    deadline = time.time() + max(0.0, float(wait_s))
    while True:
        summary = await fetch_run_summary(rid)
        if summary is not None:
            return summary
        if time.time() >= deadline:
            return None
        await asyncio.sleep(max(0.01, float(poll_interval_s)))


async def list_run_summaries(
    *,
    limit: int = 50,
    channel: str | None = None,
    since_hours: float | None = None,
) -> list[RunRecordSummary]:
    if not run_records_enabled():
        return []
    since_ts: float | None = None
    if since_hours is not None and since_hours > 0:
        since_ts = time.time() - float(since_hours) * 3600.0
    try:
        return await _get_store().list_recent(
            limit=limit,
            channel=channel,
            since_ts=since_ts,
        )
    except Exception:  # intentional — SQLite list must not break API/MCP callers
        logger.exception("list_run_summaries failed")
        return []
