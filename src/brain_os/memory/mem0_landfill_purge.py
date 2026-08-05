"""Bulk purge of Mem0 landfill: ingest-log spam + stale takeout trivia.

Pass 1 — ingest-log: content/metadata match ``ingested source`` (~10–14k).
Pass 2 — takeout trivia older than 2022 with zero hits in ``mem0_access.db``
(conservative: when in doubt keep).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

from brain_os.memory.store_policy import is_ingest_log_memory

logger = logging.getLogger(__name__)

_TAKEOUT_SRC_RX = re.compile(
    r"(takeout|mbox|data/takeout_ingest)",
    re.IGNORECASE,
)
_MAX_PAGES = 500
_PAGE_SIZE = 100


def _parse_created(memory: dict[str, Any]) -> datetime | None:
    created_at = memory.get("created_at", "")
    if not created_at:
        # Some Mem0 rows embed age in metadata only — treat as unknown → keep.
        return None
    try:
        if isinstance(created_at, str):
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        else:
            dt = created_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return None


def _memory_text(memory: dict[str, Any]) -> str:
    return str(memory.get("memory") or memory.get("content") or "")


def _is_takeout_sourced(memory: dict[str, Any]) -> bool:
    meta = memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
    src = str(meta.get("source") or meta.get("source_path") or "")
    text = _memory_text(memory)
    blob = f"{src} {text[:200]}"
    return bool(_TAKEOUT_SRC_RX.search(blob))


def is_stale_takeout_trivia(
    memory: dict[str, Any],
    *,
    access_count: int,
    cutoff_year: int = 2022,
) -> tuple[bool, str]:
    """Conservative candidate for pass-2 purge. When in doubt → keep."""
    if access_count > 0:
        return False, "accessed"
    meta = memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
    mem_type = str(meta.get("type") or "").lower()
    cat = str(meta.get("memory_category") or "").lower()
    if mem_type == "correction" or cat == "correction":
        return False, "correction"
    if mem_type == "preference" or cat == "preference":
        return False, "preference"
    if cat == "commercial_fact" or mem_type == "commercial_fact":
        return False, "commercial_fact"
    if meta.get("verified") is True:
        return False, "verified"
    try:
        if float(meta.get("confidence", 0)) >= 0.85:
            return False, "high_confidence"
    except (TypeError, ValueError):
        pass

    created = _parse_created(memory)
    if created is None:
        return False, "unknown_created_at"
    if created.year >= cutoff_year:
        return False, "too_recent_year"

    if not _is_takeout_sourced(memory):
        # Only purge clear takeout-sourced trivia older than cutoff.
        return False, "not_takeout"

    text = _memory_text(memory)
    # Keep anything that looks commercially sticky even if old.
    if re.search(r"\b(DEMO|quote|USD|INR|€|₹|warranty|lead\s*time)\b", text, re.I):
        return False, "commercial_marker"

    return True, "stale_takeout_zero_access"


async def iter_all_memories(
    long_term: Any,
    *,
    user_id: str = "global",
    max_pages: int = _MAX_PAGES,
    page_size: int = _PAGE_SIZE,
) -> list[dict[str, Any]]:
    """Page through hosted Mem0 for *user_id*."""
    out: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        rows = await long_term.list_memories(user_id, page=page, page_size=page_size)
        if not rows:
            break
        out.extend(rows)
        if len(rows) < page_size:
            break
    return out


async def plan_ingest_log_purge(
    long_term: Any,
    *,
    user_id: str = "global",
) -> dict[str, Any]:
    memories = await iter_all_memories(long_term, user_id=user_id)
    matches = [m for m in memories if is_ingest_log_memory(m)]
    samples = [
        {
            "id": m.get("id"),
            "memory": _memory_text(m)[:160],
            "created_at": m.get("created_at"),
        }
        for m in matches[:50]
    ]
    return {
        "pass": "ingest_log",
        "corpus_scanned": len(memories),
        "match_count": len(matches),
        "ids": [str(m.get("id")) for m in matches if m.get("id")],
        "samples": samples,
    }


async def plan_takeout_trivia_purge(
    long_term: Any,
    *,
    user_id: str = "global",
    cutoff_year: int = 2022,
) -> dict[str, Any]:
    from brain_os.memory.mem0_access_tracker import get_mem0_access_tracker

    memories = await iter_all_memories(long_term, user_id=user_id)
    tracker = get_mem0_access_tracker()
    ids = [str(m.get("id", "")) for m in memories]
    counts = await tracker.get_counts(ids) if ids else {}

    matches: list[dict[str, Any]] = []
    kept_reasons: dict[str, int] = {}
    for m in memories:
        mid = str(m.get("id", ""))
        ok, reason = is_stale_takeout_trivia(
            m, access_count=int(counts.get(mid, 0)), cutoff_year=cutoff_year
        )
        if ok:
            matches.append(m)
        else:
            kept_reasons[reason] = kept_reasons.get(reason, 0) + 1

    samples = [
        {
            "id": m.get("id"),
            "memory": _memory_text(m)[:160],
            "created_at": m.get("created_at"),
            "source": (m.get("metadata") or {}).get("source", "")
            if isinstance(m.get("metadata"), dict)
            else "",
        }
        for m in matches[:50]
    ]
    return {
        "pass": "takeout_trivia",
        "corpus_scanned": len(memories),
        "match_count": len(matches),
        "ids": [str(m.get("id")) for m in matches if m.get("id")],
        "samples": samples,
        "kept_reason_counts": kept_reasons,
        "cutoff_year": cutoff_year,
    }


async def execute_purge(
    long_term: Any,
    ids: list[str],
    *,
    pause_every: int = 40,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Delete the given memory ids from hosted Mem0."""
    import asyncio

    attempted = 0
    deleted = 0
    failed = 0
    total = len([i for i in ids if i])
    for mid in ids:
        if not mid:
            continue
        attempted += 1
        ok = False
        for attempt in range(max_retries):
            try:
                ok = await long_term.delete_memory(mid)
                if ok:
                    break
            except Exception:
                logger.warning(
                    "Mem0 purge delete failed id=%s attempt=%s",
                    mid,
                    attempt + 1,
                    exc_info=True,
                )
                ok = False
            if attempt + 1 < max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))
        if ok:
            deleted += 1
        else:
            failed += 1
        if pause_every and attempted % pause_every == 0:
            logger.info(
                "Mem0 purge progress attempted=%s/%s deleted=%s failed=%s",
                attempted,
                total,
                deleted,
                failed,
            )
            await asyncio.sleep(0.2)

    from brain_os.memory.mem0_access_tracker import get_mem0_access_tracker
    from brain_os.memory.mem0_search_cache import get_mem0_search_cache

    if deleted:
        try:
            await get_mem0_access_tracker().forget_ids(ids)
        except Exception:
            logger.debug("access tracker clear after purge failed", exc_info=True)
        try:
            await get_mem0_search_cache().bump_generation("global")
        except Exception:
            logger.debug("search cache bump after purge failed", exc_info=True)

    status = "ok" if failed == 0 else "partial_error"
    return {
        "status": status,
        "attempted": attempted,
        "written": deleted,  # deleted count (COUNTS-ONLY-OK shape)
        "deleted": deleted,
        "failed": failed,
    }


async def run_landfill_purge(
    *,
    pass_name: str = "ingest_log",
    execute: bool = False,
    user_id: str = "global",
    long_term: Any | None = None,
    cutoff_year: int = 2022,
) -> dict[str, Any]:
    """Dry-run (default) or execute one landfill purge pass."""
    from brain_os.memory.long_term import LongTermMemory

    ltm = long_term or LongTermMemory()
    if pass_name == "ingest_log":
        plan = await plan_ingest_log_purge(ltm, user_id=user_id)
    elif pass_name in {"takeout_trivia", "trivia"}:
        plan = await plan_takeout_trivia_purge(ltm, user_id=user_id, cutoff_year=cutoff_year)
    else:
        return {"status": "error", "reason": f"unknown_pass:{pass_name}"}

    out: dict[str, Any] = {
        "status": "ok",
        "dry_run": not execute,
        **plan,
    }
    if execute:
        purge = await execute_purge(ltm, list(plan.get("ids") or []))
        out["purge"] = purge
        out["status"] = purge.get("status", "ok")
        out["dry_run"] = False
    return out
