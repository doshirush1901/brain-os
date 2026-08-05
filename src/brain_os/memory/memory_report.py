"""Operator observability: one-screen memory subsystem report.

Powers ``brain memory report`` — counts, 7d growth proxies, access distribution,
archive/deletion stats, and top-10 most-accessed memories.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from brain_os.config import get_settings
from brain_os.memory.forgetting_exclusions import exclusion_policy_doc
from brain_os.memory.mem0_access_tracker import get_mem0_access_tracker
from brain_os.memory.mem0_archive import get_mem0_archive_store
from brain_os.memory.mem0_forgetting_ops import build_mem0_forgetting_snapshot

logger = logging.getLogger(__name__)


async def _count_mem0_live(long_term: Any | None) -> dict[str, Any]:
    if long_term is None:
        return {"status": "skipped", "reason": "no_long_term", "count": None}
    try:
        tracker = get_mem0_access_tracker()
        user_ids = set(await tracker.known_user_ids())
        user_ids.add("global")
        total = 0
        pages = 0
        for user_id in sorted(user_ids):
            for page in range(1, 11):
                rows = await long_term.list_memories(user_id, page=page, page_size=100)
                if not rows:
                    break
                total += len(rows)
                pages += 1
                if len(rows) < 100:
                    break
        return {
            "status": "ok",
            "count": total,
            "users_swept": len(user_ids),
            "pages_scanned": pages,
        }
    except Exception as exc:
        logger.debug("mem0 live count failed: %s", exc, exc_info=True)
        return {"status": "error", "reason": str(exc)[:200], "count": None}


async def _pending_count() -> int:
    try:
        from brain_os.memory.pending_memory_queue import PendingMemoryQueue

        q = PendingMemoryQueue()
        await q.initialize()
        return await q.count_pending()
    except Exception:
        return 0


async def _procedural_count() -> int:
    try:
        from brain_os.memory.procedural import ProceduralMemory

        pm = ProceduralMemory()
        await pm.initialize()
        try:
            return await pm.count_procedures()
        finally:
            await pm.close()
    except Exception:
        return 0


def _correction_ledger_count() -> int:
    try:
        import json

        from brain_os.systems.data_dir_lock import get_data_dir

        path = get_data_dir() / "brain" / "correction_ledger.json"
        if not path.is_file():
            return 0
        raw = json.loads(path.read_text(encoding="utf-8"))
        entities = raw.get("entities") or {}
        return len(entities) if isinstance(entities, dict) else 0
    except Exception:
        return 0


async def build_memory_report(
    *,
    long_term: Any | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Assemble the memory observability report (JSON-serialisable)."""
    app = get_settings().app
    now = datetime.now(UTC)
    since_7d = (now - timedelta(days=7)).isoformat()

    tracker = get_mem0_access_tracker()
    archive = get_mem0_archive_store()

    if long_term is None:
        try:
            if get_settings().memory.api_key.get_secret_value():
                from brain_os.memory.long_term import LongTermMemory

                long_term = LongTermMemory()
        except Exception:
            long_term = None

    mem0_live = await _count_mem0_live(long_term)
    access_dist = await tracker.access_distribution()
    top = await tracker.top_accessed(limit=top_n)
    archive_count = await archive.count()
    archived_7d = await archive.count_archived_since(since_iso=since_7d)
    active_7d = await tracker.hits_since(since_iso=since_7d)

    forget_snap = build_mem0_forgetting_snapshot(days=7)

    # Never-accessed % relative to live Mem0 when available.
    live_count = mem0_live.get("count")
    never_pct: float | None = None
    if isinstance(live_count, int) and live_count > 0:
        tracked = int(access_dist.get("tracked_ids") or 0)
        accessed = int(access_dist.get("accessed_tracked") or 0)
        # IDs in live store with no tracker row ≈ never accessed.
        never_live = max(0, live_count - accessed)
        never_pct = round(100.0 * never_live / live_count, 2)
        access_dist = {
            **access_dist,
            "live_count": live_count,
            "approx_never_accessed_live": never_live,
            "pct_never_accessed_live": never_pct,
            "tracked_vs_live": tracked,
        }

    pending = await _pending_count()
    procedural = await _procedural_count()
    corrections = _correction_ledger_count()

    return {
        "computed_at": now.isoformat(),
        "subsystems": {
            "mem0_live": mem0_live,
            "mem0_archive": {"count": archive_count},
            "access_tracker": access_dist,
            "pending_queue": {"count": pending},
            "procedural": {"count": procedural},
            "correction_ledger": {"count": corrections},
        },
        "growth_7d": {
            "archived": archived_7d,
            "access_rows_touched": active_7d,
            "forgetting_events": forget_snap.get("events_in_window", 0),
            "forgetting_deleted": forget_snap.get("deleted_in_window", 0),
            "forgetting_archived_or_dry": forget_snap.get("candidates_logged", 0),
        },
        "access_distribution": {
            "pct_never_accessed_tracked": access_dist.get("pct_never_accessed_tracked"),
            "pct_never_accessed_live": never_pct,
            "accessed_tracked": access_dist.get("accessed_tracked"),
            "tracked_ids": access_dist.get("tracked_ids"),
        },
        "archive_deletion": {
            "archive_count": archive_count,
            "archived_last_7d": archived_7d,
            "hard_delete_after_days": int(getattr(app, "mem0_forget_hard_delete_after_days", 30)),
            "forgetting_enabled": bool(getattr(app, "dream_mem0_forgetting_enabled", True)),
            "recent_by_reason": forget_snap.get("by_reason", {}),
            "deleted_in_window": forget_snap.get("deleted_in_window", 0),
        },
        "top_accessed": top,
        "exclusion_policy": exclusion_policy_doc(),
        "policy": {
            "dream_mem0_forgetting_enabled": bool(
                getattr(app, "dream_mem0_forgetting_enabled", True)
            ),
            "mem0_forget_min_age_days": int(getattr(app, "mem0_forget_min_age_days", 120)),
            "mem0_forget_hard_delete_after_days": int(
                getattr(app, "mem0_forget_hard_delete_after_days", 30)
            ),
            "retriever_access_boost_enabled": bool(
                getattr(app, "retriever_access_boost_enabled", True)
            ),
            "retriever_access_boost_max_influence": float(
                getattr(app, "retriever_access_boost_max_influence", 0.12)
            ),
        },
    }


def format_memory_report_text(report: dict[str, Any]) -> str:
    """Human-readable one-screen report."""
    subs = report.get("subsystems") or {}
    growth = report.get("growth_7d") or {}
    access = report.get("access_distribution") or {}
    arch = report.get("archive_deletion") or {}
    policy = report.get("policy") or {}
    lines = [
        "Brain OS memory report",
        f"computed_at: {report.get('computed_at')}",
        "",
        "— Subsystem counts —",
        f"  mem0_live:        {_fmt_count(subs.get('mem0_live'))}",
        f"  mem0_archive:     {(subs.get('mem0_archive') or {}).get('count', 0)}",
        f"  access_tracker:   {(subs.get('access_tracker') or {}).get('tracked_ids', 0)} tracked",
        f"  pending_queue:    {(subs.get('pending_queue') or {}).get('count', 0)}",
        f"  procedural:       {(subs.get('procedural') or {}).get('count', 0)}",
        f"  correction_ledger:{(subs.get('correction_ledger') or {}).get('count', 0)}",
        "",
        "— Growth (7d) —",
        f"  archived:              {growth.get('archived', 0)}",
        f"  access rows touched:   {growth.get('access_rows_touched', 0)}",
        f"  forgetting events:     {growth.get('forgetting_events', 0)}",
        f"  hard-deletes logged:   {growth.get('forgetting_deleted', 0)}",
        "",
        "— Access distribution —",
        f"  % never-accessed (tracked): {access.get('pct_never_accessed_tracked')}",
        f"  % never-accessed (live≈):   {access.get('pct_never_accessed_live')}",
        f"  accessed_tracked:           {access.get('accessed_tracked')}",
        "",
        "— Archive / deletion —",
        f"  archive_count:     {arch.get('archive_count', 0)}",
        f"  archived_last_7d:  {arch.get('archived_last_7d', 0)}",
        f"  hard_delete_after: {arch.get('hard_delete_after_days')}d",
        f"  forgetting_on:     {arch.get('forgetting_enabled')}",
        f"  recent reasons:    {arch.get('recent_by_reason')}",
        "",
        "— Policy —",
        "  access_boost: max_influence "
        f"{policy.get('retriever_access_boost_max_influence')} "
        f"enabled {policy.get('retriever_access_boost_enabled')}",
        "",
        "— Top accessed —",
    ]
    top = report.get("top_accessed") or []
    if not top:
        lines.append("  (none)")
    else:
        for i, row in enumerate(top, 1):
            lines.append(
                f"  {i:2d}. hits={row.get('hits')}  "
                f"id={row.get('memory_id')}  last={row.get('last_hit')}"
            )
    return "\n".join(lines)


def _fmt_count(block: Any) -> str:
    if not isinstance(block, dict):
        return "n/a"
    if block.get("count") is None:
        return f"n/a ({block.get('status')}: {block.get('reason', '')})"
    return str(block.get("count"))
