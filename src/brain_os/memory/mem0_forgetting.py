"""Dream stage 5b — two-stage Mem0 forgetting (archive → hard-delete).

Read-time decay only down-ranks old memories; they still occupy the store.
This stage prunes demonstrably dead weight safely:

**Stage 1 — archive (reversible)**
Move low-salience, zero-access, old memories out of live Mem0 into
``data/mem0_archive.db``.

**Stage 2 — hard-delete**
Permanently drop archive rows only after
``APP__MEM0_FORGET_HARD_DELETE_AFTER_DAYS`` (default 30) **and** still never
accessed.

Never-touch exclusion rules live in
:mod:`brain_os.memory.forgetting_exclusions` (Mnemon corrections, preferences,
verified facts, high salience, procedure/goal references, reconsolidation
window, any recorded access).

When ``APP__DREAM_MEM0_FORGETTING_ENABLED`` is false the stage runs dry —
candidates are logged to ``data/operations/mem0_forgetting_events.jsonl``
but nothing is archived or deleted.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain_os.config import get_settings
from brain_os.memory.forgetting_exclusions import (
    check_hard_exclusions,
    exclusion_policy_doc,
    load_goal_anchors,
    load_procedure_anchors,
)
from brain_os.memory.mem0_access_tracker import get_mem0_access_tracker
from brain_os.memory.mem0_archive import get_mem0_archive_store
from brain_os.memory.mem0_search_cache import get_mem0_search_cache
from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)

_TURN_SUMMARY_PREFIX = "conversation turn summary:"
_MAX_PAGES_PER_USER = 10
_PAGE_SIZE = 100


def forgetting_events_path() -> Path:
    """Append-only audit log for Mem0 archive/delete candidates."""
    p = get_data_dir() / "operations" / "mem0_forgetting_events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _events_path() -> Path:
    return forgetting_events_path()


def _append_event(event: dict[str, Any]) -> None:
    try:
        with _events_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True, default=str) + "\n")
    except OSError:
        logger.warning("Failed to append mem0 forgetting event", exc_info=True)


def _memory_age_days(memory: dict[str, Any], now: datetime) -> int | None:
    created_at = memory.get("created_at", "")
    if not created_at:
        return None
    try:
        if isinstance(created_at, str):
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        else:
            dt = created_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return max(0, (now - dt).days)
    except (ValueError, TypeError):
        return None


def is_forgetting_candidate(
    memory: dict[str, Any],
    *,
    access_count: int,
    now: datetime,
    min_age_days: int,
    max_confidence: float,
    salience_protect: float | None = None,
    procedure_anchors: frozenset[str] | set[str] | None = None,
    goal_anchors: frozenset[str] | set[str] | None = None,
) -> tuple[bool, str]:
    """Return ``(candidate, reason)`` for one memory under archive criteria.

    Reasons starting with exclusion codes (``protected_*``, ``accessed``, …)
    mean *not* a candidate. Positive reasons: ``stale_low_confidence``,
    ``turn_summary_backlog``.
    """
    if salience_protect is None:
        try:
            salience_protect = float(
                getattr(get_settings().app, "mem0_forget_salience_protect", 0.35)
            )
        except Exception:
            salience_protect = 0.35

    excluded, excl_reason = check_hard_exclusions(
        memory,
        access_count=access_count,
        salience_protect=float(salience_protect),
        procedure_anchors=procedure_anchors,
        goal_anchors=goal_anchors,
    )
    if excluded:
        return False, excl_reason

    content = str(memory.get("memory", memory.get("content", "")) or "")
    if content.strip().lower().startswith(_TURN_SUMMARY_PREFIX):
        # Legacy pollution from the removed LEARN fallback — purge regardless of age.
        return True, "turn_summary_backlog"

    age_days = _memory_age_days(memory, now)
    if age_days is None or age_days < min_age_days:
        return False, "too_young"

    meta = memory.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    try:
        confidence = float(meta.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence >= max_confidence:
        return False, "high_confidence"

    return True, "stale_low_confidence"


async def run_mem0_forgetting(
    long_term: Any,
    *,
    force_enabled: bool | None = None,
    archive_store: Any | None = None,
) -> dict[str, Any]:
    """Sweep Mem0: stage-1 archive (+ stage-2 hard-delete of aged archive rows).

    When ``force_enabled`` is set, it overrides ``APP__DREAM_MEM0_FORGETTING_ENABLED``
    for this run only (CLI ``--execute`` path).

    Returns a stage-log-friendly summary dict.
    """
    app = get_settings().app
    if force_enabled is not None:
        enabled = bool(force_enabled)
    else:
        enabled = bool(getattr(app, "dream_mem0_forgetting_enabled", True))
    min_age_days = int(getattr(app, "mem0_forget_min_age_days", 120))
    max_confidence = float(getattr(app, "mem0_forget_max_confidence", 0.7))
    run_cap = int(getattr(app, "mem0_forget_run_cap", 200))
    hard_delete_after = int(getattr(app, "mem0_forget_hard_delete_after_days", 30))
    salience_protect = float(getattr(app, "mem0_forget_salience_protect", 0.35))

    tracker = get_mem0_access_tracker()
    archive = archive_store or get_mem0_archive_store()
    user_ids = set(await tracker.known_user_ids())
    user_ids.add("global")

    procedure_anchors = await load_procedure_anchors()
    goal_anchors = await load_goal_anchors()

    now = datetime.now(UTC)
    run_started = now.isoformat()
    scanned = 0
    candidates = 0
    archived = 0
    deleted = 0  # hard-deletes from archive (stage 2)
    archived_ids: list[str] = []
    affected_users: set[str] = set()
    excluded_breakdown: Counter[str] = Counter()

    for user_id in sorted(user_ids):
        for page in range(1, _MAX_PAGES_PER_USER + 1):
            memories = await long_term.list_memories(user_id, page=page, page_size=_PAGE_SIZE)
            if not memories:
                break
            ids = [str(m.get("id", "")) for m in memories]
            counts = await tracker.get_counts(ids)
            for memory in memories:
                scanned += 1
                mem_id = str(memory.get("id", ""))
                candidate, reason = is_forgetting_candidate(
                    memory,
                    access_count=counts.get(mem_id, 0),
                    now=now,
                    min_age_days=min_age_days,
                    max_confidence=max_confidence,
                    salience_protect=salience_protect,
                    procedure_anchors=procedure_anchors,
                    goal_anchors=goal_anchors,
                )
                if not candidate:
                    if reason in {
                        "protected_type",
                        "protected_provenance",
                        "verified",
                        "accessed",
                        "high_salience",
                        "referenced_by_procedure",
                        "referenced_by_goal",
                        "too_young",
                        "high_confidence",
                    }:
                        excluded_breakdown[reason] += 1
                    continue
                if await tracker.is_recently_reconsolidated(
                    mem_id,
                    within_hours=int(getattr(app, "mem0_reconsolidation_window_hours", 48)),
                ):
                    excluded_breakdown["recently_reconsolidated"] += 1
                    continue
                # Already in the reversible archive → do not double-process.
                if mem_id and await archive.get(mem_id) is not None:
                    excluded_breakdown["already_archived"] += 1
                    continue

                candidates += 1
                event: dict[str, Any] = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "run_started": run_started,
                    "stage": "archive",
                    "user_id": user_id,
                    "memory_id": mem_id,
                    "reason": reason,
                    "content_preview": str(memory.get("memory", memory.get("content", "")) or "")[
                        :200
                    ],
                    "dry_run": not enabled,
                    "archived": False,
                    "deleted": False,
                }
                if enabled and archived < run_cap and mem_id:
                    ok_archive = await archive.archive(
                        memory,
                        user_id=user_id,
                        reason=reason,
                        access_count=counts.get(mem_id, 0),
                    )
                    if ok_archive:
                        ok_del = await long_term.delete_memory(mem_id)
                        if ok_del:
                            archived += 1
                            archived_ids.append(mem_id)
                            affected_users.add(user_id)
                            event["archived"] = True
                        else:
                            event["archive_error"] = "live_delete_failed_after_archive"
                _append_event(event)
            if len(memories) < _PAGE_SIZE:
                break

    # Stage 2: hard-delete aged archive rows that still have zero access.
    hard_delete_candidates = 0
    hard_deleted_ids: list[str] = []
    if enabled:
        hd_rows = await archive.list_hard_delete_candidates(
            min_age_days=hard_delete_after,
            now=now,
            limit=run_cap,
        )
        hard_delete_candidates = len(hd_rows)
        still_zero: list[str] = []
        for row in hd_rows:
            mid = str(row.get("memory_id") or "")
            if not mid:
                continue
            hits = (await tracker.get_counts([mid])).get(mid, 0)
            if hits > 0:
                excluded_breakdown["accessed_while_archived"] += 1
                continue
            still_zero.append(mid)
        if still_zero:
            n = await archive.hard_delete(still_zero)
            deleted = n
            hard_deleted_ids = still_zero[:n]
            for mid in hard_deleted_ids:
                _append_event(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "run_started": run_started,
                        "stage": "hard_delete",
                        "memory_id": mid,
                        "reason": "archive_aged_unaccessed",
                        "dry_run": False,
                        "archived": False,
                        "deleted": True,
                    }
                )
    else:
        # Dry-run: report how many archive rows *would* be hard-deleted.
        hd_rows = await archive.list_hard_delete_candidates(
            min_age_days=hard_delete_after,
            now=now,
            limit=run_cap,
        )
        hard_delete_candidates = len(hd_rows)
        for row in hd_rows:
            _append_event(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "run_started": run_started,
                    "stage": "hard_delete",
                    "memory_id": str(row.get("memory_id") or ""),
                    "reason": "archive_aged_unaccessed",
                    "dry_run": True,
                    "archived": False,
                    "deleted": False,
                }
            )

    if archived_ids:
        await tracker.forget_ids(archived_ids)
        cache = get_mem0_search_cache()
        for user_id in affected_users:
            await cache.bump_generation(user_id)

    summary = {
        "status": "ok",
        "enabled": enabled,
        "dry_run": not enabled,
        "users_swept": len(user_ids),
        "scanned": scanned,
        "candidates": candidates,
        "archived": archived,
        "deleted": deleted,
        "hard_delete_candidates": hard_delete_candidates,
        "hard_deleted": deleted,
        "run_cap": run_cap,
        "hard_delete_after_days": hard_delete_after,
        "excluded_by_rule": dict(excluded_breakdown),
        "exclusion_policy": exclusion_policy_doc(),
        # Backward-compatible alias used by older ops/tests (live deletes == archives).
        "live_removals": archived,
    }
    logger.info(
        "Mem0 forgetting: scanned=%d candidates=%d archived=%d hard_deleted=%d (dry_run=%s)",
        scanned,
        candidates,
        archived,
        deleted,
        not enabled,
    )
    return summary
