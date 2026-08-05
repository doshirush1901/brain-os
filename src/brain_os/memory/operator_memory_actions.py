"""Operator reinforcement actions — confirm (bump salience) / retire (archive).

Closes the morning-card loop: ``brain memory confirm <id>`` and
``brain memory retire <id>`` train the access tracker and forgetting gates.
"""

from __future__ import annotations

import logging
from typing import Any

from brain_os.memory.mem0_access_tracker import get_mem0_access_tracker
from brain_os.memory.mem0_archive import get_mem0_archive_store

logger = logging.getLogger(__name__)


async def confirm_memory(
    memory_id: str,
    *,
    user_id: str = "",
    salience_delta: float = 0.15,
) -> dict[str, Any]:
    """Bump access + salience and stamp operator confirmation."""
    mid = str(memory_id or "").strip()
    if not mid:
        return {"ok": False, "reason": "missing_memory_id"}
    tracker = get_mem0_access_tracker()
    result = await tracker.confirm_memory(mid, user_id=user_id, salience_delta=salience_delta)
    if result.get("ok"):
        logger.info(
            "OPERATOR_MEMORY_CONFIRM | id=%s hits=%s bump=%.2f",
            mid,
            result.get("hits"),
            salience_delta,
        )
    return result


async def retire_memory(
    memory_id: str,
    *,
    user_id: str = "global",
    reason: str = "operator_retire",
    long_term: Any | None = None,
) -> dict[str, Any]:
    """Archive one memory out of live Mem0 (reversible stage-1 forget)."""
    mid = str(memory_id or "").strip()
    if not mid:
        return {"ok": False, "reason": "missing_memory_id"}

    tracker = get_mem0_access_tracker()
    archive = get_mem0_archive_store()
    hits = (await tracker.get_counts([mid])).get(mid, 0)
    stats = (await tracker.get_access_stats([mid])).get(mid) or {}
    uid = (user_id or str(stats.get("user_id") or "") or "global").strip() or "global"

    content = ""
    meta: dict[str, Any] = {"operator_retire": True}
    created_at = ""
    if long_term is None:
        try:
            from brain_os.config import get_settings
            from brain_os.memory.long_term import LongTermMemory

            if get_settings().memory.api_key.get_secret_value():
                long_term = LongTermMemory()
        except Exception:
            long_term = None

    if long_term is not None and hasattr(long_term, "list_memories"):
        try:
            # Best-effort content fetch — Mem0 has no reliable get-by-id.
            for page in range(1, 4):
                rows = await long_term.list_memories(uid, page=page, page_size=100)
                if not rows:
                    break
                for row in rows:
                    if str(row.get("id") or "") == mid:
                        content = str(row.get("memory") or row.get("content") or "")
                        raw_meta = row.get("metadata") or {}
                        if isinstance(raw_meta, dict):
                            meta = {**raw_meta, "operator_retire": True}
                        created_at = str(row.get("created_at") or "")
                        break
                if content:
                    break
        except Exception:
            logger.debug("retire_memory content fetch failed", exc_info=True)

    memory_blob = {
        "id": mid,
        "memory": content or f"(retired {mid})",
        "content": content or f"(retired {mid})",
        "metadata": meta,
        "created_at": created_at,
    }
    ok_archive = await archive.archive(
        memory_blob,
        user_id=uid,
        reason=reason,
        access_count=int(hits),
    )
    if not ok_archive:
        return {"ok": False, "reason": "archive_failed", "memory_id": mid}

    deleted = False
    if long_term is not None and hasattr(long_term, "delete_memory"):
        try:
            deleted = bool(await long_term.delete_memory(mid))
        except Exception:
            logger.debug("retire_memory live delete failed", exc_info=True)
            deleted = False

    await tracker.forget_ids([mid])
    logger.info(
        "OPERATOR_MEMORY_RETIRE | id=%s archived=%s deleted_live=%s",
        mid,
        ok_archive,
        deleted,
    )
    return {
        "ok": True,
        "memory_id": mid,
        "archived": True,
        "deleted_live": deleted,
        "hits_at_retire": int(hits),
        "cli_hint": "Archived — recoverable until stage-2 hard-delete window elapses.",
    }
