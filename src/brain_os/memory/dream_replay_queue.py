"""Dream sleep replay — prioritize frequently recalled Mem0 assemblies.

Hippocampus replay analogue: during dream, bump access on high-signal memories
so they decay slower and surface stronger on the next recall.
"""

from __future__ import annotations

import logging
from typing import Any

from brain_os.config import get_settings
from brain_os.memory.mem0_access_tracker import get_mem0_access_tracker
from brain_os.memory.store_policy import classify_provenance

logger = logging.getLogger(__name__)

_MAX_PAGES = 5
_PAGE_SIZE = 100


def _replay_score(
    memory: dict[str, Any],
    *,
    access_count: int,
) -> float:
    meta = memory.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    score = float(access_count) * 2.0
    mem_type = str(meta.get("type", "")).lower()
    if mem_type == "correction":
        score += 50.0
    provenance = meta.get("provenance_class") or classify_provenance(
        str(meta.get("source", "")), meta
    )
    if provenance == "evidence":
        score += 12.0
    elif provenance == "lore":
        score += 2.0
    if meta.get("reconsolidated"):
        score += 6.0
    try:
        score += float(meta.get("salience_score", 0)) * 5.0
    except (TypeError, ValueError):
        pass
    return score


async def build_replay_queue(long_term: Any, *, top_k: int | None = None) -> list[dict[str, Any]]:
    """Return top-K Mem0 rows ranked for sleep replay (access + provenance + salience)."""
    app = get_settings().app
    limit = top_k if top_k is not None else int(getattr(app, "dream_replay_top_k", 25))
    tracker = get_mem0_access_tracker()
    user_ids = set(await tracker.known_user_ids())
    user_ids.add("global")

    ranked: list[tuple[float, dict[str, Any], str]] = []
    for user_id in sorted(user_ids):
        for page in range(1, _MAX_PAGES + 1):
            memories = await long_term.list_memories(user_id, page=page, page_size=_PAGE_SIZE)
            if not memories:
                break
            ids = [str(m.get("id", "")) for m in memories]
            counts = await tracker.get_counts(ids)
            for memory in memories:
                mem_id = str(memory.get("id", ""))
                if not mem_id:
                    continue
                hits = counts.get(mem_id, 0)
                if (
                    hits <= 0
                    and str((memory.get("metadata") or {}).get("type", "")) != "correction"
                ):
                    continue
                score = _replay_score(memory, access_count=hits)
                ranked.append((score, memory, user_id))
            if len(memories) < _PAGE_SIZE:
                break

    ranked.sort(key=lambda row: row[0], reverse=True)
    out: list[dict[str, Any]] = []
    for score, memory, user_id in ranked[:limit]:
        meta = memory.get("metadata") or {}
        out.append(
            {
                "memory_id": str(memory.get("id", "")),
                "user_id": user_id,
                "replay_score": round(score, 2),
                "content_preview": str(memory.get("memory", memory.get("content", "")) or "")[:180],
                "provenance_class": (meta.get("provenance_class") if isinstance(meta, dict) else "")
                or classify_provenance(
                    str(meta.get("source", "") if isinstance(meta, dict) else ""), meta
                ),
            }
        )
    return out


async def run_sleep_replay(long_term: Any, *, top_k: int | None = None) -> dict[str, Any]:
    """Replay top memories — one synthetic access bump per dream cycle."""
    app = get_settings().app
    if not bool(getattr(app, "dream_mem0_replay_enabled", True)):
        return {"status": "skipped", "reason": "disabled"}

    queue = await build_replay_queue(long_term, top_k=top_k)
    if not queue:
        return {"status": "ok", "replayed": 0, "queue": []}

    tracker = get_mem0_access_tracker()
    replayed = 0
    for row in queue:
        mem_id = row.get("memory_id", "")
        user_id = row.get("user_id", "global")
        if not mem_id:
            continue
        await tracker.record_hits([mem_id], user_id=user_id)
        await tracker.record_replay(mem_id, user_id=user_id)
        replayed += 1

    logger.info("Dream Mem0 replay: reinforced %d memories", replayed)
    return {"status": "ok", "replayed": replayed, "queue": queue}
