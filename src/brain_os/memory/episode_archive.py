"""Reversible archive for dream stage-5 episodic pruning.

Mirrors the Mem0 two-stage contract (``mem0_archive``):

1. **Archive** — move old episodes out of the live ``episodes`` table.
2. **Hard-delete** — permanently drop archive rows after
   ``APP__MEM0_FORGET_HARD_DELETE_AFTER_DAYS`` (default 30).

Stage 5 owns episode archival; stage 5b owns Mem0 archival. Neither path
may hard-delete live rows without first writing an archive row.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

EPISODE_ARCHIVE_DDL = """
CREATE TABLE IF NOT EXISTS episode_archive (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '',
    narrative TEXT NOT NULL DEFAULT '',
    key_topics TEXT NOT NULL DEFAULT '[]',
    decisions TEXT NOT NULL DEFAULT '[]',
    commitments TEXT NOT NULL DEFAULT '[]',
    emotional_tone TEXT NOT NULL DEFAULT '',
    relationship_impact TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    archived_at TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT ''
)
"""


async def ensure_episode_archive_table(db: aiosqlite.Connection) -> None:
    """Create ``episode_archive`` + index on the dream/conversations DB."""
    await db.execute(EPISODE_ARCHIVE_DDL)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_episode_archive_archived_at ON episode_archive(archived_at)"
    )
    await db.commit()


async def already_archived_ids(db: aiosqlite.Connection, episode_ids: list[int]) -> set[int]:
    """Return the subset of *episode_ids* already present in the archive."""
    clean = [int(i) for i in episode_ids if i is not None]
    if not clean:
        return set()
    placeholders = ",".join("?" for _ in clean)
    cursor = await db.execute(
        f"SELECT id FROM episode_archive WHERE id IN ({placeholders})",
        clean,
    )
    rows = await cursor.fetchall()
    await cursor.close()
    return {int(r[0]) for r in rows if r and r[0] is not None}


async def archive_episodes(
    db: aiosqlite.Connection,
    episode_ids: list[int],
    *,
    reason: str = "dream_stage5_llm_archive",
) -> int:
    """Move live episodes into ``episode_archive`` then delete from ``episodes``.

    Skips ids already archived (no double-process). Returns count newly archived.
    """
    clean = sorted({int(i) for i in episode_ids if i is not None})
    if not clean:
        return 0
    await ensure_episode_archive_table(db)
    existing = await already_archived_ids(db, clean)
    to_archive = [i for i in clean if i not in existing]
    if not to_archive:
        return 0

    placeholders = ",".join("?" for _ in to_archive)
    cursor = await db.execute(
        f"""
        SELECT id, user_id, narrative, key_topics, decisions, commitments,
               emotional_tone, relationship_impact, created_at
        FROM episodes WHERE id IN ({placeholders})
        """,
        to_archive,
    )
    rows = await cursor.fetchall()
    await cursor.close()
    if not rows:
        return 0

    archived_at = datetime.now(UTC).isoformat()
    reason_s = (reason or "dream_stage5_llm_archive")[:120]
    for row in rows:
        await db.execute(
            """
            INSERT OR IGNORE INTO episode_archive (
                id, user_id, narrative, key_topics, decisions, commitments,
                emotional_tone, relationship_impact, created_at, archived_at, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row[0],
                row[1] or "",
                row[2] or "",
                row[3] or "[]",
                row[4] or "[]",
                row[5] or "[]",
                row[6] or "",
                row[7] or "",
                row[8] or "",
                archived_at,
                reason_s,
            ),
        )

    archived_ids = [int(r[0]) for r in rows]
    ph = ",".join("?" for _ in archived_ids)
    await db.execute(f"DELETE FROM episodes WHERE id IN ({ph})", archived_ids)
    await db.commit()
    return len(archived_ids)


async def hard_delete_aged_episodes(
    db: aiosqlite.Connection,
    *,
    min_age_days: int = 30,
    limit: int = 200,
    now: datetime | None = None,
) -> int:
    """Permanently drop archive rows older than *min_age_days*."""
    await ensure_episode_archive_table(db)
    ts = now or datetime.now(UTC)
    cutoff = (ts - timedelta(days=max(0, int(min_age_days)))).isoformat()
    cursor = await db.execute(
        """
        SELECT id FROM episode_archive
        WHERE archived_at != '' AND archived_at <= ?
        ORDER BY archived_at ASC
        LIMIT ?
        """,
        (cutoff, max(1, int(limit))),
    )
    rows = await cursor.fetchall()
    await cursor.close()
    ids = [int(r[0]) for r in rows if r and r[0] is not None]
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    cursor = await db.execute(
        f"DELETE FROM episode_archive WHERE id IN ({placeholders})",
        ids,
    )
    await db.commit()
    return int(cursor.rowcount or 0)


def episode_looks_protected(narrative: str, topics: Any = None) -> bool:
    """Heuristic: skip archival for correction-flavoured episodes."""
    text = f"{narrative or ''} {topics or ''}".strip().lower()
    if not text:
        return False
    markers = (
        "correction",
        "corrected",
        "mnemon",
        "do not email",
        "blacklist",
        "do_not_contact",
    )
    return any(m in text for m in markers)
