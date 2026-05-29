"""FTS5 full-text index over Graphe ``cursor_sessions`` for cross-session recall.

Hermes-style session search: SQLite FTS5 over logged query, summary, and agents JSON.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_DB = Path("data/brain/cursor_sessions.db")
_FTS_TABLE = "cursor_sessions_fts"
_TRIGGER_NAMES = (
    "cursor_sessions_ai_fts",
    "cursor_sessions_au_fts",
    "cursor_sessions_ad_fts",
)


def build_fts_match_query(user_text: str) -> str:
    """Map free text to a conservative FTS5 MATCH string (prefix tokens, AND).

    Strips characters that break MATCH syntax; tokens are prefix-matched.
    """
    raw = (user_text or "").strip()
    if not raw:
        return ""
    tokens = re.findall(r"[\w.-]+", raw, flags=re.UNICODE)
    if not tokens:
        return ""
    parts: list[str] = []
    for tok in tokens[:12]:
        escaped = tok.replace('"', '""')
        parts.append(f'"{escaped}"*')
    return " AND ".join(parts)


async def _trigger_exists(db: aiosqlite.Connection, name: str) -> bool:
    cur = await db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name=?",
        (name,),
    )
    return (await cur.fetchone()) is not None


async def ensure_cursor_sessions_fts(db: aiosqlite.Connection) -> None:
    """Ensure FTS5 table, sync triggers, and incremental backfill for ``cursor_sessions``."""
    await db.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} USING fts5(
            query,
            response_summary,
            agents_used,
            tokenize = 'porter'
        )
        """
    )

    if not await _trigger_exists(db, _TRIGGER_NAMES[0]):
        await db.executescript(
            f"""
            CREATE TRIGGER {_TRIGGER_NAMES[0]} AFTER INSERT ON cursor_sessions BEGIN
                INSERT INTO {_FTS_TABLE}(rowid, query, response_summary, agents_used)
                VALUES (new.rowid, new.query, new.response_summary, new.agents_used);
            END;
            CREATE TRIGGER {_TRIGGER_NAMES[1]} AFTER UPDATE ON cursor_sessions BEGIN
                DELETE FROM {_FTS_TABLE} WHERE rowid = old.rowid;
                INSERT INTO {_FTS_TABLE}(rowid, query, response_summary, agents_used)
                VALUES (new.rowid, new.query, new.response_summary, new.agents_used);
            END;
            CREATE TRIGGER {_TRIGGER_NAMES[2]} AFTER DELETE ON cursor_sessions BEGIN
                DELETE FROM {_FTS_TABLE} WHERE rowid = old.rowid;
            END;
            """
        )

    await db.execute(
        f"""
        INSERT INTO {_FTS_TABLE}(rowid, query, response_summary, agents_used)
        SELECT c.rowid, c.query, c.response_summary, c.agents_used
        FROM cursor_sessions AS c
        WHERE NOT EXISTS (
            SELECT 1 FROM {_FTS_TABLE} AS f WHERE f.rowid = c.rowid
        )
        """
    )


async def search_cursor_sessions(
    *,
    match_query: str,
    db_path: Path | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """BM25-ranked hits over logged Cursor turns (query, summary, agents JSON)."""
    path = db_path or DEFAULT_SESSIONS_DB
    if not path.exists():
        return []
    mq = (match_query or "").strip()
    if not mq:
        return []

    async with aiosqlite.connect(str(path), timeout=30.0) as db:
        await db.execute("PRAGMA busy_timeout=30000")
        await ensure_cursor_sessions_fts(db)
        try:
            cur = await db.execute(
                f"""
                SELECT s.id, s.timestamp, s.query, s.response_summary, s.agents_used,
                       bm25({_FTS_TABLE}) AS rank
                FROM {_FTS_TABLE}
                JOIN cursor_sessions AS s ON s.rowid = {_FTS_TABLE}.rowid
                WHERE {_FTS_TABLE} MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (mq, max(1, min(limit, 100))),
            )
        except Exception as exc:
            logger.exception("cursor_session_fts search failed for MATCH %r", mq)
            raise
        rows = await cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            ts = row[1]
            iso: str | None = None
            if isinstance(ts, (int, float)):
                iso = datetime.fromtimestamp(float(ts), tz=UTC).isoformat(timespec="seconds")
            out.append(
                {
                    "id": row[0],
                    "timestamp": ts,
                    "timestamp_iso": iso,
                    "query": row[2],
                    "response_summary": row[3],
                    "agents_used": row[4],
                    "bm25": row[5],
                }
            )
    return out
