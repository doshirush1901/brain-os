"""Local hit counter for Mem0 memories (spaced-repetition signal).

Mem0's API does not expose per-memory access counts, so Brain OS keeps its own:
every time a memory id comes back from a Mem0 search (live or cached), its
hit count is bumped here. Two consumers:

* ``LongTermMemory.search`` feeds the counts into ``apply_decay`` so
  frequently recalled memories decay slower (the stability term
  ``30 * (1 + log(1 + hits))``);
* the Dream-stage forgetting job treats zero recorded hits as one of its
  deletion criteria.

Storage is a tiny aiosqlite table in the data dir. All operations are
best-effort — a tracker failure must never break a search.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)


class Mem0AccessTracker:
    """SQLite-backed hit counts for Mem0 memory ids."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else get_data_dir() / "mem0_access.db"
        self._db: aiosqlite.Connection | None = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = aiosqlite.connect(str(self._db_path), timeout=30.0)
            # aiosqlite's worker thread is non-daemon and starts on await; if a
            # connection is ever left open, interpreter exit would hang joining
            # it. Daemonize before the thread starts so exit can never block.
            worker = getattr(conn, "_thread", None)
            if worker is not None:
                worker.daemon = True
            self._db = await conn
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA busy_timeout=30000")
            await self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS mem0_access (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '',
                    hits INTEGER NOT NULL DEFAULT 0,
                    last_hit TEXT NOT NULL DEFAULT '',
                    replay_count INTEGER NOT NULL DEFAULT 0,
                    last_replay TEXT NOT NULL DEFAULT '',
                    confirmed_at TEXT NOT NULL DEFAULT '',
                    salience_bump REAL NOT NULL DEFAULT 0
                )
                """
            )
            await self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS mem0_reconsolidation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    old_memory_id TEXT NOT NULL,
                    new_memory_id TEXT NOT NULL DEFAULT '',
                    user_id TEXT NOT NULL DEFAULT '',
                    query TEXT NOT NULL DEFAULT '',
                    recalled_at TEXT NOT NULL DEFAULT '',
                    stored_at TEXT NOT NULL DEFAULT ''
                )
                """
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_recon_old ON mem0_reconsolidation(old_memory_id)"
            )
            await self._db.commit()
            await self._migrate_schema()
        return self._db

    async def _migrate_schema(self) -> None:
        """Add columns/tables on existing DBs without migrations."""
        db = self._db
        if db is None:
            return
        try:
            cursor = await db.execute("PRAGMA table_info(mem0_access)")
            cols = {str(row[1]) for row in await cursor.fetchall()}
            if "replay_count" not in cols:
                await db.execute(
                    "ALTER TABLE mem0_access ADD COLUMN replay_count INTEGER NOT NULL DEFAULT 0"
                )
            if "last_replay" not in cols:
                await db.execute(
                    "ALTER TABLE mem0_access ADD COLUMN last_replay TEXT NOT NULL DEFAULT ''"
                )
            if "confirmed_at" not in cols:
                await db.execute(
                    "ALTER TABLE mem0_access ADD COLUMN confirmed_at TEXT NOT NULL DEFAULT ''"
                )
            if "salience_bump" not in cols:
                await db.execute(
                    "ALTER TABLE mem0_access ADD COLUMN salience_bump REAL NOT NULL DEFAULT 0"
                )
            await db.commit()
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 access tracker schema migrate failed", exc_info=True)

    async def record_hits(self, ids: list[str], user_id: str = "") -> None:
        """Increment hit counts for *ids*. Best-effort; never raises."""
        clean = [str(i) for i in ids if i]
        if not clean:
            return
        try:
            db = await self._ensure_db()
            now = datetime.now(UTC).isoformat()
            await db.executemany(
                """
                INSERT INTO mem0_access (memory_id, user_id, hits, last_hit)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    hits = hits + 1,
                    last_hit = excluded.last_hit,
                    user_id = CASE
                        WHEN excluded.user_id != '' THEN excluded.user_id
                        ELSE user_id
                    END
                """,
                [(mid, user_id, now) for mid in clean],
            )
            await db.commit()
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 access tracker record_hits failed", exc_info=True)

    async def record_replay(self, memory_id: str, user_id: str = "") -> None:
        """Dream-cycle sleep replay — strengthens stability without a live recall."""
        if not memory_id:
            return
        try:
            db = await self._ensure_db()
            now = datetime.now(UTC).isoformat()
            await db.execute(
                """
                INSERT INTO mem0_access (memory_id, user_id, hits, last_hit, replay_count, last_replay)
                VALUES (?, ?, 0, '', 1, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    replay_count = replay_count + 1,
                    last_replay = excluded.last_replay,
                    user_id = CASE
                        WHEN excluded.user_id != '' THEN excluded.user_id
                        ELSE user_id
                    END
                """,
                (memory_id, user_id, now),
            )
            await db.commit()
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 access tracker record_replay failed", exc_info=True)

    async def link_reconsolidation(
        self,
        old_ids: list[str],
        new_ids: list[str],
        *,
        user_id: str = "",
        query: str = "",
        recalled_at: str = "",
    ) -> None:
        """Audit trail when a store follows a recall (memory rewrite)."""
        olds = [str(i) for i in old_ids if i]
        news = [str(i) for i in new_ids if i]
        if not olds:
            return
        new_id = news[0] if news else ""
        stored_at = datetime.now(UTC).isoformat()
        try:
            db = await self._ensure_db()
            await db.executemany(
                """
                INSERT INTO mem0_reconsolidation
                    (old_memory_id, new_memory_id, user_id, query, recalled_at, stored_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(oid, new_id, user_id, query[:500], recalled_at, stored_at) for oid in olds],
            )
            await db.commit()
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 reconsolidation link failed", exc_info=True)

    async def is_recently_reconsolidated(self, memory_id: str, *, within_hours: int = 48) -> bool:
        if not memory_id:
            return False
        try:
            db = await self._ensure_db()
            cursor = await db.execute(
                """
                SELECT stored_at FROM mem0_reconsolidation
                WHERE old_memory_id = ? OR new_memory_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (memory_id, memory_id),
            )
            row = await cursor.fetchone()
            if not row or not row[0]:
                return False
            stored = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
            if stored.tzinfo is None:
                stored = stored.replace(tzinfo=UTC)
            age_hours = (datetime.now(UTC) - stored).total_seconds() / 3600.0
            return age_hours <= within_hours
        except (aiosqlite.Error, OSError, RuntimeError, ValueError):
            logger.debug("Mem0 reconsolidation check failed", exc_info=True)
            return False

    async def get_counts(self, ids: list[str]) -> dict[str, int]:
        """Return ``{memory_id: hits}`` for known ids; missing ids are absent."""
        stats = await self.get_access_stats(ids)
        return {mid: int(row["hits"]) for mid, row in stats.items()}

    async def get_access_stats(self, ids: list[str]) -> dict[str, dict[str, int | str | float]]:
        """Return ``{memory_id: {hits, last_hit, confirmed_at, …}}`` for known ids."""
        clean = [str(i) for i in ids if i]
        if not clean:
            return {}
        try:
            db = await self._ensure_db()
            placeholders = ",".join("?" for _ in clean)
            cursor = await db.execute(
                f"SELECT memory_id, hits, last_hit, confirmed_at, salience_bump "
                f"FROM mem0_access WHERE memory_id IN ({placeholders})",
                clean,
            )
            rows = await cursor.fetchall()
            return {
                str(r[0]): {
                    "hits": int(r[1]),
                    "last_hit": str(r[2] or ""),
                    "confirmed_at": str(r[3] or ""),
                    "salience_bump": float(r[4] or 0),
                }
                for r in rows
            }
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 access tracker get_access_stats failed", exc_info=True)
            return {}

    async def access_distribution(self) -> dict[str, Any]:
        """Aggregate hit distribution for the memory report."""
        try:
            db = await self._ensure_db()
            cursor = await db.execute(
                """
                SELECT
                    COUNT(*) AS tracked,
                    COALESCE(SUM(CASE WHEN hits = 0 THEN 1 ELSE 0 END), 0) AS zero_hits,
                    COALESCE(SUM(CASE WHEN hits > 0 THEN 1 ELSE 0 END), 0) AS accessed,
                    COALESCE(SUM(hits), 0) AS total_hits
                FROM mem0_access
                """
            )
            row = await cursor.fetchone()
            tracked = int(row[0] or 0) if row else 0
            zero = int(row[1] or 0) if row else 0
            accessed = int(row[2] or 0) if row else 0
            total_hits = int(row[3] or 0) if row else 0
            return {
                "tracked_ids": tracked,
                "never_accessed_tracked": zero,
                "accessed_tracked": accessed,
                "total_hits": total_hits,
                "pct_never_accessed_tracked": (
                    round(100.0 * zero / tracked, 2) if tracked else 0.0
                ),
            }
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 access tracker access_distribution failed", exc_info=True)
            return {
                "tracked_ids": 0,
                "never_accessed_tracked": 0,
                "accessed_tracked": 0,
                "total_hits": 0,
                "pct_never_accessed_tracked": 0.0,
            }

    async def top_accessed(self, *, limit: int = 10) -> list[dict[str, Any]]:
        """Return top-N memories by hit count (then recency)."""
        try:
            db = await self._ensure_db()
            cursor = await db.execute(
                """
                SELECT memory_id, user_id, hits, last_hit, replay_count,
                       confirmed_at, salience_bump
                FROM mem0_access
                WHERE hits > 0
                ORDER BY hits DESC, last_hit DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "memory_id": str(r[0]),
                    "user_id": str(r[1] or ""),
                    "hits": int(r[2] or 0),
                    "last_hit": str(r[3] or ""),
                    "replay_count": int(r[4] or 0),
                    "confirmed_at": str(r[5] or ""),
                    "salience_bump": float(r[6] or 0),
                }
                for r in rows
            ]
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 access tracker top_accessed failed", exc_info=True)
            return []

    async def confirm_memory(
        self,
        memory_id: str,
        *,
        user_id: str = "",
        salience_delta: float = 0.15,
    ) -> dict[str, Any]:
        """Operator confirm — bump hits + salience and stamp ``confirmed_at``."""
        mid = str(memory_id or "").strip()
        if not mid:
            return {"ok": False, "reason": "missing_memory_id"}
        try:
            db = await self._ensure_db()
            now = datetime.now(UTC).isoformat()
            await db.execute(
                """
                INSERT INTO mem0_access (
                    memory_id, user_id, hits, last_hit, confirmed_at, salience_bump
                )
                VALUES (?, ?, 1, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    hits = hits + 1,
                    last_hit = excluded.last_hit,
                    confirmed_at = excluded.confirmed_at,
                    salience_bump = salience_bump + excluded.salience_bump,
                    user_id = CASE
                        WHEN excluded.user_id != '' THEN excluded.user_id
                        ELSE user_id
                    END
                """,
                (mid, user_id, now, now, float(salience_delta)),
            )
            await db.commit()
            stats = await self.get_access_stats([mid])
            row = stats.get(mid) or {}
            return {
                "ok": True,
                "memory_id": mid,
                "confirmed_at": now,
                "hits": int(row.get("hits") or 0),
                "salience_delta": float(salience_delta),
            }
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 access tracker confirm_memory failed", exc_info=True)
            return {"ok": False, "reason": "tracker_error", "memory_id": mid}

    async def stale_high_use_unconfirmed(
        self,
        *,
        min_hits: int = 5,
        stale_days: int = 90,
        limit: int = 5,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Oldest high-use memories not operator-confirmed within *stale_days*."""
        now = now or datetime.now(UTC)
        cutoff = (now - timedelta(days=max(1, int(stale_days)))).isoformat()
        try:
            db = await self._ensure_db()
            cursor = await db.execute(
                """
                SELECT memory_id, user_id, hits, last_hit, confirmed_at, salience_bump
                FROM mem0_access
                WHERE hits >= ?
                  AND (confirmed_at = '' OR confirmed_at < ?)
                ORDER BY
                    CASE WHEN confirmed_at = '' THEN 0 ELSE 1 END,
                    confirmed_at ASC,
                    hits DESC,
                    last_hit ASC
                LIMIT ?
                """,
                (max(1, int(min_hits)), cutoff, max(1, int(limit))),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "memory_id": str(r[0]),
                    "user_id": str(r[1] or ""),
                    "hits": int(r[2] or 0),
                    "last_hit": str(r[3] or ""),
                    "confirmed_at": str(r[4] or ""),
                    "salience_bump": float(r[5] or 0),
                }
                for r in rows
            ]
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 access tracker stale_high_use_unconfirmed failed", exc_info=True)
            return []

    async def hits_since(self, *, since_iso: str) -> int:
        """Count hit rows whose last_hit is on/after *since_iso* (growth proxy)."""
        if not since_iso:
            return 0
        try:
            db = await self._ensure_db()
            cursor = await db.execute(
                "SELECT COUNT(*) FROM mem0_access WHERE last_hit >= ?",
                (since_iso,),
            )
            row = await cursor.fetchone()
            return int(row[0] or 0) if row else 0
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 access tracker hits_since failed", exc_info=True)
            return 0

    async def known_user_ids(self) -> list[str]:
        """Distinct non-empty user ids seen by the tracker (forgetting sweep input)."""
        try:
            db = await self._ensure_db()
            cursor = await db.execute(
                "SELECT DISTINCT user_id FROM mem0_access WHERE user_id != ''"
            )
            rows = await cursor.fetchall()
            return [str(r[0]) for r in rows]
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 access tracker known_user_ids failed", exc_info=True)
            return []

    async def forget_ids(self, ids: list[str]) -> None:
        """Drop tracker rows for deleted memories."""
        clean = [str(i) for i in ids if i]
        if not clean:
            return
        try:
            db = await self._ensure_db()
            placeholders = ",".join("?" for _ in clean)
            await db.execute(
                f"DELETE FROM mem0_access WHERE memory_id IN ({placeholders})",
                clean,
            )
            await db.commit()
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 access tracker forget_ids failed", exc_info=True)

    async def close(self) -> None:
        if self._db is not None:
            try:
                await self._db.close()
            except aiosqlite.Error:
                logger.debug("Mem0 access tracker close failed", exc_info=True)
            self._db = None


_tracker: Mem0AccessTracker | None = None


def get_mem0_access_tracker() -> Mem0AccessTracker:
    global _tracker
    if _tracker is None:
        _tracker = Mem0AccessTracker()
    return _tracker


def reset_mem0_access_tracker() -> None:
    """Test hook: drop the singleton so the next call builds a fresh tracker."""
    global _tracker
    _tracker = None
