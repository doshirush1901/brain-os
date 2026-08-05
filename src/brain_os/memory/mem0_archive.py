"""Local reversible archive for Mem0 forgetting stage 1.

Archived memories leave the live Mem0 store but remain recoverable until
stage 2 hard-deletes them after ``APP__MEM0_FORGET_HARD_DELETE_AFTER_DAYS``
of still-zero access.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)


class Mem0ArchiveStore:
    """SQLite archive of Mem0 rows removed from the live store."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else get_data_dir() / "mem0_archive.db"
        self._db: aiosqlite.Connection | None = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = aiosqlite.connect(str(self._db_path), timeout=30.0)
            worker = getattr(conn, "_thread", None)
            if worker is not None:
                worker.daemon = True
            self._db = await conn
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA busy_timeout=30000")
            await self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS mem0_archive (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT '',
                    archived_at TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    access_count_at_archive INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_archive_archived_at ON mem0_archive(archived_at)"
            )
            await self._db.commit()
        return self._db

    async def archive(
        self,
        memory: dict[str, Any],
        *,
        user_id: str,
        reason: str,
        access_count: int = 0,
    ) -> bool:
        """Upsert one memory into the archive. Returns True on success."""
        mem_id = str(memory.get("id", "") or "").strip()
        if not mem_id:
            return False
        meta = memory.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        content = str(memory.get("memory", memory.get("content", "")) or "")
        created_at = str(memory.get("created_at", "") or "")
        archived_at = datetime.now(UTC).isoformat()
        try:
            db = await self._ensure_db()
            await db.execute(
                """
                INSERT INTO mem0_archive (
                    memory_id, user_id, content, metadata_json, created_at,
                    archived_at, reason, access_count_at_archive
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    content = excluded.content,
                    metadata_json = excluded.metadata_json,
                    archived_at = excluded.archived_at,
                    reason = excluded.reason,
                    access_count_at_archive = excluded.access_count_at_archive
                """,
                (
                    mem_id,
                    user_id,
                    content,
                    json.dumps(meta, ensure_ascii=True, default=str),
                    created_at,
                    archived_at,
                    reason[:120],
                    int(access_count),
                ),
            )
            await db.commit()
            return True
        except (aiosqlite.Error, OSError, RuntimeError, TypeError, ValueError):
            logger.warning("Mem0 archive write failed for %s", mem_id, exc_info=True)
            return False

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        if not memory_id:
            return None
        try:
            db = await self._ensure_db()
            cursor = await db.execute(
                """
                SELECT memory_id, user_id, content, metadata_json, created_at,
                       archived_at, reason, access_count_at_archive
                FROM mem0_archive WHERE memory_id = ?
                """,
                (memory_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return _row_to_dict(row)
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 archive get failed", exc_info=True)
            return None

    async def count(self) -> int:
        try:
            db = await self._ensure_db()
            cursor = await db.execute("SELECT COUNT(*) FROM mem0_archive")
            row = await cursor.fetchone()
            return int(row[0] or 0) if row else 0
        except (aiosqlite.Error, OSError, RuntimeError):
            return 0

    async def count_archived_since(self, *, since_iso: str) -> int:
        if not since_iso:
            return 0
        try:
            db = await self._ensure_db()
            cursor = await db.execute(
                "SELECT COUNT(*) FROM mem0_archive WHERE archived_at >= ?",
                (since_iso,),
            )
            row = await cursor.fetchone()
            return int(row[0] or 0) if row else 0
        except (aiosqlite.Error, OSError, RuntimeError):
            return 0

    async def list_hard_delete_candidates(
        self,
        *,
        min_age_days: int,
        now: datetime | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Archive rows older than *min_age_days* (still in archive)."""
        ts = now or datetime.now(UTC)
        cutoff = (ts - timedelta(days=max(0, int(min_age_days)))).isoformat()
        try:
            db = await self._ensure_db()
            cursor = await db.execute(
                """
                SELECT memory_id, user_id, content, metadata_json, created_at,
                       archived_at, reason, access_count_at_archive
                FROM mem0_archive
                WHERE archived_at != '' AND archived_at <= ?
                  AND access_count_at_archive = 0
                ORDER BY archived_at ASC
                LIMIT ?
                """,
                (cutoff, max(1, int(limit))),
            )
            rows = await cursor.fetchall()
            return [_row_to_dict(r) for r in rows]
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.debug("Mem0 archive hard-delete candidate list failed", exc_info=True)
            return []

    async def hard_delete(self, memory_ids: list[str]) -> int:
        """Permanently remove archive rows. Returns number deleted."""
        clean = [str(i) for i in memory_ids if i]
        if not clean:
            return 0
        try:
            db = await self._ensure_db()
            placeholders = ",".join("?" for _ in clean)
            cursor = await db.execute(
                f"DELETE FROM mem0_archive WHERE memory_id IN ({placeholders})",
                clean,
            )
            await db.commit()
            return int(cursor.rowcount or 0)
        except (aiosqlite.Error, OSError, RuntimeError):
            logger.warning("Mem0 archive hard_delete failed", exc_info=True)
            return 0

    async def close(self) -> None:
        if self._db is not None:
            try:
                await self._db.close()
            except aiosqlite.Error:
                logger.debug("Mem0 archive close failed", exc_info=True)
            self._db = None


def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    try:
        meta = json.loads(row[3] or "{}")
    except (json.JSONDecodeError, TypeError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return {
        "memory_id": str(row[0]),
        "user_id": str(row[1] or ""),
        "content": str(row[2] or ""),
        "metadata": meta,
        "created_at": str(row[4] or ""),
        "archived_at": str(row[5] or ""),
        "reason": str(row[6] or ""),
        "access_count_at_archive": int(row[7] or 0),
    }


_archive: Mem0ArchiveStore | None = None


def get_mem0_archive_store() -> Mem0ArchiveStore:
    global _archive
    if _archive is None:
        _archive = Mem0ArchiveStore()
    return _archive


def reset_mem0_archive_store() -> None:
    """Test hook: drop the singleton."""
    global _archive
    _archive = None
