"""Pending memory queue backends: SQLite (default), Postgres dual-write."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.config import get_settings
from brain_os.data.crm import CRMDatabase
from brain_os.data.pending_memory import PendingMemoryHintModel, PgPendingMemoryRepository

logger = logging.getLogger(__name__)


class PendingMemoryQueueBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def enqueue(self, body: str, *, source: str = "cli") -> int: ...

    @abstractmethod
    async def list_pending(self, *, limit: int = 50) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def count_pending(self) -> int: ...

    @abstractmethod
    async def mark_processed(self, row_id: int) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def sqlite_db_connection(self) -> Any: ...


class SqlitePendingMemoryQueueBackend(PendingMemoryQueueBackend):
    def __init__(self, db_path: Path) -> None:
        self._path = db_path

    @property
    def sqlite_db_connection(self) -> Any:
        return None

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self._path), timeout=30.0) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=30000")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_memory_hints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    body TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                )
                """
            )
            await db.commit()
        logger.info("PendingMemoryQueue (SQLite) initialised at %s", self._path)

    async def enqueue(self, body: str, *, source: str = "cli") -> int:
        text = (body or "").strip()
        if not text:
            raise ValueError("body must be non-empty")
        now = datetime.now(UTC).isoformat()
        src = (source or "cli").strip()[:80] or "cli"
        async with aiosqlite.connect(str(self._path), timeout=30.0) as db:
            await db.execute("PRAGMA busy_timeout=30000")
            cur = await db.execute(
                """
                INSERT INTO pending_memory_hints (created_at, source, body, status)
                VALUES (?, ?, ?, 'pending')
                """,
                (now, src, text),
            )
            await db.commit()
            return int(cur.lastrowid or 0)

    async def list_pending(self, *, limit: int = 50) -> list[dict[str, Any]]:
        cap = max(1, min(limit, 500))
        async with aiosqlite.connect(str(self._path), timeout=30.0) as db:
            await db.execute("PRAGMA busy_timeout=30000")
            cur = await db.execute(
                """
                SELECT id, created_at, source, body
                FROM pending_memory_hints
                WHERE status = 'pending'
                ORDER BY id ASC
                LIMIT ?
                """,
                (cap,),
            )
            rows = await cur.fetchall()
        return [{"id": r[0], "created_at": r[1], "source": r[2], "body": r[3]} for r in rows]

    async def count_pending(self) -> int:
        async with aiosqlite.connect(str(self._path), timeout=30.0) as db:
            await db.execute("PRAGMA busy_timeout=30000")
            cur = await db.execute(
                "SELECT COUNT(*) FROM pending_memory_hints WHERE status = 'pending'"
            )
            row = await cur.fetchone()
        return int(row[0] if row and row[0] is not None else 0)

    async def mark_processed(self, row_id: int) -> None:
        async with aiosqlite.connect(str(self._path), timeout=30.0) as db:
            await db.execute("PRAGMA busy_timeout=30000")
            await db.execute(
                "UPDATE pending_memory_hints SET status = 'processed' WHERE id = ?",
                (row_id,),
            )
            await db.commit()

    async def close(self) -> None:
        return None


class PgPendingMemoryQueueBackend(PendingMemoryQueueBackend):
    def __init__(self, repo: PgPendingMemoryRepository | None = None) -> None:
        if repo is None:
            repo = PgPendingMemoryRepository(CRMDatabase().session_factory)
        self._repo = repo

    @property
    def sqlite_db_connection(self) -> Any:
        return None

    async def initialize(self) -> None:
        await ensure_pending_memory_tables()

    async def enqueue(
        self,
        body: str,
        *,
        source: str = "cli",
        row_id: int | None = None,
    ) -> int:
        text = (body or "").strip()
        if not text:
            raise ValueError("body must be non-empty")
        now = datetime.now(UTC).isoformat()
        src = (source or "cli").strip()[:80] or "cli"
        return await self._repo.enqueue(text, source=src, created_at=now, row_id=row_id)

    async def list_pending(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return await self._repo.list_pending(limit=limit)

    async def count_pending(self) -> int:
        return await self._repo.count_pending()

    async def mark_processed(self, row_id: int) -> None:
        await self._repo.mark_processed(row_id)

    async def close(self) -> None:
        return None


class DualWritePendingMemoryQueueBackend(PendingMemoryQueueBackend):
    def __init__(
        self, sqlite: SqlitePendingMemoryQueueBackend, pg: PgPendingMemoryQueueBackend
    ) -> None:
        self._sqlite = sqlite
        self._pg = pg

    @property
    def sqlite_db_connection(self) -> Any:
        return self._sqlite.sqlite_db_connection

    def _read(self) -> PendingMemoryQueueBackend:
        if getattr(get_settings().app, "pg_store_pending_memory_read", False):
            return self._pg
        return self._sqlite

    async def initialize(self) -> None:
        await self._sqlite.initialize()
        await self._pg.initialize()

    async def enqueue(self, body: str, *, source: str = "cli") -> int:
        row_id = await self._sqlite.enqueue(body, source=source)
        try:
            await self._pg.enqueue(body, source=source, row_id=row_id)
        except Exception:
            logger.warning("Postgres pending_memory shadow enqueue failed", exc_info=True)
        return row_id

    async def list_pending(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return await self._read().list_pending(limit=limit)

    async def count_pending(self) -> int:
        return await self._read().count_pending()

    async def mark_processed(self, row_id: int) -> None:
        await self._sqlite.mark_processed(row_id)
        try:
            await self._pg.mark_processed(row_id)
        except Exception:
            logger.warning("Postgres pending_memory shadow mark_processed failed", exc_info=True)

    async def close(self) -> None:
        await self._sqlite.close()
        await self._pg.close()


def build_pending_memory_queue_backend(db_path: Path) -> PendingMemoryQueueBackend:
    sqlite = SqlitePendingMemoryQueueBackend(db_path)
    if not getattr(get_settings().app, "pg_store_pending_memory_enabled", False):
        return sqlite
    return DualWritePendingMemoryQueueBackend(sqlite, PgPendingMemoryQueueBackend())


async def ensure_pending_memory_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: PendingMemoryHintModel.__table__.create(sync_conn, checkfirst=True)
        )
