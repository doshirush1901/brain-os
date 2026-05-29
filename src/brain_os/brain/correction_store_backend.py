"""Correction queue backends: SQLite (default), Postgres dual-write."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.brain.correction_store_types import (
    CorrectionCategory,
    CorrectionSeverity,
    row_to_dict,
)
from brain_os.config import get_settings
from brain_os.data.crm import CRMDatabase
from brain_os.data.nemesis_corrections import NemesisCorrectionModel, PgCorrectionRepository
from brain_os.exceptions import DatabaseError

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/brain/corrections.db")


class CorrectionStoreBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def add_correction(
        self,
        entity: str,
        new_value: str,
        *,
        category: CorrectionCategory = CorrectionCategory.GENERAL,
        severity: CorrectionSeverity = CorrectionSeverity.MEDIUM,
        old_value: str = "",
        source: str = "unknown",
    ) -> int: ...

    @abstractmethod
    async def get_pending_corrections(self, limit: int = 100) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def mark_processed(self, correction_id: int) -> None: ...

    @abstractmethod
    async def get_corrections_by_entity(self, entity: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def sqlite_db_connection(self) -> aiosqlite.Connection | None: ...


class SqliteCorrectionStoreBackend(CorrectionStoreBackend):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return self._db

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path), timeout=30.0)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS corrections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entity      TEXT NOT NULL,
                category    TEXT NOT NULL,
                severity    TEXT NOT NULL DEFAULT 'MEDIUM',
                old_value   TEXT NOT NULL DEFAULT '',
                new_value   TEXT NOT NULL,
                source      TEXT NOT NULL DEFAULT 'unknown',
                created_at  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending'
            )
            """
        )
        await self._db.commit()
        logger.info("CorrectionStore (SQLite) initialised at %s", self._db_path)

    async def add_correction(
        self,
        entity: str,
        new_value: str,
        *,
        category: CorrectionCategory = CorrectionCategory.GENERAL,
        severity: CorrectionSeverity = CorrectionSeverity.MEDIUM,
        old_value: str = "",
        source: str = "unknown",
    ) -> int:
        assert self._db is not None
        now = datetime.now(UTC).isoformat()
        cursor = await self._db.execute(
            """
            INSERT INTO corrections (entity, category, severity, old_value, new_value, source, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (entity, category.value, severity.value, old_value, new_value, source, now),
        )
        await self._db.commit()
        return int(cursor.lastrowid or 0)

    async def get_pending_corrections(self, limit: int = 100) -> list[dict[str, Any]]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT id, entity, category, severity, old_value, new_value, source, created_at, status "
            "FROM corrections WHERE status = 'pending' ORDER BY created_at ASC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [row_to_dict(r) for r in rows]

    async def mark_processed(self, correction_id: int) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE corrections SET status = 'processed' WHERE id = ?",
            (correction_id,),
        )
        await self._db.commit()

    async def get_corrections_by_entity(self, entity: str) -> list[dict[str, Any]]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT id, entity, category, severity, old_value, new_value, source, created_at, status "
            "FROM corrections WHERE entity = ? ORDER BY created_at DESC",
            (entity,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [row_to_dict(r) for r in rows]

    async def get_stats(self) -> dict[str, Any]:
        assert self._db is not None
        cursor = await self._db.execute("SELECT status, COUNT(*) FROM corrections GROUP BY status")
        status_counts = dict(await cursor.fetchall())
        await cursor.close()
        cursor = await self._db.execute(
            "SELECT category, COUNT(*) FROM corrections GROUP BY category"
        )
        category_counts = dict(await cursor.fetchall())
        await cursor.close()
        cursor = await self._db.execute(
            "SELECT severity, COUNT(*) FROM corrections GROUP BY severity"
        )
        severity_counts = dict(await cursor.fetchall())
        await cursor.close()
        return {
            "total": sum(status_counts.values()),
            "by_status": status_counts,
            "by_category": category_counts,
            "by_severity": severity_counts,
        }

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


class PgCorrectionStoreBackend(CorrectionStoreBackend):
    def __init__(self, repo: PgCorrectionRepository | None = None) -> None:
        if repo is None:
            repo = PgCorrectionRepository(CRMDatabase().session_factory)
        self._repo = repo

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return None

    async def initialize(self) -> None:
        await ensure_correction_tables()

    async def add_correction(
        self,
        entity: str,
        new_value: str,
        *,
        category: CorrectionCategory = CorrectionCategory.GENERAL,
        severity: CorrectionSeverity = CorrectionSeverity.MEDIUM,
        old_value: str = "",
        source: str = "unknown",
        row_id: int | None = None,
    ) -> int:
        now = datetime.now(UTC).isoformat()
        return await self._repo.add_correction(
            entity,
            category.value,
            severity.value,
            old_value,
            new_value,
            source,
            now,
            row_id=row_id,
        )

    async def get_pending_corrections(self, limit: int = 100) -> list[dict[str, Any]]:
        return await self._repo.get_pending_corrections(limit=limit)

    async def mark_processed(self, correction_id: int) -> None:
        await self._repo.mark_processed(correction_id)

    async def get_corrections_by_entity(self, entity: str) -> list[dict[str, Any]]:
        return await self._repo.get_corrections_by_entity(entity)

    async def get_stats(self) -> dict[str, Any]:
        return await self._repo.get_stats()

    async def close(self) -> None:
        return None


class DualWriteCorrectionStoreBackend(CorrectionStoreBackend):
    def __init__(self, sqlite: SqliteCorrectionStoreBackend, pg: PgCorrectionStoreBackend) -> None:
        self._sqlite = sqlite
        self._pg = pg

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return self._sqlite.sqlite_db_connection

    def _read(self) -> CorrectionStoreBackend:
        if getattr(get_settings().app, "pg_store_corrections_read", False):
            return self._pg
        return self._sqlite

    async def initialize(self) -> None:
        await self._sqlite.initialize()
        await self._pg.initialize()

    async def add_correction(
        self,
        entity: str,
        new_value: str,
        *,
        category: CorrectionCategory = CorrectionCategory.GENERAL,
        severity: CorrectionSeverity = CorrectionSeverity.MEDIUM,
        old_value: str = "",
        source: str = "unknown",
    ) -> int:
        row_id = await self._sqlite.add_correction(
            entity,
            new_value,
            category=category,
            severity=severity,
            old_value=old_value,
            source=source,
        )
        try:
            await self._pg.add_correction(
                entity,
                new_value,
                category=category,
                severity=severity,
                old_value=old_value,
                source=source,
                row_id=row_id,
            )
        except (DatabaseError, OSError, RuntimeError):
            logger.warning("Postgres correction shadow-write failed", exc_info=True)
        return row_id

    async def get_pending_corrections(self, limit: int = 100) -> list[dict[str, Any]]:
        return await self._read().get_pending_corrections(limit=limit)

    async def mark_processed(self, correction_id: int) -> None:
        await self._sqlite.mark_processed(correction_id)
        try:
            await self._pg.mark_processed(correction_id)
        except (DatabaseError, OSError, RuntimeError):
            logger.warning("Postgres correction shadow mark_processed failed", exc_info=True)

    async def get_corrections_by_entity(self, entity: str) -> list[dict[str, Any]]:
        return await self._read().get_corrections_by_entity(entity)

    async def get_stats(self) -> dict[str, Any]:
        return await self._read().get_stats()

    async def close(self) -> None:
        await self._sqlite.close()
        await self._pg.close()


def build_correction_store_backend(db_path: str | Path | None = None) -> CorrectionStoreBackend:
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    sqlite = SqliteCorrectionStoreBackend(path)
    if not getattr(get_settings().app, "pg_store_corrections_enabled", False):
        return sqlite
    return DualWriteCorrectionStoreBackend(sqlite, PgCorrectionStoreBackend())


async def ensure_correction_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: NemesisCorrectionModel.__table__.create(sync_conn, checkfirst=True)
        )
