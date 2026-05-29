"""Relationship memory backends: SQLite (default), Postgres dual-write."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.config import get_settings
from brain_os.data.crm import CRMDatabase
from brain_os.data.models import WarmthLevel
from brain_os.data.relationships import PgRelationshipRepository, RelationshipModel

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/relationships.db")
_WARMTH_ORDER = list(WarmthLevel)


class RelationshipMemoryBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def upsert_row(
        self,
        contact_id: str,
        warmth_level: str,
        interaction_count: int,
        memorable_moments: str,
        learned_preferences: str,
        first_interaction: str | None,
        last_interaction: str | None,
    ) -> None: ...

    @abstractmethod
    async def get_row(self, contact_id: str) -> tuple[Any, ...] | None: ...

    @abstractmethod
    async def list_rows(self, min_warmth: WarmthLevel | None = None) -> list[tuple[Any, ...]]: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def sqlite_db_connection(self) -> aiosqlite.Connection | None: ...


class SqliteRelationshipMemoryBackend(RelationshipMemoryBackend):
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
            CREATE TABLE IF NOT EXISTS relationships (
                contact_id TEXT PRIMARY KEY,
                warmth_level TEXT NOT NULL DEFAULT 'STRANGER',
                interaction_count INTEGER NOT NULL DEFAULT 0,
                memorable_moments TEXT NOT NULL DEFAULT '[]',
                learned_preferences TEXT NOT NULL DEFAULT '{}',
                first_interaction TEXT,
                last_interaction TEXT
            )
            """
        )
        await self._db.commit()
        logger.info("RelationshipMemory (SQLite) initialised at %s", self._db_path)

    async def upsert_row(
        self,
        contact_id: str,
        warmth_level: str,
        interaction_count: int,
        memorable_moments: str,
        learned_preferences: str,
        first_interaction: str | None,
        last_interaction: str | None,
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT OR REPLACE INTO relationships
            (contact_id, warmth_level, interaction_count, memorable_moments, learned_preferences, first_interaction, last_interaction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                contact_id,
                warmth_level,
                interaction_count,
                memorable_moments,
                learned_preferences,
                first_interaction,
                last_interaction,
            ),
        )
        await self._db.commit()

    async def get_row(self, contact_id: str) -> tuple[Any, ...] | None:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT contact_id, warmth_level, interaction_count, memorable_moments, learned_preferences, first_interaction, last_interaction FROM relationships WHERE contact_id = ?",
            (contact_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row

    async def list_rows(self, min_warmth: WarmthLevel | None = None) -> list[tuple[Any, ...]]:
        assert self._db is not None
        if min_warmth is not None:
            idx = _WARMTH_ORDER.index(min_warmth)
            levels = [w.value for w in _WARMTH_ORDER[idx:]]
            placeholders = ",".join("?" * len(levels))
            cursor = await self._db.execute(
                f"SELECT contact_id, warmth_level, interaction_count, memorable_moments, learned_preferences, first_interaction, last_interaction FROM relationships WHERE warmth_level IN ({placeholders})",
                levels,
            )
        else:
            cursor = await self._db.execute(
                "SELECT contact_id, warmth_level, interaction_count, memorable_moments, learned_preferences, first_interaction, last_interaction FROM relationships"
            )
        rows = await cursor.fetchall()
        await cursor.close()
        return list(rows)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


class PgRelationshipMemoryBackend(RelationshipMemoryBackend):
    def __init__(self, repo: PgRelationshipRepository | None = None) -> None:
        if repo is None:
            repo = PgRelationshipRepository(CRMDatabase().session_factory)
        self._repo = repo

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return None

    async def initialize(self) -> None:
        await ensure_relationship_tables()

    async def upsert_row(
        self,
        contact_id: str,
        warmth_level: str,
        interaction_count: int,
        memorable_moments: str,
        learned_preferences: str,
        first_interaction: str | None,
        last_interaction: str | None,
    ) -> None:
        await self._repo.upsert(
            contact_id,
            warmth_level,
            interaction_count,
            memorable_moments,
            learned_preferences,
            first_interaction,
            last_interaction,
        )

    async def get_row(self, contact_id: str) -> tuple[Any, ...] | None:
        return await self._repo.get(contact_id)

    async def list_rows(self, min_warmth: WarmthLevel | None = None) -> list[tuple[Any, ...]]:
        levels = None
        if min_warmth is not None:
            idx = _WARMTH_ORDER.index(min_warmth)
            levels = [w.value for w in _WARMTH_ORDER[idx:]]
        return await self._repo.list_rows(levels)

    async def close(self) -> None:
        return None


class DualWriteRelationshipMemoryBackend(RelationshipMemoryBackend):
    def __init__(
        self,
        sqlite: SqliteRelationshipMemoryBackend,
        pg: PgRelationshipMemoryBackend,
    ) -> None:
        self._sqlite = sqlite
        self._pg = pg

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return self._sqlite.sqlite_db_connection

    def _read(self) -> RelationshipMemoryBackend:
        if getattr(get_settings().app, "pg_store_relationship_read", False):
            return self._pg
        return self._sqlite

    async def initialize(self) -> None:
        await self._sqlite.initialize()
        await self._pg.initialize()

    async def upsert_row(
        self,
        contact_id: str,
        warmth_level: str,
        interaction_count: int,
        memorable_moments: str,
        learned_preferences: str,
        first_interaction: str | None,
        last_interaction: str | None,
    ) -> None:
        await self._sqlite.upsert_row(
            contact_id,
            warmth_level,
            interaction_count,
            memorable_moments,
            learned_preferences,
            first_interaction,
            last_interaction,
        )
        try:
            await self._pg.upsert_row(
                contact_id,
                warmth_level,
                interaction_count,
                memorable_moments,
                learned_preferences,
                first_interaction,
                last_interaction,
            )
        except Exception:
            logger.warning("Postgres relationship shadow-write failed", exc_info=True)

    async def get_row(self, contact_id: str) -> tuple[Any, ...] | None:
        return await self._read().get_row(contact_id)

    async def list_rows(self, min_warmth: WarmthLevel | None = None) -> list[tuple[Any, ...]]:
        return await self._read().list_rows(min_warmth)

    async def close(self) -> None:
        await self._sqlite.close()
        await self._pg.close()


def build_relationship_backend(db_path: str | Path | None = None) -> RelationshipMemoryBackend:
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    sqlite = SqliteRelationshipMemoryBackend(path)
    if not getattr(get_settings().app, "pg_store_relationship_enabled", False):
        return sqlite
    return DualWriteRelationshipMemoryBackend(sqlite, PgRelationshipMemoryBackend())


async def ensure_relationship_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: RelationshipModel.__table__.create(sync_conn, checkfirst=True)
        )
