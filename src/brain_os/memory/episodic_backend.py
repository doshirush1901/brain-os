"""Episodic memory backends: SQLite (default), Postgres dual-write."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.config import get_settings
from brain_os.data.crm import CRMDatabase
from brain_os.data.episodes import EpisodeModel, PgEpisodeRepository

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/conversations.db")


class EpisodicMemoryBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def insert_episode(
        self,
        user_id: str,
        narrative: str,
        key_topics: str,
        decisions: str,
        commitments: str,
        emotional_tone: str,
        relationship_impact: str,
        created_at: str,
    ) -> int: ...

    @abstractmethod
    async def weave_rows(
        self, user_id: str, topic: str | None, limit: int
    ) -> list[tuple[Any, ...]]: ...

    @abstractmethod
    async def surface_rows(
        self, user_id: str, keywords: list[str], limit: int
    ) -> list[tuple[Any, ...]]: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def sqlite_db_connection(self) -> aiosqlite.Connection | None: ...


class SqliteEpisodicMemoryBackend(EpisodicMemoryBackend):
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
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                narrative TEXT NOT NULL,
                key_topics TEXT NOT NULL,
                decisions TEXT NOT NULL,
                commitments TEXT NOT NULL,
                emotional_tone TEXT NOT NULL,
                relationship_impact TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id)")
        await self._db.commit()
        logger.info("EpisodicMemory (SQLite) initialised at %s", self._db_path)

    async def insert_episode(
        self,
        user_id: str,
        narrative: str,
        key_topics: str,
        decisions: str,
        commitments: str,
        emotional_tone: str,
        relationship_impact: str,
        created_at: str,
    ) -> int:
        assert self._db is not None
        row = (
            user_id,
            narrative,
            key_topics,
            decisions,
            commitments,
            emotional_tone,
            relationship_impact,
            created_at,
        )
        sql = """
            INSERT INTO episodes (
                user_id, narrative, key_topics, decisions, commitments,
                emotional_tone, relationship_impact, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
        last_error: BaseException | None = None
        for attempt in range(5):
            try:
                cursor = await self._db.execute(sql, row)
                await self._db.commit()
                return int(cursor.lastrowid or 0)
            except sqlite3.OperationalError as e:
                last_error = e
                if "locked" not in str(e).lower() and "busy" not in str(e).lower():
                    raise
                delay = 1.0 * (attempt + 1)
                logger.warning(
                    "Episodic write failed (database locked), retry %d/5 in %.1fs: %s",
                    attempt + 1,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
        logger.error("Episodic write failed after 5 retries: %s", last_error)
        raise last_error  # type: ignore[misc]

    async def weave_rows(
        self, user_id: str, topic: str | None, limit: int
    ) -> list[tuple[Any, ...]]:
        assert self._db is not None
        if topic:
            cursor = await self._db.execute(
                """
                SELECT id, narrative, created_at FROM episodes
                WHERE user_id = ? AND key_topics LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, f"%{topic}%", limit),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT id, narrative, created_at FROM episodes
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
        rows = await cursor.fetchall()
        await cursor.close()
        return list(rows)

    async def surface_rows(
        self, user_id: str, keywords: list[str], limit: int
    ) -> list[tuple[Any, ...]]:
        assert self._db is not None
        if not keywords:
            return []
        conditions = " OR ".join(["(narrative LIKE ? OR key_topics LIKE ?)"] * len(keywords))
        params: list[Any] = [user_id]
        for kw in keywords:
            pattern = f"%{kw}%"
            params.extend([pattern, pattern])
        cursor = await self._db.execute(
            f"""
            SELECT id, narrative, key_topics, emotional_tone,
                   relationship_impact, created_at
            FROM episodes
            WHERE user_id = ? AND ({conditions})
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [*params, limit],
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return list(rows)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


class PgEpisodicMemoryBackend(EpisodicMemoryBackend):
    def __init__(self, repo: PgEpisodeRepository | None = None) -> None:
        if repo is None:
            repo = PgEpisodeRepository(CRMDatabase().session_factory)
        self._repo = repo

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return None

    async def initialize(self) -> None:
        await ensure_episode_tables()

    async def insert_episode(
        self,
        user_id: str,
        narrative: str,
        key_topics: str,
        decisions: str,
        commitments: str,
        emotional_tone: str,
        relationship_impact: str,
        created_at: str,
        *,
        row_id: int | None = None,
    ) -> int:
        return await self._repo.insert(
            user_id,
            narrative,
            key_topics,
            decisions,
            commitments,
            emotional_tone,
            relationship_impact,
            created_at,
            row_id=row_id,
        )

    async def weave_rows(
        self, user_id: str, topic: str | None, limit: int
    ) -> list[tuple[Any, ...]]:
        return await self._repo.weave_rows(user_id, topic, limit)

    async def surface_rows(
        self, user_id: str, keywords: list[str], limit: int
    ) -> list[tuple[Any, ...]]:
        return await self._repo.surface_rows(user_id, keywords, limit)

    async def close(self) -> None:
        return None


class DualWriteEpisodicMemoryBackend(EpisodicMemoryBackend):
    def __init__(self, sqlite: SqliteEpisodicMemoryBackend, pg: PgEpisodicMemoryBackend) -> None:
        self._sqlite = sqlite
        self._pg = pg

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return self._sqlite.sqlite_db_connection

    def _read(self) -> EpisodicMemoryBackend:
        if getattr(get_settings().app, "pg_store_episodes_read", False):
            return self._pg
        return self._sqlite

    async def initialize(self) -> None:
        await self._sqlite.initialize()
        await self._pg.initialize()

    async def insert_episode(
        self,
        user_id: str,
        narrative: str,
        key_topics: str,
        decisions: str,
        commitments: str,
        emotional_tone: str,
        relationship_impact: str,
        created_at: str,
    ) -> int:
        row_id = await self._sqlite.insert_episode(
            user_id,
            narrative,
            key_topics,
            decisions,
            commitments,
            emotional_tone,
            relationship_impact,
            created_at,
        )
        try:
            await self._pg.insert_episode(
                user_id,
                narrative,
                key_topics,
                decisions,
                commitments,
                emotional_tone,
                relationship_impact,
                created_at,
                row_id=row_id,
            )
        except Exception:
            logger.warning("Postgres episode shadow-write failed", exc_info=True)
        return row_id

    async def weave_rows(
        self, user_id: str, topic: str | None, limit: int
    ) -> list[tuple[Any, ...]]:
        return await self._read().weave_rows(user_id, topic, limit)

    async def surface_rows(
        self, user_id: str, keywords: list[str], limit: int
    ) -> list[tuple[Any, ...]]:
        return await self._read().surface_rows(user_id, keywords, limit)

    async def close(self) -> None:
        await self._sqlite.close()
        await self._pg.close()


def build_episodic_backend(db_path: str | Path | None = None) -> EpisodicMemoryBackend:
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    sqlite = SqliteEpisodicMemoryBackend(path)
    if not getattr(get_settings().app, "pg_store_episodes_enabled", False):
        return sqlite
    return DualWriteEpisodicMemoryBackend(sqlite, PgEpisodicMemoryBackend())


async def ensure_episode_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: EpisodeModel.__table__.create(sync_conn, checkfirst=True)
        )
