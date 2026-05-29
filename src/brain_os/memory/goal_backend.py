"""Goal manager backends: SQLite (default), Postgres dual-write."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.config import get_settings
from brain_os.data.crm import CRMDatabase
from brain_os.data.goals import GoalModel, PgGoalRepository

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/goals.db")


class GoalManagerBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def insert_goal(
        self,
        goal_id: str,
        goal_type: str,
        contact_id: str,
        status: str,
        required_slots: str,
        progress: float,
        created_at: str,
        completed_at: str | None,
    ) -> None: ...

    @abstractmethod
    async def get_row(self, goal_id: str) -> tuple[Any, ...] | None: ...

    @abstractmethod
    async def update_goal(
        self,
        goal_id: str,
        required_slots: str,
        progress: float,
        status: str,
        completed_at: str | None,
    ) -> None: ...

    @abstractmethod
    async def get_active_row(self, contact_id: str) -> tuple[Any, ...] | None: ...

    @abstractmethod
    async def list_stalled_active(self, cutoff: str) -> list[tuple[Any, ...]]: ...

    @abstractmethod
    async def abandon(self, goal_id: str, status: str) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def sqlite_db_connection(self) -> aiosqlite.Connection | None: ...


class SqliteGoalManagerBackend(GoalManagerBackend):
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
            CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                goal_type TEXT NOT NULL,
                contact_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                required_slots TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_goals_contact_status ON goals(contact_id, status)"
        )
        await self._db.commit()
        logger.info("GoalManager (SQLite) initialised at %s", self._db_path)

    async def insert_goal(
        self,
        goal_id: str,
        goal_type: str,
        contact_id: str,
        status: str,
        required_slots: str,
        progress: float,
        created_at: str,
        completed_at: str | None,
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO goals (id, goal_type, contact_id, status, required_slots, progress, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                goal_id,
                goal_type,
                contact_id,
                status,
                required_slots,
                progress,
                created_at,
                completed_at,
            ),
        )
        await self._db.commit()

    async def get_row(self, goal_id: str) -> tuple[Any, ...] | None:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT id, goal_type, contact_id, status, required_slots, progress, created_at, completed_at FROM goals WHERE id = ?",
            (goal_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row

    async def update_goal(
        self,
        goal_id: str,
        required_slots: str,
        progress: float,
        status: str,
        completed_at: str | None,
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            UPDATE goals SET required_slots = ?, progress = ?, status = ?, completed_at = ?
            WHERE id = ?
            """,
            (required_slots, progress, status, completed_at, goal_id),
        )
        await self._db.commit()

    async def get_active_row(self, contact_id: str) -> tuple[Any, ...] | None:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT id, goal_type, contact_id, status, required_slots, progress, created_at, completed_at
            FROM goals WHERE contact_id = ? AND status = 'ACTIVE'
            ORDER BY created_at DESC LIMIT 1
            """,
            (contact_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row

    async def list_stalled_active(self, cutoff: str) -> list[tuple[Any, ...]]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT id, goal_type, contact_id, status, required_slots, progress, "
            "created_at, completed_at "
            "FROM goals WHERE status = 'ACTIVE' AND created_at < ? "
            "ORDER BY created_at ASC",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return list(rows)

    async def abandon(self, goal_id: str, status: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE goals SET status = ? WHERE id = ?",
            (status, goal_id),
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


class PgGoalManagerBackend(GoalManagerBackend):
    def __init__(self, repo: PgGoalRepository | None = None) -> None:
        if repo is None:
            repo = PgGoalRepository(CRMDatabase().session_factory)
        self._repo = repo

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return None

    async def initialize(self) -> None:
        await ensure_goal_tables()

    async def insert_goal(
        self,
        goal_id: str,
        goal_type: str,
        contact_id: str,
        status: str,
        required_slots: str,
        progress: float,
        created_at: str,
        completed_at: str | None,
    ) -> None:
        await self._repo.insert_goal(
            goal_id,
            goal_type,
            contact_id,
            status,
            required_slots,
            progress,
            created_at,
            completed_at,
        )

    async def get_row(self, goal_id: str) -> tuple[Any, ...] | None:
        return await self._repo.get_row(goal_id)

    async def update_goal(
        self,
        goal_id: str,
        required_slots: str,
        progress: float,
        status: str,
        completed_at: str | None,
    ) -> None:
        await self._repo.update_goal(goal_id, required_slots, progress, status, completed_at)

    async def get_active_row(self, contact_id: str) -> tuple[Any, ...] | None:
        return await self._repo.get_active_row(contact_id)

    async def list_stalled_active(self, cutoff: str) -> list[tuple[Any, ...]]:
        return await self._repo.list_stalled_active(cutoff)

    async def abandon(self, goal_id: str, status: str) -> None:
        await self._repo.abandon(goal_id, status)

    async def close(self) -> None:
        return None


class DualWriteGoalManagerBackend(GoalManagerBackend):
    def __init__(self, sqlite: SqliteGoalManagerBackend, pg: PgGoalManagerBackend) -> None:
        self._sqlite = sqlite
        self._pg = pg

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return self._sqlite.sqlite_db_connection

    def _read(self) -> GoalManagerBackend:
        if getattr(get_settings().app, "pg_store_goals_read", False):
            return self._pg
        return self._sqlite

    async def initialize(self) -> None:
        await self._sqlite.initialize()
        await self._pg.initialize()

    async def insert_goal(
        self,
        goal_id: str,
        goal_type: str,
        contact_id: str,
        status: str,
        required_slots: str,
        progress: float,
        created_at: str,
        completed_at: str | None,
    ) -> None:
        await self._sqlite.insert_goal(
            goal_id,
            goal_type,
            contact_id,
            status,
            required_slots,
            progress,
            created_at,
            completed_at,
        )
        try:
            await self._pg.insert_goal(
                goal_id,
                goal_type,
                contact_id,
                status,
                required_slots,
                progress,
                created_at,
                completed_at,
            )
        except Exception:
            logger.warning("Postgres goal shadow-write failed", exc_info=True)

    async def get_row(self, goal_id: str) -> tuple[Any, ...] | None:
        return await self._read().get_row(goal_id)

    async def update_goal(
        self,
        goal_id: str,
        required_slots: str,
        progress: float,
        status: str,
        completed_at: str | None,
    ) -> None:
        await self._sqlite.update_goal(goal_id, required_slots, progress, status, completed_at)
        try:
            await self._pg.update_goal(goal_id, required_slots, progress, status, completed_at)
        except Exception:
            logger.warning("Postgres goal shadow-update failed", exc_info=True)

    async def get_active_row(self, contact_id: str) -> tuple[Any, ...] | None:
        return await self._read().get_active_row(contact_id)

    async def list_stalled_active(self, cutoff: str) -> list[tuple[Any, ...]]:
        return await self._read().list_stalled_active(cutoff)

    async def abandon(self, goal_id: str, status: str) -> None:
        await self._sqlite.abandon(goal_id, status)
        try:
            await self._pg.abandon(goal_id, status)
        except Exception:
            logger.warning("Postgres goal shadow-abandon failed", exc_info=True)

    async def close(self) -> None:
        await self._sqlite.close()
        await self._pg.close()


def build_goal_backend(db_path: str | Path | None = None) -> GoalManagerBackend:
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    sqlite = SqliteGoalManagerBackend(path)
    if not getattr(get_settings().app, "pg_store_goals_enabled", False):
        return sqlite
    return DualWriteGoalManagerBackend(sqlite, PgGoalManagerBackend())


async def ensure_goal_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: GoalModel.__table__.create(sync_conn, checkfirst=True)
        )
