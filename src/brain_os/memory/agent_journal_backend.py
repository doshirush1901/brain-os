"""Agent journal backends: SQLite (default), Postgres dual-write."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.config import get_settings
from brain_os.data.agent_journal import (
    AgentJournalDailyActionModel,
    AgentJournalEntryModel,
    PgAgentJournalRepository,
)
from brain_os.data.crm import CRMDatabase
from brain_os.memory import agent_journal_clock as _clock

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/brain/agent_journals.db")


class AgentJournalBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def log_action(
        self,
        agent_name: str,
        action_text: str,
        outcome: str,
        *,
        at_date: date | None = None,
    ) -> None: ...

    @abstractmethod
    async def get_actions_for_date(self, agent_name: str, d: date) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def get_agents_with_actions_for_date(self, d: date) -> list[str]: ...

    @abstractmethod
    async def get_agents_with_actions_since_hours(self, hours: float = 24.0) -> list[str]: ...

    @abstractmethod
    async def get_actions_since_hours(
        self, agent_name: str, hours: float = 24.0
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def save_journal_entry(
        self,
        agent_name: str,
        reflection_text: str,
        mood: str = "",
        *,
        at_date: date | None = None,
    ) -> None: ...

    @abstractmethod
    async def search_past_journals(
        self,
        agent_name: str,
        query: str = "",
        limit: int = 10,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def get_latest_journal_entry(self, agent_name: str) -> str | None: ...

    @abstractmethod
    async def get_latest_journal_created_at(self, agent_name: str) -> datetime | None: ...

    @abstractmethod
    async def get_actions_since_datetime(
        self, agent_name: str, since: datetime
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def sqlite_db_connection(self) -> aiosqlite.Connection | None: ...


class SqliteAgentJournalBackend(AgentJournalBackend):
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
            CREATE TABLE IF NOT EXISTS daily_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                date TEXT NOT NULL,
                action_text TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                date TEXT NOT NULL,
                reflection_text TEXT NOT NULL,
                mood TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_actions_agent_date "
            "ON daily_actions(agent_name, date)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_journal_entries_agent_date "
            "ON journal_entries(agent_name, date)"
        )
        await self._db.commit()
        logger.info("AgentJournal (SQLite) initialised at %s", self._db_path)

    async def log_action(
        self,
        agent_name: str,
        action_text: str,
        outcome: str,
        *,
        at_date: date | None = None,
    ) -> None:
        if self._db is None:
            return
        d = at_date or _clock.date.today()
        now = _clock.datetime.now(_clock.UTC).isoformat()
        try:
            await self._db.execute(
                """
                INSERT INTO daily_actions (agent_name, date, action_text, outcome, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (agent_name, d.isoformat(), action_text, outcome, now),
            )
            await self._db.commit()
        except Exception as exc:
            logger.warning("AgentJournal.log_action failed for %s", agent_name, exc_info=True)

    async def get_actions_for_date(self, agent_name: str, d: date) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        try:
            cursor = await self._db.execute(
                """
                SELECT action_text, outcome, created_at
                FROM daily_actions
                WHERE agent_name = ? AND date = ?
                ORDER BY created_at ASC
                """,
                (agent_name, d.isoformat()),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return [{"action_text": r[0], "outcome": r[1], "created_at": r[2]} for r in rows]
        except Exception as exc:
            logger.warning("AgentJournal.get_actions_for_date failed", exc_info=True)
            return []

    async def get_agents_with_actions_for_date(self, d: date) -> list[str]:
        if self._db is None:
            return []
        try:
            cursor = await self._db.execute(
                "SELECT DISTINCT agent_name FROM daily_actions WHERE date = ?",
                (d.isoformat(),),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return [r[0] for r in rows]
        except Exception as exc:
            logger.warning("AgentJournal.get_agents_with_actions_for_date failed", exc_info=True)
            return []

    async def get_agents_with_actions_since_hours(self, hours: float = 24.0) -> list[str]:
        if self._db is None:
            return []
        since = (_clock.datetime.now(_clock.UTC) - _clock.timedelta(hours=hours)).isoformat()
        try:
            cursor = await self._db.execute(
                "SELECT DISTINCT agent_name FROM daily_actions WHERE created_at >= ? ORDER BY agent_name",
                (since,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return [r[0] for r in rows]
        except Exception as exc:
            logger.warning("AgentJournal.get_agents_with_actions_since_hours failed", exc_info=True)
            return []

    async def get_actions_since_hours(
        self, agent_name: str, hours: float = 24.0
    ) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        since = (_clock.datetime.now(_clock.UTC) - _clock.timedelta(hours=hours)).isoformat()
        try:
            cursor = await self._db.execute(
                """
                SELECT action_text, outcome, created_at
                FROM daily_actions
                WHERE agent_name = ? AND created_at >= ?
                ORDER BY created_at ASC
                """,
                (agent_name, since),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return [{"action_text": r[0], "outcome": r[1], "created_at": r[2]} for r in rows]
        except Exception as exc:
            logger.warning("AgentJournal.get_actions_since_hours failed", exc_info=True)
            return []

    async def save_journal_entry(
        self,
        agent_name: str,
        reflection_text: str,
        mood: str = "",
        *,
        at_date: date | None = None,
    ) -> None:
        if self._db is None:
            return
        d = at_date or _clock.date.today()
        now = _clock.datetime.now(_clock.UTC).isoformat()
        try:
            await self._db.execute(
                """
                INSERT INTO journal_entries (agent_name, date, reflection_text, mood, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (agent_name, d.isoformat(), reflection_text, mood, now),
            )
            await self._db.commit()
        except Exception as exc:
            logger.warning(
                "AgentJournal.save_journal_entry failed for %s", agent_name, exc_info=True
            )

    async def search_past_journals(
        self,
        agent_name: str,
        query: str = "",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        try:
            if query.strip():
                cursor = await self._db.execute(
                    """
                    SELECT date, reflection_text, mood, created_at
                    FROM journal_entries
                    WHERE agent_name = ? AND reflection_text LIKE ?
                    ORDER BY date DESC, created_at DESC
                    LIMIT ?
                    """,
                    (agent_name, f"%{query.strip()}%", limit),
                )
            else:
                cursor = await self._db.execute(
                    """
                    SELECT date, reflection_text, mood, created_at
                    FROM journal_entries
                    WHERE agent_name = ?
                    ORDER BY date DESC, created_at DESC
                    LIMIT ?
                    """,
                    (agent_name, limit),
                )
            rows = await cursor.fetchall()
            await cursor.close()
            return [
                {"date": r[0], "reflection_text": r[1], "mood": r[2], "created_at": r[3]}
                for r in rows
            ]
        except Exception as exc:
            logger.warning("AgentJournal.search_past_journals failed", exc_info=True)
            return []

    async def get_latest_journal_entry(self, agent_name: str) -> str | None:
        if self._db is None:
            return None
        try:
            cursor = await self._db.execute(
                """
                SELECT reflection_text
                FROM journal_entries
                WHERE agent_name = ?
                ORDER BY date DESC, created_at DESC
                LIMIT 1
                """,
                (agent_name,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            return row[0] if row else None
        except Exception as exc:
            logger.warning("AgentJournal.get_latest_journal_entry failed", exc_info=True)
            return None

    async def get_latest_journal_created_at(self, agent_name: str) -> datetime | None:
        if self._db is None:
            return None
        try:
            cursor = await self._db.execute(
                """
                SELECT created_at
                FROM journal_entries
                WHERE agent_name = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (agent_name,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if not row or not row[0]:
                return None
            dt = _clock.datetime.fromisoformat(row[0].replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_clock.UTC)
            return dt
        except Exception as exc:
            logger.warning("AgentJournal.get_latest_journal_created_at failed", exc_info=True)
            return None

    async def get_actions_since_datetime(
        self, agent_name: str, since: datetime
    ) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        since_iso = since.isoformat()
        try:
            cursor = await self._db.execute(
                """
                SELECT action_text, outcome, created_at
                FROM daily_actions
                WHERE agent_name = ? AND created_at >= ?
                ORDER BY created_at ASC
                """,
                (agent_name, since_iso),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return [{"action_text": r[0], "outcome": r[1], "created_at": r[2]} for r in rows]
        except Exception as exc:
            logger.warning("AgentJournal.get_actions_since_datetime failed", exc_info=True)
            return []

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


class PgAgentJournalBackend(AgentJournalBackend):
    def __init__(self, repo: PgAgentJournalRepository | None = None) -> None:
        if repo is None:
            crm = CRMDatabase()
            repo = PgAgentJournalRepository(crm.session_factory)
        self._repo = repo

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return None

    async def initialize(self) -> None:
        await ensure_agent_journal_tables()

    async def log_action(
        self,
        agent_name: str,
        action_text: str,
        outcome: str,
        *,
        at_date: date | None = None,
    ) -> None:
        try:
            await self._repo.log_action(agent_name, action_text, outcome, at_date=at_date)
        except Exception as exc:
            logger.warning("AgentJournal.log_action failed for %s", agent_name, exc_info=True)

    async def get_actions_for_date(self, agent_name: str, d: date) -> list[dict[str, Any]]:
        try:
            return await self._repo.get_actions_for_date(agent_name, d)
        except Exception as exc:
            logger.warning("AgentJournal.get_actions_for_date failed", exc_info=True)
            return []

    async def get_agents_with_actions_for_date(self, d: date) -> list[str]:
        try:
            return await self._repo.get_agents_with_actions_for_date(d)
        except Exception as exc:
            logger.warning("AgentJournal.get_agents_with_actions_for_date failed", exc_info=True)
            return []

    async def get_agents_with_actions_since_hours(self, hours: float = 24.0) -> list[str]:
        try:
            return await self._repo.get_agents_with_actions_since_hours(hours=hours)
        except Exception as exc:
            logger.warning("AgentJournal.get_agents_with_actions_since_hours failed", exc_info=True)
            return []

    async def get_actions_since_hours(
        self, agent_name: str, hours: float = 24.0
    ) -> list[dict[str, Any]]:
        try:
            return await self._repo.get_actions_since_hours(agent_name, hours=hours)
        except Exception as exc:
            logger.warning("AgentJournal.get_actions_since_hours failed", exc_info=True)
            return []

    async def save_journal_entry(
        self,
        agent_name: str,
        reflection_text: str,
        mood: str = "",
        *,
        at_date: date | None = None,
    ) -> None:
        try:
            await self._repo.save_journal_entry(agent_name, reflection_text, mood, at_date=at_date)
        except Exception as exc:
            logger.warning(
                "AgentJournal.save_journal_entry failed for %s", agent_name, exc_info=True
            )

    async def search_past_journals(
        self,
        agent_name: str,
        query: str = "",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        try:
            return await self._repo.search_past_journals(agent_name, query=query, limit=limit)
        except Exception as exc:
            logger.warning("AgentJournal.search_past_journals failed", exc_info=True)
            return []

    async def get_latest_journal_entry(self, agent_name: str) -> str | None:
        try:
            return await self._repo.get_latest_journal_entry(agent_name)
        except Exception as exc:
            logger.warning("AgentJournal.get_latest_journal_entry failed", exc_info=True)
            return None

    async def get_latest_journal_created_at(self, agent_name: str) -> datetime | None:
        try:
            return await self._repo.get_latest_journal_created_at(agent_name)
        except Exception as exc:
            logger.warning("AgentJournal.get_latest_journal_created_at failed", exc_info=True)
            return None

    async def get_actions_since_datetime(
        self, agent_name: str, since: datetime
    ) -> list[dict[str, Any]]:
        try:
            return await self._repo.get_actions_since_datetime(agent_name, since)
        except Exception as exc:
            logger.warning("AgentJournal.get_actions_since_datetime failed", exc_info=True)
            return []

    async def close(self) -> None:
        return None


class DualWriteAgentJournalBackend(AgentJournalBackend):
    def __init__(self, sqlite: SqliteAgentJournalBackend, pg: PgAgentJournalBackend) -> None:
        self._sqlite = sqlite
        self._pg = pg

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return self._sqlite.sqlite_db_connection

    def _read_backend(self) -> AgentJournalBackend:
        if getattr(get_settings().app, "pg_store_agent_journal_read", False):
            return self._pg
        return self._sqlite

    async def initialize(self) -> None:
        await self._sqlite.initialize()
        await self._pg.initialize()

    async def log_action(
        self,
        agent_name: str,
        action_text: str,
        outcome: str,
        *,
        at_date: date | None = None,
    ) -> None:
        await self._sqlite.log_action(agent_name, action_text, outcome, at_date=at_date)
        try:
            await self._pg.log_action(agent_name, action_text, outcome, at_date=at_date)
        except Exception:
            logger.warning("Postgres agent journal shadow log_action failed", exc_info=True)

    async def get_actions_for_date(self, agent_name: str, d: date) -> list[dict[str, Any]]:
        return await self._read_backend().get_actions_for_date(agent_name, d)

    async def get_agents_with_actions_for_date(self, d: date) -> list[str]:
        return await self._read_backend().get_agents_with_actions_for_date(d)

    async def get_agents_with_actions_since_hours(self, hours: float = 24.0) -> list[str]:
        return await self._read_backend().get_agents_with_actions_since_hours(hours=hours)

    async def get_actions_since_hours(
        self, agent_name: str, hours: float = 24.0
    ) -> list[dict[str, Any]]:
        return await self._read_backend().get_actions_since_hours(agent_name, hours=hours)

    async def save_journal_entry(
        self,
        agent_name: str,
        reflection_text: str,
        mood: str = "",
        *,
        at_date: date | None = None,
    ) -> None:
        await self._sqlite.save_journal_entry(agent_name, reflection_text, mood, at_date=at_date)
        try:
            await self._pg.save_journal_entry(agent_name, reflection_text, mood, at_date=at_date)
        except Exception:
            logger.warning("Postgres agent journal shadow save_journal_entry failed", exc_info=True)

    async def search_past_journals(
        self,
        agent_name: str,
        query: str = "",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return await self._read_backend().search_past_journals(agent_name, query=query, limit=limit)

    async def get_latest_journal_entry(self, agent_name: str) -> str | None:
        return await self._read_backend().get_latest_journal_entry(agent_name)

    async def get_latest_journal_created_at(self, agent_name: str) -> datetime | None:
        return await self._read_backend().get_latest_journal_created_at(agent_name)

    async def get_actions_since_datetime(
        self, agent_name: str, since: datetime
    ) -> list[dict[str, Any]]:
        return await self._read_backend().get_actions_since_datetime(agent_name, since)

    async def close(self) -> None:
        await self._sqlite.close()
        await self._pg.close()


def build_agent_journal_backend(db_path: str | Path | None) -> AgentJournalBackend:
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    sqlite = SqliteAgentJournalBackend(path)
    if not getattr(get_settings().app, "pg_store_agent_journal_enabled", False):
        return sqlite
    return DualWriteAgentJournalBackend(sqlite, PgAgentJournalBackend())


async def ensure_agent_journal_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: AgentJournalDailyActionModel.__table__.create(
                sync_conn, checkfirst=True
            )
        )
        await conn.run_sync(
            lambda sync_conn: AgentJournalEntryModel.__table__.create(sync_conn, checkfirst=True)
        )
