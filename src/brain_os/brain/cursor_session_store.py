"""Cursor session log backends: SQLite (default), Postgres dual-write."""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.config import get_settings
from brain_os.data.crm import CRMDatabase
from brain_os.data.cursor_sessions import CursorSessionModel, PgCursorSessionRepository
from brain_os.exceptions import DatabaseError
from brain_os.memory.cursor_session_fts import ensure_cursor_sessions_fts

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/brain/cursor_sessions.db")


class CursorSessionStoreBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def insert_session(
        self,
        query: str,
        agents_used: list[str],
        response_summary: str,
        tool_calls: list[dict[str, Any]] | None,
        sources: list[str] | None,
        email_threads: list[str] | None,
        user_feedback: str | None,
        run_id: str | None,
        work_context: dict[str, Any] | None,
        learning_meta: dict[str, Any] | None,
    ) -> str: ...

    @abstractmethod
    async def get_recent_sessions(self, limit: int) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def fetch_by_run_id(self, run_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def sqlite_db_connection(self) -> aiosqlite.Connection | None: ...

    @property
    @abstractmethod
    def db_path(self) -> Path: ...


class SqliteCursorSessionStoreBackend(CursorSessionStoreBackend):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return self._db

    @property
    def db_path(self) -> Path:
        return self._db_path

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path), timeout=30.0)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._ensure_schema(self._db)
        await ensure_cursor_sessions_fts(self._db)
        await self._db.commit()
        logger.info("CursorSessionStore (SQLite) initialised at %s", self._db_path)

    async def _ensure_schema(self, db: aiosqlite.Connection) -> None:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS cursor_sessions (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                query TEXT NOT NULL,
                agents_used TEXT NOT NULL,
                response_summary TEXT NOT NULL,
                tool_calls TEXT NOT NULL DEFAULT '[]',
                sources TEXT NOT NULL DEFAULT '[]',
                email_threads TEXT NOT NULL DEFAULT '[]',
                user_feedback TEXT,
                run_id TEXT,
                work_context_json TEXT,
                learning_meta_json TEXT
            )
            """
        )
        cur = await db.execute("PRAGMA table_info(cursor_sessions)")
        existing = {row[1] for row in await cur.fetchall()}
        if "run_id" not in existing:
            await db.execute("ALTER TABLE cursor_sessions ADD COLUMN run_id TEXT")
        if "work_context_json" not in existing:
            await db.execute(
                "ALTER TABLE cursor_sessions ADD COLUMN work_context_json TEXT",
            )
        if "tool_calls" not in existing:
            await db.execute(
                "ALTER TABLE cursor_sessions ADD COLUMN tool_calls TEXT NOT NULL DEFAULT '[]'",
            )
        if "learning_meta_json" not in existing:
            await db.execute(
                "ALTER TABLE cursor_sessions ADD COLUMN learning_meta_json TEXT",
            )

    async def insert_session(
        self,
        query: str,
        agents_used: list[str],
        response_summary: str,
        tool_calls: list[dict[str, Any]] | None,
        sources: list[str] | None,
        email_threads: list[str] | None,
        user_feedback: str | None,
        run_id: str | None,
        work_context: dict[str, Any] | None,
        learning_meta: dict[str, Any] | None,
        *,
        session_id: str | None = None,
    ) -> str:
        if self._db is None:
            await self.initialize()
        assert self._db is not None
        sid = session_id or str(uuid.uuid4())
        wc_json = json.dumps(work_context, default=str) if work_context else None
        lm_json = json.dumps(learning_meta, default=str) if learning_meta else None
        await self._db.execute(
            """
            INSERT INTO cursor_sessions
                (id, timestamp, query, agents_used, response_summary, tool_calls, sources, email_threads, user_feedback, run_id, work_context_json, learning_meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid,
                time.time(),
                query,
                json.dumps(agents_used),
                response_summary[:500],
                json.dumps(tool_calls or []),
                json.dumps(sources or []),
                json.dumps(email_threads or []),
                user_feedback,
                run_id,
                wc_json,
                lm_json,
            ),
        )
        await ensure_cursor_sessions_fts(self._db)
        await self._db.commit()
        return sid

    async def get_recent_sessions(self, limit: int) -> list[dict[str, Any]]:
        if not self._db_path.exists():
            return []
        if self._db is None:
            async with aiosqlite.connect(str(self._db_path), timeout=30.0) as db:
                await db.execute("PRAGMA busy_timeout=30000")
                return await self._fetch_recent(db, limit)
        await self._db.execute("PRAGMA busy_timeout=30000")
        return await self._fetch_recent(self._db, limit)

    async def _fetch_recent(self, db: aiosqlite.Connection, limit: int) -> list[dict[str, Any]]:
        cursor = await db.execute(
            "SELECT * FROM cursor_sessions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        await cursor.close()
        return [dict(zip(cols, row)) for row in rows]

    async def fetch_by_run_id(self, run_id: str) -> dict[str, Any] | None:
        rid = (run_id or "").strip()
        if not rid or not self._db_path.exists():
            return None
        async with aiosqlite.connect(str(self._db_path), timeout=30.0) as db:
            await db.execute("PRAGMA busy_timeout=30000")
            cur = await db.execute(
                "SELECT * FROM cursor_sessions WHERE run_id = ? ORDER BY timestamp DESC LIMIT 1",
                (rid,),
            )
            row = await cur.fetchone()
            if row is None:
                await cur.close()
                return None
            cols = [d[0] for d in cur.description]
            await cur.close()
        return dict(zip(cols, row))

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


class PgCursorSessionStoreBackend(CursorSessionStoreBackend):
    def __init__(self, repo: PgCursorSessionRepository | None = None) -> None:
        if repo is None:
            repo = PgCursorSessionRepository(CRMDatabase().session_factory)
        self._repo = repo
        self._db_path = _DEFAULT_DB_PATH

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return None

    @property
    def db_path(self) -> Path:
        return self._db_path

    async def initialize(self) -> None:
        await ensure_cursor_session_tables()

    async def insert_session(
        self,
        query: str,
        agents_used: list[str],
        response_summary: str,
        tool_calls: list[dict[str, Any]] | None,
        sources: list[str] | None,
        email_threads: list[str] | None,
        user_feedback: str | None,
        run_id: str | None,
        work_context: dict[str, Any] | None,
        learning_meta: dict[str, Any] | None,
        *,
        session_id: str | None = None,
    ) -> str:
        sid = session_id or str(uuid.uuid4())
        wc_json = json.dumps(work_context, default=str) if work_context else None
        lm_json = json.dumps(learning_meta, default=str) if learning_meta else None
        await self._repo.insert(
            sid,
            time.time(),
            query,
            json.dumps(agents_used),
            response_summary[:500],
            json.dumps(tool_calls or []),
            json.dumps(sources or []),
            json.dumps(email_threads or []),
            user_feedback,
            run_id,
            wc_json,
            lm_json,
        )
        return sid

    async def get_recent_sessions(self, limit: int) -> list[dict[str, Any]]:
        return await self._repo.get_recent(limit)

    async def fetch_by_run_id(self, run_id: str) -> dict[str, Any] | None:
        rid = (run_id or "").strip()
        if not rid:
            return None
        return await self._repo.fetch_by_run_id(rid)

    async def close(self) -> None:
        return None


class DualWriteCursorSessionStoreBackend(CursorSessionStoreBackend):
    def __init__(
        self, sqlite: SqliteCursorSessionStoreBackend, pg: PgCursorSessionStoreBackend
    ) -> None:
        self._sqlite = sqlite
        self._pg = pg

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return self._sqlite.sqlite_db_connection

    @property
    def db_path(self) -> Path:
        return self._sqlite.db_path

    def _read(self) -> CursorSessionStoreBackend:
        if getattr(get_settings().app, "pg_store_cursor_sessions_read", False):
            return self._pg
        return self._sqlite

    async def initialize(self) -> None:
        await self._sqlite.initialize()
        await self._pg.initialize()

    async def insert_session(
        self,
        query: str,
        agents_used: list[str],
        response_summary: str,
        tool_calls: list[dict[str, Any]] | None,
        sources: list[str] | None,
        email_threads: list[str] | None,
        user_feedback: str | None,
        run_id: str | None,
        work_context: dict[str, Any] | None,
        learning_meta: dict[str, Any] | None,
        *,
        session_id: str | None = None,
    ) -> str:
        sid = session_id or str(uuid.uuid4())
        await self._sqlite.insert_session(
            query,
            agents_used,
            response_summary,
            tool_calls,
            sources,
            email_threads,
            user_feedback,
            run_id,
            work_context,
            learning_meta,
            session_id=sid,
        )
        try:
            await self._pg.insert_session(
                query,
                agents_used,
                response_summary,
                tool_calls,
                sources,
                email_threads,
                user_feedback,
                run_id,
                work_context,
                learning_meta,
                session_id=sid,
            )
        except (DatabaseError, OSError, RuntimeError):
            logger.warning("Postgres cursor session shadow-write failed", exc_info=True)
        return sid

    async def get_recent_sessions(self, limit: int) -> list[dict[str, Any]]:
        return await self._read().get_recent_sessions(limit)

    async def fetch_by_run_id(self, run_id: str) -> dict[str, Any] | None:
        return await self._read().fetch_by_run_id(run_id)

    async def close(self) -> None:
        await self._sqlite.close()
        await self._pg.close()


def build_cursor_session_store(db_path: str | Path | None = None) -> CursorSessionStoreBackend:
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    sqlite = SqliteCursorSessionStoreBackend(path)
    if not getattr(get_settings().app, "pg_store_cursor_sessions_enabled", False):
        return sqlite
    return DualWriteCursorSessionStoreBackend(sqlite, PgCursorSessionStoreBackend())


async def ensure_cursor_session_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: CursorSessionModel.__table__.create(sync_conn, checkfirst=True)
        )
