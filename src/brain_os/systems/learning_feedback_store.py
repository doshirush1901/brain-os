"""Learning-hub feedback backends: SQLite (default), Postgres dual-write."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from brain_os.config import get_settings
from brain_os.data.crm import CRMDatabase
from brain_os.data.learning_feedback import LearningFeedbackModel, PgLearningFeedbackRepository

logger = logging.getLogger(__name__)

_PG_DUAL_WRITE_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    asyncio.TimeoutError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    SQLAlchemyError,
)

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id TEXT NOT NULL,
    feedback_score INTEGER NOT NULL,
    correction TEXT,
    correction_analysis TEXT DEFAULT '{}',
    gap_analysis TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
)
"""


def _run_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("learning feedback store sync API cannot run inside a running event loop")


class LearningFeedbackStore(ABC):
    @abstractmethod
    def ensure_initialized(self) -> None: ...

    @abstractmethod
    def load_all(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def insert(
        self,
        interaction_id: str,
        feedback_score: int,
        correction: str | None,
        correction_analysis: dict[str, Any],
        gap_analysis: dict[str, Any],
        created_at: str,
    ) -> None: ...


class SqliteLearningFeedbackStore(LearningFeedbackStore):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def ensure_initialized(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.executescript(_INIT_SQL)
            conn.commit()
        finally:
            conn.close()

    def load_all(self) -> list[dict[str, Any]]:
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            rows = conn.execute(
                "SELECT id, interaction_id, feedback_score, correction, "
                "correction_analysis, gap_analysis, created_at "
                "FROM feedback ORDER BY id"
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "id": row[0],
                "interaction_id": row[1],
                "feedback_score": row[2],
                "correction": row[3],
                "correction_analysis": json.loads(row[4]) if row[4] else {},
                "gap_analysis": json.loads(row[5]) if row[5] else {},
                "created_at": row[6],
            }
            for row in rows
        ]

    def insert_sync(
        self,
        interaction_id: str,
        feedback_score: int,
        correction: str | None,
        correction_analysis: dict[str, Any],
        gap_analysis: dict[str, Any],
        created_at: str,
    ) -> int:
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            cursor = conn.execute(
                "INSERT INTO feedback "
                "(interaction_id, feedback_score, correction, correction_analysis, "
                "gap_analysis, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    interaction_id,
                    feedback_score,
                    correction,
                    json.dumps(correction_analysis),
                    json.dumps(gap_analysis),
                    created_at,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)
        finally:
            conn.close()

    async def insert(
        self,
        interaction_id: str,
        feedback_score: int,
        correction: str | None,
        correction_analysis: dict[str, Any],
        gap_analysis: dict[str, Any],
        created_at: str,
    ) -> None:
        await asyncio.to_thread(
            self.insert_sync,
            interaction_id,
            feedback_score,
            correction,
            correction_analysis,
            gap_analysis,
            created_at,
        )


class PgLearningFeedbackStore(LearningFeedbackStore):
    def __init__(self, repo: PgLearningFeedbackRepository) -> None:
        self._repo = repo

    def ensure_initialized(self) -> None:
        _run_sync(ensure_learning_feedback_tables())

    def load_all(self) -> list[dict[str, Any]]:
        return _run_sync(self._repo.load_all())

    async def insert(
        self,
        interaction_id: str,
        feedback_score: int,
        correction: str | None,
        correction_analysis: dict[str, Any],
        gap_analysis: dict[str, Any],
        created_at: str,
        *,
        row_id: int | None = None,
    ) -> None:
        await self._repo.insert(
            interaction_id,
            feedback_score,
            correction,
            correction_analysis,
            gap_analysis,
            created_at,
            row_id=row_id,
        )


class DualWriteLearningFeedbackStore(LearningFeedbackStore):
    def __init__(self, sqlite: SqliteLearningFeedbackStore, pg: PgLearningFeedbackStore) -> None:
        self._sqlite = sqlite
        self._pg = pg

    def ensure_initialized(self) -> None:
        self._sqlite.ensure_initialized()
        try:
            self._pg.ensure_initialized()
        except _PG_DUAL_WRITE_ERRORS:
            logger.warning("Postgres learning-feedback table init failed", exc_info=True)

    def load_all(self) -> list[dict[str, Any]]:
        if getattr(get_settings().app, "pg_store_learning_feedback_read", False):
            try:
                return self._pg.load_all()
            except _PG_DUAL_WRITE_ERRORS:
                logger.warning(
                    "Postgres learning-feedback load failed; using SQLite", exc_info=True
                )
        return self._sqlite.load_all()

    async def insert(
        self,
        interaction_id: str,
        feedback_score: int,
        correction: str | None,
        correction_analysis: dict[str, Any],
        gap_analysis: dict[str, Any],
        created_at: str,
    ) -> None:
        row_id = await asyncio.to_thread(
            self._sqlite.insert_sync,
            interaction_id,
            feedback_score,
            correction,
            correction_analysis,
            gap_analysis,
            created_at,
        )
        try:
            await self._pg.insert(
                interaction_id,
                feedback_score,
                correction,
                correction_analysis,
                gap_analysis,
                created_at,
                row_id=row_id,
            )
        except _PG_DUAL_WRITE_ERRORS:
            logger.warning("Postgres learning-feedback shadow-write failed", exc_info=True)


def build_learning_feedback_store(db_path: Path) -> LearningFeedbackStore:
    sqlite = SqliteLearningFeedbackStore(db_path)
    if not getattr(get_settings().app, "pg_store_learning_feedback_enabled", False):
        return sqlite
    repo = PgLearningFeedbackRepository(CRMDatabase().session_factory)
    return DualWriteLearningFeedbackStore(sqlite, PgLearningFeedbackStore(repo))


async def ensure_learning_feedback_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: LearningFeedbackModel.__table__.create(sync_conn, checkfirst=True)
        )
