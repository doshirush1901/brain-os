"""Procedural memory backends: SQLite (default), Postgres dual-write."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import aiosqlite

from brain_os.config import get_settings
from brain_os.data.crm import CRMDatabase
from brain_os.data.procedures import PgProcedureRepository, ProcedureModel

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/conversations.db")


class ProceduralMemoryBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def load_all_rows(self) -> list[tuple[Any, ...]]: ...

    @abstractmethod
    async def insert_procedure(
        self,
        trigger_pattern: str,
        steps: str,
        last_used: str,
        *,
        trigger_embedding: str | None = None,
    ) -> int: ...

    @abstractmethod
    async def update_trigger_embedding(self, procedure_id: int, trigger_embedding: str) -> None: ...

    @abstractmethod
    async def update_procedure_merge(
        self,
        procedure_id: int,
        last_used: str,
        success_rate: float,
        steps: str | None,
    ) -> None: ...

    @abstractmethod
    async def update_success_rate(self, procedure_id: int, success_rate: float) -> None: ...

    @abstractmethod
    async def touch_procedure(self, procedure_id: int, last_used: str) -> None: ...

    @abstractmethod
    async def delete_procedure(self, procedure_id: int) -> None: ...

    @abstractmethod
    async def get_top_rows(self, limit: int) -> list[tuple[Any, ...]]: ...

    @abstractmethod
    async def list_rows(
        self, *, limit: int, sort: Literal["last_used", "score"]
    ) -> list[tuple[Any, ...]]: ...

    @abstractmethod
    async def count_procedures(self) -> int: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def sqlite_db_connection(self) -> aiosqlite.Connection | None: ...


class SqliteProceduralMemoryBackend(ProceduralMemoryBackend):
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
            CREATE TABLE IF NOT EXISTS procedures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_pattern TEXT NOT NULL,
                steps TEXT NOT NULL,
                success_rate REAL NOT NULL DEFAULT 1.0,
                times_used INTEGER NOT NULL DEFAULT 1,
                last_used TEXT NOT NULL,
                trigger_embedding TEXT
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_procedures_pattern ON procedures(trigger_pattern)"
        )
        # Migrate older DBs that predate the embedding cache column.
        cursor = await self._db.execute("PRAGMA table_info(procedures)")
        cols = {row[1] for row in await cursor.fetchall()}
        await cursor.close()
        if "trigger_embedding" not in cols:
            await self._db.execute("ALTER TABLE procedures ADD COLUMN trigger_embedding TEXT")
        await self._db.commit()
        logger.info("ProceduralMemory (SQLite) initialised at %s", self._db_path)

    async def load_all_rows(self) -> list[tuple[Any, ...]]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT id, trigger_pattern, steps, success_rate, times_used, last_used,
                   trigger_embedding
            FROM procedures
            """
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return list(rows)

    async def insert_procedure(
        self,
        trigger_pattern: str,
        steps: str,
        last_used: str,
        *,
        trigger_embedding: str | None = None,
    ) -> int:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO procedures (
                trigger_pattern, steps, success_rate, times_used, last_used, trigger_embedding
            )
            VALUES (?, ?, 1.0, 1, ?, ?)
            """,
            (trigger_pattern, steps, last_used, trigger_embedding),
        )
        await self._db.commit()
        cursor = await self._db.execute("SELECT last_insert_rowid()")
        row = await cursor.fetchone()
        await cursor.close()
        return int(row[0] if row else 0)

    async def update_trigger_embedding(self, procedure_id: int, trigger_embedding: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE procedures SET trigger_embedding = ? WHERE id = ?",
            (trigger_embedding, procedure_id),
        )
        await self._db.commit()

    async def update_procedure_merge(
        self,
        procedure_id: int,
        last_used: str,
        success_rate: float,
        steps: str | None,
    ) -> None:
        assert self._db is not None
        if steps is not None:
            await self._db.execute(
                """
                UPDATE procedures
                SET times_used = times_used + 1, last_used = ?, success_rate = ?, steps = ?
                WHERE id = ?
                """,
                (last_used, success_rate, steps, procedure_id),
            )
        else:
            await self._db.execute(
                """
                UPDATE procedures
                SET times_used = times_used + 1, last_used = ?, success_rate = ?
                WHERE id = ?
                """,
                (last_used, success_rate, procedure_id),
            )
        await self._db.commit()

    async def update_success_rate(self, procedure_id: int, success_rate: float) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE procedures SET success_rate = ? WHERE id = ?",
            (success_rate, procedure_id),
        )
        await self._db.commit()

    async def touch_procedure(self, procedure_id: int, last_used: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE procedures SET last_used = ? WHERE id = ?",
            (last_used, procedure_id),
        )
        await self._db.commit()

    async def delete_procedure(self, procedure_id: int) -> None:
        assert self._db is not None
        await self._db.execute("DELETE FROM procedures WHERE id = ?", (procedure_id,))
        await self._db.commit()

    async def get_top_rows(self, limit: int) -> list[tuple[Any, ...]]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT id, trigger_pattern, steps, success_rate, times_used, last_used,
                   trigger_embedding
            FROM procedures
            ORDER BY (success_rate * times_used) DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return list(rows)

    async def list_rows(
        self, *, limit: int, sort: Literal["last_used", "score"]
    ) -> list[tuple[Any, ...]]:
        assert self._db is not None
        order = "last_used DESC" if sort == "last_used" else "(success_rate * times_used) DESC"
        cursor = await self._db.execute(
            f"""
            SELECT id, trigger_pattern, steps, success_rate, times_used, last_used,
                   trigger_embedding
            FROM procedures ORDER BY {order} LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return list(rows)

    async def count_procedures(self) -> int:
        assert self._db is not None
        cursor = await self._db.execute("SELECT count(*) FROM procedures")
        row = await cursor.fetchone()
        await cursor.close()
        return row[0] if row else 0

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


class PgProceduralMemoryBackend(ProceduralMemoryBackend):
    def __init__(self, repo: PgProcedureRepository | None = None) -> None:
        if repo is None:
            repo = PgProcedureRepository(CRMDatabase().session_factory)
        self._repo = repo

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return None

    async def initialize(self) -> None:
        await ensure_procedural_tables()

    async def load_all_rows(self) -> list[tuple[Any, ...]]:
        return await self._repo.load_all()

    async def insert_procedure(
        self,
        trigger_pattern: str,
        steps: str,
        last_used: str,
        *,
        row_id: int | None = None,
        trigger_embedding: str | None = None,
    ) -> int:
        return await self._repo.insert(
            trigger_pattern,
            steps,
            1.0,
            1,
            last_used,
            row_id=row_id,
            trigger_embedding=trigger_embedding,
        )

    async def update_trigger_embedding(self, procedure_id: int, trigger_embedding: str) -> None:
        await self._repo.update_embedding(procedure_id, trigger_embedding)

    async def update_procedure_merge(
        self,
        procedure_id: int,
        last_used: str,
        success_rate: float,
        steps: str | None,
    ) -> None:
        await self._repo.update_merge(procedure_id, last_used, success_rate, steps)

    async def update_success_rate(self, procedure_id: int, success_rate: float) -> None:
        await self._repo.update_success_rate(procedure_id, success_rate)

    async def touch_procedure(self, procedure_id: int, last_used: str) -> None:
        await self._repo.touch(procedure_id, last_used)

    async def delete_procedure(self, procedure_id: int) -> None:
        await self._repo.delete(procedure_id)

    async def get_top_rows(self, limit: int) -> list[tuple[Any, ...]]:
        return await self._repo.get_top(limit)

    async def list_rows(
        self, *, limit: int, sort: Literal["last_used", "score"]
    ) -> list[tuple[Any, ...]]:
        order = "last_used DESC" if sort == "last_used" else "score"
        return await self._repo.list_ordered(order, limit)

    async def count_procedures(self) -> int:
        return await self._repo.count()

    async def close(self) -> None:
        return None


class DualWriteProceduralMemoryBackend(ProceduralMemoryBackend):
    def __init__(
        self, sqlite: SqliteProceduralMemoryBackend, pg: PgProceduralMemoryBackend
    ) -> None:
        self._sqlite = sqlite
        self._pg = pg

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return self._sqlite.sqlite_db_connection

    def _read(self) -> ProceduralMemoryBackend:
        if getattr(get_settings().app, "pg_store_procedural_read", False):
            return self._pg
        return self._sqlite

    async def initialize(self) -> None:
        await self._sqlite.initialize()
        await self._pg.initialize()

    async def load_all_rows(self) -> list[tuple[Any, ...]]:
        return await self._read().load_all_rows()

    async def insert_procedure(
        self,
        trigger_pattern: str,
        steps: str,
        last_used: str,
        *,
        trigger_embedding: str | None = None,
    ) -> int:
        row_id = await self._sqlite.insert_procedure(
            trigger_pattern, steps, last_used, trigger_embedding=trigger_embedding
        )
        try:
            await self._pg.insert_procedure(
                trigger_pattern,
                steps,
                last_used,
                row_id=row_id,
                trigger_embedding=trigger_embedding,
            )
        except Exception:
            logger.warning("Postgres procedural shadow-insert failed", exc_info=True)
        return row_id

    async def update_trigger_embedding(self, procedure_id: int, trigger_embedding: str) -> None:
        await self._sqlite.update_trigger_embedding(procedure_id, trigger_embedding)
        try:
            await self._pg.update_trigger_embedding(procedure_id, trigger_embedding)
        except Exception:
            logger.warning("Postgres procedural shadow-embedding failed", exc_info=True)

    async def update_procedure_merge(
        self,
        procedure_id: int,
        last_used: str,
        success_rate: float,
        steps: str | None,
    ) -> None:
        await self._sqlite.update_procedure_merge(procedure_id, last_used, success_rate, steps)
        try:
            await self._pg.update_procedure_merge(procedure_id, last_used, success_rate, steps)
        except Exception:
            logger.warning("Postgres procedural shadow-update failed", exc_info=True)

    async def update_success_rate(self, procedure_id: int, success_rate: float) -> None:
        await self._sqlite.update_success_rate(procedure_id, success_rate)
        try:
            await self._pg.update_success_rate(procedure_id, success_rate)
        except Exception:
            logger.warning("Postgres procedural shadow-rate failed", exc_info=True)

    async def touch_procedure(self, procedure_id: int, last_used: str) -> None:
        await self._sqlite.touch_procedure(procedure_id, last_used)
        try:
            await self._pg.touch_procedure(procedure_id, last_used)
        except Exception:
            logger.warning("Postgres procedural shadow-touch failed", exc_info=True)

    async def delete_procedure(self, procedure_id: int) -> None:
        await self._sqlite.delete_procedure(procedure_id)
        try:
            await self._pg.delete_procedure(procedure_id)
        except Exception:
            logger.warning("Postgres procedural shadow-delete failed", exc_info=True)

    async def get_top_rows(self, limit: int) -> list[tuple[Any, ...]]:
        return await self._read().get_top_rows(limit)

    async def list_rows(
        self, *, limit: int, sort: Literal["last_used", "score"]
    ) -> list[tuple[Any, ...]]:
        return await self._read().list_rows(limit=limit, sort=sort)

    async def count_procedures(self) -> int:
        return await self._read().count_procedures()

    async def close(self) -> None:
        await self._sqlite.close()
        await self._pg.close()


def build_procedural_backend(db_path: str | Path | None = None) -> ProceduralMemoryBackend:
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    sqlite = SqliteProceduralMemoryBackend(path)
    if not getattr(get_settings().app, "pg_store_procedural_enabled", False):
        return sqlite
    return DualWriteProceduralMemoryBackend(sqlite, PgProceduralMemoryBackend())


async def ensure_procedural_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: ProcedureModel.__table__.create(sync_conn, checkfirst=True)
        )

        # Add embedding column on older PG schemas created before revision 038.
        def _ensure_embedding_col(sync_conn: Any) -> None:
            from sqlalchemy import inspect, text

            insp = inspect(sync_conn)
            if not insp.has_table("procedures"):
                return
            cols = {c["name"] for c in insp.get_columns("procedures")}
            if "trigger_embedding" not in cols:
                sync_conn.execute(text("ALTER TABLE procedures ADD COLUMN trigger_embedding TEXT"))

        await conn.run_sync(_ensure_embedding_col)
