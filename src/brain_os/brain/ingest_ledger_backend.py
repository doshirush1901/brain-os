"""Ingest ledger backends: SQLite (default), Postgres dual-write."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from brain_os.config import get_settings
from brain_os.data.crm import CRMDatabase
from brain_os.data.ingest_ledger import IngestedFileModel, PgIngestLedgerRepository
from brain_os.exceptions import DatabaseError

logger = logging.getLogger(__name__)

_DEFAULT_LEDGER_PATH = Path("data/ingested_files.db")

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS ingested_files (
    path        TEXT PRIMARY KEY,
    hash        TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
)
"""


def init_sqlite_ledger(db_path: Path) -> sqlite3.Connection:
    """Create/open the SQLite ingest ledger (exported for characterization tests)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_INIT_SQL)
    conn.commit()
    return conn


class IngestLedgerBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def is_already_ingested(self, path: str, file_hash: str) -> bool: ...

    @abstractmethod
    async def record_ingestion(self, path: str, file_hash: str, chunk_count: int) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def sqlite_connection(self) -> sqlite3.Connection | None: ...


class SqliteIngestLedgerBackend(IngestLedgerBackend):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def sqlite_connection(self) -> sqlite3.Connection | None:
        return self._conn

    async def initialize(self) -> None:
        self._conn = await asyncio.to_thread(init_sqlite_ledger, self._db_path)

    async def is_already_ingested(self, path: str, file_hash: str) -> bool:
        assert self._conn is not None

        def _check() -> bool:
            row = self._conn.execute(
                "SELECT hash FROM ingested_files WHERE path = ?", (path,)
            ).fetchone()
            return row is not None and row[0] == file_hash

        return await asyncio.to_thread(_check)

    async def record_ingestion(self, path: str, file_hash: str, chunk_count: int) -> None:
        assert self._conn is not None
        ingested_at = datetime.now(UTC).isoformat()

        def _write() -> None:
            self._conn.execute(
                """
                INSERT INTO ingested_files (path, hash, chunk_count, ingested_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    hash = excluded.hash,
                    chunk_count = excluded.chunk_count,
                    ingested_at = excluded.ingested_at
                """,
                (path, file_hash, chunk_count, ingested_at),
            )
            self._conn.commit()

        await asyncio.to_thread(_write)

    def close(self) -> None:
        if self._conn is not None:
            conn = self._conn
            self._conn = None
            try:
                conn.close()
            except sqlite3.ProgrammingError:
                pass


class PgIngestLedgerBackend(IngestLedgerBackend):
    def __init__(self, repo: PgIngestLedgerRepository | None = None) -> None:
        if repo is None:
            repo = PgIngestLedgerRepository(CRMDatabase().session_factory)
        self._repo = repo

    @property
    def sqlite_connection(self) -> sqlite3.Connection | None:
        return None

    async def initialize(self) -> None:
        await ensure_ingest_ledger_tables()

    async def is_already_ingested(self, path: str, file_hash: str) -> bool:
        return await self._repo.is_already_ingested(path, file_hash)

    async def record_ingestion(self, path: str, file_hash: str, chunk_count: int) -> None:
        ingested_at = datetime.now(UTC).isoformat()
        await self._repo.record_ingestion(path, file_hash, chunk_count, ingested_at)

    def close(self) -> None:
        return None


class DualWriteIngestLedgerBackend(IngestLedgerBackend):
    def __init__(self, sqlite: SqliteIngestLedgerBackend, pg: PgIngestLedgerBackend) -> None:
        self._sqlite = sqlite
        self._pg = pg

    @property
    def sqlite_connection(self) -> sqlite3.Connection | None:
        return self._sqlite.sqlite_connection

    def _read(self) -> IngestLedgerBackend:
        if getattr(get_settings().app, "pg_store_ingest_read", False):
            return self._pg
        return self._sqlite

    async def initialize(self) -> None:
        await self._sqlite.initialize()
        await self._pg.initialize()

    async def is_already_ingested(self, path: str, file_hash: str) -> bool:
        return await self._read().is_already_ingested(path, file_hash)

    async def record_ingestion(self, path: str, file_hash: str, chunk_count: int) -> None:
        await self._sqlite.record_ingestion(path, file_hash, chunk_count)
        try:
            await self._pg.record_ingestion(path, file_hash, chunk_count)
        except (DatabaseError, OSError, RuntimeError):
            logger.warning("Postgres ingest-ledger shadow-write failed", exc_info=True)

    def close(self) -> None:
        self._sqlite.close()
        self._pg.close()


def build_ingest_ledger(ledger_path: Path | None = None) -> IngestLedgerBackend:
    path = ledger_path or _DEFAULT_LEDGER_PATH
    sqlite = SqliteIngestLedgerBackend(path)
    if not getattr(get_settings().app, "pg_store_ingest_enabled", False):
        return sqlite
    return DualWriteIngestLedgerBackend(sqlite, PgIngestLedgerBackend())


async def ensure_ingest_ledger_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: IngestedFileModel.__table__.create(sync_conn, checkfirst=True)
        )
