"""Run-record backends: SQLite (default), Postgres dual-write."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.config import get_settings
from brain_os.data.crm import CRMDatabase
from brain_os.data.run_records_pg import PgRunRecordRepository, RunRecordModel
from brain_os.exceptions import DatabaseError
from brain_os.schemas.run_record import RunRecord, RunRecordSummary
from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)

_PRUNE_EVERY_N = 100


def run_records_db_path() -> Path:
    raw = (get_settings().app.run_record_db_path or "").strip()
    if not raw:
        raw = "brain/run_records.sqlite"
    candidate = Path(raw)
    if candidate.is_absolute():
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate
    p = get_data_dir() / raw.lstrip("/")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class RunRecordStore(ABC):
    """Persistence for :class:`~brain_os.schemas.run_record.RunRecord` JSON payloads."""

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def save(self, record: RunRecord) -> None: ...

    @abstractmethod
    async def get(self, run_id: str) -> RunRecord | None: ...

    @abstractmethod
    async def row_count(self) -> int: ...

    @abstractmethod
    async def list_all_records(
        self,
        *,
        since_ts: float | None = None,
        limit: int | None = None,
    ) -> list[RunRecord]: ...

    @abstractmethod
    async def list_recent(
        self,
        *,
        limit: int = 50,
        channel: str | None = None,
        since_ts: float | None = None,
    ) -> list[RunRecordSummary]: ...

    @abstractmethod
    async def list_recent_records(
        self,
        *,
        limit: int = 50,
        channel: str | None = None,
        since_ts: float | None = None,
    ) -> list[RunRecord]: ...


class SqliteRunRecordStore(RunRecordStore):
    """SQLite store for :class:`~brain_os.schemas.run_record.RunRecord` JSON payloads."""

    def __init__(self, *, db_path: Path | None = None) -> None:
        self._db_path = db_path or run_records_db_path()
        self._db: aiosqlite.Connection | None = None
        self._initialized = False
        self._writes_since_prune = 0

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._db = await aiosqlite.connect(str(self._db_path), timeout=30.0)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS run_records (
                run_id TEXT PRIMARY KEY,
                ts REAL NOT NULL,
                channel TEXT,
                outcome TEXT,
                payload_json TEXT NOT NULL
            )
            """
        )
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_run_records_ts ON run_records(ts)")
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_run_records_channel_ts ON run_records(channel, ts)"
        )
        await self._db.commit()
        self._initialized = True

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
        self._initialized = False

    async def save(self, record: RunRecord) -> None:
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        payload = json.dumps(record.model_dump(mode="json"), default=str)
        await self._db.execute(
            """
            INSERT OR REPLACE INTO run_records (run_id, ts, channel, outcome, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.run_id[:128],
                float(record.ts_end),
                (record.channel or "")[:64],
                record.outcome[:32],
                payload,
            ),
        )
        await self._db.commit()
        self._writes_since_prune += 1
        if self._writes_since_prune >= _PRUNE_EVERY_N:
            self._writes_since_prune = 0
            await self._prune()

    async def get(self, run_id: str) -> RunRecord | None:
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        rid = (run_id or "").strip()
        if not rid:
            return None
        cur = await self._db.execute(
            "SELECT payload_json FROM run_records WHERE run_id = ?",
            (rid[:128],),
        )
        row = await cur.fetchone()
        await cur.close()
        if not row or not row[0]:
            return None
        try:
            data = json.loads(row[0])
            return RunRecord.model_validate(data)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.debug("run_records get: invalid payload for %s", rid, exc_info=True)
            return None

    async def row_count(self) -> int:
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        cur = await self._db.execute("SELECT COUNT(*) FROM run_records")
        row = await cur.fetchone()
        await cur.close()
        return int(row[0]) if row and row[0] is not None else 0

    async def list_all_records(
        self,
        *,
        since_ts: float | None = None,
        limit: int | None = None,
    ) -> list[RunRecord]:
        """Return full run records oldest-first (for Neo4j backfill)."""
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        clauses: list[str] = []
        params: list[Any] = []
        if since_ts is not None:
            clauses.append("ts >= ?")
            params.append(float(since_ts))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT payload_json FROM run_records{where} ORDER BY ts ASC"
        if limit is not None:
            lim = max(1, min(int(limit), 100_000))
            sql += " LIMIT ?"
            params.append(lim)
        cur = await self._db.execute(sql, tuple(params))
        rows = await cur.fetchall()
        await cur.close()
        out: list[RunRecord] = []
        for row in rows:
            if not row or not row[0]:
                continue
            try:
                out.append(RunRecord.model_validate(json.loads(row[0])))
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.debug("list_all_records: skip invalid row", exc_info=True)
        return out

    async def list_recent(
        self,
        *,
        limit: int = 50,
        channel: str | None = None,
        since_ts: float | None = None,
    ) -> list[RunRecordSummary]:
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        lim = max(1, min(int(limit), 500))
        clauses: list[str] = []
        params: list[Any] = []
        if channel:
            clauses.append("channel = ?")
            params.append(channel[:64])
        if since_ts is not None:
            clauses.append("ts >= ?")
            params.append(float(since_ts))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT payload_json FROM run_records{where} ORDER BY ts DESC LIMIT ?"
        params.append(lim)
        cur = await self._db.execute(sql, tuple(params))
        rows = await cur.fetchall()
        await cur.close()
        out: list[RunRecordSummary] = []
        for row in rows:
            if not row or not row[0]:
                continue
            try:
                data = json.loads(row[0])
                record = RunRecord.model_validate(data)
                out.append(record.summary())
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.debug("run_records list_recent: skip invalid row", exc_info=True)
        return out

    async def list_recent_records(
        self,
        *,
        limit: int = 50,
        channel: str | None = None,
        since_ts: float | None = None,
    ) -> list[RunRecord]:
        """Full :class:`RunRecord` payloads, newest first (for branch telemetry rollup)."""
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        lim = max(1, min(int(limit), 500))
        clauses: list[str] = []
        params: list[Any] = []
        if channel:
            clauses.append("channel = ?")
            params.append(channel[:64])
        if since_ts is not None:
            clauses.append("ts >= ?")
            params.append(float(since_ts))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT payload_json FROM run_records{where} ORDER BY ts DESC LIMIT ?"
        params.append(lim)
        cur = await self._db.execute(sql, tuple(params))
        rows = await cur.fetchall()
        await cur.close()
        out: list[RunRecord] = []
        for row in rows:
            if not row or not row[0]:
                continue
            try:
                out.append(RunRecord.model_validate(json.loads(row[0])))
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.debug("run_records list_recent_records: skip invalid row", exc_info=True)
        return out

    async def _prune(self) -> None:
        assert self._db is not None
        cfg = get_settings().app
        cutoff = time.time() - float(cfg.run_record_retention_days) * 86400.0
        try:
            await self._db.execute("DELETE FROM run_records WHERE ts < ?", (cutoff,))
            await self._db.commit()
        except aiosqlite.Error:
            logger.debug("run_records retention prune failed", exc_info=True)
        max_rows = int(cfg.run_record_max_rows)
        if max_rows <= 0:
            return
        try:
            cur = await self._db.execute("SELECT COUNT(*) FROM run_records")
            row = await cur.fetchone()
            await cur.close()
            n = int(row[0]) if row and row[0] is not None else 0
        except aiosqlite.Error:
            logger.debug("run_records count failed", exc_info=True)
            return
        excess = n - max_rows
        if excess <= 0:
            return
        try:
            await self._db.execute(
                """
                DELETE FROM run_records WHERE run_id IN (
                    SELECT run_id FROM run_records ORDER BY ts ASC LIMIT ?
                )
                """,
                (excess,),
            )
            await self._db.commit()
        except aiosqlite.Error:
            logger.debug("run_records max_rows prune failed", exc_info=True)


class PgRunRecordStore(RunRecordStore):
    def __init__(self, repo: PgRunRecordRepository | None = None) -> None:
        if repo is None:
            repo = PgRunRecordRepository(CRMDatabase().session_factory)
        self._repo = repo
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await ensure_run_record_tables()
        self._initialized = True

    async def close(self) -> None:
        self._initialized = False

    async def save(self, record: RunRecord) -> None:
        if not self._initialized:
            await self.initialize()
        payload = json.dumps(record.model_dump(mode="json"), default=str)
        await self._repo.upsert(
            run_id=record.run_id[:128],
            ts=float(record.ts_end),
            channel=(record.channel or "")[:64] or None,
            outcome=record.outcome[:32],
            payload_json=payload,
        )

    async def get(self, run_id: str) -> RunRecord | None:
        if not self._initialized:
            await self.initialize()
        rid = (run_id or "").strip()
        if not rid:
            return None
        raw = await self._repo.get_payload(rid[:128])
        if not raw:
            return None
        try:
            return RunRecord.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.debug("run_records pg get: invalid payload for %s", rid, exc_info=True)
            return None

    async def row_count(self) -> int:
        if not self._initialized:
            await self.initialize()
        return await self._repo.row_count()

    async def list_all_records(
        self,
        *,
        since_ts: float | None = None,
        limit: int | None = None,
    ) -> list[RunRecord]:
        if not self._initialized:
            await self.initialize()
        lim = None if limit is None else max(1, min(int(limit), 100_000))
        payloads = await self._repo.list_payloads(since_ts=since_ts, order_desc=False, limit=lim)
        out: list[RunRecord] = []
        for raw in payloads:
            try:
                out.append(RunRecord.model_validate(json.loads(raw)))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return out

    async def list_recent(
        self,
        *,
        limit: int = 50,
        channel: str | None = None,
        since_ts: float | None = None,
    ) -> list[RunRecordSummary]:
        records = await self.list_recent_records(limit=limit, channel=channel, since_ts=since_ts)
        return [r.summary() for r in records]

    async def list_recent_records(
        self,
        *,
        limit: int = 50,
        channel: str | None = None,
        since_ts: float | None = None,
    ) -> list[RunRecord]:
        if not self._initialized:
            await self.initialize()
        lim = max(1, min(int(limit), 500))
        ch = channel[:64] if channel else None
        payloads = await self._repo.list_payloads(
            channel=ch, since_ts=since_ts, order_desc=True, limit=lim
        )
        out: list[RunRecord] = []
        for raw in payloads:
            try:
                out.append(RunRecord.model_validate(json.loads(raw)))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return out


class DualWriteRunRecordStore(RunRecordStore):
    def __init__(self, sqlite: SqliteRunRecordStore, pg: PgRunRecordStore) -> None:
        self._sqlite = sqlite
        self._pg = pg

    def _read(self) -> RunRecordStore:
        if getattr(get_settings().app, "pg_store_run_records_read", False):
            return self._pg
        return self._sqlite

    async def initialize(self) -> None:
        await self._sqlite.initialize()
        try:
            await self._pg.initialize()
        except (DatabaseError, OSError, RuntimeError):
            logger.warning("Postgres run_records table init failed", exc_info=True)

    async def close(self) -> None:
        await self._sqlite.close()
        await self._pg.close()

    async def save(self, record: RunRecord) -> None:
        await self._sqlite.save(record)
        try:
            await self._pg.save(record)
        except (DatabaseError, OSError, RuntimeError):
            logger.warning("Postgres run_records shadow-write failed", exc_info=True)

    async def get(self, run_id: str) -> RunRecord | None:
        return await self._read().get(run_id)

    async def row_count(self) -> int:
        return await self._read().row_count()

    async def list_all_records(
        self,
        *,
        since_ts: float | None = None,
        limit: int | None = None,
    ) -> list[RunRecord]:
        return await self._read().list_all_records(since_ts=since_ts, limit=limit)

    async def list_recent(
        self,
        *,
        limit: int = 50,
        channel: str | None = None,
        since_ts: float | None = None,
    ) -> list[RunRecordSummary]:
        return await self._read().list_recent(limit=limit, channel=channel, since_ts=since_ts)

    async def list_recent_records(
        self,
        *,
        limit: int = 50,
        channel: str | None = None,
        since_ts: float | None = None,
    ) -> list[RunRecord]:
        return await self._read().list_recent_records(
            limit=limit, channel=channel, since_ts=since_ts
        )


def build_run_record_store(*, db_path: Path | None = None) -> RunRecordStore:
    sqlite = SqliteRunRecordStore(db_path=db_path)
    if not getattr(get_settings().app, "pg_store_run_records_enabled", False):
        return sqlite
    return DualWriteRunRecordStore(sqlite, PgRunRecordStore())


async def ensure_run_record_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: RunRecordModel.__table__.create(sync_conn, checkfirst=True)
        )
