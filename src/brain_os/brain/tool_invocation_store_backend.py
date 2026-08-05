"""Tool invocation store backends: SQLite (default), Postgres dual-write."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.config import get_settings
from brain_os.data.crm import CRMDatabase
from brain_os.data.tool_invocations import PgToolInvocationRepository, ToolInvocationModel
from brain_os.exceptions import DatabaseError

logger = logging.getLogger(__name__)

_PRUNE_EVERY_N = 100


class ToolInvocationStoreBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def record(
        self,
        *,
        agent: str,
        tool: str,
        success: bool,
        error_code: str | None = None,
        run_id: str | None = None,
        duration_ms: int | None = None,
    ) -> int: ...

    @abstractmethod
    async def aggregate_pairs(
        self,
        *,
        since_ts: float,
        min_invocations: int,
        max_success_rate: float,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def row_count(self) -> int: ...

    @property
    @abstractmethod
    def sqlite_db_connection(self) -> aiosqlite.Connection | None: ...


class SqliteToolInvocationStoreBackend(ToolInvocationStoreBackend):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._initialized = False
        self._writes_since_prune = 0

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return self._db

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path), timeout=30.0)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_invocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                agent TEXT NOT NULL,
                tool TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_code TEXT,
                run_id TEXT,
                duration_ms INTEGER
            )
            """
        )
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_tool_inv_ts ON tool_invocations(ts)")
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_inv_agent_tool ON tool_invocations(agent, tool)"
        )
        await self._db.commit()
        self._initialized = True
        logger.info("ToolInvocationStore (SQLite) initialised at %s", self._db_path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
        self._initialized = False

    async def record(
        self,
        *,
        agent: str,
        tool: str,
        success: bool,
        error_code: str | None = None,
        run_id: str | None = None,
        duration_ms: int | None = None,
    ) -> int:
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        now = time.time()
        cursor = await self._db.execute(
            """
            INSERT INTO tool_invocations (ts, agent, tool, success, error_code, run_id, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                (agent or "").strip().lower()[:128],
                (tool or "").strip()[:256],
                1 if success else 0,
                (error_code or "")[:128] or None,
                (run_id or "")[:128] or None,
                int(duration_ms) if duration_ms is not None else None,
            ),
        )
        await self._db.commit()
        row_id = int(cursor.lastrowid or 0)
        self._writes_since_prune += 1
        if self._writes_since_prune >= _PRUNE_EVERY_N:
            self._writes_since_prune = 0
            await self._prune()
        return row_id

    async def _prune(self) -> None:
        assert self._db is not None
        from brain_os.brain import tool_invocation_store as _store_mod

        cfg = _store_mod.get_settings().app
        cutoff = _store_mod.time.time() - float(cfg.gepa_tool_invocations_retention_days) * 86400.0
        try:
            await self._db.execute("DELETE FROM tool_invocations WHERE ts < ?", (cutoff,))
            await self._db.commit()
        except aiosqlite.Error:
            logger.debug("tool_invocations retention prune failed", exc_info=True)
        max_rows = int(cfg.gepa_tool_invocations_max_rows)
        if max_rows <= 0:
            return
        try:
            cur = await self._db.execute("SELECT COUNT(*) FROM tool_invocations")
            row = await cur.fetchone()
            await cur.close()
            n = int(row[0]) if row and row[0] is not None else 0
        except aiosqlite.Error:
            logger.debug("tool_invocations count failed", exc_info=True)
            return
        excess = n - max_rows
        if excess <= 0:
            return
        try:
            await self._db.execute(
                """
                DELETE FROM tool_invocations WHERE id IN (
                    SELECT id FROM tool_invocations ORDER BY ts ASC LIMIT ?
                )
                """,
                (excess,),
            )
            await self._db.commit()
        except aiosqlite.Error:
            logger.debug("tool_invocations max_rows prune failed", exc_info=True)

    async def aggregate_pairs(
        self,
        *,
        since_ts: float,
        min_invocations: int,
        max_success_rate: float,
    ) -> list[dict[str, Any]]:
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        min_inv = max(1, int(min_invocations))
        cur = await self._db.execute(
            """
            SELECT agent, tool,
                   SUM(success) AS successes,
                   COUNT(*) AS total
            FROM tool_invocations
            WHERE ts >= ?
              AND (run_id IS NULL OR run_id NOT LIKE 'seed-%')
            GROUP BY agent, tool
            HAVING total >= ?
            """,
            (since_ts, min_inv),
        )
        rows = await cur.fetchall()
        await cur.close()
        out: list[dict[str, Any]] = []
        for agent, tool, successes, total in rows:
            t = int(total or 0)
            s = int(successes or 0)
            if t <= 0:
                continue
            rate = s / t
            if rate <= float(max_success_rate):
                out.append(
                    {
                        "agent": str(agent),
                        "tool": str(tool),
                        "successes": s,
                        "failures": t - s,
                        "total": t,
                        "success_rate": round(rate, 4),
                    }
                )
        for row in out:
            row["top_error_codes"] = await self._top_error_codes_for_pair(
                str(row["agent"]),
                str(row["tool"]),
                since_ts=since_ts,
                limit=3,
            )
        out.sort(key=lambda r: (r["success_rate"], -r["total"]))
        return out

    async def _top_error_codes_for_pair(
        self,
        agent: str,
        tool: str,
        *,
        since_ts: float,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        cap = max(1, min(int(limit), 10))
        cur = await self._db.execute(
            """
            SELECT error_code, COUNT(*) AS n
            FROM tool_invocations
            WHERE ts >= ? AND agent = ? AND tool = ? AND success = 0
              AND error_code IS NOT NULL AND TRIM(error_code) != ''
            GROUP BY error_code
            ORDER BY n DESC
            LIMIT ?
            """,
            (since_ts, agent, tool, cap),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [{"error_code": str(code), "count": int(n or 0)} for code, n in rows]

    async def row_count(self) -> int:
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        cur = await self._db.execute("SELECT COUNT(*) FROM tool_invocations")
        row = await cur.fetchone()
        await cur.close()
        return int(row[0]) if row and row[0] is not None else 0


class PgToolInvocationStoreBackend(ToolInvocationStoreBackend):
    def __init__(self, repo: PgToolInvocationRepository | None = None) -> None:
        if repo is None:
            repo = PgToolInvocationRepository(CRMDatabase().session_factory)
        self._repo = repo
        self._writes_since_prune = 0

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return None

    async def initialize(self) -> None:
        await ensure_tool_invocation_tables()

    async def close(self) -> None:
        return None

    async def record(
        self,
        *,
        agent: str,
        tool: str,
        success: bool,
        error_code: str | None = None,
        run_id: str | None = None,
        duration_ms: int | None = None,
        row_id: int | None = None,
    ) -> int:
        from brain_os.brain import tool_invocation_store as _store_mod

        now = _store_mod.time.time()
        rid = await self._repo.record(
            ts=now,
            agent=(agent or "").strip().lower()[:128],
            tool=(tool or "").strip()[:256],
            success=success,
            error_code=(error_code or "")[:128] or None,
            run_id=(run_id or "")[:128] or None,
            duration_ms=int(duration_ms) if duration_ms is not None else None,
            row_id=row_id,
        )
        self._writes_since_prune += 1
        if self._writes_since_prune >= _PRUNE_EVERY_N:
            self._writes_since_prune = 0
            await self._prune()
        return rid

    async def _prune(self) -> None:
        from brain_os.brain import tool_invocation_store as _store_mod

        cfg = _store_mod.get_settings().app
        cutoff = _store_mod.time.time() - float(cfg.gepa_tool_invocations_retention_days) * 86400.0
        try:
            await self._repo.delete_before_ts(cutoff)
        except (DatabaseError, OSError, RuntimeError):
            logger.debug("tool_invocations PG retention prune failed", exc_info=True)
        max_rows = int(cfg.gepa_tool_invocations_max_rows)
        if max_rows <= 0:
            return
        try:
            await self._repo.trim_to_max_rows(max_rows)
        except (DatabaseError, OSError, RuntimeError):
            logger.debug("tool_invocations PG max_rows prune failed", exc_info=True)

    async def aggregate_pairs(
        self,
        *,
        since_ts: float,
        min_invocations: int,
        max_success_rate: float,
    ) -> list[dict[str, Any]]:
        return await self._repo.aggregate_pairs(
            since_ts=since_ts,
            min_invocations=min_invocations,
            max_success_rate=max_success_rate,
        )

    async def row_count(self) -> int:
        return await self._repo.row_count()


class DualWriteToolInvocationStoreBackend(ToolInvocationStoreBackend):
    def __init__(
        self, sqlite: SqliteToolInvocationStoreBackend, pg: PgToolInvocationStoreBackend
    ) -> None:
        self._sqlite = sqlite
        self._pg = pg

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return self._sqlite.sqlite_db_connection

    def _read(self) -> ToolInvocationStoreBackend:
        if getattr(get_settings().app, "pg_store_tool_invocations_read", False):
            return self._pg
        return self._sqlite

    async def initialize(self) -> None:
        await self._sqlite.initialize()
        await self._pg.initialize()

    async def close(self) -> None:
        await self._sqlite.close()
        await self._pg.close()

    async def record(
        self,
        *,
        agent: str,
        tool: str,
        success: bool,
        error_code: str | None = None,
        run_id: str | None = None,
        duration_ms: int | None = None,
    ) -> int:
        row_id = await self._sqlite.record(
            agent=agent,
            tool=tool,
            success=success,
            error_code=error_code,
            run_id=run_id,
            duration_ms=duration_ms,
        )
        try:
            await self._pg.record(
                agent=agent,
                tool=tool,
                success=success,
                error_code=error_code,
                run_id=run_id,
                duration_ms=duration_ms,
                row_id=row_id,
            )
        except (DatabaseError, OSError, RuntimeError):
            logger.warning("Postgres tool_invocations shadow-write failed", exc_info=True)
        return row_id

    async def aggregate_pairs(
        self,
        *,
        since_ts: float,
        min_invocations: int,
        max_success_rate: float,
    ) -> list[dict[str, Any]]:
        return await self._read().aggregate_pairs(
            since_ts=since_ts,
            min_invocations=min_invocations,
            max_success_rate=max_success_rate,
        )

    async def row_count(self) -> int:
        return await self._read().row_count()


def build_tool_invocation_store(db_path: Path) -> ToolInvocationStoreBackend:
    sqlite = SqliteToolInvocationStoreBackend(db_path)
    if not getattr(get_settings().app, "pg_store_tool_invocations_enabled", False):
        return sqlite
    return DualWriteToolInvocationStoreBackend(sqlite, PgToolInvocationStoreBackend())


async def ensure_tool_invocation_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: ToolInvocationModel.__table__.create(sync_conn, checkfirst=True)
        )
