"""Persisted per-tool invocation telemetry for GEPA-style offline analysis."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.config import get_settings
from brain_os.systems.data_dir_lock import coerce_config_path, get_data_dir

logger = logging.getLogger(__name__)

_PRUNE_EVERY_N = 100

_DEFAULT_DB_REL = "brain/tool_invocations.sqlite"


def _resolve_gepa_db_rel_path(raw: object) -> str:
    """Validate the configured DB path before it is ever joined or ``mkdir``'d.

    Guards against an unconfigured test ``MagicMock`` leaking into the filesystem:
    a stringified mock such as ``mock.app.gepa_tool_invocations_db_path...`` must
    never become a real directory under ``data/``.
    """
    return coerce_config_path(raw, default=_DEFAULT_DB_REL).lstrip("/") or _DEFAULT_DB_REL


def tool_invocations_db_path() -> Path:
    rel = _resolve_gepa_db_rel_path(get_settings().app.gepa_tool_invocations_db_path)
    p = get_data_dir() / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class ToolInvocationStore:
    """SQLite append-only tool outcomes with TTL / max-row pruning."""

    def __init__(self, *, db_path: Path | None = None) -> None:
        self._db_path = db_path or tool_invocations_db_path()
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
    ) -> None:
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        now = time.time()
        await self._db.execute(
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
        self._writes_since_prune += 1
        if self._writes_since_prune >= _PRUNE_EVERY_N:
            self._writes_since_prune = 0
            await self._prune()

    async def _prune(self) -> None:
        assert self._db is not None
        cfg = get_settings().app
        cutoff = time.time() - float(cfg.gepa_tool_invocations_retention_days) * 86400.0
        try:
            await self._db.execute("DELETE FROM tool_invocations WHERE ts < ?", (cutoff,))
            await self._db.commit()
        except aiosqlite.Error as exc:
            logger.debug("tool_invocations retention prune failed", exc_info=True)
        max_rows = int(cfg.gepa_tool_invocations_max_rows)
        if max_rows <= 0:
            return
        try:
            cur = await self._db.execute("SELECT COUNT(*) FROM tool_invocations")
            row = await cur.fetchone()
            await cur.close()
            n = int(row[0]) if row and row[0] is not None else 0
        except aiosqlite.Error as exc:
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
        except aiosqlite.Error as exc:
            logger.debug("tool_invocations max_rows prune failed", exc_info=True)

    async def aggregate_pairs(
        self,
        *,
        since_ts: float,
        min_invocations: int,
        max_success_rate: float,
    ) -> list[dict[str, Any]]:
        """Return (agent, tool) rows with enough volume and success rate at or below *max_success_rate*."""
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
