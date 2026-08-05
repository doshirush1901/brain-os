"""Operator context backends: SQLite (default), Postgres dual-write."""

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
from brain_os.data.operator_context_pg import (
    OperatorContextRunModel,
    PgOperatorContextRepository,
)
from brain_os.exceptions import DatabaseError
from brain_os.schemas.operator_context import (
    ContextPrecedent,
    OperatorContextKind,
    OperatorContextOutcome,
    OperatorContextRun,
)
from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)

_PRUNE_EVERY_N = 50


def operator_context_db_path() -> Path:
    raw = (get_settings().app.operator_context_db_path or "").strip()
    if not raw:
        raw = "brain/operator_context.sqlite"
    candidate = Path(raw)
    if candidate.is_absolute():
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate
    p = get_data_dir() / raw.lstrip("/")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def operator_context_enabled() -> bool:
    return bool(get_settings().app.operator_context_enabled)


class OperatorContextStore(ABC):
    """Persistence for :class:`~brain_os.schemas.operator_context.OperatorContextRun`."""

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def save(self, record: OperatorContextRun) -> None: ...

    @abstractmethod
    async def get(self, run_id: str) -> OperatorContextRun | None: ...

    @abstractmethod
    async def list_precedents(
        self,
        *,
        company_key: str,
        domain: str | None = None,
        machine_model: str | None = None,
        exclude_run_id: str | None = None,
        limit: int = 5,
    ) -> list[ContextPrecedent]: ...

    @abstractmethod
    async def get_recent_account_brief(
        self,
        company_key: str,
        *,
        max_age_hours: float,
        contact_email: str | None = None,
    ) -> OperatorContextRun | None: ...

    @abstractmethod
    async def mark_success(
        self,
        run_id: str,
        *,
        outcome: OperatorContextOutcome | None = None,
        success: bool = True,
    ) -> bool: ...

    @abstractmethod
    async def list_all_runs(
        self,
        *,
        since_ts: float | None = None,
        limit: int | None = None,
    ) -> list[OperatorContextRun]: ...

    @abstractmethod
    async def count_runs(self, *, since_ts: float | None = None) -> int: ...

    @abstractmethod
    async def latest_run_id_for_company(
        self,
        company_key: str,
        *,
        kind: OperatorContextKind | None = None,
    ) -> str | None: ...


class SqliteOperatorContextStore(OperatorContextStore):
    """SQLite store for :class:`~brain_os.schemas.operator_context.OperatorContextRun`."""

    def __init__(self, *, db_path: Path | None = None) -> None:
        self._db_path = db_path or operator_context_db_path()
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
            CREATE TABLE IF NOT EXISTS operator_context_runs (
                run_id TEXT PRIMARY KEY,
                ts REAL NOT NULL,
                company_key TEXT NOT NULL,
                domain TEXT,
                kind TEXT NOT NULL,
                machine_model TEXT,
                outcome TEXT,
                success INTEGER,
                payload_json TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_opctx_company_ts ON operator_context_runs(company_key, ts)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_opctx_domain_ts ON operator_context_runs(domain, ts)"
        )
        await self._db.commit()
        self._initialized = True

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
        self._initialized = False

    async def save(self, record: OperatorContextRun) -> None:
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        payload = json.dumps(record.model_dump(mode="json"), default=str)
        success_val: int | None
        if record.success is None:
            success_val = None
        else:
            success_val = 1 if record.success else 0
        await self._db.execute(
            """
            INSERT OR REPLACE INTO operator_context_runs (
                run_id, ts, company_key, domain, kind, machine_model, outcome, success, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id[:128],
                float(record.ts),
                record.company_key[:256],
                (record.domain or "")[:128] or None,
                record.kind[:32],
                (record.machine_model or "")[:128] or None,
                record.outcome[:32],
                success_val,
                payload,
            ),
        )
        await self._db.commit()
        self._writes_since_prune += 1
        if self._writes_since_prune >= _PRUNE_EVERY_N:
            self._writes_since_prune = 0
            await self._prune()

    async def get(self, run_id: str) -> OperatorContextRun | None:
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        rid = (run_id or "").strip()
        if not rid:
            return None
        cur = await self._db.execute(
            "SELECT payload_json FROM operator_context_runs WHERE run_id = ?",
            (rid[:128],),
        )
        row = await cur.fetchone()
        await cur.close()
        if not row or not row[0]:
            return None
        try:
            return OperatorContextRun.model_validate(json.loads(row[0]))
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.debug("operator_context get: invalid payload for %s", rid, exc_info=True)
            return None

    async def list_precedents(
        self,
        *,
        company_key: str,
        domain: str | None = None,
        machine_model: str | None = None,
        exclude_run_id: str | None = None,
        limit: int = 5,
    ) -> list[ContextPrecedent]:
        """Return recent operator runs for this account (company, then domain, then machine)."""
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        lim = max(1, min(int(limit), 20))
        exclude = (exclude_run_id or "").strip()[:128]
        seen: set[str] = set()
        out: list[ContextPrecedent] = []

        async def _load_rows(
            where_sql: str,
            params: tuple[Any, ...],
            match_reason: str,
        ) -> None:
            nonlocal out
            if len(out) >= lim:
                return
            sql = (
                f"SELECT payload_json FROM operator_context_runs WHERE {where_sql} "
                "ORDER BY ts DESC LIMIT ?"
            )
            cur = await self._db.execute(sql, (*params, lim * 3))
            rows = await cur.fetchall()
            await cur.close()
            for row in rows:
                if len(out) >= lim:
                    break
                if not row or not row[0]:
                    continue
                try:
                    rec = OperatorContextRun.model_validate(json.loads(row[0]))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
                if rec.run_id in seen or (exclude and rec.run_id == exclude):
                    continue
                seen.add(rec.run_id)
                out.append(
                    ContextPrecedent(
                        run_id=rec.run_id,
                        kind=rec.kind,
                        ts=rec.ts,
                        company_name=rec.company_name,
                        outcome=rec.outcome,
                        success=rec.success,
                        machine_model=rec.machine_model,
                        crm_stage=rec.crm_stage,
                        summary=rec.summary,
                        match_reason=match_reason,
                    )
                )

        ck = (company_key or "").strip()
        if ck:
            await _load_rows("company_key = ?", (ck,), "same_company")
        dom = (domain or "").strip().lower()
        if dom and len(out) < lim:
            await _load_rows(
                "domain = ? AND company_key != ?",
                (dom, ck or "__none__"),
                "same_domain",
            )
        model = (machine_model or "").strip()
        if model and len(out) < lim:
            await _load_rows(
                "machine_model = ? AND company_key != ?",
                (model, ck or "__none__"),
                "same_machine_model",
            )

        def _sort_key(p: ContextPrecedent) -> tuple[int, float]:
            success_rank = 0 if p.success is True else (1 if p.success is None else 2)
            return (success_rank, -p.ts)

        out.sort(key=_sort_key)
        return out[:lim]

    async def get_recent_account_brief(
        self,
        company_key: str,
        *,
        max_age_hours: float,
        contact_email: str | None = None,
    ) -> OperatorContextRun | None:
        """Latest account_brief run with brief_snapshot within max_age_hours."""
        if max_age_hours <= 0:
            return None
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        ck = (company_key or "").strip()
        if not ck:
            return None
        cutoff = time.time() - max_age_hours * 3600.0
        cur = await self._db.execute(
            """
            SELECT payload_json FROM operator_context_runs
            WHERE company_key = ? AND kind = 'account_brief' AND ts >= ?
            ORDER BY ts DESC LIMIT 20
            """,
            (ck[:256], cutoff),
        )
        rows = await cur.fetchall()
        await cur.close()
        want_contact = (contact_email or "").strip().lower()
        for row in rows:
            if not row or not row[0]:
                continue
            try:
                rec = OperatorContextRun.model_validate(json.loads(row[0]))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if not rec.brief_snapshot:
                continue
            if want_contact and rec.contact_email:
                stored = str(rec.contact_email)
                if stored.startswith("redacted:"):
                    pass
                elif stored.lower() != want_contact:
                    continue
            return rec
        return None

    async def mark_success(
        self,
        run_id: str,
        *,
        outcome: OperatorContextOutcome | None = None,
        success: bool = True,
    ) -> bool:
        record = await self.get(run_id)
        if record is None:
            return False
        updates: dict[str, Any] = {"success": success}
        if outcome:
            updates["outcome"] = outcome
        updated = record.model_copy(update=updates)
        await self.save(updated)
        return True

    async def list_all_runs(
        self,
        *,
        since_ts: float | None = None,
        limit: int | None = None,
    ) -> list[OperatorContextRun]:
        """Return operator context runs oldest-first (for Neo4j backfill)."""
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        clauses: list[str] = []
        params: list[Any] = []
        if since_ts is not None:
            clauses.append("ts >= ?")
            params.append(float(since_ts))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT payload_json FROM operator_context_runs{where} ORDER BY ts ASC"
        if limit is not None:
            lim = max(1, min(int(limit), 100_000))
            sql += " LIMIT ?"
            params.append(lim)
        cur = await self._db.execute(sql, tuple(params))
        rows = await cur.fetchall()
        await cur.close()
        out: list[OperatorContextRun] = []
        for row in rows:
            if not row or not row[0]:
                continue
            try:
                out.append(OperatorContextRun.model_validate(json.loads(row[0])))
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.debug("list_all_runs: skip invalid row", exc_info=True)
        return out

    async def count_runs(self, *, since_ts: float | None = None) -> int:
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        if since_ts is not None:
            cur = await self._db.execute(
                "SELECT COUNT(*) FROM operator_context_runs WHERE ts >= ?",
                (float(since_ts),),
            )
        else:
            cur = await self._db.execute("SELECT COUNT(*) FROM operator_context_runs")
        row = await cur.fetchone()
        await cur.close()
        return int(row[0]) if row and row[0] is not None else 0

    async def latest_run_id_for_company(
        self,
        company_key: str,
        *,
        kind: OperatorContextKind | None = None,
    ) -> str | None:
        if not self._initialized:
            await self.initialize()
        assert self._db is not None
        ck = (company_key or "").strip()
        if not ck:
            return None
        clauses = ["company_key = ?"]
        params: list[Any] = [ck]
        if kind:
            clauses.append("kind = ?")
            params.append(kind[:32])
        sql = (
            f"SELECT run_id FROM operator_context_runs WHERE {' AND '.join(clauses)} "
            "ORDER BY ts DESC LIMIT 1"
        )
        cur = await self._db.execute(sql, tuple(params))
        row = await cur.fetchone()
        await cur.close()
        if row and row[0]:
            return str(row[0])
        return None

    async def _prune(self) -> None:
        assert self._db is not None
        cfg = get_settings().app
        cutoff = time.time() - float(cfg.operator_context_retention_days) * 86400.0
        try:
            await self._db.execute(
                "DELETE FROM operator_context_runs WHERE ts < ?",
                (cutoff,),
            )
            await self._db.commit()
        except aiosqlite.Error:
            logger.debug("operator_context retention prune failed", exc_info=True)
        max_rows = int(cfg.operator_context_max_rows)
        if max_rows <= 0:
            return
        try:
            cur = await self._db.execute("SELECT COUNT(*) FROM operator_context_runs")
            row = await cur.fetchone()
            await cur.close()
            n = int(row[0]) if row and row[0] is not None else 0
        except aiosqlite.Error:
            return
        excess = n - max_rows
        if excess <= 0:
            return
        try:
            await self._db.execute(
                """
                DELETE FROM operator_context_runs WHERE run_id IN (
                    SELECT run_id FROM operator_context_runs ORDER BY ts ASC LIMIT ?
                )
                """,
                (excess,),
            )
            await self._db.commit()
        except aiosqlite.Error:
            logger.debug("operator_context max_rows prune failed", exc_info=True)


def new_operator_context_run_id() -> str:
    return f"opctx-{uuid.uuid4().hex[:16]}"


class PgOperatorContextStore(OperatorContextStore):
    def __init__(self, repo: PgOperatorContextRepository | None = None) -> None:
        if repo is None:
            repo = PgOperatorContextRepository(CRMDatabase().session_factory)
        self._repo = repo
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await ensure_operator_context_tables()
        self._initialized = True

    async def close(self) -> None:
        self._initialized = False

    async def save(self, record: OperatorContextRun) -> None:
        if not self._initialized:
            await self.initialize()
        payload = json.dumps(record.model_dump(mode="json"), default=str)
        if record.success is None:
            success_val = None
        else:
            success_val = 1 if record.success else 0
        await self._repo.upsert(
            run_id=record.run_id[:128],
            ts=float(record.ts),
            company_key=record.company_key[:256],
            domain=(record.domain or "")[:128] or None,
            kind=record.kind[:32],
            machine_model=(record.machine_model or "")[:128] or None,
            outcome=record.outcome[:32],
            success=success_val,
            payload_json=payload,
        )

    async def get(self, run_id: str) -> OperatorContextRun | None:
        if not self._initialized:
            await self.initialize()
        rid = (run_id or "").strip()
        if not rid:
            return None
        raw = await self._repo.get_payload(rid[:128])
        if not raw:
            return None
        try:
            return OperatorContextRun.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.debug("operator_context pg get: invalid payload for %s", rid, exc_info=True)
            return None

    async def list_precedents(
        self,
        *,
        company_key: str,
        domain: str | None = None,
        machine_model: str | None = None,
        exclude_run_id: str | None = None,
        limit: int = 5,
    ) -> list[ContextPrecedent]:
        if not self._initialized:
            await self.initialize()
        lim = max(1, min(int(limit), 20))
        exclude = (exclude_run_id or "").strip()[:128]
        seen: set[str] = set()
        out: list[ContextPrecedent] = []

        async def _load(where_clauses: list[Any], match_reason: str) -> None:
            nonlocal out
            if len(out) >= lim:
                return
            payloads = await self._repo.list_payloads(
                where_clauses=where_clauses, order_desc=True, limit=lim * 3
            )
            for raw in payloads:
                if len(out) >= lim:
                    break
                try:
                    rec = OperatorContextRun.model_validate(json.loads(raw))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
                if rec.run_id in seen or (exclude and rec.run_id == exclude):
                    continue
                seen.add(rec.run_id)
                out.append(
                    ContextPrecedent(
                        run_id=rec.run_id,
                        kind=rec.kind,
                        ts=rec.ts,
                        company_name=rec.company_name,
                        outcome=rec.outcome,
                        success=rec.success,
                        machine_model=rec.machine_model,
                        crm_stage=rec.crm_stage,
                        summary=rec.summary,
                        match_reason=match_reason,
                    )
                )

        ck = (company_key or "").strip()
        if ck:
            await _load([OperatorContextRunModel.company_key == ck], "same_company")
        dom = (domain or "").strip().lower()
        if dom and len(out) < lim:
            await _load(
                [
                    OperatorContextRunModel.domain == dom,
                    OperatorContextRunModel.company_key != (ck or "__none__"),
                ],
                "same_domain",
            )
        model = (machine_model or "").strip()
        if model and len(out) < lim:
            await _load(
                [
                    OperatorContextRunModel.machine_model == model,
                    OperatorContextRunModel.company_key != (ck or "__none__"),
                ],
                "same_machine_model",
            )

        def _sort_key(p: ContextPrecedent) -> tuple[int, float]:
            success_rank = 0 if p.success is True else (1 if p.success is None else 2)
            return (success_rank, -p.ts)

        out.sort(key=_sort_key)
        return out[:lim]

    async def get_recent_account_brief(
        self,
        company_key: str,
        *,
        max_age_hours: float,
        contact_email: str | None = None,
    ) -> OperatorContextRun | None:
        if max_age_hours <= 0:
            return None
        if not self._initialized:
            await self.initialize()
        ck = (company_key or "").strip()
        if not ck:
            return None
        cutoff = time.time() - max_age_hours * 3600.0
        payloads = await self._repo.list_payloads(
            where_clauses=[
                OperatorContextRunModel.company_key == ck[:256],
                OperatorContextRunModel.kind == "account_brief",
                OperatorContextRunModel.ts >= cutoff,
            ],
            order_desc=True,
            limit=20,
        )
        want_contact = (contact_email or "").strip().lower()
        for raw in payloads:
            try:
                rec = OperatorContextRun.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if not rec.brief_snapshot:
                continue
            if want_contact and rec.contact_email:
                stored = str(rec.contact_email)
                if stored.startswith("redacted:"):
                    pass
                elif stored.lower() != want_contact:
                    continue
            return rec
        return None

    async def mark_success(
        self,
        run_id: str,
        *,
        outcome: OperatorContextOutcome | None = None,
        success: bool = True,
    ) -> bool:
        record = await self.get(run_id)
        if record is None:
            return False
        updates: dict[str, Any] = {"success": success}
        if outcome:
            updates["outcome"] = outcome
        await self.save(record.model_copy(update=updates))
        return True

    async def list_all_runs(
        self,
        *,
        since_ts: float | None = None,
        limit: int | None = None,
    ) -> list[OperatorContextRun]:
        if not self._initialized:
            await self.initialize()
        clauses: list[Any] = []
        if since_ts is not None:
            clauses.append(OperatorContextRunModel.ts >= float(since_ts))
        lim = None if limit is None else max(1, min(int(limit), 100_000))
        payloads = await self._repo.list_payloads(
            where_clauses=clauses, order_desc=False, limit=lim
        )
        out: list[OperatorContextRun] = []
        for raw in payloads:
            try:
                out.append(OperatorContextRun.model_validate(json.loads(raw)))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return out

    async def count_runs(self, *, since_ts: float | None = None) -> int:
        if not self._initialized:
            await self.initialize()
        return await self._repo.count_runs(since_ts=since_ts)

    async def latest_run_id_for_company(
        self,
        company_key: str,
        *,
        kind: OperatorContextKind | None = None,
    ) -> str | None:
        if not self._initialized:
            await self.initialize()
        ck = (company_key or "").strip()
        if not ck:
            return None
        return await self._repo.latest_run_id(ck, kind=kind[:32] if kind else None)


class DualWriteOperatorContextStore(OperatorContextStore):
    def __init__(self, sqlite: SqliteOperatorContextStore, pg: PgOperatorContextStore) -> None:
        self._sqlite = sqlite
        self._pg = pg

    def _read(self) -> OperatorContextStore:
        if getattr(get_settings().app, "pg_store_operator_context_read", False):
            return self._pg
        return self._sqlite

    async def initialize(self) -> None:
        await self._sqlite.initialize()
        try:
            await self._pg.initialize()
        except (DatabaseError, OSError, RuntimeError):
            logger.warning("Postgres operator_context table init failed", exc_info=True)

    async def close(self) -> None:
        await self._sqlite.close()
        await self._pg.close()

    async def save(self, record: OperatorContextRun) -> None:
        await self._sqlite.save(record)
        try:
            await self._pg.save(record)
        except (DatabaseError, OSError, RuntimeError):
            logger.warning("Postgres operator_context shadow-write failed", exc_info=True)

    async def get(self, run_id: str) -> OperatorContextRun | None:
        return await self._read().get(run_id)

    async def list_precedents(
        self,
        *,
        company_key: str,
        domain: str | None = None,
        machine_model: str | None = None,
        exclude_run_id: str | None = None,
        limit: int = 5,
    ) -> list[ContextPrecedent]:
        return await self._read().list_precedents(
            company_key=company_key,
            domain=domain,
            machine_model=machine_model,
            exclude_run_id=exclude_run_id,
            limit=limit,
        )

    async def get_recent_account_brief(
        self,
        company_key: str,
        *,
        max_age_hours: float,
        contact_email: str | None = None,
    ) -> OperatorContextRun | None:
        return await self._read().get_recent_account_brief(
            company_key, max_age_hours=max_age_hours, contact_email=contact_email
        )

    async def mark_success(
        self,
        run_id: str,
        *,
        outcome: OperatorContextOutcome | None = None,
        success: bool = True,
    ) -> bool:
        ok = await self._sqlite.mark_success(run_id, outcome=outcome, success=success)
        if ok:
            try:
                await self._pg.mark_success(run_id, outcome=outcome, success=success)
            except (DatabaseError, OSError, RuntimeError):
                logger.warning("Postgres operator_context mark_success failed", exc_info=True)
        return ok

    async def list_all_runs(
        self,
        *,
        since_ts: float | None = None,
        limit: int | None = None,
    ) -> list[OperatorContextRun]:
        return await self._read().list_all_runs(since_ts=since_ts, limit=limit)

    async def count_runs(self, *, since_ts: float | None = None) -> int:
        return await self._read().count_runs(since_ts=since_ts)

    async def latest_run_id_for_company(
        self,
        company_key: str,
        *,
        kind: OperatorContextKind | None = None,
    ) -> str | None:
        return await self._read().latest_run_id_for_company(company_key, kind=kind)


def build_operator_context_store(*, db_path: Path | None = None) -> OperatorContextStore:
    sqlite = SqliteOperatorContextStore(db_path=db_path)
    if not getattr(get_settings().app, "pg_store_operator_context_enabled", False):
        return sqlite
    return DualWriteOperatorContextStore(sqlite, PgOperatorContextStore())


async def ensure_operator_context_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: OperatorContextRunModel.__table__.create(sync_conn, checkfirst=True)
        )
