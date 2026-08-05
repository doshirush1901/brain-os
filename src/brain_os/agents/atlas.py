"""Atlas — Project Manager agent.

Tracks projects, logs events, monitors production schedules, and
alerts on overdue payment milestones using a local SQLite logbook.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.brain.asana_planning_mapper import normalize_task_record, pair_procurement_events
from brain_os.brain.atlas_production_portfolio import (
    format_portfolio_brief,
    portfolio_project_names,
)
from brain_os.prompt_loader import load_prompt
from brain_os.service_keys import ServiceKey as SK

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("atlas_system")

_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "brain" / "atlas_logbook.db"
_ASANA_IMPORTS_DIR = Path(__file__).resolve().parents[3] / "data" / "imports" / "23_Asana"

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    customer    TEXT    NOT NULL DEFAULT '',
    status      TEXT    NOT NULL DEFAULT 'active',
    created_at  TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES projects(id),
    event_type  TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL
);
"""


class Atlas(BaseAgent):
    name = "atlas"
    role = "Project Manager"
    description = (
        "Project tracking, event logging, production scheduling, and payment milestone alerts"
    )
    knowledge_categories = [
        "orders_and_pos",
        "production",
        "current machine orders",
        "project_case_studies",
        "company_internal",
        "contracts_and_legal",
        "business plans",
    ]
    # Bridge Atlas planning domains to imports metadata doc types so shop-floor
    # datasets (e.g. Asana exports/gold sets) can be retrieved via category filters.
    category_aliases: dict[str, list[str]] = {
        "orders_and_pos": ["order", "invoice", "quote", "spreadsheet"],
        "production": ["technical_spec", "manual", "report", "spreadsheet"],
        "current machine orders": ["order", "spreadsheet"],
        "project_case_studies": ["report", "presentation", "manual"],
        "company_internal": ["report", "spreadsheet", "other"],
        "contracts_and_legal": ["contract", "invoice"],
        "business plans": ["presentation", "report"],
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._db_initialised = False

    # ── tool registration ─────────────────────────────────────────────────

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="get_project_status",
                description="Get a full project summary including logbook events and KB data.",
                parameters={"project_id": "Project name or identifier"},
                handler=self._tool_get_project_status,
            )
        )
        self.register_tool(
            AgentTool(
                name="log_project_event",
                description="Log an event to a project's logbook (creates the project if new).",
                parameters={
                    "project_id": "Project name or identifier",
                    "event": "Description of the event to log",
                },
                handler=self._tool_log_project_event,
            )
        )
        self.register_tool(
            AgentTool(
                name="get_overdue_milestones",
                description="Check for overdue payment milestones and pending invoices across projects.",
                parameters={},
                handler=self._tool_get_overdue_milestones,
            )
        )
        self.register_tool(
            AgentTool(
                name="get_production_schedule",
                description="Get the current production schedule with active project timelines.",
                parameters={"project_id": "Optional project name to filter (empty for all)"},
                handler=self._tool_get_production_schedule,
            )
        )
        self.register_tool(
            AgentTool(
                name="get_order_book",
                description=(
                    "Operator-maintained order book: installation-mode (recently shipped) "
                    "vs current in-plant projects."
                ),
                parameters={},
                handler=self._tool_get_order_book,
            )
        )
        self.register_tool(
            AgentTool(
                name="get_eto_daily_report",
                description="Generate a deterministic ETO daily report from 23_Asana exports.",
                parameters={
                    "max_files": "Optional number of recent CSV exports to scan (default 8)"
                },
                handler=self._tool_get_eto_daily_report,
            )
        )

        if self._services.get(SK.PANTHEON):
            self.register_tool(
                AgentTool(
                    name="ask_hephaestus",
                    description="Delegate a production/machine specs question to the Hephaestus agent.",
                    parameters={"query": "The production or machine question"},
                    handler=self._tool_ask_hephaestus,
                )
            )
            self.register_tool(
                AgentTool(
                    name="ask_asclepius",
                    description="Delegate quality/FAT/punch-list checks to Asclepius.",
                    parameters={"query": "Quality/punch-list question"},
                    handler=self._tool_ask_asclepius,
                )
            )

    # ── tool handlers ─────────────────────────────────────────────────────

    async def _tool_get_project_status(self, project_id: str) -> str:
        return await self.project_summary(project_id)

    async def _tool_log_project_event(self, project_id: str, event: str) -> str:
        return await self.log_event(project_id, "note", event)

    async def _tool_get_overdue_milestones(self) -> str:
        return await self.payment_alerts()

    async def _tool_get_production_schedule(self, project_id: str = "") -> str:
        return await self.production_schedule()

    async def _tool_get_order_book(self) -> str:
        brief = format_portfolio_brief(include_kb_hint=False)
        return brief or json.dumps({"status": "empty", "reason": "portfolio file missing"})

    async def _tool_get_eto_daily_report(self, max_files: int = 8) -> str:
        return await self.eto_daily_report(max_files=max_files)

    async def _tool_ask_hephaestus(self, query: str) -> str:
        return await self._tool_ask_agent("hephaestus", query)

    async def _tool_ask_asclepius(self, query: str) -> str:
        return await self._tool_ask_agent("asclepius", query)

    # ── DB setup ──────────────────────────────────────────────────────────

    async def _ensure_db(self) -> None:
        if self._db_initialised:
            return
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(_DB_PATH), timeout=30.0) as db:
            await db.execute("PRAGMA busy_timeout=30000")
            await db.executescript(_INIT_SQL)
            await db.commit()
        await self._sync_portfolio_to_logbook()
        self._db_initialised = True

    async def _sync_portfolio_to_logbook(self) -> None:
        """Upsert canonical portfolio rows into SQLite (idempotent)."""
        rows = portfolio_project_names()
        if not rows:
            return
        now = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(str(_DB_PATH), timeout=30.0) as db:
            await db.execute("PRAGMA busy_timeout=30000")
            for name, customer, status in rows:
                async with db.execute("SELECT id FROM projects WHERE name = ?", (name,)) as cursor:
                    existing = await cursor.fetchone()
                if existing:
                    await db.execute(
                        "UPDATE projects SET customer = ?, status = ? WHERE name = ?",
                        (customer, status, name),
                    )
                else:
                    await db.execute(
                        "INSERT INTO projects (name, customer, status, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (name, customer, status, now),
                    )
            await db.commit()

    # ── main handler ──────────────────────────────────────────────────────

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        await self._ensure_db()
        ctx = context or {}
        action = ctx.get("action", "")

        if action == "project_summary":
            return await self.project_summary(ctx.get("project_name", query))

        if action == "log_event":
            return await self.log_event(
                ctx.get("project_name", ""),
                ctx.get("event_type", "note"),
                ctx.get("description", query),
            )

        if action == "production_schedule":
            return await self.production_schedule(plain_text=bool(ctx.get("slack_brief")))

        if action == "payment_alerts":
            return await self.payment_alerts()

        if action == "eto_daily_report":
            return await self.eto_daily_report(int(ctx.get("max_files", 8)))

        if action == "meeting_notes":
            return await self.use_skill(
                "generate_meeting_notes",
                transcript=ctx.get("transcript", query),
                attendees=ctx.get("attendees", []),
            )

        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)

    # ── existing methods ──────────────────────────────────────────────────

    def _atlas_search_categories(self) -> list[str]:
        """Return Atlas categories expanded with imports/doc-type aliases."""
        ordered: list[str] = []
        seen: set[str] = set()
        for category in self.knowledge_categories:
            for value in [category, *self.category_aliases.get(category, [])]:
                cleaned = value.strip()
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    ordered.append(cleaned)
        return ordered

    async def search_domain_knowledge(
        self,
        query: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Search Atlas planning categories with alias expansion and dedupe."""
        categories = self._atlas_search_categories()
        if not categories:
            return await self.search_knowledge(query, limit=limit)

        per_cat = max(2, limit // max(1, min(len(categories), limit)))
        results_lists = await asyncio.gather(
            *(self.search_category(query, cat, limit=per_cat) for cat in categories)
        )

        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for results in results_lists:
            for result in results:
                key = result.get("source", "") + result.get("content", "")[:120]
                if key not in seen:
                    seen.add(key)
                    merged.append(result)

        if not merged:
            return await self.search_knowledge(query, limit=limit)

        merged.sort(key=lambda r: r.get("score", 0), reverse=True)
        return merged[:limit]

    @staticmethod
    def _read_csv_rows(filepath: Path) -> list[dict[str, str]]:
        with filepath.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh)
            return [dict(row) for row in reader]

    @staticmethod
    def _canonical_export_name(filepath: Path) -> str:
        """Normalize Asana export filenames so copied exports can be deduped."""
        stem = filepath.stem.strip().lower()
        stem = re.sub(r"\s+\(\d+\)$", "", stem)
        return f"{stem}{filepath.suffix.lower()}"

    async def eto_daily_report(self, max_files: int = 8) -> str:
        """Build an ETO planning report from Asana project-export CSV files."""
        if not _ASANA_IMPORTS_DIR.exists():
            return json.dumps(
                {
                    "status": "unavailable",
                    "reason": "23_Asana imports folder not found",
                }
            )

        candidates = sorted(
            _ASANA_IMPORTS_DIR.glob("*.csv"),
            key=lambda fp: fp.stat().st_mtime,
            reverse=True,
        )
        csv_files: list[Path] = []
        seen_exports: set[str] = set()
        for fp in candidates:
            export_key = self._canonical_export_name(fp)
            if export_key in seen_exports:
                continue
            seen_exports.add(export_key)
            csv_files.append(fp)
            if len(csv_files) >= max(1, max_files):
                break
        if not csv_files:
            return json.dumps(
                {
                    "status": "empty",
                    "reason": "No Asana CSV exports found",
                }
            )

        rows: list[dict[str, str]] = []
        for fp in csv_files:
            try:
                rows.extend(await asyncio.to_thread(self._read_csv_rows, fp))
            except OSError:
                logger.warning("Unable to read Asana CSV: %s", fp, exc_info=True)

        normalized = [normalize_task_record(row) for row in rows]
        completed = [task for task in normalized if task.completed_at is not None]
        open_tasks = [task for task in normalized if task.completed_at is None]

        gates: dict[str, dict[str, int]] = {}
        for task in normalized:
            gate = task.phase_std
            stats = gates.setdefault(gate, {"total": 0, "completed": 0, "open": 0})
            stats["total"] += 1
            if task.completed_at is not None:
                stats["completed"] += 1
            else:
                stats["open"] += 1

        pairs = pair_procurement_events(rows)
        received_pairs = [pair for pair in pairs if pair["status"] == "received"]
        open_pairs = [pair for pair in pairs if pair["status"] == "open"]
        lead_days = [
            int(pair["lead_time_days"])
            for pair in received_pairs
            if pair["lead_time_days"] is not None
        ]
        avg_lead = round(sum(lead_days) / len(lead_days), 1) if lead_days else None

        unblock_candidates: list[dict[str, Any]] = []
        for row, task in zip(rows, normalized, strict=False):
            if task.completed_at is not None:
                continue
            blocked = str(row.get("Blocked By (Dependencies)", "")).strip()
            blocking = str(row.get("Blocking (Dependencies)", "")).strip()
            if not blocked and not blocking:
                continue
            blocked_count = len([x for x in blocked.split(",") if x.strip()]) if blocked else 0
            blocking_count = len([x for x in blocking.split(",") if x.strip()]) if blocking else 0
            unblock_candidates.append(
                {
                    "task_id": task.task_id,
                    "task_name": task.task_name,
                    "phase_std": task.phase_std,
                    "blocked_by_count": blocked_count,
                    "blocking_count": blocking_count,
                    "priority": (blocking_count * 2) + blocked_count,
                }
            )

        unblock_candidates.sort(key=lambda item: item["priority"], reverse=True)

        projects = {
            str(row.get("Projects", "")).strip()
            for row in rows
            if str(row.get("Projects", "")).strip()
        }
        report = {
            "status": "ok",
            "generated_at": datetime.now(UTC).isoformat(),
            "source": {
                "folder": str(_ASANA_IMPORTS_DIR),
                "csv_files_scanned": len(csv_files),
                "tasks_scanned": len(rows),
                "projects_scanned": len(projects),
            },
            "execution": {
                "completed_tasks": len(completed),
                "open_tasks": len(open_tasks),
            },
            "gate_status": gates,
            "procurement": {
                "pairs_total": len(pairs),
                "pairs_received": len(received_pairs),
                "pairs_open": len(open_pairs),
                "avg_lead_time_days": avg_lead,
            },
            "top_unblockers": unblock_candidates[:10],
        }
        return json.dumps(report, indent=2)

    async def project_summary(self, project_name: str) -> str:
        """Combine KB data with logbook entries for a project summary."""
        kb_results = await self.search_domain_knowledge(
            f"project {project_name} status timeline",
            limit=8,
        )
        kb_context = self._format_context(kb_results)

        logbook_context = ""
        async with aiosqlite.connect(str(_DB_PATH), timeout=30.0) as db:
            await db.execute("PRAGMA busy_timeout=30000")
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM projects WHERE name = ? COLLATE NOCASE",
                (project_name,),
            ) as cursor:
                project = await cursor.fetchone()

            if project:
                pid = project["id"]
                async with db.execute(
                    "SELECT event_type, description, created_at "
                    "FROM events WHERE project_id = ? ORDER BY created_at DESC LIMIT 20",
                    (pid,),
                ) as cursor:
                    events = await cursor.fetchall()

                logbook_context = (
                    f"Project: {project['name']} | Customer: {project['customer']} | "
                    f"Status: {project['status']} | Created: {project['created_at']}\n"
                    f"Recent events ({len(events)}):\n"
                ) + "\n".join(
                    f"  [{e['event_type']}] {e['description']} ({e['created_at']})" for e in events
                )

        portfolio_brief = format_portfolio_brief()
        return await self.call_llm(
            _SYSTEM_PROMPT,
            f"Provide a project summary for: {project_name}\n\n"
            f"{portfolio_brief or ''}\n\n"
            f"Logbook data:\n{logbook_context or '(no logbook entries)'}\n\n"
            f"Knowledge base data:\n{kb_context}",
        )

    async def log_event(
        self,
        project_name: str,
        event_type: str,
        description: str,
    ) -> str:
        """Log an event to the project logbook, creating the project if needed."""
        now = datetime.now(UTC).isoformat()

        async with aiosqlite.connect(str(_DB_PATH), timeout=30.0) as db:
            await db.execute("PRAGMA busy_timeout=30000")
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id FROM projects WHERE name = ? COLLATE NOCASE",
                (project_name,),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                pid = row["id"]
            else:
                cursor = await db.execute(
                    "INSERT INTO projects (name, customer, status, created_at) VALUES (?, '', 'active', ?)",
                    (project_name, now),
                )
                pid = cursor.lastrowid

            await db.execute(
                "INSERT INTO events (project_id, event_type, description, created_at) VALUES (?, ?, ?, ?)",
                (pid, event_type, description, now),
            )
            await db.commit()

        return json.dumps(
            {
                "status": "logged",
                "project": project_name,
                "event_type": event_type,
                "description": description,
                "timestamp": now,
            }
        )

    async def production_schedule(self, *, plain_text: bool = False) -> str:
        """Search KB for current production schedule data."""
        kb_limit = 3 if plain_text else 10
        results = await self.search_domain_knowledge(
            "production schedule timeline delivery dates manufacturing active orders",
            limit=kb_limit,
        )
        kb_context = self._format_context(results)

        logbook_context = ""
        async with aiosqlite.connect(str(_DB_PATH), timeout=30.0) as db:
            await db.execute("PRAGMA busy_timeout=30000")
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT p.name, p.customer, p.status, "
                "(SELECT COUNT(*) FROM events e WHERE e.project_id = p.id) as event_count "
                "FROM projects p WHERE p.status IN ('active', 'installation') "
                "ORDER BY p.created_at DESC",
            ) as cursor:
                projects = await cursor.fetchall()

            if projects:
                logbook_context = "Active projects in logbook:\n" + "\n".join(
                    f"  - {p['name']} (customer: {p['customer'] or 'N/A'}, events: {p['event_count']})"
                    for p in projects
                )

        portfolio_brief = format_portfolio_brief()
        user_msg = (
            "Compile the current production schedule for Acme Corp.\n\n"
            f"{portfolio_brief or '(no canonical portfolio file)'}\n\n"
            f"Logbook:\n{logbook_context or '(no active projects in logbook)'}\n\n"
            f"Knowledge base data:\n{kb_context or '(no KB excerpts)'}"
        )
        if plain_text:
            system = (
                "You are Atlas, Acme's project manager. "
                "Reply in plain text for Slack: 5–8 short lines, use bullets. "
                "No JSON, no ReAct, no 'thought' field. "
                "Split answer into: (1) installation-mode recently shipped, "
                "(2) current in-plant projects. "
                "Trust the CANONICAL ORDER BOOK block over KB when they conflict. "
                "Never list StaleCorp Demo as active production. "
                "Label older KB dates as historical, not active production."
            )
            return await self.call_llm(system, user_msg, temperature=0.2)

        return await self.call_llm(_SYSTEM_PROMPT, user_msg)

    async def payment_alerts(self) -> str:
        """Check open commercial payment milestones + invoice registry (structured)."""
        try:
            from brain_os.data.crm import CRMDatabase
            from brain_os.services.invoice_service import (
                InvoiceService,
                format_payment_alerts_text,
            )

            svc = InvoiceService(CRMDatabase().session_factory)
            payload = await svc.payment_alerts_structured()
            return format_payment_alerts_text(payload)
        except Exception:
            logger.warning(
                "structured payment_alerts failed; falling back to logbook summary",
                exc_info=True,
            )

        logbook_context = ""
        try:
            async with aiosqlite.connect(str(_DB_PATH), timeout=30.0) as db:
                await db.execute("PRAGMA busy_timeout=30000")
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT p.name, e.event_type, e.description, e.created_at "
                    "FROM events e JOIN projects p ON e.project_id = p.id "
                    "WHERE e.event_type IN ('payment', 'invoice', 'milestone') "
                    "ORDER BY e.created_at DESC LIMIT 20",
                ) as cursor:
                    events = await cursor.fetchall()
                if events:
                    logbook_context = "Payment-related events:\n" + "\n".join(
                        f"  - [{e['event_type']}] {e['name']}: {e['description']} "
                        f"({e['created_at']})"
                        for e in events
                    )
        except Exception:
            logger.warning("atlas logbook payment query failed", exc_info=True)

        return (
            "PAYMENT ALERTS (fallback — structured tables unavailable)\n"
            f"{logbook_context or 'No payment events in SQLite logbook.'}\n"
            "Run migration 029 + seed work_orders for structured receivables."
        )
