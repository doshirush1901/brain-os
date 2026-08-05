"""Graphe — Logger / Scribe agent.

Records Cursor chat sessions to a structured store so Brain OS can learn
from them during dream/sleep cycles.  Runs at the end of the pipeline
after the response is shaped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from brain_os.agents.base_agent import _AGENT_ENRICHMENT_ERRORS, AgentTool, BaseAgent
from brain_os.brain.cursor_session_store import (
    CursorSessionStoreBackend,
    build_cursor_session_store,
)
from brain_os.memory.cursor_session_fts import (
    build_fts_match_query,
    search_cursor_sessions,
)
from brain_os.prompt_loader import load_prompt
from brain_os.service_keys import ServiceKey as SK

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("graphe_system")
_DB_PATH = Path("data/brain/cursor_sessions.db")


def _cursor_session_store() -> CursorSessionStoreBackend:
    return build_cursor_session_store(_DB_PATH)


class Graphe(BaseAgent):
    name = "graphe"
    role = "Logger / Scribe"
    description = "Records Cursor chat sessions for dream/sleep learning"
    knowledge_categories = []
    timeout = 15

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="log_session",
                description="Record a Cursor chat turn to the session log.",
                parameters={
                    "query": "The user's query",
                    "agents_used": "Comma-separated list of agents that handled the query",
                    "response_summary": "1-2 sentence summary of the response",
                    "sources": "Sources cited (comma-separated)",
                    "email_threads": "Email thread IDs accessed (comma-separated, or empty)",
                },
                handler=self._tool_log_session,
            )
        )

        self.register_tool(
            AgentTool(
                name="get_recent_sessions",
                description="Retrieve recent session logs for analysis.",
                parameters={"limit": "Number of recent sessions to retrieve (default 20)"},
                handler=self._tool_get_recent,
            )
        )

        self.register_tool(
            AgentTool(
                name="search_sessions",
                description=(
                    "Full-text search over logged Cursor sessions (query, response summary, agents). "
                    "Use keywords from what the user asked or topics discussed."
                ),
                parameters={
                    "q": "Space-separated keywords (e.g. mailbox digest CRM)",
                    "limit": "Max hits to return (default 15, max 50)",
                },
                handler=self._tool_search_sessions,
            )
        )

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)

    async def log_turn(
        self,
        query: str,
        agents_used: list[str],
        response_summary: str,
        tool_calls: list[dict[str, Any]] | None = None,
        sources: list[str] | None = None,
        email_threads: list[str] | None = None,
        user_feedback: str | None = None,
        run_id: str | None = None,
        work_context: dict[str, Any] | None = None,
        learning_meta: dict[str, Any] | None = None,
    ) -> None:
        """Direct API for pipeline to log a turn without the ReAct loop."""
        store = _cursor_session_store()
        await store.initialize()
        await store.insert_session(
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
        )
        journal = self._services.get(SK.AGENT_JOURNAL)
        if journal is not None:
            try:
                await journal.log_action(
                    agent_name=self.name,
                    action_text=f"Handled query: {query[:100]}",
                    outcome="success",
                )
            except _AGENT_ENRICHMENT_ERRORS:
                logger.warning("GRAPHE | failed to write AgentJournal action", exc_info=True)
        logger.info("GRAPHE | logged session: %s → %s", query[:60], agents_used)

    async def get_recent_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retrieve recent sessions for dream learning."""
        store = _cursor_session_store()
        await store.initialize()
        return await store.get_recent_sessions(limit)

    async def _tool_log_session(
        self,
        query: str,
        agents_used: str = "",
        response_summary: str = "",
        sources: str = "",
        email_threads: str = "",
    ) -> str:
        agent_list = [a.strip() for a in agents_used.split(",") if a.strip()]
        from brain_os.brain.learning_telemetry import build_learning_meta

        learning_meta = build_learning_meta(
            metis_result=None,
            pipeline_ms=0.0,
            email_scope="no_email",
            agents_used=agent_list,
            raw_response=response_summary,
            had_provenance_warning=False,
            had_dlp_flag=False,
            route_method="graphe_tool",
        )
        await self.log_turn(
            query=query,
            agents_used=agent_list,
            response_summary=response_summary,
            sources=[s.strip() for s in sources.split(",") if s.strip()],
            email_threads=[t.strip() for t in email_threads.split(",") if t.strip()],
            learning_meta=learning_meta,
        )
        return "Session logged."

    async def _tool_get_recent(self, limit: str = "20") -> str:
        sessions = await self.get_recent_sessions(int(limit))
        if not sessions:
            return "No sessions logged yet."
        lines = []
        for s in sessions:
            lines.append(
                f"- [{s.get('timestamp', '?')}] Q: {s.get('query', '?')[:80]} "
                f"→ agents: {s.get('agents_used', '[]')}"
            )
        return "\n".join(lines)

    async def _tool_search_sessions(self, q: str, limit: str = "15") -> str:
        lim = max(1, min(int(limit or "15"), 50))
        mq = build_fts_match_query(q)
        if not mq:
            return "Provide non-empty search keywords."
        hits = await search_cursor_sessions(match_query=mq, db_path=_DB_PATH, limit=lim)
        if not hits:
            return "No matching logged sessions."
        lines: list[str] = []
        for h in hits:
            ts = h.get("timestamp_iso") or h.get("timestamp")
            qtext = (h.get("query") or "")[:120]
            summ = (h.get("response_summary") or "")[:120]
            lines.append(f"- [{ts}] {qtext}\n  summary: {summ}\n  id={h.get('id')}")
        return "\n".join(lines)


async def fetch_cursor_turn_by_run_id(run_id: str) -> dict[str, Any] | None:
    """Latest Graphe row for a pipeline ``run_id`` (for learning / feedback by run)."""
    rid = (run_id or "").strip()
    if not rid:
        return None
    store = _cursor_session_store()
    await store.initialize()
    return await store.fetch_by_run_id(rid)
