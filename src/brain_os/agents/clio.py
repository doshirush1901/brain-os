"""Clio — Research Director agent.

Searches the knowledge base and answers factual questions using
retrieved context.  Uses skills for document summarization, fact
extraction, and document comparison.  Equipped with ReAct tools
for Qdrant search, cross-agent delegation, and fact verification.
"""

from __future__ import annotations

import logging
from typing import Any

from brain_os.agents.base_agent import (
    AgentTool,
    BaseAgent,
    _coerce_tool_limit,
    _coerce_tool_query,
)
from brain_os.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("clio_system")


class Clio(BaseAgent):
    name = "clio"
    role = "Research Director"
    description = "Searches the knowledge base and answers factual questions"
    knowledge_categories = [
        "company_internal",
        "market_research_and_analysis",
        "project_case_studies",
        "product_catalogues",
    ]
    timeout = 120

    # ── tool registration ────────────────────────────────────────────────

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="search_qdrant",
                description="Search the Qdrant vector store, optionally filtered by category.",
                parameters={
                    "query": "string (required) — non-empty search text",
                    "category": "string (optional) — category filter; empty for all",
                    "limit": "integer (optional, default 10, max 50)",
                },
                handler=self._tool_search_qdrant,
                json_schema={
                    "type": "object",
                    "required": ["query"],
                    "additionalProperties": False,
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "category": {"type": "string", "default": ""},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                    },
                },
            )
        )
        self.register_tool(
            AgentTool(
                name="ask_alexandros",
                description="Delegate a question to Alexandros, the document archive librarian.",
                parameters={"query": "Question for Alexandros"},
                handler=self._tool_ask_alexandros,
            )
        )
        self.register_tool(
            AgentTool(
                name="ask_iris",
                description="Delegate a question to Iris, the external intelligence agent (web search, news).",
                parameters={"query": "Question for Iris"},
                handler=self._tool_ask_iris,
            )
        )
        self.register_tool(
            AgentTool(
                name="verify_with_vera",
                description="Ask Vera to fact-check a claim against the knowledge base.",
                parameters={"claim": "The claim to verify"},
                handler=self._tool_verify_with_vera,
            )
        )

    # ── tool handlers ────────────────────────────────────────────────────

    async def _tool_search_qdrant(
        self,
        query: Any = "",
        category: Any = "",
        limit: Any = "10",
        **_extra: Any,
    ) -> str:
        q = _coerce_tool_query(query)
        if not q:
            return (
                "No results found. Provide a non-empty query string "
                '(JSON: {"query": "<text>", "limit": <int optional>}).'
            )
        lim = _coerce_tool_limit(limit)
        cat = _coerce_tool_query(category)
        try:
            if cat:
                results = await self.search_category(q, cat, limit=lim)
            else:
                results = await self.search_knowledge(q, limit=lim)
        except Exception as exc:
            logger.warning("search_qdrant degraded for clio (%s: %s)", type(exc).__name__, exc)
            return f"No results found (knowledge search unavailable: {type(exc).__name__})."
        if not results:
            return "No results found."
        from brain_os.brain.untrusted_input import fence_untrusted

        lines = []
        for r in results:
            if not isinstance(r, dict):
                continue
            content = r.get("content", "")
            content_s = content if isinstance(content, str) else str(content or "")
            lines.append(f"- [{r.get('source', '?')}] {content_s[:400]}")
        if not lines:
            return "No results found."
        return fence_untrusted("\n".join(lines), source="kb_search")

    async def _tool_ask_alexandros(self, query: str) -> str:
        return await self._tool_ask_agent("alexandros", query)

    async def _tool_ask_iris(self, query: str) -> str:
        return await self._tool_ask_agent("iris", query)

    async def _tool_verify_with_vera(self, claim: str) -> str:
        return await self._tool_ask_agent("vera", claim)

    # ── handle ───────────────────────────────────────────────────────────

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        task = ctx.get("task", "")

        if task == "summarize":
            return await self.use_skill(
                "summarize_document",
                text=ctx.get("text", query),
            )

        if task == "extract_facts":
            return await self.use_skill(
                "extract_key_facts",
                text=ctx.get("text", query),
            )

        if task == "compare":
            return await self.use_skill(
                "compare_documents",
                documents=ctx.get("documents", []),
            )

        if task == "search":
            return await self.use_skill(
                "search_knowledge_base",
                query=query,
                category=ctx.get("category"),
                limit=ctx.get("limit", 10),
            )

        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)
