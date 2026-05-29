"""Vera — Fact Checker agent.

Verifies claims and statements against the knowledge base,
flagging inaccuracies and providing corrections.
Now operates via the ReAct loop with knowledge-search,
external-verification, and Guardrails-based validation tools
including competitor detection and confidentiality checks.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.agents.creative_react_tools import register_creative_mind_tools
from brain_os.agents.formula_react_tools import register_formula_tools
from brain_os.agents.pf1_react_tools import register_pf1_icp_tools
from brain_os.exceptions import LLMError
from brain_os.prompt_loader import load_prompt
from brain_os.service_keys import ServiceKey as SK

logger = logging.getLogger(__name__)

_GUARDRAIL_TOOL_ERRORS = (
    ImportError,
    httpx.HTTPError,
    LLMError,
    json.JSONDecodeError,
    ValueError,
    TypeError,
    RuntimeError,
)

_SYSTEM_PROMPT = load_prompt("vera_system")


class Vera(BaseAgent):
    name = "vera"
    role = "Fact Checker"
    description = "Verifies claims against the knowledge base"
    knowledge_categories = [
        "company_internal",
        "sales_and_crm",
        "contracts_and_legal",
        "project_case_studies",
    ]

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="search_qdrant",
                description=(
                    "Search the internal knowledge base for evidence to verify a claim. "
                    "Optionally filter by category."
                ),
                parameters={
                    "query": "Search query for verification evidence",
                    "category": "Optional knowledge category to filter by (leave empty for all)",
                },
                handler=self._tool_search_qdrant,
            )
        )

        self.register_tool(
            AgentTool(
                name="ask_iris",
                description="Delegate to Iris for external web/news verification of a claim.",
                parameters={"query": "The claim or question to verify externally"},
                handler=self._tool_ask_iris,
            )
        )
        self.register_tool(
            AgentTool(
                name="ask_mnemon",
                description=(
                    "Delegate to Mnemon to check whether retrieved facts conflict with correction ledger truth."
                ),
                parameters={"query": "Claim or fact set to correction-check"},
                handler=self._tool_ask_mnemon,
            )
        )

        self.register_tool(
            AgentTool(
                name="validate_output",
                description=(
                    "Run Guardrails AI validators on a text to check for PII leakage, "
                    "toxic language, and other output quality issues. Returns a validation report."
                ),
                parameters={"text": "The text to validate"},
                handler=self._tool_validate_output,
            )
        )

        self.register_tool(
            AgentTool(
                name="check_faithfulness",
                description=(
                    "Check whether a response is faithful to its source context using "
                    "LLM-based entailment verification. Detects unsupported claims "
                    "and hallucinations with semantic understanding."
                ),
                parameters={
                    "response": "The response text to check",
                    "context": "The source context documents (pipe-separated if multiple)",
                },
                handler=self._tool_check_faithfulness,
            )
        )

        self.register_tool(
            AgentTool(
                name="check_competitors",
                description=(
                    "Check whether a response mentions or praises competitors. "
                    "Flags competitor names with surrounding context for review."
                ),
                parameters={"text": "The text to check for competitor mentions"},
                handler=self._tool_check_competitors,
            )
        )

        self.register_tool(
            AgentTool(
                name="check_confidentiality",
                description=(
                    "Check whether a response leaks confidential internal data such as "
                    "margins, cost prices, salaries, or vendor pricing. Use for any "
                    "response that will be sent externally."
                ),
                parameters={
                    "text": "The text to check for confidential data",
                    "direction": "Target audience: 'external' (strict) or 'internal' (lenient)",
                },
                handler=self._tool_check_confidentiality,
            )
        )
        self.register_tool(
            AgentTool(
                name="run_governance_check",
                description="Run a governance policy check for externally-facing responses.",
                parameters={
                    "text": "Response text",
                    "audience": "Audience scope (external/internal)",
                },
                handler=self._tool_run_governance_check,
            )
        )
        self.register_tool(
            AgentTool(
                name="audit_decision_log",
                description="Create an evidence-backed decision audit for verification outcomes.",
                parameters={
                    "decision": "Decision statement to audit",
                    "evidence": "Optional evidence notes",
                },
                handler=self._tool_audit_decision_log,
            )
        )
        register_pf1_icp_tools(
            self,
            include_read_icp=False,
            include_pf1_fit_brief=False,
            include_format_crm_note=False,
            include_physics=True,
            include_maestro=False,
        )
        register_formula_tools(self, include_registry=True, include_math_mode=False)
        register_creative_mind_tools(self, profile="verify")

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)

    async def _tool_search_qdrant(self, query: str, category: str = "") -> str:
        if category.strip():
            results = await self._retriever.search_by_category(query, category.strip())
        else:
            results = await self._retriever.search(query)
        return self._format_context(results)

    async def _tool_ask_iris(self, query: str) -> str:
        if not self._services.get(SK.PANTHEON):
            return "Pantheon service not available; cannot delegate to Iris."
        return await self._tool_ask_agent("iris", query)

    async def _tool_ask_mnemon(self, query: str) -> str:
        if not self._services.get(SK.PANTHEON):
            return "Pantheon service not available; cannot delegate to Mnemon."
        return await self._tool_ask_agent("mnemon", query)

    async def _tool_validate_output(self, text: str) -> str:
        try:
            from brain_os.brain.guardrails import validate_output

            result = await validate_output(text)
            return json.dumps(result, default=str)
        except _GUARDRAIL_TOOL_ERRORS as exc:
            return f"Validation error: {exc}"

    async def _tool_check_faithfulness(self, response: str, context: str = "") -> str:
        try:
            from brain_os.brain.guardrails import check_faithfulness

            context_docs = [c.strip() for c in context.split("|") if c.strip()]
            result = await check_faithfulness(response, context_docs)
            return json.dumps(result, default=str)
        except _GUARDRAIL_TOOL_ERRORS as exc:
            return f"Faithfulness check error: {exc}"

    async def _tool_check_competitors(self, text: str) -> str:
        try:
            from brain_os.brain.guardrails import check_competitor_mentions

            result = await check_competitor_mentions(text)
            return json.dumps(result, default=str)
        except _GUARDRAIL_TOOL_ERRORS as exc:
            return f"Competitor check error: {exc}"

    async def _tool_check_confidentiality(self, text: str, direction: str = "external") -> str:
        try:
            from brain_os.brain.guardrails import check_confidentiality

            result = await check_confidentiality(text, direction=direction)
            return json.dumps(result, default=str)
        except _GUARDRAIL_TOOL_ERRORS as exc:
            return f"Confidentiality check error: {exc}"

    async def _tool_run_governance_check(
        self,
        text: str,
        audience: str = "external",
    ) -> str:
        return await self.use_skill(
            "run_governance_check",
            text=text,
            audience=audience,
        )

    async def _tool_audit_decision_log(self, decision: str, evidence: str = "") -> str:
        return await self.use_skill(
            "audit_decision_log",
            decision=decision,
            evidence=evidence,
        )
