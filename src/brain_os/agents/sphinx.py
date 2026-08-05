"""Sphinx — Gatekeeper / Clarifier agent.

When a query is ambiguous or incomplete, Sphinx asks targeted
clarifying questions before routing to specialist agents.
Now operates via the ReAct loop with query-analysis and
clarification tools.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import ClarityAssessment

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("sphinx_system")


class Sphinx(BaseAgent):
    name = "sphinx"
    role = "Gatekeeper / Clarifier"
    description = "Asks clarifying questions when queries are ambiguous"
    knowledge_categories = [
        "company_internal",
        "webcall transcripts",
        "market_research_and_analysis",
    ]
    timeout = 30

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="analyze_query",
                description=(
                    "Assess the clarity and intent of a user query. "
                    "Returns JSON with 'clear' (bool), 'ambiguity_reason' (str), and 'intent' (str)."
                ),
                parameters={"query": "The user query to analyze"},
                handler=self._tool_analyze_query,
            )
        )

        self.register_tool(
            AgentTool(
                name="suggest_clarifications",
                description=(
                    "Generate a list of clarifying questions for an ambiguous query. "
                    "Returns a JSON list of question strings."
                ),
                parameters={"query": "The ambiguous query to generate clarifications for"},
                handler=self._tool_suggest_clarifications,
            )
        )
        self.register_tool(
            AgentTool(
                name="run_governance_check",
                description="Check clarification text against governance/policy boundaries.",
                parameters={
                    "text": "Clarification question or response draft",
                    "audience": "Audience scope (external/internal)",
                },
                handler=self._tool_run_governance_check,
            )
        )

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)

    async def assess_clarity(self, query: str) -> ClarityAssessment:
        """Structured clarity check used by the task orchestrator.

        Unlike ``handle()`` (which returns ``[CLEAR]``/``[CLARIFY]`` tagged
        text via the ReAct loop), this method returns a validated Pydantic
        model suitable for programmatic branching.
        """
        await self._ensure_llm()
        system_prompt = self._compose_system_prompt(_SYSTEM_PROMPT)
        if inspect.isawaitable(system_prompt):
            system_prompt = await system_prompt
        return await self._llm.generate_structured(
            str(system_prompt),
            (
                f"Assess whether this request is clear enough to act on: {query}\n\n"
                "If it IS clear, set clear=true. "
                "If it is NOT clear, set clear=false, explain why in ambiguity_reason, "
                "provide 2-3 specific clarifying questions, and fill missing_slots with "
                "the key unresolved fields (e.g., company, project, machine, date_range, scope). "
                "Set can_answer_partially=true only for low-risk asks that can proceed with explicit assumptions."
            ),
            ClarityAssessment,
            name="sphinx.assess_clarity",
            model_profile="fast",
        )

    async def _tool_analyze_query(self, query: str) -> str:
        return await self.call_llm(
            "You are a query clarity analyst. Respond ONLY with valid JSON.",
            (
                f"Assess the clarity of this query: {query}\n\n"
                'Return JSON: {"clear": bool, "ambiguity_reason": "...", "intent": "..."}'
            ),
        )

    async def _tool_suggest_clarifications(self, query: str) -> str:
        return await self.call_llm(
            "You are a clarification specialist. Respond ONLY with a valid JSON list of strings.",
            (
                f"This query is ambiguous: {query}\n\n"
                "Generate 2-4 targeted clarifying questions that would resolve the ambiguity. "
                'Return a JSON list: ["question1", "question2", ...]'
            ),
        )

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
