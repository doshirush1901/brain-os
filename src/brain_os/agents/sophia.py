"""Sophia — Reflector / Learner agent.

Reviews past decisions and interactions to identify patterns,
suggest improvements, and surface lessons learned.
Now operates via the ReAct loop with correction-history and
improvement-suggestion tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.exceptions import DatabaseError
from brain_os.prompt_loader import load_prompt
from brain_os.service_keys import ServiceKey as SK

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("sophia_system")


class Sophia(BaseAgent):
    name = "sophia"
    role = "Reflector / Learner"
    description = "Reviews past decisions and suggests improvements"
    knowledge_categories = [
        "company_internal",
        "webcall transcripts",
        "sales_and_crm",
        "project_case_studies",
    ]

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="get_correction_history",
                description="Retrieve recent corrections from the Nemesis training system.",
                parameters={},
                handler=self._tool_get_correction_history,
            )
        )

        self.register_tool(
            AgentTool(
                name="suggest_improvement",
                description="Format a structured improvement suggestion for a specific agent.",
                parameters={
                    "agent_name": "Name of the agent to improve",
                    "observation": "What was observed that needs improvement",
                },
                handler=self._tool_suggest_improvement,
            )
        )
        self.register_tool(
            AgentTool(
                name="audit_decision_log",
                description="Create auditable decision traces for reflective analysis.",
                parameters={
                    "decision": "Decision/response text",
                    "evidence": "Optional supporting context",
                },
                handler=self._tool_audit_decision_log,
            )
        )
        self.register_tool(
            AgentTool(
                name="validate_correction_consistency",
                description="Check reflective statements against correction consistency constraints.",
                parameters={
                    "statement": "Statement to validate",
                    "ledger_context": "Optional correction context",
                },
                handler=self._tool_validate_correction_consistency,
            )
        )

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        ctx = dict(context or {})

        ep_mem = self._services.get(SK.EPISODIC_MEMORY)
        if ep_mem is not None:
            try:
                episodes = await ep_mem.surface_relevant_episodes(query, "global")
                if episodes:
                    ctx["recent_episodes"] = [e.get("narrative", "")[:300] for e in episodes[:3]]
            except (DatabaseError, Exception):
                logger.debug("Sophia: episodic memory lookup failed")

        rel_mem = self._services.get(SK.RELATIONSHIP_MEMORY)
        if rel_mem is not None:
            contact_id = (ctx.get("perception") or {}).get("email", "")
            if contact_id:
                try:
                    rel = await rel_mem.get_relationship(contact_id)
                    ctx["relationship_context"] = {
                        "warmth": getattr(rel.warmth_level, "value", str(rel.warmth_level)),
                        "interaction_count": rel.interaction_count,
                    }
                except (DatabaseError, Exception):
                    logger.debug("Sophia: relationship memory lookup failed")

        return await self.run(query, ctx, system_prompt=_SYSTEM_PROMPT)

    async def _tool_get_correction_history(self) -> str:
        pantheon = self._services.get(SK.PANTHEON)
        if pantheon is None:
            return "Correction store not available."
        nemesis = pantheon.get_agent("nemesis")
        if nemesis is None or not hasattr(nemesis, "get_pending_corrections"):
            return "Correction store not available."
        try:
            corrections = await nemesis.get_pending_corrections(limit=20)
            if not corrections:
                return "No recent corrections found."
            return json.dumps(
                [
                    {
                        "entity": c.get("entity", "?"),
                        "category": c.get("category", "?"),
                        "old_value": str(c.get("old_value", ""))[:200],
                        "new_value": str(c.get("new_value", ""))[:200],
                        "severity": c.get("severity", "?"),
                        "status": c.get("status", "?"),
                    }
                    for c in corrections
                ],
                default=str,
            )
        except (DatabaseError, Exception) as exc:
            logger.debug("Failed to fetch correction history: %s", exc)
            return "Correction store not available."

    async def _tool_suggest_improvement(self, agent_name: str, observation: str) -> str:
        return (
            f"IMPROVEMENT SUGGESTION\n"
            f"Agent: {agent_name}\n"
            f"Observation: {observation}\n"
            f"Recommended action: Review and adjust {agent_name}'s behaviour "
            f"based on the above observation."
        )

    async def _tool_audit_decision_log(self, decision: str, evidence: str = "") -> str:
        return await self.use_skill(
            "audit_decision_log",
            decision=decision,
            evidence=evidence,
        )

    async def _tool_validate_correction_consistency(
        self,
        statement: str,
        ledger_context: str = "",
    ) -> str:
        return await self.use_skill(
            "validate_correction_consistency",
            statement=statement,
            ledger_context=ledger_context,
        )
