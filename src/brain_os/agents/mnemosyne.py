"""Mnemosyne — Memory Keeper agent.

Manages long-term memory storage and retrieval, ensuring important
information persists across conversations.
Now operates via the ReAct loop with recall, store, episodic,
relationship, and goal tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.exceptions import DatabaseError
from brain_os.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("mnemosyne_system")


class Mnemosyne(BaseAgent):
    name = "mnemosyne"
    role = "Memory Keeper"
    description = "Manages long-term memory storage and retrieval"
    knowledge_categories = [
        "company_internal",
        "sales_and_crm",
        "project_case_studies",
    ]

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        if self._services.get("long_term_memory"):
            self.register_tool(
                AgentTool(
                    name="recall_long_term",
                    description="Search long-term semantic memory for past facts, context, and stored knowledge.",
                    parameters={
                        "query": "What to search for in long-term memory",
                        "user_id": "User ID scope (default 'global')",
                    },
                    handler=self._tool_recall_long_term,
                )
            )

            self.register_tool(
                AgentTool(
                    name="store_long_term",
                    description="Store an important fact or insight in long-term memory for future recall.",
                    parameters={
                        "content": "The fact or insight to store",
                        "user_id": "User ID scope (default 'global')",
                    },
                    handler=self._tool_store_long_term,
                )
            )

        if self._services.get("episodic_memory"):
            self.register_tool(
                AgentTool(
                    name="get_episodic_memory",
                    description="Search episodic memory for past interaction episodes and experiences.",
                    parameters={"query": "What to search for in episodic memory"},
                    handler=self._tool_get_episodic_memory,
                )
            )

        if self._services.get("relationship_memory"):
            self.register_tool(
                AgentTool(
                    name="get_relationship",
                    description="Look up the full relationship profile for a contact.",
                    parameters={"contact_id": "Contact identifier"},
                    handler=self._tool_get_relationship,
                )
            )

        if self._services.get("goal_manager"):
            self.register_tool(
                AgentTool(
                    name="get_goals",
                    description="Get the active goal and slot-filling progress for a contact.",
                    parameters={"contact_id": "Contact identifier"},
                    handler=self._tool_get_goals,
                )
            )
        self.register_tool(
            AgentTool(
                name="validate_correction_consistency",
                description="Check whether a memory statement conflicts with known corrections.",
                parameters={
                    "statement": "Statement to validate",
                    "ledger_context": "Optional correction ledger context",
                },
                handler=self._tool_validate_correction_consistency,
            )
        )

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)

    async def _tool_recall_long_term(self, query: str, user_id: str = "global") -> str:
        mem = self._services["long_term_memory"]
        results = await mem.search(query, user_id=user_id)
        if not results:
            return "No long-term memories found."
        lines = [f"- {m.get('memory', m.get('content', ''))}" for m in results]
        return "\n".join(lines)

    async def _tool_store_long_term(self, content: str, user_id: str = "global") -> str:
        mem = self._services["long_term_memory"]
        result = await mem.store(content, user_id=user_id)
        return f"Stored successfully. ({len(result)} memory entries affected)"

    async def _tool_get_episodic_memory(self, query: str) -> str:
        episodic = self._services.get("episodic_memory")
        if episodic is None:
            return "Episodic memory not available."
        try:
            results = await episodic.search(query)
            if not results:
                return "No episodic memories found."
            lines = [f"- {e.get('summary', e.get('content', ''))[:300]}" for e in results]
            return "\n".join(lines)
        except (DatabaseError, Exception) as exc:
            return f"Episodic memory error: {exc}"

    async def _tool_get_relationship(self, contact_id: str) -> str:
        rel_mem = self._services["relationship_memory"]
        rel = await rel_mem.get_relationship(contact_id)
        return json.dumps(
            {
                "contact_id": rel.contact_id,
                "warmth_level": rel.warmth_level.value
                if hasattr(rel.warmth_level, "value")
                else str(rel.warmth_level),
                "interaction_count": rel.interaction_count,
                "memorable_moments": rel.memorable_moments[:5],
                "learned_preferences": rel.learned_preferences,
            },
            default=str,
        )

    async def _tool_get_goals(self, contact_id: str) -> str:
        gm = self._services["goal_manager"]
        goal = await gm.get_active_goal(contact_id)
        if goal is None:
            return f"No active goal for contact '{contact_id}'."
        return json.dumps(
            {
                "id": str(goal.id),
                "type": goal.goal_type.value,
                "status": goal.status.value,
                "progress": goal.progress,
                "slots": goal.required_slots,
            },
            default=str,
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
