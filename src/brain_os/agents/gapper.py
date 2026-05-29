"""Gapper — The Gap Resolver.

Activated when a draft response contains missing data (prices, specs,
dates, contacts).  Gapper uses every available tool — email search,
document archive, knowledge base, web search, CRM, and inter-agent
delegation — to fill gaps before the response reaches the user.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("gapper_system")

_GAP_INDICATORS = re.compile(
    r"(?:"
    r"\b(?:not specified|not available|unknown|TBD|N/?A|missing|unclear)\b"
    r"|—"
    r"|\((?:in (?:offer|PDF|quote|PO|attached)|details? (?:not|pending))[^)]*\)"
    r"|\?\s*$"
    r")",
    re.IGNORECASE,
)


def detect_gaps(text: str | Any) -> list[str]:
    """Return a list of lines from *text* that contain gap indicators."""
    if not isinstance(text, str):
        if text is None:
            return []
        logger.warning(
            "detect_gaps: expected str, got %s — coercing for robustness",
            type(text).__name__,
        )
        text = str(text)
    gaps: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _GAP_INDICATORS.search(stripped):
            gaps.append(stripped)
    return gaps


class Gapper(BaseAgent):
    name = "gapper"
    role = "Gap Resolver"
    description = (
        "Finds and fills missing data in reports and responses. "
        "Uses email search, document archive, knowledge base, web search, "
        "CRM, and inter-agent delegation to resolve every gap."
    )
    knowledge_categories = [
        "sales_and_crm",
        "quotes_and_proposals",
        "orders_and_pos",
        "project_case_studies",
    ]
    timeout = 120

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="search_archive_for_document",
                description=(
                    "Ask Alexandros to search the document archive for a specific "
                    "file (PO, quote, order confirmation) and extract its content."
                ),
                parameters={
                    "query": "What to search for (e.g. 'DemoCo PO', 'Acme quote PDF')",
                },
                handler=self._tool_search_archive,
            )
        )

        self.register_tool(
            AgentTool(
                name="ask_specialist",
                description=(
                    "Ask a specialist agent a targeted question to fill a gap. "
                    "Use 'prometheus' for sales/CRM, 'atlas' for project status, "
                    "'plutus' for finance, 'hephaestus' for production, "
                    "'alexandros' for document archive, 'iris' for web research."
                ),
                parameters={
                    "agent_name": "Agent to ask (prometheus, atlas, plutus, hephaestus, alexandros, iris)",
                    "question": "Specific question to fill the gap",
                },
                handler=self._tool_ask_specialist,
            )
        )
        self.register_tool(
            AgentTool(
                name="search_knowledge_base_skill",
                description="Search internal knowledge for missing data resolution.",
                parameters={"query": "Search query"},
                handler=self._tool_search_knowledge_base_skill,
            )
        )
        self.register_tool(
            AgentTool(
                name="extract_key_facts_skill",
                description="Extract structured missing facts from draft sections.",
                parameters={"text": "Text block to extract facts from"},
                handler=self._tool_extract_key_facts_skill,
            )
        )
        self.register_tool(
            AgentTool(
                name="run_governance_check",
                description="Run governance policy checks before returning filled gaps.",
                parameters={
                    "text": "Response text",
                    "audience": "Audience scope (external/internal)",
                },
                handler=self._tool_run_governance_check,
            )
        )

    async def _tool_search_archive(self, query: str) -> str:
        return await self._tool_ask_agent(
            "alexandros",
            f"{query}\n\nPlease treat this as action=ask and synthesize=true.",
        )

    async def _tool_ask_specialist(self, agent_name: str, question: str) -> str:
        return await self._tool_ask_agent(agent_name.lower(), question)

    async def _tool_search_knowledge_base_skill(self, query: str) -> str:
        return await self.use_skill("search_knowledge_base", query=query)

    async def _tool_extract_key_facts_skill(self, text: str) -> str:
        return await self.use_skill("extract_key_facts", text=text)

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

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        """Resolve gaps in a draft response.

        *query* should be the draft text with gaps.  Gapper will identify
        the gaps and use its tools to fill them.
        """
        ctx = context or {}

        gaps = detect_gaps(query)
        if not gaps:
            return query

        gap_summary = "\n".join(f"- {g}" for g in gaps[:20])
        enriched_query = (
            f"The following draft response has {len(gaps)} gaps that need filling:\n\n"
            f"GAPS FOUND:\n{gap_summary}\n\n"
            f"FULL DRAFT:\n{query[:6000]}\n\n"
            f"For each gap, use your tools to find the missing data. "
            f"Return the COMPLETE response with all gaps filled and sources cited."
        )

        return await self.run(enriched_query, ctx, system_prompt=_SYSTEM_PROMPT)

    async def resolve_gaps(self, draft: str, original_query: str) -> str:
        """Pipeline-callable method: detect gaps and resolve them.

        Returns the original draft unchanged if no gaps are found,
        or an enriched version with gaps filled.
        """
        gaps = detect_gaps(draft)
        if not gaps:
            logger.info("Gapper: no gaps detected, returning draft unchanged")
            return draft

        logger.info("Gapper: detected %d gaps, resolving...", len(gaps))

        gap_summary = "\n".join(f"- {g}" for g in gaps[:15])
        enriched_query = (
            f"Original user query: {original_query}\n\n"
            f"Draft response with {len(gaps)} data gaps:\n\n"
            f"GAPS FOUND:\n{gap_summary}\n\n"
            f"FULL DRAFT:\n{draft[:6000]}\n\n"
            f"Fill every gap using your tools. Return ONLY the missing data "
            f"you found, formatted as a list of corrections with sources."
        )

        try:
            corrections = await self.run(
                enriched_query,
                system_prompt=_SYSTEM_PROMPT,
            )
            if corrections and len(corrections) > 50:
                return f"{draft}\n\n---\n**Gap Resolution (by Gapper):**\n{corrections}"
            return draft
        except Exception as exc:  # intentional — Gapper must return draft on resolution failure
            logger.exception("Gapper resolution failed")
            return draft
