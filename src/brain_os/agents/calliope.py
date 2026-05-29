"""Calliope — Writer agent.

Drafts and polishes all external communication: emails, reports,
proposals, and presentations.  Uses skills for polishing, translation,
and proposal generation.  Equipped with ReAct tools for proposal
drafting, text polishing, translation, and research delegation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.agents.creative_react_tools import register_creative_mind_tools
from brain_os.brain.machine_mapper import recommend_machines
from brain_os.knowledge.outbound_proof_registry import (
    format_artifacts_markdown,
    match_artifacts,
)
from brain_os.prompt_loader import load_prompt
from brain_os.services import sales_decisions as sd

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("calliope_system")


class Calliope(BaseAgent):
    name = "calliope"
    role = "Chief Writer"
    description = "Drafts emails, reports, proposals, and all external communication"
    preferred_model_profile = "writing"
    knowledge_categories = [
        "quotes_and_proposals",
        "project_case_studies",
        "presentations",
        "webcall transcripts",
    ]

    # ── tool registration ────────────────────────────────────────────────

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="draft_proposal",
                description="Draft a business proposal for a customer.",
                parameters={
                    "customer": "Customer name or company",
                    "machine_model": "Machine model (optional)",
                    "context": "Additional context or requirements (optional)",
                },
                handler=self._tool_draft_proposal,
            )
        )
        self.register_tool(
            AgentTool(
                name="polish_text",
                description="Polish and improve a piece of text.",
                parameters={
                    "text": "The text to polish",
                    "tone": "Desired tone (default: professional)",
                },
                handler=self._tool_polish_text,
            )
        )
        self.register_tool(
            AgentTool(
                name="translate_text",
                description="Translate text to a target language.",
                parameters={
                    "text": "The text to translate",
                    "target_language": "Target language (e.g. German, Hindi, Dutch)",
                },
                handler=self._tool_translate_text,
            )
        )
        self.register_tool(
            AgentTool(
                name="ask_clio",
                description="Delegate a research question to Clio, the research director.",
                parameters={"query": "Research question for Clio"},
                handler=self._tool_ask_clio,
            )
        )
        self.register_tool(
            AgentTool(
                name="list_outbound_proof_artifacts",
                description=(
                    "List approved outbound proof links and allowed claims from "
                    "data/knowledge/outbound_proof_artifacts.json when present, "
                    "else examples/public_demo/outbound_proof_artifacts.json. "
                    "Use before drafting sales/follow-up emails (Evidence step). "
                    "Pass tag_filter with region, application, machine model, or competitor "
                    "(e.g. 'north_america US DEMO-X' or 'EU Illig DEMO-X-1210 Netherlands')."
                ),
                parameters={
                    "tag_filter": "Optional keywords to rank matching artifacts (comma/space ok)",
                    "limit": "Max artifacts to return (default 6)",
                },
                handler=self._tool_list_outbound_proof_artifacts,
            )
        )
        self.register_tool(
            AgentTool(
                name="evidence_backing_check",
                description=(
                    "Check draft outbound text against approved proof artifacts; "
                    "returns backed/weak/unverified claims and redlines."
                ),
                parameters={
                    "draft_text": "Email or proposal draft to verify",
                    "tag_filter": "Optional proof registry tags",
                },
                handler=self._tool_evidence_backing_check,
            )
        )
        self.register_tool(
            AgentTool(
                name="map_machine_from_inquiry",
                description=(
                    "Deterministically map inquiry details to likely machine models/families "
                    "with caveats and budget posture."
                ),
                parameters={
                    "query": "Inquiry text or draft context",
                    "region": "Optional region context, e.g. India, UAE, EU",
                    "limit": "Max recommendations (default 3)",
                },
                handler=self._tool_map_machine_from_inquiry,
            )
        )
        register_creative_mind_tools(self, profile="full")

    # ── tool handlers ────────────────────────────────────────────────────

    async def _tool_draft_proposal(
        self,
        customer: str,
        machine_model: str = "",
        context: str = "",
    ) -> str:
        enriched_context = await self._augment_context_with_machine_mapping(
            customer=customer,
            machine_model=machine_model,
            context=context,
        )
        return await self.use_skill(
            "draft_proposal",
            customer=customer,
            machine_model=machine_model,
            context=enriched_context,
        )

    async def _tool_polish_text(
        self,
        text: str,
        tone: str = "professional",
    ) -> str:
        return await self.use_skill("polish_text", text=text, tone=tone)

    async def _tool_translate_text(self, text: str, target_language: str) -> str:
        return await self.use_skill(
            "translate_text",
            text=text,
            language=target_language,
        )

    async def _tool_ask_clio(self, query: str) -> str:
        return await self._tool_ask_agent("clio", query)

    async def _tool_list_outbound_proof_artifacts(
        self,
        tag_filter: str = "",
        limit: str = "6",
    ) -> str:
        try:
            lim = max(1, min(20, int(limit)))
        except (TypeError, ValueError):
            lim = 6
        arts = match_artifacts(tag_filter=tag_filter or "", limit=lim)
        return format_artifacts_markdown(arts)

    async def _tool_evidence_backing_check(
        self,
        draft_text: str,
        tag_filter: str = "",
        **_kwargs: str,
    ) -> str:
        payload = await sd.run_evidence_backing_check(
            draft_text=draft_text,
            tag_filter=tag_filter,
        )
        return json.dumps(payload, indent=2, default=str)

    async def _tool_map_machine_from_inquiry(
        self,
        query: str,
        region: str = "",
        limit: str = "3",
    ) -> str:
        try:
            lim = max(1, min(6, int(limit)))
        except (TypeError, ValueError):
            lim = 3
        mapped = recommend_machines(query=query, region=region or None, limit=lim)
        return json.dumps(mapped, ensure_ascii=True)

    async def _augment_context_with_machine_mapping(
        self,
        customer: str,
        machine_model: str,
        context: str,
    ) -> str:
        region_hint = ""
        if (context and "india" in context.lower()) or (customer and "india" in customer.lower()):
            region_hint = "India"
        mapping_input = " ".join(part for part in [machine_model, context] if part).strip()
        if not mapping_input:
            return context
        mapped = recommend_machines(query=mapping_input, region=region_hint or None, limit=3)
        if not mapped:
            return context
        mapping_json = json.dumps(mapped, ensure_ascii=True)
        return (
            f"{context}\n\n"
            f"Machine mapping (deterministic first-pass): {mapping_json}\n"
            "Use this mapping to ground model-fit and pricing posture; "
            "state caveats and keep pricing subject to configuration and current pricing."
        ).strip()

    # ── handle ───────────────────────────────────────────────────────────

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        task = ctx.get("task", ctx.get("draft_type", ""))

        if ctx.get("draft_type") == "email" and ctx.get("interconnection_brief"):
            brief = ctx["interconnection_brief"]
            selection = brief.get("selection", {}) if isinstance(brief, dict) else {}
            must_include = selection.get("must_include", [])
            must_avoid = selection.get("must_avoid", [])
            cta = selection.get("cta", {}) if isinstance(selection, dict) else {}
            query = (
                f"{query}\n\n"
                "Interconnection brief:\n"
                f"- Must include: {must_include}\n"
                f"- Must avoid: {must_avoid}\n"
                f"- CTA primary: {cta.get('primary', '')}\n"
                f"- CTA secondary: {cta.get('secondary', '')}\n"
                "Write the email with these constraints."
            ).strip()

        if task == "proposal":
            return await self._tool_draft_proposal(
                customer=ctx.get("recipient", ctx.get("customer", "")),
                machine_model=ctx.get("machine_model", ""),
                context=query,
            )

        if task == "polish":
            return await self.use_skill(
                "polish_text",
                text=query,
                tone=ctx.get("tone", "professional"),
            )

        if task == "translate":
            return await self.use_skill(
                "translate_text",
                text=query,
                language=ctx.get("language", ctx.get("target_language", "")),
            )

        if task == "meeting_notes":
            return await self.use_skill(
                "generate_meeting_notes",
                transcript=query,
                attendees=ctx.get("attendees", []),
            )

        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)
