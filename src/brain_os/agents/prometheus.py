"""Prometheus — Sales / CRO agent.

Manages the sales pipeline, tracks deals, analyses conversion rates,
and provides strategic sales advice.  When the CRM is available
(injected via ``services``), Prometheus queries live deal, contact, and
interaction data.  Otherwise falls back to knowledge-base search.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.agents.creative_react_tools import register_creative_mind_tools
from brain_os.agents.formula_react_tools import register_formula_tools
from brain_os.agents.pf1_react_tools import register_pf1_icp_tools
from brain_os.exceptions import DatabaseError, IraError
from brain_os.prompt_loader import load_prompt
from brain_os.service_keys import ServiceKey as SK
from brain_os.services import sales_decisions as sd

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("prometheus_system")


class Prometheus(BaseAgent):
    name = "prometheus"
    role = "Chief Revenue Officer"
    description = "Sales pipeline management, deal tracking, and revenue strategy"
    knowledge_categories = [
        "sales_and_crm",
        "orders_and_pos",
        "leads_and_contacts",
        "quotes_and_proposals",
        "current machine orders",
        "webcall transcripts",
    ]

    @property
    def _crm(self) -> Any | None:
        return self._services.get(SK.CRM)

    @property
    def _quotes(self) -> Any | None:
        return self._services.get(SK.QUOTES)

    # ── tool registration ─────────────────────────────────────────────────

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="search_sales_knowledge",
                description="Search the knowledge base across all sales categories: quotes, orders, leads, contacts, webcall transcripts. This is your RICHEST data source.",
                parameters={"query": "Search query about leads, deals, companies, contacts"},
                handler=self._tool_search_sales_knowledge,
            )
        )

        if self._crm:
            self.register_tool(
                AgentTool(
                    name="search_contacts",
                    description="Search CRM contacts by name, email, or company.",
                    parameters={"query": "Search term (name, email, company)"},
                    handler=self._tool_search_contacts,
                )
            )
            self.register_tool(
                AgentTool(
                    name="get_deal",
                    description="Get full details for a specific deal by ID.",
                    parameters={"deal_id": "The deal identifier"},
                    handler=self._tool_get_deal,
                )
            )
            self.register_tool(
                AgentTool(
                    name="get_pipeline_summary",
                    description="Get an overview of the current sales pipeline (stages, counts, values).",
                    parameters={},
                    handler=self._tool_get_pipeline_summary,
                )
            )
            self.register_tool(
                AgentTool(
                    name="list_demo_programme_deals",
                    description=(
                        "List all CRM deals in the Demo-Programme quote+reply programme "
                        "(titles start with [Demo-Programme]). Returns company, contact email, stage, value, deal id. "
                        "Optional source_mailbox (e.g. sales@example.com): only deals whose contact has "
                        "CRM email interactions logged from that Gmail account (after sales mailbox rescan/sync)."
                    ),
                    parameters={
                        "source_mailbox": (
                            "Optional. e.g. sales@example.com — filter to deals touched via that mailbox; "
                            "leave empty for all Demo-Programme deals."
                        ),
                    },
                    handler=self._tool_list_demo_programme_deals,
                )
            )
            self.register_tool(
                AgentTool(
                    name="get_stale_leads",
                    description="List leads with no activity in the given number of days.",
                    parameters={"days": "Inactivity threshold in days (default 14)"},
                    handler=self._tool_get_stale_leads,
                )
            )
            self.register_tool(
                AgentTool(
                    name="get_warm_contacts",
                    description="List all WARM and TRUSTED contacts from the CRM — these are high-value existing relationships.",
                    parameters={"limit": "Max contacts to return (default 20)"},
                    handler=self._tool_get_warm_contacts,
                )
            )
            self.register_tool(
                AgentTool(
                    name="get_active_leads",
                    description="List contacts classified as LEAD_WITH_INTERACTIONS — these are the active sales opportunities that need follow-up. THIS IS YOUR MOST IMPORTANT TOOL. There are 200+ active leads.",
                    parameters={"limit": "Max leads to return (default 50)"},
                    handler=self._tool_get_active_leads,
                )
            )
            self.register_tool(
                AgentTool(
                    name="get_outbound_readiness",
                    description=(
                        "Return outbound-readiness for top leads, including buyer-role clarity, "
                        "motivation-fit, channel friction, and discovery gaps that should block send."
                    ),
                    parameters={"limit": "Max rows to return (default 25)"},
                    handler=self._tool_get_outbound_readiness,
                )
            )
            self.register_tool(
                AgentTool(
                    name="get_contact_email_history",
                    description="Get all emails (and other interactions) sent to or from a contact: what was sent, when, subject, snippet. Use to see full context for any contact on your list.",
                    parameters={"contact_email": "Contact email address (or contact ID)"},
                    handler=self._tool_get_contact_email_history,
                )
            )
            self.register_tool(
                AgentTool(
                    name="get_punch_list_for_customer",
                    description="Get punch list (customer complaints, open issues, resolutions) for a customer. Delegates to Asclepius/Atlas. Use to see quality issues and how we solved them.",
                    parameters={
                        "company_name_or_project": "Company name or project identifier (e.g. Acme Plastics, Demo DEMO)",
                    },
                    handler=self._tool_get_punch_list_for_customer,
                )
            )
            self.register_tool(
                AgentTool(
                    name="log_customer_issue",
                    description="Log a customer complaint or punch item and its resolution in the CRM. Stored as an interaction so the CRM has full history of issues and how we solved them. Stays in touch with project manager (Atlas) and quality (Asclepius) data.",
                    parameters={
                        "contact_email_or_company": "Contact email or company name to attach the issue to",
                        "description": "Short description of the customer complaint or issue",
                        "resolution": "How we resolved it (or 'Open' if not yet resolved)",
                    },
                    handler=self._tool_log_customer_issue,
                )
            )

        if self._quotes:
            self.register_tool(
                AgentTool(
                    name="get_quote_analytics",
                    description="Get analytics on quotes (totals, conversion rates, follow-ups due).",
                    parameters={},
                    handler=self._tool_get_quote_analytics,
                )
            )

        if self._services.get(SK.PANTHEON):
            self.register_tool(
                AgentTool(
                    name="ask_quotebuilder",
                    description="Delegate a quoting question to the Quotebuilder agent.",
                    parameters={"query": "The quoting question or request"},
                    handler=self._tool_ask_quotebuilder,
                )
            )
            self.register_tool(
                AgentTool(
                    name="ask_atlas",
                    description=(
                        "Delegate timeline, milestone, or project-logbook checks to Atlas for "
                        "revenue-truth cross-referencing."
                    ),
                    parameters={"query": "Project timeline/status question"},
                    handler=self._tool_ask_atlas,
                )
            )

        register_pf1_icp_tools(
            self,
            include_pf1_fit_brief=True,
            include_physics=False,
            include_maestro=False,
        )
        self._register_sales_decision_tools()
        register_formula_tools(
            self,
            include_registry=True,
            include_math_mode=True,
            include_deal_risk=False,
        )
        register_creative_mind_tools(self, profile="strategy")

    def _register_sales_decision_tools(self) -> None:
        """Deterministic deal/account decision primitives (shared with MCP sales tools)."""
        self.register_tool(
            AgentTool(
                name="next_best_account_action",
                description=(
                    "Recommend one next best action for an account using brief + Math Mode "
                    "(readiness, cadence, priority). Use after CRM/KB context is gathered."
                ),
                parameters={
                    "company": "Company name",
                    "contact_email": "Optional primary contact email",
                },
                handler=self._tool_next_best_account_action,
            )
        )
        self.register_tool(
            AgentTool(
                name="deal_risk_audit",
                description=(
                    "Audit deal risk flags (single-threaded, stale activity, proof gaps) "
                    "before stage changes or outbound."
                ),
                parameters={
                    "deal_id": "Optional CRM deal id",
                    "company": "Company name if deal_id omitted",
                },
                handler=self._tool_deal_risk_audit,
            )
        )
        self.register_tool(
            AgentTool(
                name="stage_exit_criteria_check",
                description=(
                    "Validate CRM stage-exit criteria before moving a deal (CONTACTED through WON)."
                ),
                parameters={
                    "deal_id": "CRM deal id",
                    "target_stage": "Target stage e.g. PROPOSAL",
                },
                handler=self._tool_stage_exit_criteria_check,
            )
        )
        self.register_tool(
            AgentTool(
                name="stalled_deal_revival_playbook",
                description="Structured revival plan for stalled deals (channels, angles, signals).",
                parameters={
                    "deal_id": "Optional deal id",
                    "company": "Company if no deal_id",
                    "stall_days": "Inactivity threshold (default 21)",
                },
                handler=self._tool_stalled_deal_revival_playbook,
            )
        )
        self.register_tool(
            AgentTool(
                name="buying_committee_map",
                description="Map stakeholders and committee-role gaps for an account.",
                parameters={"company": "Company name", "domain": "Optional domain override"},
                handler=self._tool_buying_committee_map,
            )
        )
        self.register_tool(
            AgentTool(
                name="objection_rebuttal_generator",
                description=(
                    "Evidence-backed rebuttal points for price, lead_time, downtime, "
                    "switch_risk, or quality objections."
                ),
                parameters={
                    "company": "Company name",
                    "objection_type": "price|lead_time|downtime|switch_risk|quality",
                    "persona": "Optional persona hint",
                },
                handler=self._tool_objection_rebuttal_generator,
            )
        )
        self.register_tool(
            AgentTool(
                name="quote_competitiveness_check",
                description="Assess quote band, margin risk, and approval need before sharing.",
                parameters={
                    "company": "Company name",
                    "quote_payload": "Quote text or JSON summary",
                },
                handler=self._tool_quote_competitiveness_check,
            )
        )

    def _sales_ctx(self) -> sd.SalesDecisionContext:
        return sd.SalesDecisionContext.from_services(self._services, crm=self._crm)

    # ── tool handlers ─────────────────────────────────────────────────────

    async def _tool_search_sales_knowledge(self, query: str, **_kwargs: str) -> str:
        """Search across sales categories AND the full KB for maximum coverage."""
        domain_results = await self.search_domain_knowledge(query, limit=8)
        general_results = await self.search_knowledge(query, limit=5)

        seen: set[str] = set()
        merged: list[dict] = []
        for r in domain_results + general_results:
            key = r.get("content", "")[:100]
            if key not in seen:
                seen.add(key)
                merged.append(r)

        if not merged:
            return "No results found in sales knowledge base."
        return self._format_context(merged[:12])

    async def _tool_search_contacts(self, query: str) -> str:
        results = await self._crm.search_contacts(query)
        return json.dumps(results, default=str) if results else "No contacts found."

    async def _tool_get_deal(self, deal_id: str) -> str:
        deal = await self._crm.get_deal(deal_id)
        return json.dumps(deal.to_dict(), default=str) if deal else f"Deal '{deal_id}' not found."

    async def _tool_get_pipeline_summary(self) -> str:
        summary = await self._crm.get_pipeline_summary()
        return json.dumps(summary, default=str)

    async def _tool_list_demo_programme_deals(self, source_mailbox: str = "", **_kwargs: str) -> str:
        from brain_os.systems.demo_programme_crm_seed import DEAL_TITLE_PREFIX

        mb = (source_mailbox or "").strip() or None
        rows = await self._crm.list_deals_by_title_prefix(
            DEAL_TITLE_PREFIX,
            limit=64,
            source_mailbox=mb,
        )
        if not rows:
            hint = (
                " No rows for this mailbox until emails are ingested from that account "
                "(ira email rescan --mailbox …) so interactions carry source_mailbox."
                if mb
                else ""
            )
            return (
                "No Demo-Programme programme deals found in CRM (seed with: ira crm seed-demo-programme)."
                + hint
            )
        return json.dumps(rows, default=str)

    async def _tool_get_stale_leads(self, days: str = "14") -> str:
        leads = await self._crm.get_stale_leads(days=int(days))
        return json.dumps(leads, default=str) if leads else "No stale leads found."

    async def _tool_get_warm_contacts(self, limit: str = "20") -> str:
        from sqlalchemy import text

        async with self._crm.session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT c.name, c.email, co.name as company, c.lead_score, "
                    "c.warmth_level, c.contact_type "
                    "FROM contacts c LEFT JOIN companies co ON c.company_id = co.id "
                    "WHERE c.warmth_level IN ('WARM', 'TRUSTED') "
                    "ORDER BY c.lead_score DESC LIMIT :lim"
                ),
                {"lim": int(limit)},
            )
            rows = result.fetchall()
        if not rows:
            return "No warm/trusted contacts found."
        lines = []
        for r in rows:
            lines.append(
                f"- {r[0]} ({r[1]}) | Company: {r[2] or '?'} | Score: {r[3]} | {r[4]} | {r[5]}"
            )
        return "\n".join(lines)

    async def _tool_get_active_leads(self, limit: str = "50") -> str:
        """List contacts classified as active leads — these are the sales opportunities."""
        from sqlalchemy import text

        async with self._crm.session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT c.name, c.email, co.name as company, c.lead_score, "
                    "c.warmth_level, c.contact_type "
                    "FROM contacts c LEFT JOIN companies co ON c.company_id = co.id "
                    "WHERE c.contact_type IN ('LEAD_WITH_INTERACTIONS', 'ACTIVE_LEAD', 'PROSPECT') "
                    "ORDER BY c.lead_score DESC, c.updated_at DESC LIMIT :lim"
                ),
                {"lim": int(limit)},
            )
            rows = result.fetchall()
        if not rows:
            return "No active leads found."
        lines = []
        for r in rows:
            lines.append(
                f"- {r[0]} ({r[1]}) | Company: {r[2] or '?'} | Score: {r[3]} | {r[4]} | {r[5]}"
            )
        return "\n".join(lines)

    async def _tool_get_outbound_readiness(self, limit: str = "25") -> str:
        rows = await self._crm.list_deals_with_lead_score(
            limit=max(1, int(limit)),
            sort_by_score="desc",
            engagement_only=True,
        )
        if not rows:
            return "No ready outbound leads found."

        out: list[dict[str, Any]] = []
        for row in rows:
            discovery_gap_fields: list[str] = []
            if not row.get("buyer_role"):
                discovery_gap_fields.append("buyer_role")
            if not row.get("job_to_be_done"):
                discovery_gap_fields.append("job_to_be_done")
            if not row.get("motivation_signal"):
                discovery_gap_fields.append("motivation_signal")
            if row.get("channel_friction_score") is None:
                discovery_gap_fields.append("channel_friction_score")

            discovery_gap = len(discovery_gap_fields) > 0
            recommended_next_action = (
                "Run customer-discovery before outbound send"
                if discovery_gap
                else "Proceed with tailored outbound"
            )
            out.append(
                {
                    "company_name": row.get("company_name"),
                    "contact_name": row.get("contact_name"),
                    "contact_email": row.get("contact_email"),
                    "lead_score": row.get("lead_score"),
                    "motivation_fit_score": row.get("motivation_fit_score"),
                    "buyer_role": row.get("buyer_role"),
                    "job_to_be_done": row.get("job_to_be_done"),
                    "channel_friction_score": row.get("channel_friction_score"),
                    "discovery_gap": discovery_gap,
                    "discovery_gap_fields": discovery_gap_fields,
                    "recommended_next_action": recommended_next_action,
                }
            )
        return json.dumps(out, default=str)

    async def _tool_get_contact_email_history(self, contact_email: str) -> str:
        """Return all interactions (emails sent/received) for this contact."""
        contact_email = (contact_email or "").strip()
        if not contact_email:
            return "Provide a contact email or contact ID."
        if "@" in contact_email:
            contact = await self._crm.get_contact_by_email(contact_email)
        else:
            contact = await self._crm.get_contact(contact_email)
        if contact is None:
            return f"No CRM contact found for '{contact_email}'."
        interactions = await self._crm.get_interactions_for_contact(str(contact.id))
        if not interactions:
            return f"No email or interaction history for {contact.name or contact_email}."
        lines = []
        for i in interactions[:50]:
            direction = i.get("direction", "?")
            subject = (i.get("subject") or "(no subject)")[:80]
            created = i.get("created_at", "?")
            content_preview = (i.get("content") or "")[:150].replace("\n", " ")
            smb = i.get("source_mailbox")
            mb_tag = f" [{smb}]" if smb else ""
            lines.append(f"- [{direction}]{mb_tag} {created} | {subject} | {content_preview}...")
        return (
            f"Contact: {contact.name} ({contact.email})\n\nInteractions ({len(interactions)} total, showing up to 50):\n"
            + "\n".join(lines)
        )

    async def _tool_get_punch_list_for_customer(self, company_name_or_project: str) -> str:
        """Delegate to Asclepius for punch list."""
        return await self._tool_ask_agent(
            "asclepius",
            (
                f"What is the punch list for project or customer: {company_name_or_project}? "
                "List open and recently closed items, and any resolutions."
            ),
        )

    async def _tool_log_customer_issue(
        self,
        contact_email_or_company: str,
        description: str,
        resolution: str,
    ) -> str:
        """Log a customer complaint and resolution in the CRM."""
        from brain_os.data.models import Channel, Direction

        contact = None
        if (contact_email_or_company or "").strip() and "@" in (contact_email_or_company or ""):
            contact = await self._crm.get_contact_by_email(contact_email_or_company.strip())
        if contact is None and (contact_email_or_company or "").strip():
            results = await self._crm.search_contacts(contact_email_or_company.strip())
            if results:
                contact = await self._crm.get_contact(results[0]["id"])
        if contact is None:
            return f"No contact found for '{contact_email_or_company}'. Log issues against an existing contact email or company name."
        subject = f"[Customer issue] {(description or '')[:200]}"
        content = f"Description: {description or 'N/A'}\n\nResolution: {resolution or 'Open'}"
        await self._crm.create_interaction(
            contact_id=str(contact.id),
            channel=Channel.WEB,
            direction=Direction.INBOUND,
            subject=subject,
            content=content[:4000],
        )
        d_preview = (description or "")[:80]
        r_preview = (resolution or "")[:80]
        return f"Logged customer issue for {contact.name or contact.email}. Description: {d_preview}{'...' if len(description or '') > 80 else ''}. Resolution: {r_preview}{'...' if len(resolution or '') > 80 else ''}."

    async def _tool_get_quote_analytics(self) -> str:
        analytics = await self._quotes.get_quote_analytics()
        return json.dumps(analytics, default=str)

    async def _tool_ask_quotebuilder(self, query: str) -> str:
        return await self._tool_ask_agent("quotebuilder", query)

    async def _tool_ask_atlas(self, query: str) -> str:
        """Delegate project/timeline checks to Atlas."""
        return await self._tool_ask_agent("atlas", query)

    async def _tool_next_best_account_action(
        self, company: str, contact_email: str = "", **_kwargs: str
    ) -> str:
        payload = await sd.run_next_best_account_action(
            self._sales_ctx(),
            company=company,
            contact_email=contact_email,
        )
        return json.dumps(payload, indent=2, default=str)

    async def _tool_deal_risk_audit(
        self, deal_id: str = "", company: str = "", **_kwargs: str
    ) -> str:
        payload = await sd.run_deal_risk_audit(
            self._sales_ctx(),
            deal_id=deal_id,
            company=company,
        )
        return json.dumps(payload, indent=2, default=str)

    async def _tool_stage_exit_criteria_check(
        self, deal_id: str, target_stage: str, **_kwargs: str
    ) -> str:
        payload = await sd.run_stage_exit_criteria_check(
            self._sales_ctx(),
            deal_id=deal_id,
            target_stage=target_stage,
        )
        return json.dumps(payload, indent=2, default=str)

    async def _tool_stalled_deal_revival_playbook(
        self,
        deal_id: str = "",
        company: str = "",
        stall_days: str = "21",
        **_kwargs: str,
    ) -> str:
        payload = await sd.run_stalled_deal_revival_playbook(
            self._sales_ctx(),
            deal_id=deal_id,
            company=company,
            stall_days=int(stall_days or 21),
        )
        return json.dumps(payload, indent=2, default=str)

    async def _tool_buying_committee_map(
        self, company: str, domain: str = "", **_kwargs: str
    ) -> str:
        payload = await sd.run_buying_committee_map(
            self._sales_ctx(),
            company=company,
            domain=domain,
        )
        return json.dumps(payload, indent=2, default=str)

    async def _tool_objection_rebuttal_generator(
        self,
        company: str,
        objection_type: str,
        persona: str = "",
        **_kwargs: str,
    ) -> str:
        payload = await sd.run_objection_rebuttal_generator(
            self._sales_ctx(),
            company=company,
            objection_type=objection_type,
            persona=persona,
        )
        return json.dumps(payload, indent=2, default=str)

    async def _tool_quote_competitiveness_check(
        self, company: str, quote_payload: str, **_kwargs: str
    ) -> str:
        payload = await sd.run_quote_competitiveness_check(
            self._sales_ctx(),
            company=company,
            quote_payload=quote_payload,
        )
        return json.dumps(payload, indent=2, default=str)

    # ── main handler ──────────────────────────────────────────────────────

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        task = ctx.get("task", "")

        if task == "qualify_lead":
            return await self.use_skill(
                "qualify_lead",
                email=ctx.get("email", ""),
                company=ctx.get("company", ""),
            )
        if task == "deal_summary":
            return await self.use_skill(
                "generate_deal_summary",
                deal_id=ctx.get("deal_id", ""),
            )
        if task == "update_crm":
            return await self.use_skill(
                "update_crm_record",
                type=ctx.get("record_type", "deal"),
                id=ctx.get("record_id", ""),
                updates=ctx.get("updates", {}),
            )

        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)

    # ── private helpers (kept for backward compat) ────────────────────────

    async def _build_sales_intel_context(self, query: str, ctx: dict[str, Any]) -> str:
        try:
            from brain_os.brain.sales_intelligence import SalesIntelligence

            si = SalesIntelligence(
                retriever=self._retriever,
                crm=self._crm,
                pricing_engine=self._services.get(SK.PRICING_ENGINE),
            )
            contact_email = ctx.get("perception", {}).get("resolved_contact", {}).get("email")
            if contact_email:
                health = await si.assess_customer_health(contact_email)
                if health:
                    return f"Customer health: {health}"
            return "(No customer health data available for this contact.)"
        except (IraError, Exception) as exc:
            logger.debug("SalesIntelligence not available: %s", exc)
            return (
                f"Error: Could not retrieve customer health; SalesIntelligence unavailable ({exc})."
            )

    async def _build_crm_context(self, query: str, ctx: dict[str, Any]) -> str:
        try:
            parts: list[str] = []

            contact_email = ctx.get("perception", {}).get("resolved_contact", {}).get("email")
            if contact_email:
                contacts = await self._crm.search_contacts(contact_email)
                if contacts:
                    contact = contacts[0]
                    parts.append(f"Contact: {contact.get('name')} ({contact.get('email')})")
                    parts.append(f"  Company: {contact.get('company_id', 'N/A')}")
                    parts.append(f"  Lead score: {contact.get('lead_score', 0)}")
                    parts.append(f"  Warmth: {contact.get('warmth_level', 'N/A')}")

                    deals = await self._crm.get_deals_for_contact(contact["id"])
                    if deals:
                        parts.append(f"  Active deals ({len(deals)}):")
                        for d in deals[:5]:
                            parts.append(
                                f"    - {d.get('title')} | {d.get('stage')} | "
                                f"{d.get('currency', 'USD')} {d.get('value', 0):,.2f}"
                            )
                            machine = d.get("machine_model", "")
                            company_name = contact.get("company_name", "") or contact.get(
                                "company", ""
                            )
                            if machine and company_name:
                                await self.report_relationship(
                                    "Company",
                                    company_name,
                                    "INTERESTED_IN",
                                    "Machine",
                                    machine,
                                )

                    interactions = await self._crm.get_interactions_for_contact(contact["id"])
                    if interactions:
                        parts.append(f"  Recent interactions ({len(interactions)}):")
                        for i in interactions[:3]:
                            parts.append(
                                f"    - [{i.get('channel')}] {i.get('subject', 'N/A')[:80]} "
                                f"({i.get('created_at', '?')})"
                            )

            summary = await self._crm.get_pipeline_summary()
            if summary.get("total_count", 0) > 0:
                parts.append(f"\nPipeline overview: {json.dumps(summary, default=str)}")

            stale = await self._crm.get_stale_leads(days=14)
            if stale:
                parts.append(f"\nStale leads (>14 days): {len(stale)}")
                for s in stale[:3]:
                    parts.append(f"  - {s.get('name')} ({s.get('email')})")

            return "\n".join(parts) if parts else "(No CRM data found)"
        except (DatabaseError, Exception):
            logger.exception("CRM context build failed in Prometheus")
            return "(CRM query failed)"

    async def _build_quote_context(self, query: str, ctx: dict[str, Any]) -> str:
        try:
            analytics = await self._quotes.get_quote_analytics()
            followups = await self._quotes.get_quotes_due_for_followup()

            parts: list[str] = []
            if analytics.get("total_quotes", 0) > 0:
                parts.append(f"Quote analytics: {json.dumps(analytics, default=str)}")
            if followups:
                parts.append(f"Quotes due for follow-up: {len(followups)}")
                for q in followups[:3]:
                    parts.append(
                        f"  - {q.company_name or 'N/A'} | {q.machine_model or 'N/A'} | "
                        f"Status: {q.status.value if hasattr(q.status, 'value') else q.status}"
                    )

            return "\n".join(parts) if parts else "(No quote data found.)"
        except (DatabaseError, Exception) as exc:
            logger.exception("Quote context build failed in Prometheus")
            return f"Error: Could not retrieve quote context; quotes service failed ({exc})."
