"""Artemis — Mailbox Intelligence & Lead Hunter agent.

Scans historical email at scale, extracts sales intelligence, and
produces structured reports across four categories:

1. Customer journeys (how they became customers, conversation map)
2. Delivered machines (specs, price, delivery, open issues)
3. Hot leads (quotes sent, last interaction, days since contact)
4. Missed leads (inbound emails that never got a reply)

Artemis works with:
- **Alexandros** — seeds the scan with known accounts from inquiry forms,
  lead lists, customer spreadsheets, and exhibition data in data/imports/
- **Delphi** — classifies individual high-value emails (Artemis only
  delegates emails that pass batch triage, not all 40k)
- **Clio** — enriches analysis with KB context (machine specs, order history)
- **Prometheus** — receives enriched lead/deal data for pipeline updates

Key innovation: batch triage.  Instead of running a full ReAct loop per
email, Artemis classifies emails in batches of 20 with a single
lightweight LLM call, reducing 40k Delphi calls to ~2k batch calls.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.prompt_loader import load_prompt
from brain_os.service_keys import ServiceKey as SK

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("artemis_system")

_BATCH_TRIAGE_PROMPT = """You are a triage classifier for Machinecraft machine sales emails ONLY.

Machinecraft sells industrial thermoforming and vacuum forming machines.
Models: PF1, PF2, ATF, AM, IMG, FCS, SAM (and variants like PF1-XL, PF1-C, AM-V, PF1-X-1210, etc.)

Classify each email as:
- SALES: Directly about Machinecraft machine sales — quotes, proposals, pricing, orders, delivery, installation, machine inquiries, customer negotiations, techno-commercial offers, support/complaints about delivered machines. MUST be processed.
- SKIP: Everything else — newsletters, bank alerts, notifications, personal, internal dev tools, marketing from other companies, logistics not about MC machines, HR, finance, social media. Skip entirely.

Return ONLY valid JSON — an array of objects:
[{"id": "msg_001", "category": "SALES"}, {"id": "msg_002", "category": "SKIP"}, ...]

CRITICAL: Only SALES if the email is clearly about Machinecraft machine sales, quotes, orders, delivery, or customer support for MC machines. Vendor bills, shipping advisories, and general business correspondence are SKIP unless they specifically mention a Machinecraft machine model or deal.
"""


class Artemis(BaseAgent):
    name = "artemis"
    role = "Mailbox Intelligence & Lead Hunter"
    description = (
        "Scans historical email at scale, extracts sales intelligence, "
        "builds customer journey maps, detects missed leads, and produces "
        "structured CRM reports. Works with Alexandros for seed data."
    )
    knowledge_categories = [
        "sales_and_crm",
        "leads_and_contacts",
        "quotes_and_proposals",
        "orders_and_pos",
    ]

    @property
    def _crm(self) -> Any | None:
        return self._services.get(SK.CRM)

    @property
    def _email_processor(self) -> Any | None:
        return self._services.get(SK.EMAIL_PROCESSOR)

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="batch_triage_emails",
                description=(
                    "Classify a batch of email summaries (up to 20) as "
                    "BUSINESS_HIGH, BUSINESS_LOW, or NOISE in a single LLM call. "
                    "Input: JSON array of {id, subject, from, snippet}."
                ),
                parameters={"batch_json": "JSON array of email summaries"},
                handler=self._tool_batch_triage,
            )
        )

        self.register_tool(
            AgentTool(
                name="get_seed_accounts",
                description=(
                    "Ask Alexandros to extract company names, contact emails, and "
                    "machine models from inquiry forms, lead lists, customer "
                    "spreadsheets, and exhibition data in data/imports/. Returns "
                    "a list of known accounts to search for in the mailbox."
                ),
                parameters={"query": "What kind of accounts to look for"},
                handler=self._tool_get_seed_accounts,
            )
        )

        self.register_tool(
            AgentTool(
                name="search_mailbox",
                description=(
                    "Search Gmail for emails matching a query (from address, "
                    "subject keywords, date range). Returns parsed email summaries."
                ),
                parameters={
                    "query": "Gmail search query or keywords",
                    "after": "Start date YYYY/MM/DD (optional)",
                    "before": "End date YYYY/MM/DD (optional)",
                    "max_results": "Max results to return (default 20)",
                },
                handler=self._tool_search_mailbox,
            )
        )

        self.register_tool(
            AgentTool(
                name="analyze_thread",
                description=(
                    "Fetch and analyze a full email thread by thread_id. Returns "
                    "the conversation arc: participants, timeline, key decisions, "
                    "machine models mentioned, pricing, and current status."
                ),
                parameters={"thread_id": "Gmail thread ID"},
                handler=self._tool_analyze_thread,
            )
        )

        if self._crm:
            self.register_tool(
                AgentTool(
                    name="get_crm_contacts",
                    description=(
                        "List CRM contacts filtered by type (LIVE_CUSTOMER, "
                        "PAST_CUSTOMER, LEAD_WITH_INTERACTIONS, LEAD_NO_INTERACTIONS)."
                    ),
                    parameters={"contact_type": "Contact type filter (optional)"},
                    handler=self._tool_get_crm_contacts,
                )
            )

            self.register_tool(
                AgentTool(
                    name="get_contact_history",
                    description=(
                        "Get full interaction history for a contact by email address: "
                        "deals, interactions, company info."
                    ),
                    parameters={"email": "Contact email address"},
                    handler=self._tool_get_contact_history,
                )
            )

            self.register_tool(
                AgentTool(
                    name="get_pipeline_deals",
                    description=(
                        "Get all deals at a specific stage or all stages. Returns "
                        "deal title, value, machine model, stage, and dates."
                    ),
                    parameters={"stage": "Deal stage filter (optional, e.g. PROPOSAL)"},
                    handler=self._tool_get_pipeline_deals,
                )
            )

            self.register_tool(
                AgentTool(
                    name="get_stale_leads",
                    description="Get leads with no interaction in the last N days.",
                    parameters={"days": "Days of inactivity (default 30)"},
                    handler=self._tool_get_stale_leads,
                )
            )

        self.register_tool(
            AgentTool(
                name="ask_alexandros",
                description=(
                    "Ask Alexandros to search the document archive for customer "
                    "data, inquiry forms, order books, quote documents, or lead "
                    "lists. Alexandros has 700+ catalogued files."
                ),
                parameters={"question": "What to search for in the archive"},
                handler=self._tool_ask_alexandros,
            )
        )

        self.register_tool(
            AgentTool(
                name="ask_delphi",
                description=(
                    "Ask Delphi to classify a single email's intent, urgency, "
                    "and suggested agent. Use only for high-value emails that "
                    "need deep classification."
                ),
                parameters={"email_body": "Email body text", "subject": "Email subject"},
                handler=self._tool_ask_delphi,
            )
        )

        self.register_tool(
            AgentTool(
                name="ask_prometheus",
                description=(
                    "Ask Prometheus about the sales pipeline, deal status, "
                    "revenue data, or lead qualification."
                ),
                parameters={"question": "Sales/pipeline question"},
                handler=self._tool_ask_prometheus,
            )
        )
        self.register_tool(
            AgentTool(
                name="ask_argus",
                description=(
                    "Ask Argus for a single-company/domain deep dossier after mailbox triage "
                    "(Apollo + site + Gmail + KB pass)."
                ),
                parameters={"question": "Single-account dossier request"},
                handler=self._tool_ask_argus,
            )
        )
        self.register_tool(
            AgentTool(
                name="qualify_lead_skill",
                description="Qualify a lead using canonical CRM skill scoring.",
                parameters={"email": "Lead contact email"},
                handler=self._tool_qualify_lead_skill,
            )
        )
        self.register_tool(
            AgentTool(
                name="build_lead_report_skill",
                description="Build stale-lead report using canonical lead report skill.",
                parameters={"days": "Stale threshold days (default 14)"},
                handler=self._tool_build_lead_report_skill,
            )
        )
        self.register_tool(
            AgentTool(
                name="search_knowledge_base_skill",
                description="Search internal knowledge for lead/company enrichment context.",
                parameters={"query": "Knowledge query"},
                handler=self._tool_search_knowledge_base_skill,
            )
        )
        self.register_tool(
            AgentTool(
                name="pull_email_data_skill",
                description=(
                    "Data pulling from email past conversations: run pull_contact_email_history "
                    "and download_email_attachments for a contact. Use when asked to pull all "
                    "context from email for a lead (threads, logic tree, PDFs, quote data)."
                ),
                parameters={
                    "email": "Contact email address",
                    "folder": "Subfolder name (e.g. demo_buyer_acme) under downloaded_from_emails",
                    "analyze": "Run PDF analysis and extract quote data (default true)",
                    "store_memory": "Store summary in long-term memory (default false)",
                    "to_send_path": "Optional path to TO_SEND.md to fill price placeholders",
                    "name": "Contact name for context",
                },
                handler=self._tool_pull_email_data_skill,
            )
        )

    # ── Tool implementations ─────────────────────────────────────────────

    @staticmethod
    def _fallback_telemetry(event: str, **fields: Any) -> None:
        """Emit structured fallback telemetry for learning/audit loops."""
        logger.warning(
            "artemis_fallback event=%s payload=%s", event, json.dumps(fields, default=str)
        )

    async def _tool_batch_triage(self, batch_json: str) -> str:
        """Classify a batch of email summaries in a single LLM call."""
        try:
            batch = json.loads(batch_json) if isinstance(batch_json, str) else batch_json
        except (json.JSONDecodeError, TypeError):
            return '{"error": "Invalid JSON input"}'

        summaries = []
        for item in batch[:20]:
            summaries.append(
                f"ID: {item.get('id', '?')} | "
                f"From: {item.get('from', '?')} | "
                f"Subject: {item.get('subject', '?')} | "
                f"Snippet: {item.get('snippet', '')[:150]}"
            )

        user_msg = _BATCH_TRIAGE_PROMPT + "\n\nEMAILS:\n" + "\n".join(summaries)

        await self._ensure_llm()
        result = await self._llm.generate_text(
            "You are a fast email classifier. Return only valid JSON.",
            user_msg,
            temperature=0.0,
            max_tokens=1000,
        )
        return result

    async def _tool_get_seed_accounts(self, query: str) -> str:
        """Ask Alexandros for known accounts from the document archive."""
        return await self._tool_ask_agent(
            "alexandros",
            (
                f"Search the archive for {query}. Focus on folders: "
                "07_Leads_and_Contacts, 01_Quotes_and_Proposals, 02_Orders_and_POs, "
                "08_Sales_and_CRM. Prioritize doc types lead_list, customer_data, "
                "quote, and order. Extract company names, contact emails, machine "
                "models mentioned, and any inquiry/lead details. Return as a "
                "structured list."
            ),
        )

    async def _tool_search_mailbox(
        self,
        query: str,
        after: str = "",
        before: str = "",
        max_results: str = "20",
    ) -> str:
        """Search Gmail via the EmailProcessor."""
        ep = self._email_processor
        if not ep:
            return "EmailProcessor not available."

        try:
            limit = int(max_results)
        except (ValueError, TypeError):
            limit = 20

        emails = await ep.search_emails(
            query=query,
            after=after,
            before=before,
            max_results=min(limit, 50),
        )

        if not emails:
            return "No emails found."

        lines = []
        for e in emails:
            lines.append(
                f"- [{e.received_at.strftime('%Y-%m-%d')}] "
                f"From: {e.from_address} | "
                f"Subject: {e.subject} | "
                f"Thread: {e.thread_id or 'N/A'} | "
                f"Snippet: {e.body[:120].replace(chr(10), ' ')}"
            )
        return "\n".join(lines)

    async def _tool_analyze_thread(self, thread_id: str) -> str:
        """Fetch a thread and produce a conversation arc summary.

        Convergence policy: search -> fetch thread -> summarize with stage checks.
        """
        ep = self._email_processor
        if not ep:
            return "EmailProcessor not available."

        # Stage 1: validate input
        tid = (thread_id or "").strip()
        if not tid:
            self._fallback_telemetry("thread_analyze_missing_thread_id")
            return (
                "Thread analysis could not start (missing thread_id). "
                "status=inferred confidence=low"
            )

        # Stage 2: fetch full thread evidence
        emails = await ep.get_thread(thread_id)
        if not emails:
            self._fallback_telemetry("thread_analyze_no_messages", thread_id=tid)
            return (
                f"No messages found in thread {thread_id}. "
                "status=inferred confidence=low evidence=missing_thread"
            )

        messages = []
        for e in emails:
            messages.append(
                f"[{e.received_at.strftime('%Y-%m-%d %H:%M')}] "
                f"From: {e.from_address} → To: {e.to_address}\n"
                f"Subject: {e.subject}\n"
                f"Body: {e.body[:500]}\n"
            )

        if len(messages) < 1:
            self._fallback_telemetry("thread_analyze_empty_messages", thread_id=tid)
            return (
                f"Thread {thread_id} had no analyzable content. "
                "status=inferred confidence=low evidence=insufficient_content"
            )

        # Stage 3: summarize evidence-backed thread
        thread_text = "\n---\n".join(messages)

        summary = await self._llm.generate_text(
            "You are a sales intelligence analyst for Machinecraft.",
            (
                f"Analyze this email thread from Machinecraft (industrial machinery).\n\n"
                f"{thread_text}\n\n"
                "Return a structured summary:\n"
                "1. Participants (names, companies, roles)\n"
                "2. Timeline (key dates and what happened)\n"
                "3. Machine models mentioned (PF1, PF2, ATF, AM, IMG, FCS, SAM)\n"
                "4. Pricing/value discussed\n"
                "5. Current status (won, lost, pending, no response)\n"
                "6. Next action recommended"
            ),
            temperature=0.1,
            max_tokens=1500,
        )
        return (
            f"{summary}\n\n"
            "evidence_status=thread_verified confidence=high "
            f"message_count={len(emails)}"
        )

    async def _tool_get_crm_contacts(self, contact_type: str = "") -> str:
        """List CRM contacts, optionally filtered by type."""
        if not self._crm:
            return "CRM not available."

        filters = {}
        if contact_type:
            filters["contact_type"] = contact_type

        contacts = await self._crm.list_contacts(filters or None)
        if not contacts:
            return "No contacts found."

        lines = []
        for c in contacts[:50]:
            ct = c.contact_type.value if c.contact_type else "?"
            lines.append(
                f"- {c.name} <{c.email}> | Type: {ct} | "
                f"Score: {c.lead_score:.0f} | Source: {c.source or '?'}"
            )
        return (
            f"{len(contacts)} contacts total (showing first {min(len(contacts), 50)}):\n"
            + "\n".join(lines)
        )

    async def _tool_get_contact_history(self, email: str) -> str:
        """Get full history for a contact."""
        if not self._crm:
            return "CRM not available."

        contact = await self._crm.get_contact_by_email(email.strip().lower())
        if not contact:
            return f"No contact found for {email}."

        parts = [
            f"Name: {contact.name}",
            f"Email: {contact.email}",
            f"Type: {contact.contact_type.value if contact.contact_type else '?'}",
            f"Score: {contact.lead_score:.0f}",
            f"Source: {contact.source or '?'}",
        ]

        deals = await self._crm.get_deals_for_contact(str(contact.id))
        if deals:
            parts.append(f"\nDeals ({len(deals)}):")
            for d in deals:
                parts.append(
                    f"  - {d.get('title', '?')} | Stage: {d.get('stage', '?')} | "
                    f"Value: {d.get('currency', 'USD')} {d.get('value', 0):,.0f} | "
                    f"Machine: {d.get('machine_model', '?')}"
                )

        interactions = await self._crm.get_interactions_for_contact(str(contact.id))
        if interactions:
            parts.append(f"\nInteractions ({len(interactions)}, showing last 10):")
            for ix in interactions[:10]:
                parts.append(
                    f"  - [{ix.get('created_at', '?')[:10]}] "
                    f"{ix.get('channel', '?')} {ix.get('direction', '?')} | "
                    f"{ix.get('subject', '?')}"
                )

        return "\n".join(parts)

    async def _tool_get_pipeline_deals(self, stage: str = "") -> str:
        """Get deals, optionally filtered by stage."""
        if not self._crm:
            return "CRM not available."

        filters = {}
        if stage:
            filters["stage"] = stage

        deals = await self._crm.list_deals(filters or None)
        if not deals:
            return "No deals found."

        lines = []
        for d in deals[:50]:
            lines.append(
                f"- {d.title} | Stage: {d.stage.value if d.stage else '?'} | "
                f"Value: {d.currency} {float(d.value):,.0f} | "
                f"Machine: {d.machine_model or '?'} | "
                f"Created: {d.created_at.strftime('%Y-%m-%d') if d.created_at else '?'}"
            )
        return f"{len(deals)} deals total:\n" + "\n".join(lines)

    async def _tool_get_stale_leads(self, days: str = "30") -> str:
        """Get leads with no recent interaction."""
        if not self._crm:
            return "CRM not available."

        try:
            d = int(days)
        except (ValueError, TypeError):
            d = 30

        stale = await self._crm.get_stale_leads(days=d)
        if not stale:
            return f"No stale leads (>{d} days)."

        lines = []
        for s in stale[:30]:
            last_interaction = s.get("last_interaction_at") or "unknown"
            has_thread_proof = bool(s.get("last_thread_id"))
            if not has_thread_proof or str(last_interaction).strip() in {"", "unknown", "None"}:
                lines.append(
                    f"- {s.get('name', '?')} <{s.get('email', '?')}> | "
                    f"last_interaction={last_interaction} | "
                    "status=inferred confidence=low evidence=crm_only"
                )
                self._fallback_telemetry(
                    "stale_lead_inferred",
                    email=s.get("email"),
                    last_interaction=last_interaction,
                    has_thread_proof=has_thread_proof,
                )
            else:
                lines.append(
                    f"- {s.get('name', '?')} <{s.get('email', '?')}> | "
                    f"last_interaction={last_interaction} | "
                    "status=verified confidence=high evidence=crm_plus_thread"
                )
        return f"{len(stale)} stale leads (>{d} days):\n" + "\n".join(lines)

    async def _tool_ask_alexandros(self, question: str) -> str:
        """Delegate to Alexandros for archive search."""
        return await self._tool_ask_agent("alexandros", question)

    async def _tool_ask_delphi(self, email_body: str, subject: str = "") -> str:
        """Delegate to Delphi for deep email classification."""
        return await self._tool_ask_agent(
            "delphi",
            (
                "Classify this email thread content with task=classify_email.\n"
                f"Subject: {subject}\n\n"
                f"Body:\n{email_body[:2000]}"
            ),
        )

    async def _tool_ask_prometheus(self, question: str) -> str:
        """Delegate to Prometheus for pipeline/sales data."""
        return await self._tool_ask_agent("prometheus", question)

    async def _tool_ask_argus(self, question: str) -> str:
        """Delegate to Argus for single-account deep intelligence."""
        return await self._tool_ask_agent("argus", question)

    async def _tool_qualify_lead_skill(self, email: str) -> str:
        return await self.use_skill("qualify_lead", email=email)

    async def _tool_build_lead_report_skill(self, days: str = "14") -> str:
        try:
            age = int(days)
        except ValueError:
            age = 14
        return await self.use_skill("build_lead_report", days=age)

    async def _tool_search_knowledge_base_skill(self, query: str) -> str:
        return await self.use_skill("search_knowledge_base", query=query)

    async def _tool_pull_email_data_skill(
        self,
        email: str,
        folder: str = "",
        analyze: str = "true",
        store_memory: str = "false",
        to_send_path: str = "",
        name: str = "",
    ) -> str:
        return await self.use_skill(
            "data_pulling_from_email_past_conversations",
            email=email,
            folder=folder or email.split("@")[0].replace(".", "_"),
            analyze=analyze.lower() in ("true", "1", "yes"),
            store_memory=store_memory.lower() in ("true", "1", "yes"),
            to_send_path=to_send_path or "",
            name=name or "",
        )

    # ── Handle ───────────────────────────────────────────────────────────

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        task = ctx.get("task", "")

        if task == "batch_triage":
            return await self._tool_batch_triage(ctx.get("batch", "[]"))

        return await self.run(query, ctx, system_prompt=_SYSTEM_PROMPT)
