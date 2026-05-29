"""Async handler functions for every skill in the SKILL_MATRIX.

Each function accepts arbitrary keyword arguments and returns a status
string.  The :func:`use_skill` dispatcher at the bottom of this module
is the single entry-point used by agents.

Services (CRM, PricingEngine, retriever, etc.) are injected at startup
via :func:`bind_services`.  Handlers that need a service gracefully
degrade when it isn't available.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any

from brain_os.config import get_settings
from brain_os.exceptions import DatabaseError, IraError
from brain_os.service_keys import ServiceKey as SK
from brain_os.services.tool_runner import is_retryable_transient, run_tool
from brain_os.skills import SKILL_MATRIX

logger = logging.getLogger(__name__)

_SERVICES: dict[str, Any] = {}
_SKILL_STATS: dict[str, dict[str, Any]] = defaultdict(
    lambda: {
        "calls": 0,
        "success": 0,
        "failure": 0,
        "total_ms": 0.0,
        "last_error": "",
    }
)


def bind_services(services: dict[str, Any]) -> None:
    """Called once at startup to make shared services available to handlers."""
    _SERVICES.update(services)
    logger.info("Skill handlers bound to services: %s", sorted(services))


def _svc(name: str) -> Any | None:
    return _SERVICES.get(name)


def get_skill_stats() -> dict[str, dict[str, Any]]:
    """Return per-skill usage and reliability metrics."""
    stats: dict[str, dict[str, Any]] = {}
    for skill in sorted(SKILL_MATRIX):
        raw = _SKILL_STATS[skill]
        calls = int(raw["calls"])
        success = int(raw["success"])
        failure = int(raw["failure"])
        total_ms = float(raw["total_ms"])
        avg_ms = (total_ms / calls) if calls else 0.0
        failure_rate = (failure / calls) if calls else 0.0
        stats[skill] = {
            "calls": calls,
            "success": success,
            "failure": failure,
            "failure_rate": round(failure_rate, 4),
            "avg_ms": round(avg_ms, 2),
            "total_ms": round(total_ms, 2),
            "last_error": str(raw.get("last_error", "")),
        }
    return stats


def reset_skill_stats() -> None:
    """Reset in-memory skill metrics (useful for tests)."""
    _SKILL_STATS.clear()


async def _llm_call(system: str, user: str, *, temperature: float = 0.3) -> str:
    """Shared LLM helper with automatic Anthropic fallback."""
    from brain_os.services.llm_client import get_llm_client

    llm = get_llm_client()
    return await llm.generate_text_with_fallback(
        system,
        user,
        temperature=temperature,
        name="skill_handler",
        model_profile="fast",
    )


# ── Research & Knowledge ─────────────────────────────────────────────────


async def summarize_document(**kwargs: Any) -> str:
    text = kwargs.get("text", "")
    if not text:
        return "Error: 'text' argument required"

    retriever = _svc(SK.RETRIEVER)
    context = ""
    if retriever:
        results = await retriever.search(text[:500], limit=3)
        context = "\n".join(r.get("content", "")[:300] for r in results)

    return await _llm_call(
        "You are a document summarization specialist. Produce a concise executive "
        "summary capturing the key points, decisions, and action items.",
        f"Document to summarize ({len(text)} chars):\n\n{text[:8000]}"
        + (f"\n\nRelated context from knowledge base:\n{context}" if context else ""),
    )


async def extract_key_facts(**kwargs: Any) -> str:
    text = kwargs.get("text", "")
    if not text:
        return "Error: 'text' argument required"

    return await _llm_call(
        "Extract structured facts from the text. Return a JSON object with keys: "
        "companies (list), people (list of {name, role, email}), machines (list), "
        "prices (list of {item, amount, currency}), dates (list of {event, date}), "
        "key_decisions (list of strings).",
        f"Text to extract from:\n\n{text[:8000]}",
        temperature=0.1,
    )


async def compare_documents(**kwargs: Any) -> str:
    docs = kwargs.get("documents", [])
    if len(docs) < 2:
        return "Error: at least 2 documents required in 'documents' list"

    formatted = "\n\n---\n\n".join(f"DOCUMENT {i + 1}:\n{d[:4000]}" for i, d in enumerate(docs))

    return await _llm_call(
        "Compare the provided documents. Identify: (1) key differences, "
        "(2) common elements, (3) contradictions if any, (4) which is more "
        "recent or authoritative. Use a structured format.",
        formatted,
    )


async def search_knowledge_base(**kwargs: Any) -> str:
    retriever = _svc(SK.RETRIEVER)
    query = kwargs.get("query", "")
    if not query:
        return "Error: 'query' argument required"
    if not retriever:
        return "Knowledge base not available"

    limit = kwargs.get("limit", 10)
    category = kwargs.get("category")

    if category:
        results = await retriever.search_by_category(query, category, limit=limit)
    else:
        results = await retriever.search(query, limit=limit)

    if not results:
        return f"No results found for: {query}"

    lines = [f"Found {len(results)} results for '{query}':"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r.get('source', 'unknown')}] {r.get('content', '')[:300]}")
    return "\n".join(lines)


async def fetch_youtube_transcript(**kwargs: Any) -> str:
    from brain_os.tools.youtube_transcript import fetch_youtube_transcript as _fetch
    from brain_os.tools.youtube_transcript import format_youtube_transcript_result

    url = kwargs.get("url", kwargs.get("youtube_url", ""))
    if not url:
        return "Error: 'url' or 'youtube_url' argument required"

    languages = kwargs.get("languages")
    include_timestamps = kwargs.get("include_timestamps", True)
    if isinstance(include_timestamps, str):
        include_timestamps = include_timestamps.strip().lower() not in {"0", "false", "no"}

    payload = await _fetch(
        str(url),
        languages=languages,
        include_timestamps=bool(include_timestamps),
    )
    return format_youtube_transcript_result(payload)


# ── Sales & CRM ──────────────────────────────────────────────────────────


async def draft_outreach_email(**kwargs: Any) -> str:
    contact_email = kwargs.get("email", "")
    company = kwargs.get("company", "")
    stage = kwargs.get("stage", "INTRO")
    region = kwargs.get("region", "")
    context_notes = kwargs.get("context", "")

    retriever = _svc(SK.RETRIEVER)
    crm = _svc(SK.CRM)

    from brain_os.data import crm_tier1_bridge as t1

    crm_context = ""
    if crm and contact_email:
        contact = await t1.get_contact_by_email(crm, contact_email)
        if contact:
            deals = await t1.get_deals_for_contact(crm, str(contact.id))
            interactions = await crm.get_interactions_for_contact(str(contact.id))
            crm_context = (
                f"Contact: {contact.name} ({contact.email})\n"
                f"Company: {contact.company_id or 'N/A'}\n"
                f"Deals: {len(deals)} | Interactions: {len(interactions)}\n"
                f"Lead score: {contact.lead_score}"
            )

    kb_context = ""
    if retriever:
        search_term = company or contact_email or "Acme Corp outreach"
        results = await retriever.search(
            f"customer reference {search_term} industrial forming",
            limit=5,
        )
        kb_context = "\n".join(r.get("content", "")[:300] for r in results)

    regional_tones = {
        "germany": "formal, precise, engineering-focused",
        "india": "warm, relationship-first",
        "middle_east": "respectful, relationship-building",
        "usa": "professional, results-oriented",
        "europe": "professional, consultative",
    }
    tone = regional_tones.get(region.lower(), "professional, consultative")

    return await _llm_call(
        "You are a B2B sales email specialist for Acme Corp, an industrial "
        "industrial forming machine manufacturer. Write compelling, personalised "
        "outreach emails that demonstrate domain expertise.",
        f"Draft a {stage} stage outreach email.\n\n"
        f"Recipient: {contact_email or 'unknown'}\n"
        f"Company: {company or 'unknown'}\n"
        f"Regional tone: {tone}\n"
        f"Additional context: {context_notes}\n\n"
        f"CRM data:\n{crm_context or '(none)'}\n\n"
        f"Reference material:\n{kb_context or '(none)'}",
        temperature=0.5,
    )


async def qualify_lead(**kwargs: Any) -> str:
    crm = _svc(SK.CRM)
    email = kwargs.get("email", "")
    if not email:
        return "Error: 'email' argument required"
    if not crm:
        return "CRM not available for lead qualification"

    from brain_os.data import crm_tier1_bridge as t1

    contact = await t1.get_contact_by_email(crm, email)
    if not contact:
        return f"No contact found for {email}"

    deals = await t1.get_deals_for_contact(crm, str(contact.id))
    interactions = await crm.get_interactions_for_contact(str(contact.id))

    return json.dumps(
        {
            "contact": contact.to_dict(),
            "deal_count": len(deals),
            "interaction_count": len(interactions),
            "lead_score": contact.lead_score,
            "warmth": contact.warmth_level.value if contact.warmth_level else "unknown",
        },
        default=str,
        indent=2,
    )


async def generate_deal_summary(**kwargs: Any) -> str:
    crm = _svc(SK.CRM)
    deal_id = kwargs.get("deal_id", "")
    contact_email = kwargs.get("email", "")

    if not crm:
        return "CRM not available"

    from brain_os.data import crm_tier1_bridge as t1

    if deal_id:
        deal = await t1.get_deal(crm, deal_id)
        if not deal:
            return f"Deal {deal_id} not found"
        payload = deal.to_dict() if hasattr(deal, "to_dict") else deal
        return json.dumps(payload, default=str, indent=2)

    if contact_email:
        contact = await t1.get_contact_by_email(crm, contact_email)
        if not contact:
            return f"No contact found for {contact_email}"
        deals = await t1.get_deals_for_contact(crm, str(contact.id))
        if not deals:
            return f"No deals found for {contact_email}"
        return json.dumps(deals, default=str, indent=2)

    return "Error: 'deal_id' or 'email' argument required"


async def update_crm_record(**kwargs: Any) -> str:
    crm = _svc(SK.CRM)
    if not crm:
        return "CRM not available"

    record_type = kwargs.get("type", "")
    record_id = kwargs.get("id", "")
    updates = kwargs.get("updates", {})

    if not record_type or not record_id or not updates:
        return "Error: 'type', 'id', and 'updates' arguments required"

    try:
        if record_type == "deal":
            from brain_os.data import crm_tier1_bridge as t1

            result = await t1.update_deal(crm, record_id, **updates)
        elif record_type == "contact":
            result = await crm.update_contact(record_id, **updates)
        elif record_type == "company":
            result = await crm.update_company(record_id, **updates)
        else:
            return f"Unknown record type: {record_type}"
    except (DatabaseError, Exception) as exc:
        return f"CRM update failed for {record_type} {record_id}: {exc}"

    if result is None:
        return f"{record_type} {record_id} not found"
    return f"Updated {record_type} {record_id} successfully"


# ── Finance & Pricing ────────────────────────────────────────────────────


async def calculate_quote(**kwargs: Any) -> str:
    pricing = _svc(SK.PRICING_ENGINE)
    if not pricing:
        return "Pricing engine not available"

    machine_model = kwargs.get("machine_model", "")
    configuration = kwargs.get("configuration", {})

    if not machine_model:
        return "Error: 'machine_model' argument required"

    estimate = await pricing.estimate_price(machine_model, configuration)
    return json.dumps(estimate, default=str, indent=2)


async def analyze_revenue(**kwargs: Any) -> str:
    crm = _svc(SK.CRM)
    if not crm:
        return "CRM not available for revenue analysis"

    from brain_os.data import crm_tier1_bridge as t1

    summary = await t1.get_pipeline_summary(crm, kwargs.get("filters"))
    velocity = await crm.get_deal_velocity()
    return json.dumps({"pipeline": summary, "velocity": velocity}, default=str, indent=2)


async def forecast_pipeline(**kwargs: Any) -> str:
    crm = _svc(SK.CRM)
    if not crm:
        return "CRM not available for pipeline forecasting"

    from brain_os.data import crm_tier1_bridge as t1

    summary = await t1.get_pipeline_summary(crm)
    quotes_mgr = _svc(SK.QUOTES)
    analytics = {}
    if quotes_mgr:
        analytics = await quotes_mgr.get_quote_analytics()

    return json.dumps(
        {
            "pipeline_summary": summary,
            "quote_analytics": analytics,
        },
        default=str,
        indent=2,
    )


async def generate_invoice(**kwargs: Any) -> str:
    customer = kwargs.get("customer", "")
    quote_id = kwargs.get("quote_id", "")
    items = kwargs.get("items", [])

    if not customer:
        return "Error: 'customer' argument required"

    retriever = _svc(SK.RETRIEVER)
    kb_context = ""
    if retriever and quote_id:
        results = await retriever.search(f"quote {quote_id} {customer}", limit=5)
        kb_context = "\n".join(r.get("content", "")[:400] for r in results)

    items_text = ""
    if items:
        items_text = "\n".join(
            f"  - {it.get('description', '?')}: {it.get('amount', '?')} {it.get('currency', 'USD')}"
            for it in items
        )

    return await _llm_call(
        "You are a finance specialist at Acme Corp. Generate a professional "
        "invoice document in markdown format with proper line items, taxes, "
        "payment terms, and bank details placeholder.",
        f"Generate an invoice for:\n"
        f"Customer: {customer}\n"
        f"Quote reference: {quote_id or 'N/A'}\n"
        f"Line items:\n{items_text or '(derive from quote data)'}\n\n"
        f"Quote/order data:\n{kb_context or '(none)'}",
    )


# ── Marketing & Campaigns ────────────────────────────────────────────────


async def create_drip_sequence(**kwargs: Any) -> str:
    contact_email = kwargs.get("email", "")
    company = kwargs.get("company", "")
    region = kwargs.get("region", "")
    machine_interest = kwargs.get("machine", "")
    num_stages = kwargs.get("stages", 6)

    retriever = _svc(SK.RETRIEVER)
    kb_context = ""
    if retriever:
        search_term = f"{company} {machine_interest} industrial forming".strip()
        results = await retriever.search(search_term, limit=5)
        kb_context = "\n".join(r.get("content", "")[:300] for r in results)

    return await _llm_call(
        "You are a B2B drip campaign designer for Acme Corp, an industrial "
        "industrial forming machine manufacturer. Design multi-stage email sequences "
        "that nurture leads from awareness to purchase decision.",
        f"Design a {num_stages}-stage drip campaign.\n\n"
        f"Target: {contact_email or 'segment'}\n"
        f"Company: {company or 'unknown'}\n"
        f"Region: {region or 'unknown'}\n"
        f"Machine interest: {machine_interest or 'general'}\n\n"
        f"For each stage provide:\n"
        f"1. Stage name and goal\n"
        f"2. Days after previous email\n"
        f"3. Subject line\n"
        f"4. Email body\n"
        f"5. CTA (call to action)\n\n"
        f"Reference material:\n{kb_context or '(none)'}",
        temperature=0.5,
    )


async def generate_social_post(**kwargs: Any) -> str:
    topic = kwargs.get("topic", "")
    platform = kwargs.get("platform", "linkedin")
    if not topic:
        return "Error: 'topic' argument required"

    retriever = _svc(SK.RETRIEVER)
    kb_context = ""
    if retriever:
        results = await retriever.search(topic, limit=5)
        kb_context = "\n".join(r.get("content", "")[:300] for r in results)

    return await _llm_call(
        "You are a social media content specialist for Acme Corp, an industrial "
        "industrial forming machine manufacturer. Create engaging, professional posts "
        "that showcase domain expertise.",
        f"Draft a {platform} post about: {topic}\n\n"
        f"Guidelines:\n"
        f"- Professional but engaging tone\n"
        f"- 150-250 words for LinkedIn, shorter for Twitter\n"
        f"- Hook in the first line\n"
        f"- End with CTA or thought-provoking question\n"
        f"- 3-5 relevant hashtags\n\n"
        f"Reference material:\n{kb_context or '(none)'}",
        temperature=0.6,
    )


async def build_lead_report(**kwargs: Any) -> str:
    crm = _svc(SK.CRM)
    if not crm:
        return "CRM not available"

    stale = await crm.get_stale_leads(days=kwargs.get("days", 14))
    return json.dumps(
        {
            "stale_leads": stale[:20],
            "total_stale": len(stale),
        },
        default=str,
        indent=2,
    )


async def data_pulling_from_email_past_conversations(**kwargs: Any) -> str:
    """Run pull_contact_email_history and download_email_attachments for a contact.
    Requires: email, folder. Optional: output_path, analyze, store_memory, to_send_path, name."""
    import asyncio
    from pathlib import Path

    email = kwargs.get("email", "").strip()
    folder = kwargs.get("folder", "").strip()
    if not email or not folder:
        return "Error: 'email' and 'folder' arguments required (e.g. folder=demo_buyer_acme)"

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    output_path = kwargs.get("output_path") or str(
        project_root / "data" / "imports" / "24_WebSite_Leads" / f"{folder}_email_history.md"
    )
    analyze = kwargs.get("analyze", True)
    store_memory = kwargs.get("store_memory", False)
    to_send_path = kwargs.get("to_send_path", "")
    contact_name = kwargs.get("name", "")

    results = []

    # 1. Pull contact email history
    pull_cmd = [
        "poetry",
        "run",
        "python",
        str(project_root / "scripts" / "pull_contact_email_history.py"),
        "--email",
        email,
        "--output",
        output_path,
    ]
    if store_memory:
        pull_cmd.append("--store-memory")
    pull_cmd.append("--summarize")
    if contact_name:
        pull_cmd.extend(["--name", contact_name])

    proc = await asyncio.create_subprocess_exec(
        *pull_cmd,
        cwd=str(project_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        results.append(f"Pull history failed (exit {proc.returncode}): {stderr.decode()[:500]}")
    else:
        results.append(f"Pull history OK: {output_path}")

    # 2. Download PDF attachments
    download_cmd = [
        "poetry",
        "run",
        "python",
        str(project_root / "scripts" / "download_email_attachments.py"),
        "--email",
        email,
        "--folder",
        folder,
    ]
    if analyze:
        download_cmd.append("--analyze")
    if store_memory:
        download_cmd.append("--memory")
    if to_send_path:
        download_cmd.extend(["--to-send", to_send_path])
    if contact_name:
        download_cmd.extend(["--name", contact_name])

    proc2 = await asyncio.create_subprocess_exec(
        *download_cmd,
        cwd=str(project_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout2, stderr2 = await proc2.communicate()
    if proc2.returncode != 0:
        results.append(f"Download PDFs failed (exit {proc2.returncode}): {stderr2.decode()[:500]}")
    else:
        results.append("Download PDFs OK: examples/acme/docs/downloaded_from_emails/" + folder + "/")
        if stdout2:
            results.append(stdout2.decode()[-800:])

    return "\n".join(results)


async def schedule_campaign(**kwargs: Any) -> str:
    campaign_name = kwargs.get("name", "")
    target_segment = kwargs.get("segment", {})
    start_date = kwargs.get("start_date", "")

    if not campaign_name:
        return "Error: 'name' argument required"

    crm = _svc(SK.CRM)
    if crm:
        campaign = await crm.create_campaign(
            name=campaign_name,
            target_segment=target_segment,
        )
        return json.dumps(
            {
                "status": "scheduled",
                "campaign_id": str(campaign.id),
                "name": campaign_name,
                "start_date": start_date or "immediate",
            },
            default=str,
            indent=2,
        )

    return json.dumps(
        {
            "status": "planned",
            "name": campaign_name,
            "note": "CRM not available — campaign recorded but not persisted",
        }
    )


# ── Writing & Communication ──────────────────────────────────────────────


async def draft_proposal(**kwargs: Any) -> str:
    customer = kwargs.get("customer", "")
    machine_model = kwargs.get("machine_model", "")
    context_notes = kwargs.get("context", "")

    if not customer:
        return "Error: 'customer' argument required"

    retriever = _svc(SK.RETRIEVER)
    kb_context = ""
    if retriever:
        search_terms = [customer, machine_model, "proposal"]
        search_q = " ".join(t for t in search_terms if t)
        results = await retriever.search(search_q, limit=8)
        kb_context = "\n".join(r.get("content", "")[:400] for r in results)

    pricing_context = ""
    pricing = _svc(SK.PRICING_ENGINE)
    if pricing and machine_model:
        try:
            estimate = await pricing.estimate_price(machine_model, {})
            ep = estimate.get("estimated_price", {})
            if ep:
                pricing_context = (
                    f"Price guidance: {ep.get('currency', 'USD')} "
                    f"{ep.get('low', '?')} – {ep.get('high', '?')} "
                    f"(mid: {ep.get('mid', '?')})"
                )
        except (IraError, Exception):
            logger.debug("Pricing engine not available for proposal", exc_info=True)

    return await _llm_call(
        "You are a proposal writer for Acme Corp, an industrial industrial forming "
        "machine manufacturer. Write professional, compelling proposals that "
        "highlight technical capabilities and business value.",
        f"Draft a business proposal for:\n"
        f"Customer: {customer}\n"
        f"Machine: {machine_model or 'to be determined'}\n"
        f"Context: {context_notes}\n\n"
        f"Include sections:\n"
        f"1. Executive summary\n"
        f"2. Understanding of requirements\n"
        f"3. Proposed solution\n"
        f"4. Technical specifications\n"
        f"5. Pricing overview (use [TBD] for unconfirmed values)\n"
        f"6. Timeline and delivery\n"
        f"7. Why Acme Corp\n"
        f"8. Next steps\n\n"
        f"Pricing intelligence:\n{pricing_context or '(none)'}\n\n"
        f"Reference material:\n{kb_context or '(none)'}",
    )


async def polish_text(**kwargs: Any) -> str:
    text = kwargs.get("text", "")
    tone = kwargs.get("tone", "professional")
    if not text:
        return "Error: 'text' argument required"

    return await _llm_call(
        "You are a professional editor. Improve the grammar, clarity, tone, and "
        "flow of the text while preserving the original meaning and technical "
        "accuracy. Do not add information that isn't in the original.",
        f"Polish this text (target tone: {tone}):\n\n{text}",
    )


async def translate_text(**kwargs: Any) -> str:
    text = kwargs.get("text", "")
    target_language = kwargs.get("language", "")
    if not text or not target_language:
        return "Error: 'text' and 'language' arguments required"

    return await _llm_call(
        "You are a professional translator specializing in B2B industrial "
        "communication. Translate accurately while preserving technical terms, "
        "tone, and cultural appropriateness for the target audience.",
        f"Translate to {target_language}:\n\n{text}",
    )


async def generate_meeting_notes(**kwargs: Any) -> str:
    transcript = kwargs.get("transcript", kwargs.get("text", ""))
    attendees = kwargs.get("attendees", [])
    if not transcript:
        return "Error: 'transcript' or 'text' argument required"

    attendees_str = ", ".join(attendees) if attendees else "(not specified)"

    return await _llm_call(
        "You are a meeting notes specialist. Produce structured, actionable "
        "meeting minutes from the provided transcript or summary.",
        f"Generate meeting notes.\n\n"
        f"Attendees: {attendees_str}\n\n"
        f"Include:\n"
        f"1. Meeting summary (2-3 sentences)\n"
        f"2. Key discussion points\n"
        f"3. Decisions made\n"
        f"4. Action items (with owner and deadline if mentioned)\n"
        f"5. Open questions\n\n"
        f"Transcript/summary:\n{transcript[:8000]}",
    )


# ── Production & HR ──────────────────────────────────────────────────────


async def lookup_machine_spec(**kwargs: Any) -> str:
    retriever = _svc(SK.RETRIEVER)
    machine = kwargs.get("machine_model", kwargs.get("query", ""))
    if not machine:
        return "Error: 'machine_model' or 'query' argument required"

    if retriever:
        results = await retriever.search_by_category(
            f"{machine} specifications technical specs",
            category="machine_manuals_and_specs",
            limit=5,
        )
        if results:
            lines = [f"Specs for {machine}:"]
            for r in results:
                lines.append(f"  - {r.get('content', '')[:400]}")
            return "\n".join(lines)

    try:
        import json as _json
        from pathlib import Path

        mk_path = Path("data/machine_knowledge.json")
        if mk_path.exists():
            data = _json.loads(mk_path.read_text())
            catalog = data.get("machine_catalog", {}) if isinstance(data, dict) else {}
            for model_key, specs in catalog.items():
                if machine.upper() in model_key.upper():
                    return _json.dumps({"model": model_key, **specs}, indent=2)
    except (IraError, Exception):
        logger.debug("Machine knowledge lookup failed for %s", machine, exc_info=True)

    return f"No specs found for {machine}"


async def estimate_production_time(**kwargs: Any) -> str:
    machine_model = kwargs.get("machine_model", "")
    configuration = kwargs.get("configuration", {})
    quantity = kwargs.get("quantity", 1)

    if not machine_model:
        return "Error: 'machine_model' argument required"

    retriever = _svc(SK.RETRIEVER)
    kb_context = ""
    if retriever:
        results = await retriever.search(
            f"{machine_model} production lead time manufacturing delivery schedule",
            limit=8,
        )
        kb_context = "\n".join(r.get("content", "")[:400] for r in results)

    config_text = (
        ", ".join(f"{k}: {v}" for k, v in configuration.items()) if configuration else "standard"
    )

    return await _llm_call(
        "You are Acme's production planning specialist. Estimate lead "
        "times based on machine complexity, current order book, and historical "
        "production data. Be specific about phases: design, procurement, "
        "fabrication, assembly, testing, shipping.",
        f"Estimate production time for:\n"
        f"Machine: {machine_model}\n"
        f"Configuration: {config_text}\n"
        f"Quantity: {quantity}\n\n"
        f"Production knowledge:\n{kb_context or '(none)'}",
    )


async def lookup_employee(**kwargs: Any) -> str:
    name = kwargs.get("name", kwargs.get("query", ""))
    if not name:
        return "Error: 'name' or 'query' argument required"

    retriever = _svc(SK.RETRIEVER)
    if retriever:
        results = await retriever.search(
            f"employee {name} role contact team",
            limit=5,
        )
        if results:
            lines = [f"Employee lookup for '{name}':"]
            for r in results:
                lines.append(f"  - {r.get('content', '')[:300]}")
            return "\n".join(lines)

    return f"No employee data found for '{name}'"


async def generate_org_chart(**kwargs: Any) -> str:
    department = kwargs.get("department", "")

    retriever = _svc(SK.RETRIEVER)
    kb_context = ""
    if retriever:
        query = f"organization structure team roles {department}".strip()
        results = await retriever.search(query, limit=10)
        kb_context = "\n".join(r.get("content", "")[:300] for r in results)

    return await _llm_call(
        "You are an HR specialist. Generate an organizational chart in text "
        "format showing reporting lines, roles, and team structure.",
        f"Generate an org chart{f' for {department}' if department else ''}.\n\n"
        f"Available data:\n{kb_context or '(no HR data available)'}",
    )


# ── Procurement, Quality, Governance & Memory ────────────────────────────


async def evaluate_vendor_risk(**kwargs: Any) -> str:
    vendor = kwargs.get("vendor", kwargs.get("supplier", ""))
    context_notes = kwargs.get("context", "")
    if not vendor:
        return "Error: 'vendor' or 'supplier' argument required"

    retriever = _svc(SK.RETRIEVER)
    kb_context = ""
    if retriever:
        results = await retriever.search(
            f"{vendor} supplier performance quality delays payment terms risk",
            limit=8,
        )
        kb_context = "\n".join(r.get("content", "")[:300] for r in results)

    return await _llm_call(
        "You are a procurement risk analyst for Acme Corp. Evaluate supplier risk "
        "across dimensions: delivery reliability, quality consistency, commercial "
        "stability, and operational dependency.",
        f"Vendor: {vendor}\n"
        f"Context notes: {context_notes or '(none)'}\n\n"
        f"Evidence:\n{kb_context or '(no historical evidence)'}\n\n"
        "Return: risk score (1-10), risk tier, key risk factors, mitigation actions.",
    )


async def compare_supplier_quotes(**kwargs: Any) -> str:
    requirement = kwargs.get("requirement", kwargs.get("part", ""))
    quotes = kwargs.get("quotes", [])
    quotes_text = json.dumps(quotes, indent=2, default=str) if quotes else "(none provided)"

    retriever = _svc(SK.RETRIEVER)
    kb_context = ""
    if retriever and requirement:
        results = await retriever.search(
            f"{requirement} supplier quotes lead time quality history",
            limit=6,
        )
        kb_context = "\n".join(r.get("content", "")[:250] for r in results)

    return await _llm_call(
        "You are a strategic sourcing specialist. Compare supplier offers using total "
        "value, not just price. Include lead-time risk and quality confidence.",
        f"Requirement: {requirement or '(unspecified)'}\n"
        f"Supplier quotes:\n{quotes_text}\n\n"
        f"Historical context:\n{kb_context or '(none)'}\n\n"
        "Return a ranked comparison with recommendation and rationale.",
    )


async def forecast_component_lead_time(**kwargs: Any) -> str:
    component = kwargs.get("component", kwargs.get("part", ""))
    quantity = kwargs.get("quantity", 1)
    if not component:
        return "Error: 'component' or 'part' argument required"

    retriever = _svc(SK.RETRIEVER)
    kb_context = ""
    if retriever:
        results = await retriever.search(
            f"{component} procurement lead time vendor delivery history quantity {quantity}",
            limit=8,
        )
        kb_context = "\n".join(r.get("content", "")[:320] for r in results)

    return await _llm_call(
        "You are a supply-chain planner. Forecast lead time with a range and confidence "
        "based on procurement history, supplier reliability, and quantity impact.",
        f"Component: {component}\n"
        f"Quantity: {quantity}\n\n"
        f"Context:\n{kb_context or '(none)'}\n\n"
        "Return expected lead-time range, confidence, risks, and contingency options.",
    )


async def triage_punch_list(**kwargs: Any) -> str:
    items = kwargs.get("items", kwargs.get("punch_list", []))
    if isinstance(items, str):
        items_text = items
    else:
        items_text = json.dumps(items, indent=2, default=str)

    if not items_text or items_text == "[]":
        return "Error: 'items' or 'punch_list' argument required"

    return await _llm_call(
        "You are a quality manager. Triage punch-list items using severity, safety, "
        "customer impact, and dispatch/FAT blocking risk.",
        f"Punch-list items:\n{items_text}\n\n"
        "Return priorities (P0/P1/P2), owners, recommended sequence, and blockers.",
    )


async def generate_fat_plan(**kwargs: Any) -> str:
    machine_model = kwargs.get("machine_model", "")
    standards = kwargs.get("standards", "")
    if not machine_model:
        return "Error: 'machine_model' argument required"

    retriever = _svc(SK.RETRIEVER)
    kb_context = ""
    if retriever:
        results = await retriever.search(
            f"{machine_model} FAT checklist test plan acceptance criteria",
            limit=8,
        )
        kb_context = "\n".join(r.get("content", "")[:320] for r in results)

    return await _llm_call(
        "You are an industrial QA specialist. Build a FAT execution plan with clear "
        "acceptance criteria and evidence capture steps.",
        f"Machine model: {machine_model}\n"
        f"Applicable standards: {standards or '(none specified)'}\n\n"
        f"Reference context:\n{kb_context or '(none)'}\n\n"
        "Return pre-FAT checks, test sequence, pass/fail criteria, and sign-off checklist.",
    )


async def analyze_service_root_cause(**kwargs: Any) -> str:
    issue = kwargs.get("issue", "")
    observations = kwargs.get("observations", "")
    if not issue:
        return "Error: 'issue' argument required"

    retriever = _svc(SK.RETRIEVER)
    kb_context = ""
    if retriever:
        results = await retriever.search(
            f"{issue} service failure root cause corrective action",
            limit=8,
        )
        kb_context = "\n".join(r.get("content", "")[:300] for r in results)

    return await _llm_call(
        "You are a service reliability engineer. Perform root-cause analysis and "
        "propose corrective and preventive actions.",
        f"Issue: {issue}\n"
        f"Field observations: {observations or '(none)'}\n\n"
        f"Historical context:\n{kb_context or '(none)'}\n\n"
        "Return likely root causes, confidence level, immediate containment, and CAPA actions.",
    )


async def run_governance_check(**kwargs: Any) -> str:
    text = kwargs.get("text", kwargs.get("response", ""))
    audience = kwargs.get("audience", "external")
    if not text:
        return "Error: 'text' or 'response' argument required"

    return await _llm_call(
        "You are a governance reviewer for Acme Corp AI outputs. Flag policy risks: "
        "unverified claims, confidential disclosure, unauthorized commitments, and "
        "actions requiring human approval.",
        f"Audience: {audience}\n\n"
        f"Text to review:\n{text[:8000]}\n\n"
        "Return PASS/FAIL, detected risks, and exact remediation instructions.",
        temperature=0.1,
    )


async def audit_decision_log(**kwargs: Any) -> str:
    decision = kwargs.get("decision", kwargs.get("text", ""))
    evidence = kwargs.get("evidence", "")
    if not decision:
        return "Error: 'decision' or 'text' argument required"

    return await _llm_call(
        "You are a decision-audit specialist. Convert decisions into an auditable "
        "trace showing claims, evidence sources, assumptions, risks, and open gaps.",
        f"Decision:\n{decision}\n\n"
        f"Evidence/context:\n{evidence or '(none provided)'}\n\n"
        "Return a structured audit log with risk level and follow-up actions.",
        temperature=0.1,
    )


async def validate_correction_consistency(**kwargs: Any) -> str:
    statement = kwargs.get("statement", kwargs.get("text", ""))
    ledger_context = kwargs.get("ledger_context", "")
    if not statement:
        return "Error: 'statement' or 'text' argument required"

    return await _llm_call(
        "You are a correction-consistency checker. Determine whether a statement "
        "conflicts with known corrections or appears stale relative to updated truth.",
        f"Statement:\n{statement}\n\n"
        f"Correction ledger context:\n{ledger_context or '(not provided)'}\n\n"
        "Return CONSISTENT/CONFLICT/UNCERTAIN with reasoning and remediation.",
        temperature=0.1,
    )


async def formpack_akash_quick_quote_workflow(**kwargs: Any) -> str:
    """Return the human-defined steps for PartnerPack / AskAkash quick quotes (no external API)."""
    _ = kwargs  # optional context ignored; supports agent use_skill(...) with extra keys
    return (
        "FORMPACK / ASKAKASH QUICK QUOTE WORKFLOW (indicative costs — verify before binding quotes)\n"
        "\n"
        "1. Open the PartnerPack Quick Quote / AskAkash chat: https://askaskash.manus.space/chat\n"
        "2. Use access code: 8340\n"
        "3. Chat with the bot: describe the part (material, thickness, geometry context as needed).\n"
        "4. Use bot outputs for indicative per-piece cost, trimming cost, and tooling splits "
        "(e.g. FRP forming tool, cutting fixture) plus machine class hints where the bot provides them.\n"
        "5. Copy figures into internal review; Plutus / sales still own final commercial numbers. "
        "For customer-facing email, mark figures approximate and subject to drawing approval.\n"
        "\n"
        "This skill documents provenance only — it does not call the chat URL."
    )


# ── Dispatcher ───────────────────────────────────────────────────────────

_HANDLERS: dict[str, Any] = {
    name: func
    for name in SKILL_MATRIX
    if (func := globals().get(name)) is not None and callable(func)
}


async def use_skill(skill_name: str, **kwargs: Any) -> str:
    """Look up *skill_name* in the handler registry and execute it.

    Raises :class:`ValueError` if the skill name is not recognised.
    """
    handler = _HANDLERS.get(skill_name)
    if handler is None:
        raise ValueError(f"Unknown skill '{skill_name}'. Available skills: {sorted(SKILL_MATRIX)}")
    app_cfg = get_settings().app
    max_retries = max(0, int(app_cfg.react_tool_transient_retries))
    tool_timeout = float(app_cfg.react_tool_timeout_seconds)

    async def _invoke() -> Any:
        return await handler(**kwargs)

    started = time.perf_counter()
    _SKILL_STATS[skill_name]["calls"] += 1
    tool_result = await run_tool(
        f"skill.{skill_name}",
        _invoke,
        retries=max_retries,
        backoff_seconds=0.3,
        timeout_seconds=tool_timeout,
        is_retryable=is_retryable_transient,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    _SKILL_STATS[skill_name]["total_ms"] += elapsed_ms
    if tool_result.ok:
        _SKILL_STATS[skill_name]["success"] += 1
        return tool_result.value
    _SKILL_STATS[skill_name]["failure"] += 1
    _SKILL_STATS[skill_name]["last_error"] = (tool_result.error or "unknown")[:500]
    if tool_result.last_exception is not None:
        raise tool_result.last_exception
    raise RuntimeError(tool_result.error or f"skill '{skill_name}' failed")
