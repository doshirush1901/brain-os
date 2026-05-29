"""CRM MCP tools.

Tools fronting ``srv._crm`` (Acme Corp CRM):
- Deals: get_deal, list_deals, create_contact, update_deal, get_stale_leads
- Search / pipeline: search_crm, get_pipeline_summary
- Apollo.io enrichment + discovery: sync_crm_apollo, enrich_contact_apollo, search_people_apollo

Apollo tools live here despite not all touching ``srv._crm`` directly — they
enrich the same data domain (contacts, companies) and the call sites overlap.

All handlers go through the ``mcp_server`` lazy facade so test monkeypatches
(``mcp_mod._crm = mock``) keep working.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool

logger = logging.getLogger(__name__)


async def get_deal(deal_id: str) -> str:
    """Get a specific CRM deal by its ID.

    Returns deal title, value, stage, machine model, contact,
    expected close date, and notes.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._crm is None:
        return "CRM not available."

    try:
        deal = await srv._crm.get_deal(deal_id)
        if deal is None:
            return f"Deal '{deal_id}' not found."
        return json.dumps(srv._model_to_dict(deal), indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP get_deal failed")
        return f"Error: {exc}"


async def list_deals(
    stage: str = "",
    contact_id: str = "",
    limit: int = 20,
    source_mailbox: str = "",
) -> str:
    """List CRM deals with optional filters.

    Filter by stage (e.g. 'new', 'proposal', 'negotiation', 'won', 'lost')
    and/or contact_id. Optional source_mailbox (e.g. sales@example.com):
    only deals whose contact has an interaction from that Gmail account.
    Returns up to `limit` deals.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._crm is None:
        return "CRM not available."

    try:
        filters: dict[str, Any] = {}
        if stage:
            filters["stage"] = stage
        if contact_id:
            filters["contact_id"] = contact_id
        if (source_mailbox or "").strip():
            filters["source_mailbox"] = source_mailbox.strip()
        deals = await srv._crm.list_deals(filters=filters if filters else None)
        result = [srv._model_to_dict(d) for d in deals[:limit]]
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP list_deals failed")
        return f"Error: {exc}"


async def create_contact(
    name: str,
    email: str,
    company_name: str = "",
    role: str = "",
) -> str:
    """Create a new contact in the Acme Corp CRM.

    Returns the created contact record with its assigned ID.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._crm is None:
        return "CRM not available."

    try:
        kwargs: dict[str, Any] = {"name": name, "email": email}
        cname = (company_name or "").strip()
        if cname:
            company_id: str | None = None
            for co in await srv._crm.list_companies():
                if (co.name or "").strip().lower() == cname.lower():
                    company_id = str(co.id)
                    break
            if company_id is None:
                co = await srv._crm.create_company(name=cname)
                company_id = str(co.id)
            kwargs["company_id"] = company_id
        if role:
            kwargs["role"] = role
        contact = await srv._crm.create_contact(**kwargs)
        return json.dumps(srv._model_to_dict(contact), indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP create_contact failed")
        return f"Error: {exc}"


async def update_deal(
    deal_id: str,
    stage: str = "",
    value: str = "",
    notes: str = "",
) -> str:
    """Update an existing CRM deal.

    Provide the deal_id and any fields to update: stage, value, or notes.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._crm is None:
        return "CRM not available."

    try:
        kwargs: dict[str, Any] = {}
        if stage:
            kwargs["stage"] = stage
        if value:
            kwargs["value"] = value
        if notes:
            kwargs["notes"] = notes
        if not kwargs:
            return "No fields to update. Provide at least one of: stage, value, notes."
        deal = await srv._crm.update_deal(deal_id, **kwargs)
        if deal is None:
            return f"Deal '{deal_id}' not found."
        return json.dumps(srv._model_to_dict(deal), indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP update_deal failed")
        return f"Error: {exc}"


async def get_stale_leads(days: int = 14) -> str:
    """Find CRM leads with no activity in the specified number of days.

    Returns contacts that need follow-up attention.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._crm is None:
        return "CRM not available."

    try:
        leads = await srv._crm.get_stale_leads(days=days)
        return json.dumps(leads, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP get_stale_leads failed")
        return f"Error: {exc}"


async def search_crm(query: str) -> str:
    """Search the Acme Corp CRM for contacts, companies, and deals.

    Returns matching CRM records. Use for customer lookups, deal status,
    and pipeline queries.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._crm is None:
        return "CRM not available."

    try:
        contact_dicts = await srv._crm.search_contacts(query)
        contacts = [
            {
                "name": c.get("name", ""),
                "email": c.get("email", ""),
                "company": c.get("company_name", ""),
                "role": c.get("role", ""),
            }
            for c in contact_dicts[:10]
        ]

        all_companies = await srv._crm.list_companies()
        query_lower = query.lower()
        companies = [
            {
                "name": co.name,
                "region": co.region or "",
                "industry": co.industry or "",
            }
            for co in all_companies
            if query_lower in (co.name or "").lower()
            or query_lower in (co.region or "").lower()
            or query_lower in (co.industry or "").lower()
        ][:10]

        return json.dumps({"contacts": contacts, "companies": companies}, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP search_crm failed")
        return f"Error: {exc}"


async def get_pipeline_summary() -> str:
    """Get the current Acme Corp sales pipeline summary.

    Returns active deals, total value, stage breakdown, and top deals.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._crm is None:
        return "CRM not available."

    try:
        summary = await srv._crm.get_pipeline_summary()
        return json.dumps(summary, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP get_pipeline_summary failed")
        return f"Error: {exc}"


async def sync_crm_apollo(
    dry_run: bool = False,
    limit: int = 0,
    contacts_only: bool = False,
) -> str:
    """Sync CRM with Apollo.io: enrich contacts (role, LinkedIn) and companies (industry, website, employees, region).

    Uses Apollo credits. Set limit to 0 for no limit, or a positive number to process only that many contacts.
    Set dry_run=True to only report what would be updated. Set contacts_only=True to skip company enrichment.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._crm is None:
        return "CRM not available."

    try:
        from brain_os.config import get_settings

        if not get_settings().apollo.api_key.get_secret_value():
            return "APOLLO_API_KEY not set; cannot run Apollo sync."
        from brain_os.systems.apollo_crm_sync import sync_crm_with_apollo

        result = await sync_crm_with_apollo(
            srv._crm,
            dry_run=dry_run,
            limit=limit or None,
            contact_type=None,
            contacts_only=contacts_only,
        )
        if result.get("contact_type_error"):
            return result["contact_type_error"]
        return (
            f"Apollo sync done. Contacts updated: {result['contacts_updated']}, "
            f"companies updated: {result['companies_updated']}, "
            f"skipped (no email): {result['skipped_no_email']}, no match: {result['no_match']}, errors: {result['errors']}."
        )
    except Exception as exc:
        logger.exception("MCP sync_crm_apollo failed")
        return f"Error: {exc}"


async def enrich_contact_apollo(
    email: str = "",
    name: str = "",
    company_or_domain: str = "",
) -> str:
    """Enrich one person via Apollo people/match (title, company, LinkedIn). Uses credits.

    Pass email and/or name; optional company_or_domain as company name or domain (e.g. acme.com).
    Does not run email reveal. Does not write to CRM.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    try:
        from brain_os.config import get_settings

        if not get_settings().apollo.api_key.get_secret_value():
            return "APOLLO_API_KEY not set; cannot enrich via Apollo."

        if (
            not (email or "").strip()
            and not (name or "").strip()
            and not (company_or_domain or "").strip()
        ):
            return "Provide at least one of: email, name, or company_or_domain."

        from brain_os.systems.apollo_client import enrich_person_async

        domain = None
        org = None
        if (company_or_domain or "").strip():
            s = company_or_domain.strip()
            if "." in s and " " not in s:
                domain = s.replace("www.", "")
            else:
                org = s

        result = await enrich_person_async(
            email=email.strip() or None,
            name=name.strip() or None,
            domain=domain,
            organization_name=org,
        )
        if not result:
            return "No Apollo match for the given identifiers."
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP enrich_contact_apollo failed")
        return f"Error: {exc}"


async def search_people_apollo(
    person_titles: str,
    organization_name: str = "",
    organization_domains: str = "",
    page: int = 1,
    per_page: int = 10,
    max_email_reveals: int = 3,
) -> str:
    """Discover people via Apollo.io (search + optional email reveal). Uses credits.

    person_titles: comma-separated job titles (required), e.g. "CEO,VP Sales".
    organization_name: optional company keyword for q_keywords.
    organization_domains: optional comma-separated domains to narrow results.
    Requires an Apollo API key with api_search access for mixed_people/api_search.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    try:
        from brain_os.config import get_settings

        if not get_settings().apollo.api_key.get_secret_value():
            return "APOLLO_API_KEY not set; cannot search Apollo."

        def _parts(raw: str) -> list[str]:
            return [p.strip() for p in (raw or "").split(",") if p.strip()]

        titles = _parts(person_titles)
        if not titles:
            return "person_titles is required (comma-separated), e.g. 'CEO,Head of Procurement'."

        from brain_os.systems.apollo_client import search_people_async

        org_names = [organization_name.strip()] if organization_name.strip() else []
        domains = _parts(organization_domains) or None
        page_i = max(1, int(page))
        per_i = min(25, max(1, int(per_page)))
        reveals = min(5, max(0, int(max_email_reveals)))

        rows = await search_people_async(
            organization_names=org_names,
            person_titles=titles,
            organization_domains=domains,
            page=page_i,
            per_page=per_i,
            max_email_reveals=reveals,
        )
        if not rows:
            return (
                "No people returned (no matches, blocked api_search, or reveals found nothing). "
                "Try different titles/domains."
            )
        return json.dumps(rows, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP search_people_apollo failed")
        return f"Error: {exc}"


def register(mcp: FastMCP) -> None:
    """Register CRM tools on the given FastMCP instance."""
    mcp.tool()(hardened_mcp_tool(get_deal))
    mcp.tool()(hardened_mcp_tool(list_deals))
    mcp.tool()(hardened_mcp_tool(create_contact))
    mcp.tool()(hardened_mcp_tool(update_deal))
    mcp.tool()(hardened_mcp_tool(get_stale_leads))
    mcp.tool()(hardened_mcp_tool(search_crm))
    mcp.tool()(hardened_mcp_tool(get_pipeline_summary))
    mcp.tool()(hardened_mcp_tool(sync_crm_apollo))
    mcp.tool()(hardened_mcp_tool(enrich_contact_apollo))
    mcp.tool()(hardened_mcp_tool(search_people_apollo))
