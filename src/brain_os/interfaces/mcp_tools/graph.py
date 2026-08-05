"""Graph / knowledge-graph MCP tools.

Tools backed by ``srv._knowledge_graph`` (Neo4j knowledge graph).
Access patterns are read-only — entity discovery, contact lookup, quote lookup.
"""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool

logger = logging.getLogger(__name__)


async def find_related_entities(name: str, max_hops: int = 2) -> str:
    """Explore the Neo4j knowledge graph around an entity.

    Returns nodes and relationships within max_hops of the named entity
    (person, company, machine, quote). Useful for understanding connections.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    kg = srv._knowledge_graph
    if kg is None:
        return "Knowledge graph not available."

    try:
        result = await kg.find_related_entities(name, max_hops=max_hops)
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP find_related_entities failed")
        return f"Error: {exc}"


async def find_company_contacts(company_name: str) -> str:
    """Find all contacts associated with a company in the knowledge graph.

    Returns names, emails, and roles of people linked to the company.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._knowledge_graph is None:
        return "Knowledge graph not available."

    try:
        contacts = await srv._knowledge_graph.find_company_contacts(company_name)
        return json.dumps(contacts, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP find_company_contacts failed")
        return f"Error: {exc}"


async def find_company_quotes(company_name: str) -> str:
    """Find all quotes associated with a company in the knowledge graph.

    Returns quote IDs, values, dates, statuses, and machine models.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._knowledge_graph is None:
        return "Knowledge graph not available."

    try:
        quotes = await srv._knowledge_graph.find_company_quotes(company_name)
        return json.dumps(quotes, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP find_company_quotes failed")
        return f"Error: {exc}"


async def find_similar_companies_mcp(
    company_name: str,
    limit: int = 5,
    min_score: float = 0.0,
) -> str:
    """Peer accounts by Voyage embedding cosine similarity on Company nodes (P3).

    Embeds the anchor company if missing. Requires Neo4j Company nodes.
    """
    from brain_os.brain.company_similarity import find_similar_companies

    try:
        peers = await find_similar_companies(
            company_name,
            limit=max(1, min(int(limit), 20)),
            min_score=min_score if min_score > 0 else None,
        )
        return json.dumps(
            [p.model_dump(mode="json") for p in peers],
            indent=2,
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("MCP find_similar_companies failed")
        return f"Error: {exc}"


async def get_company_context_graph(
    company_name: str,
    as_json: bool = True,
    include_similar: bool = True,
    include_sqlite_precedents: bool = True,
    similar_limit: int = 5,
) -> str:
    """Read-only context-graph slice for one account (P5).

    Returns 1-hop Neo4j neighborhood (contacts, quotes, operator/pipeline runs),
    embedding-ranked similar companies, and SQLite operator precedents.
    Does not run Cypher from the model — fixed read paths only.
    """
    from brain_os.brain.company_context_graph import (
        fetch_company_context_graph,
        format_company_context_graph_plain,
    )
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._knowledge_graph is None:
        return "Knowledge graph not available."

    try:
        bundle = await fetch_company_context_graph(
            company_name,
            include_similar=include_similar,
            include_sqlite_precedents=include_sqlite_precedents,
            similar_limit=max(1, min(int(similar_limit), 20)),
            graph=srv._knowledge_graph,
        )
    except Exception as exc:
        logger.exception("MCP get_company_context_graph failed")
        return f"Error: {exc}"

    if as_json:
        return json.dumps(
            bundle.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    return format_company_context_graph_plain(bundle)


async def get_company_icp_profile(
    company_name: str = "",
    domain: str = "",
    website: str = "",
) -> str:
    """Read Neo4j :Company ICP fields (gauge_tier, is_thermoformer, icp_category).

    Match by company name, domain host, or full website URL.
    """
    from brain_os.brain.icp_profile_access import fetch_company_icp_profile, normalize_icp_lookup
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    co, dom, web = normalize_icp_lookup(
        company_name=company_name,
        domain=domain,
        website=website,
    )

    if srv._knowledge_graph is None:
        return json.dumps(
            {
                "ok": False,
                "error": "knowledge_graph_unavailable",
                "company_name": co,
                "domain": dom,
                "website": web,
            },
            indent=2,
        )

    try:
        row = await fetch_company_icp_profile(
            knowledge_graph=srv._knowledge_graph,
            company_name=co,
            domain=dom,
            website=web,
        )
    except Exception as exc:
        logger.exception("MCP get_company_icp_profile failed")
        return json.dumps(
            {"ok": False, "error": str(exc)[:300], "company_name": co, "domain": dom},
            indent=2,
        )

    if row is None:
        return json.dumps(
            {
                "ok": False,
                "error": "company_not_found",
                "company_name": co,
                "domain": dom,
                "website": web,
                "hint": "Run brain leads classify-thermoformers --sync-neo4j or industrial former site ingest.",
            },
            indent=2,
        )
    return json.dumps({"ok": True, "icp": row}, indent=2, default=str)


async def find_intro_paths_mcp(
    target_company: str,
    max_hops: int = 5,
    limit: int = 5,
) -> str:
    """Intro paths from Acme Corp-strong nodes to a target company (G3).

    Plain Cypher shortestPath via shared contacts / exhibitions / vendors.
    For Argus / NA-sales dossiers — never sends mail.
    """
    from brain_os.interfaces import mcp_server as srv
    from brain_os.services.graph_analytics import find_intro_paths

    await srv._ensure_initialized()
    kg = srv._knowledge_graph
    if kg is None:
        return json.dumps({"ok": False, "error": "Knowledge graph not available."})

    try:
        result = await find_intro_paths(
            kg,
            target_company,
            max_hops=max(2, min(int(max_hops), 6)),
            limit=max(1, min(int(limit), 20)),
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP find_intro_paths failed")
        return json.dumps({"ok": False, "error": str(exc)[:300]})


def register(mcp: FastMCP) -> None:
    """Register graph tools on the given FastMCP instance."""
    mcp.tool()(hardened_mcp_tool(find_related_entities))
    mcp.tool()(hardened_mcp_tool(find_company_contacts))
    mcp.tool()(hardened_mcp_tool(get_company_icp_profile))
    mcp.tool()(hardened_mcp_tool(find_company_quotes))
    mcp.tool()(hardened_mcp_tool(find_similar_companies_mcp))
    mcp.tool()(hardened_mcp_tool(get_company_context_graph))
    mcp.tool()(hardened_mcp_tool(find_intro_paths_mcp))
