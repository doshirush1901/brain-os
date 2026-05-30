"""Graph MCP registration split by license tier."""

from __future__ import annotations

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool
from brain_os.interfaces.mcp_tools.graph import (
    find_company_contacts,
    find_company_quotes,
    find_related_entities,
    find_similar_companies_mcp,
    get_company_context_graph,
    get_company_icp_profile,
)
from mcp.server.fastmcp import FastMCP


def register_community(mcp: FastMCP) -> None:
    mcp.tool()(hardened_mcp_tool(find_related_entities))
    mcp.tool()(hardened_mcp_tool(find_company_contacts))


def register_pro(mcp: FastMCP) -> None:
    mcp.tool()(hardened_mcp_tool(find_company_quotes))
    mcp.tool()(hardened_mcp_tool(find_similar_companies_mcp))
    mcp.tool()(hardened_mcp_tool(get_company_context_graph))
    mcp.tool()(hardened_mcp_tool(get_company_icp_profile))
