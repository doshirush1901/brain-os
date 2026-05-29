"""CRM MCP registration split by license tier."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool
from brain_os.interfaces.mcp_tools.crm import (
    create_contact,
    get_deal,
    get_pipeline_summary,
    get_stale_leads,
    list_deals,
    search_crm,
    update_deal,
)


def register_community(mcp: FastMCP) -> None:
    mcp.tool()(hardened_mcp_tool(get_deal))
    mcp.tool()(hardened_mcp_tool(list_deals))
    mcp.tool()(hardened_mcp_tool(search_crm))
    mcp.tool()(hardened_mcp_tool(get_pipeline_summary))


def register_pro(mcp: FastMCP) -> None:
    mcp.tool()(hardened_mcp_tool(create_contact))
    mcp.tool()(hardened_mcp_tool(update_deal))
    mcp.tool()(hardened_mcp_tool(get_stale_leads))
