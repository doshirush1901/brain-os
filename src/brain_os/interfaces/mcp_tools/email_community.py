"""Community-tier email MCP tools (search, read, draft — no send)."""

from __future__ import annotations

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool
from brain_os.interfaces.mcp_tools.email import draft_email, read_email_thread, search_emails
from mcp.server.fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    mcp.tool()(hardened_mcp_tool(search_emails))
    mcp.tool()(hardened_mcp_tool(read_email_thread))
    mcp.tool()(hardened_mcp_tool(draft_email))
