"""Pro-only email MCP tools (send, operator-adjacent mail)."""

from __future__ import annotations

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool
from brain_os.interfaces.mcp_tools.email import (
    email_touch_audit,
    get_account_mail_journey,
    send_email,
)
from mcp.server.fastmcp import FastMCP


def register_pro(mcp: FastMCP) -> None:
    mcp.tool()(hardened_mcp_tool(send_email))
    mcp.tool()(hardened_mcp_tool(email_touch_audit))
    mcp.tool()(hardened_mcp_tool(get_account_mail_journey))
