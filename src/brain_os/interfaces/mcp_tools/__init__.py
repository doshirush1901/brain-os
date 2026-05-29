"""Brain OS MCP tool registration — Community vs Pro tiers."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from brain_os.licensing.tiers import is_pro


def register_all_tools(mcp: FastMCP) -> None:
    """Register MCP tools allowed for the active license tier."""
    from brain_os.interfaces.mcp_tools import (
        agents,
        brief,
        corrections,
        crm_register,
        email_community,
        graph_register,
        ingest,
        memory,
        query,
        system,
    )

    query.register(mcp)
    agents.register(mcp)
    graph_register.register_community(mcp)
    crm_register.register_community(mcp)
    ingest.register(mcp)
    memory.register(mcp)
    corrections.register(mcp)
    brief.register(mcp)
    system.register(mcp)
    email_community.register(mcp)

    if is_pro():
        from brain_os.interfaces.mcp_tools import email_pro, operator

        graph_register.register_pro(mcp)
        crm_register.register_pro(mcp)
        operator.register(mcp)
        email_pro.register_pro(mcp)
