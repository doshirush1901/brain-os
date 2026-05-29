"""Pantheon agent MCP tools.

- ``get_agent_list`` — enumerate all registered Pantheon agents
- ``ask_agent``      — route a question to a named agent; optional
  ``handoff_json`` (an ``AgentHandoffBrief``) merges structured context
  into the query.

Both tools read ``srv._pantheon`` via the lazy facade.
"""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool
from brain_os.schemas.agent_handoff import AgentHandoffBrief

logger = logging.getLogger(__name__)


async def get_agent_list() -> str:
    """List all Ira Pantheon agents with their roles and descriptions."""
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._pantheon is None:
        return "Pantheon not available."

    try:
        agents = []
        for name, agent in srv._pantheon.agents.items():
            agents.append(
                {
                    "name": name,
                    "role": getattr(agent, "role", ""),
                    "description": getattr(agent, "description", ""),
                }
            )
        return json.dumps(agents, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP get_agent_list failed")
        return f"Error: {exc}"


async def ask_agent(agent_name: str, question: str, handoff_json: str = "") -> str:
    """Ask a specific Ira agent a question directly.

    Available agents include: athena, clio, prometheus, hephaestus,
    plutus, calliope, vera, hermes, atlas, quotebuilder, and others.
    Use get_agent_list to see all available agents.
    Optional ``handoff_json``: JSON for AgentHandoffBrief (goal, bullets, …) merged into the query.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._pantheon is None:
        return "Pantheon not available."

    try:
        agent = srv._pantheon.get_agent(agent_name.lower())
        if agent is None:
            return f"Agent '{agent_name}' not found."

        composed = question
        raw_hj = (handoff_json or "").strip()
        if raw_hj:
            brief = AgentHandoffBrief.model_validate_json(raw_hj)
            if not brief.source_agent:
                brief = brief.model_copy(update={"source_agent": "mcp"})
            composed = AgentHandoffBrief.compose_query(brief, question)
        return await agent.handle(composed)
    except Exception as exc:
        logger.exception("MCP ask_agent failed")
        return f"Error: {exc}"


def register(mcp: FastMCP) -> None:
    """Register agent tools on the given FastMCP instance."""
    mcp.tool()(hardened_mcp_tool(get_agent_list))
    mcp.tool()(hardened_mcp_tool(ask_agent))
