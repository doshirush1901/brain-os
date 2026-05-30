"""Correction submission MCP tool.

``submit_correction`` records a factual correction in the CorrectionStore
for Nemesis to process during the next Dream Mode cycle. Self-contained:
opens its own CorrectionStore instance per call, no module-global state.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool

logger = logging.getLogger(__name__)


async def submit_correction(
    entity: str,
    wrong_value: str,
    correct_value: str,
    category: str = "GENERAL",
) -> str:
    """Submit a factual correction so Nemesis can train Brain OS during Dream Mode.

    Use this when Brain OS gets a fact wrong — pricing, specs, customer info, etc.
    Valid categories: PRICING, SPECS, CUSTOMER, COMPETITOR, GENERAL.
    """
    from brain_os.brain.correction_store import CorrectionCategory, CorrectionStore

    try:
        cat = CorrectionCategory[category.upper()]
    except KeyError:
        cat = CorrectionCategory.GENERAL

    try:
        store = CorrectionStore()
        await store.initialize()
        row_id = await store.add_correction(
            entity=entity,
            new_value=correct_value,
            old_value=wrong_value,
            category=cat,
            source="cursor_mcp",
        )
        await store.close()
        return (
            f"Correction #{row_id} logged for entity '{entity}' "
            f"(category={cat.value}). Nemesis will process this during "
            "the next Dream Mode cycle."
        )
    except Exception as exc:
        logger.exception("MCP submit_correction failed")
        return f"Error: {exc}"


async def submit_praise(
    query: str,
    response: str,
    message: str = "well done",
    run_id: str = "",
) -> str:
    """Record operator praise and reinforce the method used for a turn (``brain praise``)."""
    import json

    try:
        from brain_os.interfaces.cursor_feedback import process_cursor_feedback

        result = await process_cursor_feedback(
            query=query,
            response=response,
            correction="",
            feedback=message,
            run_id=(run_id or "").strip() or None,
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP submit_praise failed")
        return f"Error: {exc}"


def register(mcp: FastMCP) -> None:
    """Register correction tools on the given FastMCP instance."""
    mcp.tool()(hardened_mcp_tool(submit_correction))
    mcp.tool()(hardened_mcp_tool(submit_praise))
