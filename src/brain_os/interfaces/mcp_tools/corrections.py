"""Correction submission MCP tool.

``submit_correction`` records a factual correction in the CorrectionStore
*and* writes Mnemon's correction ledger immediately so overrides beat
nostalgia without waiting for Dream. Self-contained: opens its own
CorrectionStore instance per call, no module-global state.
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

    Also writes Mnemon's ledger immediately and returns an OVERRIDE ACTIVE
    receipt. Valid categories: PRICING, SPECS, CUSTOMER, COMPETITOR, GENERAL.
    """
    from brain_os.agents.mnemon import apply_ledger_correction
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
    except Exception as exc:
        logger.exception("MCP submit_correction failed")
        return f"Error: {exc}"

    stale = [wrong_value.strip()] if (wrong_value or "").strip() else None
    try:
        ledger = await apply_ledger_correction(
            entity=entity,
            current_status=correct_value,
            stale_values=stale,
            source="cursor_mcp",
        )
        receipt = str(ledger.get("override_receipt") or "")
    except Exception as exc:
        logger.exception("MCP submit_correction ledger write failed")
        return (
            f"Correction #{row_id} logged for entity '{entity}' "
            f"(category={cat.value}), but Mnemon ledger write failed: {exc}"
        )

    return (
        f"Correction #{row_id} logged for entity '{entity}' "
        f"(category={cat.value}). Nemesis will reinforce in Dream Mode.\n"
        f"{receipt}"
    )


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
