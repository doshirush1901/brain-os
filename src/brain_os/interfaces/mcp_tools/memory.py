"""Memory MCP tools.

Tools backed by Brain OS's memory subsystems:

- ``_long_term_memory`` — Mem0 semantic store (``recall_memory``, ``store_memory``)
- ``_conversation_memory`` — recent-history buffer (``get_conversation_history``)
- ``_relationship_memory`` — per-contact relationship profile (``check_relationship``)
- ``_goal_manager`` — active-goal tracker (``check_goals``)
- ``ProceduralMemory`` — learned routing (``list_procedures``, ``match_procedure``)
- ``MemoryBlockStore`` — pinned core memory (``read_memory_block``, ``update_memory_block``)

All handlers use the ``mcp_server`` lazy facade so test monkeypatches
(``mcp_mod._long_term_memory = mock`` etc.) keep working.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool

logger = logging.getLogger(__name__)


def _resolve_long_term_memory() -> Any:
    """Return the Mem0-backed store after MCP bootstrap (runtime + test facade)."""
    from brain_os.interfaces import mcp_runtime as rt
    from brain_os.interfaces import mcp_server as srv

    ltm = rt._long_term_memory or getattr(srv, "_long_term_memory", None)
    if ltm is not None:
        return ltm
    pipeline = rt._pipeline if rt._pipeline is not None else getattr(srv, "_pipeline", None)
    if pipeline is not None:
        return getattr(pipeline, "_long_term", None)
    return None


async def recall_memory(query: str, user_id: str = "global", limit: int = 5) -> str:
    """Search Brain OS's long-term semantic memory (Mem0).

    Returns relevant memories with content, score, and metadata.
    Use for recalling facts, preferences, and learned information.
    """
    from brain_os.interfaces import mcp_runtime as rt

    await rt._ensure_initialized()
    ltm = _resolve_long_term_memory()
    if ltm is None:
        return "Long-term memory not available."

    try:
        results = await ltm.search(query, user_id=user_id, limit=limit)
        return json.dumps(results, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP recall_memory failed")
        return f"Error: {exc}"


async def store_memory(content: str, user_id: str = "global", metadata: str = "") -> str:
    """Store a fact or learning in Brain OS's long-term memory.

    Provide the content to remember and an optional metadata JSON string
    (e.g. '{"source": "meeting", "topic": "pricing"}').
    """
    from brain_os.interfaces import mcp_runtime as rt

    await rt._ensure_initialized()
    ltm = _resolve_long_term_memory()
    if ltm is None:
        return "Long-term memory not available."

    try:
        meta = json.loads(metadata) if metadata else None
        meta_dict = dict(meta) if isinstance(meta, dict) else {}
        cat = str(meta_dict.get("memory_category") or "session")
        meta_dict.setdefault("memory_category", cat)
        gated = await ltm.store_gated(
            content,
            user_id=user_id,
            metadata=meta_dict,
            source="mcp:store_memory",
            category=cat,
        )
        return json.dumps(gated, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP store_memory failed")
        return f"Error: {exc}"


async def get_conversation_history(
    user_id: str,
    channel: str = "mcp",
    limit: int = 20,
) -> str:
    """Retrieve recent conversation history for a user.

    Returns the last N messages with role, content, and timestamp.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._conversation_memory is None:
        return "Conversation memory not available."

    try:
        history = await srv._conversation_memory.get_history(user_id, channel, limit=limit)
        return json.dumps(history, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP get_conversation_history failed")
        return f"Error: {exc}"


async def check_relationship(contact_name: str) -> str:
    """Look up Brain OS's relationship profile with a contact.

    Returns warmth level, interaction count, memorable moments,
    learned preferences, and interaction dates.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._relationship_memory is None:
        return "Relationship memory not available."

    try:
        rel = await srv._relationship_memory.get_relationship(contact_name)
        return json.dumps(srv._model_to_dict(rel), indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP check_relationship failed")
        return f"Error: {exc}"


async def list_procedures(
    agent: str = "",
    limit: int = 50,
    sort: str = "last_used",
) -> str:
    """List learned procedural routing patterns (read-only)."""
    from brain_os.memory.procedural import procedure_to_dict
    from brain_os.runtime.procedural_factory import build_procedural_memory

    sort_key = sort if sort in ("last_used", "score") else "last_used"
    pm = build_procedural_memory()
    await pm.initialize()
    try:
        rows = await pm.list_procedures(
            limit=max(1, min(limit, 500)),
            agent=agent.strip() or None,
            sort=sort_key,  # type: ignore[arg-type]
        )
        return json.dumps(
            {"count": len(rows), "procedures": [procedure_to_dict(p) for p in rows]},
            indent=2,
            default=str,
        )
    finally:
        await pm.close()


async def match_procedure(query: str) -> str:
    """Dry-run procedural routing for a query (same gates as pipeline step 4)."""
    from brain_os.memory.procedural import procedure_match_to_dict
    from brain_os.runtime.procedural_factory import build_procedural_memory

    pm = build_procedural_memory()
    await pm.initialize()
    try:
        explanation = await pm.explain_procedure_match(query)
        return json.dumps(procedure_match_to_dict(explanation), indent=2, default=str)
    finally:
        await pm.close()


async def read_memory_block(contact_id: str, label: str = "") -> str:
    """Read pinned core-memory blocks for a contact (Letta-style)."""
    from brain_os.interfaces import mcp_server as srv
    from brain_os.memory.blocks import BLOCK_SPECS, DEFAULT_BLOCK_LABELS
    from brain_os.service_keys import ServiceKey as SK

    await srv._ensure_initialized()
    store = srv._shared_services.get(SK.MEMORY_BLOCK_STORE)
    if store is None:
        return "Memory block store not available."

    scope = contact_id.strip().lower()
    wanted = (label.strip(),) if label.strip() else DEFAULT_BLOCK_LABELS
    if label.strip() and label.strip() not in BLOCK_SPECS:
        return f"Unknown label '{label}'. Valid: {', '.join(DEFAULT_BLOCK_LABELS)}"
    try:
        return await store.render_for_scope(scope, wanted)
    except Exception as exc:
        logger.exception("MCP read_memory_block failed")
        return f"Error: {exc}"


async def update_memory_block(contact_id: str, label: str, value: str) -> str:
    """Update a pinned core-memory block for a contact."""
    from brain_os.interfaces import mcp_server as srv
    from brain_os.memory.blocks import BLOCK_SPECS
    from brain_os.service_keys import ServiceKey as SK

    await srv._ensure_initialized()
    store = srv._shared_services.get(SK.MEMORY_BLOCK_STORE)
    if store is None:
        return "Memory block store not available."

    scope = contact_id.strip().lower()
    key = label.strip()
    if key not in BLOCK_SPECS:
        return f"Unknown label '{label}'. Valid: {', '.join(BLOCK_SPECS)}"
    try:
        block = await store.set_block(scope, key, value)
        return f"Updated {key} for {scope} ({len(block.value)} chars)."
    except Exception as exc:
        logger.exception("MCP update_memory_block failed")
        return f"Error: {exc}"


async def check_goals(contact_name: str) -> str:
    """Check active goals for a contact (e.g. quote follow-up, onboarding).

    Returns goal type, status, required slots, and progress.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._goal_manager is None:
        return "Goal manager not available."

    try:
        goal = await srv._goal_manager.get_active_goal(contact_name)
        if goal is None:
            return json.dumps({"contact": contact_name, "active_goal": None})
        return json.dumps(srv._model_to_dict(goal), indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP check_goals failed")
        return f"Error: {exc}"


def register(mcp: FastMCP) -> None:
    """Register memory tools on the given FastMCP instance."""
    mcp.tool()(hardened_mcp_tool(recall_memory))
    mcp.tool()(hardened_mcp_tool(store_memory))
    mcp.tool()(hardened_mcp_tool(get_conversation_history))
    mcp.tool()(hardened_mcp_tool(check_relationship))
    mcp.tool()(hardened_mcp_tool(check_goals))
    mcp.tool()(hardened_mcp_tool(list_procedures))
    mcp.tool()(hardened_mcp_tool(match_procedure))
    mcp.tool()(hardened_mcp_tool(read_memory_block))
    mcp.tool()(hardened_mcp_tool(update_memory_block))
