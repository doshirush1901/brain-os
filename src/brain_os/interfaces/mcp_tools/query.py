"""Query / knowledge MCP tools.

Four retrieval / pipeline-driven query tools:

- ``query_ira``                — full 17-step pipeline (Perceive → Remember →
  Route → Enrich → Execute → Compliance → DLP → Corrections → Gaps →
  Faithfulness → Assess → Reflect → Shape → Learn → Return). Highest-traffic
  MCP tool. Supports progressive tool discovery + fastlane.
- ``discover_tools_for_query`` — preflight tool-shortlist helper
- ``search_knowledge``         — direct retriever search (no synthesis)
- ``quick_answer``             — retrieval-only fastlane response

All four front ``srv.*`` lazy-facade attributes on ``mcp_server``. ``query_ira``
additionally reads ``is_low_confidence_response`` from ``brain_os.brain.tool_discovery``
at module top — preserved as a top-level import (called inside the
low-confidence retry branch only).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from brain_os.brain.tool_discovery import is_low_confidence_response
from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool

logger = logging.getLogger(__name__)


async def query_ira(question: str, user_id: str | None = None) -> str:
    """Ask Ira a question about Machinecraft.

    Routes through the full 17-step pipeline (see AGENTS.md): perceive,
    remember, route, enrich, execute, compliance, DLP, corrections, gaps,
    faithfulness, assess, reflect, shape, learn, and return.
    Ira will delegate to the appropriate specialist agents automatically.

    ``user_id`` scopes conversation history and Mem0 (same as API ``user_id``).
    When omitted or blank, uses ``APP__DEFAULT_USER_ID`` from the host ``.env``
    (same as ``ira ask``); if that is unset, falls back to ``mcp_user``. Pass
    ``user_id="mcp_user"`` explicitly to keep a shared MCP-only namespace.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._pipeline is None:
        return "Ira pipeline not available."

    try:
        from brain_os.config import get_settings

        app = get_settings().app
        progressive = bool(app.progressive_tool_discovery)
        top_k = int(app.progressive_tool_discovery_top_k)
        expand_on_low_conf = bool(app.progressive_tool_discovery_expand_on_low_confidence)
        telemetry = bool(app.tool_selection_telemetry)
        fastlane_enabled = bool(app.mcp_fastlane_enabled)
        fastlane_min_score = float(app.mcp_fastlane_min_score)
        fastlane_max_snippets = int(app.mcp_fastlane_max_snippets)

        shortlisted_tools: list[str] = []
        expanded_tools: list[str] = []
        effective_question = question
        quick_attempted = False
        quick_confident = False
        quick_meta: dict[str, Any] = {}
        mcp_runtime_hints = {
            "route_prefer_cloud_fast": bool(app.mcp_route_prefer_cloud_fast),
            "ollama_retry_budget": int(app.mcp_ollama_retry_budget),
            "fast_fail_to_openai": bool(app.mcp_fast_fail_to_openai),
            "mcp_fastlane_enabled": fastlane_enabled,
        }
        if fastlane_enabled and srv._is_short_factual_query(question):
            quick_attempted = True
            quick_text, quick_confident, quick_meta = await srv._quick_answer_core(
                question,
                max_snippets=fastlane_max_snippets,
                min_score=fastlane_min_score,
            )
            if quick_confident:
                return (
                    f"{quick_text}\n\n[Path: quick_answer]\n[Quick meta: {json.dumps(quick_meta)}]"
                )

        if progressive:
            ranked = srv._rank_tools_for_query(question)
            shortlisted_tools = ranked[:top_k]
            effective_question = (
                f"{question}\n\n[mcp_tool_shortlist: {', '.join(shortlisted_tools)}]"
            )
            if telemetry:
                logger.info(
                    "mcp_tool_discovery shortlist selected=%d total=%d top_k=%d tools=%s",
                    len(shortlisted_tools),
                    len(ranked),
                    top_k,
                    ",".join(shortlisted_tools),
                )

        sender_id = srv._mcp_query_sender_id(user_id)
        response, agents_used, _ = await srv._pipeline.process_request(
            raw_input=effective_question,
            channel="mcp",
            sender_id=sender_id,
            metadata={"mcp_runtime_hints": mcp_runtime_hints},
        )

        if progressive and expand_on_low_conf and is_low_confidence_response(response):
            expanded_tools = srv._rank_tools_for_query(question)[: min(top_k * 2, 50)]
            if telemetry:
                logger.info(
                    "mcp_tool_discovery expansion retry selected=%d tools=%s",
                    len(expanded_tools),
                    ",".join(expanded_tools),
                )
            retry_question = (
                f"{question}\n\n[mcp_tool_shortlist_expanded: {', '.join(expanded_tools)}]"
            )
            response, agents_used, _ = await srv._pipeline.process_request(
                raw_input=retry_question,
                channel="mcp",
                sender_id=sender_id,
                metadata={"mcp_runtime_hints": mcp_runtime_hints},
            )

        suffix = ""
        if agents_used:
            suffix = f"\n\n[Agents consulted: {', '.join(agents_used)}]"
        if progressive and telemetry:
            discovery_meta = {
                "shortlisted_count": len(shortlisted_tools),
                "expanded_count": len(expanded_tools),
                "expanded": bool(expanded_tools),
            }
            suffix += f"\n[Tool discovery: {json.dumps(discovery_meta)}]"
        if telemetry:
            suffix += (
                "\n[MCP runtime: "
                + json.dumps(
                    {
                        "quick_attempted": quick_attempted,
                        "quick_confident": quick_confident,
                        "quick_meta": quick_meta,
                        "runtime_hints": mcp_runtime_hints,
                    }
                )
                + "]"
            )
        return response + suffix
    except Exception as exc:
        logger.exception("MCP query_ira failed")
        return f"Error: {exc}"


async def discover_tools_for_query(query: str, top_k: int = 8) -> str:
    """Return a ranked MCP tool shortlist for a query (progressive discovery helper).

    Useful for clients that want to preflight tool selection before issuing
    a full `query_ira` request.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    try:
        from brain_os.config import get_settings

        app = get_settings().app
        k = max(1, min(int(top_k), 50))
        ranked = srv._rank_tools_for_query(query)
        selected = ranked[:k]
        return json.dumps(
            {
                "query": query,
                "selected_tools": selected,
                "selected_count": len(selected),
                "total_tools": len(ranked),
                "progressive_tool_discovery_enabled": bool(app.progressive_tool_discovery),
            },
            indent=2,
            default=str,
        )
    except Exception as exc:
        logger.exception("MCP discover_tools_for_query failed")
        return f"Error: {exc}"


async def search_knowledge(query: str, limit: int = 10) -> str:
    """Search Ira's knowledge base across Qdrant, Neo4j, and Mem0.

    Returns the top results with content, scores, and source metadata.
    Use this for direct knowledge retrieval without agent reasoning.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._retriever is None:
        return "Retriever not available."

    try:
        results = await srv._retriever.search(query, limit=limit)
        formatted = []
        for r in results[:limit]:
            formatted.append(
                {
                    "content": r.get("content", "")[:500],
                    "score": round(r.get("score", 0), 3),
                    "source": r.get("source", ""),
                    "source_type": r.get("source_type", ""),
                }
            )
        return json.dumps(formatted, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP search_knowledge failed")
        return f"Error: {exc}"


async def quick_answer(
    question: str,
    max_snippets: int = 3,
    min_score: float = 0.45,
) -> str:
    """Fast retrieval-only answer for short factual prompts.

    Uses retriever search + compact formatting without full pipeline synthesis.
    """
    from brain_os.interfaces import mcp_server as srv

    text, _ok, _meta = await srv._quick_answer_core(
        question,
        max_snippets=max(1, min(int(max_snippets), 10)),
        min_score=max(0.0, min(float(min_score), 1.0)),
    )
    return text


def register(mcp: FastMCP) -> None:
    """Register query tools on the given FastMCP instance."""
    mcp.tool()(hardened_mcp_tool(query_ira))
    mcp.tool()(hardened_mcp_tool(discover_tools_for_query))
    mcp.tool()(hardened_mcp_tool(search_knowledge))
    mcp.tool()(hardened_mcp_tool(quick_answer))
