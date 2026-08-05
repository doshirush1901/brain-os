"""Query / knowledge MCP tools.

Four retrieval / pipeline-driven query tools:

- ``query_brain``                — full 17-step pipeline (Perceive → Remember →
  Route → Enrich → Execute → Compliance → DLP → Corrections → Gaps →
  Faithfulness → Assess → Reflect → Shape → Learn → Return). Highest-traffic
  MCP tool. Supports progressive tool discovery + fastlane.
- ``discover_tools_for_query`` — preflight tool-shortlist helper
- ``recommend_mcp_profile``  — which Cursor MCP fleet key to enable/use
- ``search_knowledge``         — direct retriever search (no synthesis)
- ``quick_answer``             — retrieval-only fastlane response
- ``socratic_gate_status``     — peek pending Sphinx gate by gate_id / user
- ``socratic_gate_resume``     — structured resume (``go`` or answers_json)

All six front ``srv.*`` lazy-facade attributes on ``mcp_server``. ``query_brain``
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


async def query_brain(
    question: str,
    user_id: str | None = None,
    ask_first: bool = False,
) -> str:
    """Ask Brain OS a question about Acme Corp.

    Routes through the full 17-step pipeline (see AGENTS.md): perceive,
    remember, route, enrich, execute, compliance, DLP, corrections, gaps,
    faithfulness, assess, reflect, shape, learn, and return.
    Brain OS will delegate to the appropriate specialist agents automatically.

    ``user_id`` scopes conversation history and Mem0 (same as API ``user_id``).
    When omitted or blank, uses ``APP__DEFAULT_USER_ID`` from the host ``.env``
    (same as ``brain ask``); if that is unset, falls back to ``mcp_user``. Pass
    ``user_id="mcp_user"`` explicitly to keep a shared MCP-only namespace.

    ``ask_first=True`` forces Sphinx v2 Socratic counter-questions when any
    ambiguity is detected (high-stakes decisions with proposed defaults),
    before agent execution. Reply ``go`` on the next turn to accept defaults.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._pipeline is None:
        return "Brain OS pipeline not available."

    try:
        from brain_os.config import get_settings

        app = get_settings().app
        progressive = bool(app.progressive_tool_discovery)
        top_k = int(app.progressive_tool_discovery_top_k)
        expand_on_low_conf = bool(app.progressive_tool_discovery_expand_on_low_confidence)
        telemetry = bool(app.tool_selection_telemetry)
        fastlane_enabled = bool(app.mcp_fastlane_enabled) and not ask_first
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
        meta: dict[str, Any] = {
            "mcp_runtime_hints": mcp_runtime_hints,
            "ask_first": bool(ask_first),
            "socratic_ask_first": bool(ask_first),
            "session_id": sender_id,
            "conversation_id": sender_id,
        }
        response, agents_used, _ = await srv._pipeline.process_request(
            raw_input=effective_question,
            channel="mcp",
            sender_id=sender_id,
            metadata=meta,
        )

        # Ensure Socratic gate text is the primary tool-result content in Cursor.
        trace = meta.get("pipeline_trace") if isinstance(meta.get("pipeline_trace"), dict) else {}
        if (
            isinstance(trace, dict)
            and trace.get("early_exit") == "sphinx_socratic"
            and "IRA NEEDS INPUT" not in (response or "")[:80]
        ):
            # Defensive: if something stripped the header, re-prefix from agents path.
            response = f"IRA NEEDS INPUT\n\n{response}"

        if progressive and expand_on_low_conf and is_low_confidence_response(response):
            # Never expand/retry when Sphinx is waiting for operator answers.
            if not (isinstance(trace, dict) and trace.get("early_exit") == "sphinx_socratic"):
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

        # Keep Socratic questions as the primary body; skip noisy suffixes.
        if isinstance(trace, dict) and trace.get("early_exit") == "sphinx_socratic":
            return response

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
        logger.exception("MCP query_brain failed")
        return f"Error: {exc}"


async def discover_tools_for_query(query: str, top_k: int = 8) -> str:
    """Return a ranked MCP tool shortlist for a query (progressive discovery helper).

    Useful for clients that want to preflight tool selection before issuing
    a full `query_brain` request.
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


async def recommend_mcp_profile(
    query: str,
    enabled_keys: str | None = None,
) -> str:
    """Recommend which Cursor MCP fleet key to use for a job (deterministic).

    Cursor cannot hot-swap MCP servers mid-chat. Returns primary key, confidence,
    next_action (``use_tools_if_enabled`` vs ``ask_operator_to_enable``), and notes.

    ``enabled_keys`` — optional comma-separated Cursor keys already in mcp.json
    (e.g. ``ira-sales,ira-leads``). When set, ``next_action`` reflects whether the
    primary key is already enabled.
    """
    try:
        from brain_os.interfaces.mcp_profile_router import recommend_mcp_profile as _route

        keys: list[str] | None = None
        raw = (enabled_keys or "").strip()
        if raw:
            keys = [part.strip() for part in raw.split(",") if part.strip()]
        result = _route(query, enabled_keys=keys)
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP recommend_mcp_profile failed")
        return f"Error: {exc}"


async def search_knowledge(
    query: str,
    limit: int = 10,
    company: str | None = None,
) -> str:
    """Search Brain OS's knowledge base across Qdrant, Neo4j, and Mem0.

    Returns the top results with content, scores, and source metadata.
    Use this for direct knowledge retrieval without agent reasoning.
    Optional *company* filters Qdrant via ``metadata.graph_entity_ids``.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._retriever is None:
        return "Retriever not available."

    try:
        results = await srv._retriever.search(query, limit=limit, company=company)
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


async def socratic_gate_status(
    gate_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """Show pending Sphinx Socratic gate state for this MCP user (or a gate_id).

    Returns JSON: gate_id, questions, defaults, slots, created_at, expired.
    """
    from brain_os.interfaces import mcp_server as srv
    from brain_os.services.socratic_gate import pending_is_expired

    await srv._ensure_initialized()
    if srv._pipeline is None:
        return json.dumps({"error": "pipeline_unavailable"})

    sender_id = srv._mcp_query_sender_id(user_id)
    pending = await srv._pipeline._get_clarification(
        sender_id=None if gate_id else sender_id,
        gate_id=gate_id,
    )
    if not pending:
        return json.dumps(
            {"pending": False, "sender_id": sender_id, "gate_id": gate_id},
            indent=2,
        )
    expired = pending_is_expired(pending)
    slots = pending.get("socratic_slots") or []
    questions = pending.get("socratic_questions") or pending.get("questions") or []
    from brain_os.services.socratic_gate import (
        build_socratic_cards,
        open_slots,
        questions_from_slots,
        slots_from_pending,
    )

    open_qs = questions_from_slots(open_slots(slots_from_pending(pending)))
    cards = build_socratic_cards(open_qs if open_qs else questions)
    return json.dumps(
        {
            "pending": True,
            "expired": expired,
            "gate_id": pending.get("gate_id"),
            "sender_id": pending.get("sender_id") or sender_id,
            "original_query": pending.get("original_query"),
            "created_at": pending.get("created_at"),
            "questions": questions,
            "defaults": pending.get("socratic_defaults") or {},
            "slots": slots,
            "cards": cards,
            "slots_open": [
                s.get("context_key")
                for s in slots
                if isinstance(s, dict) and s.get("status") == "open"
            ],
            "slots_filled": [
                s.get("context_key")
                for s in slots
                if isinstance(s, dict) and s.get("status") == "filled"
            ],
            "clarification_question": pending.get("clarification_question"),
        },
        indent=2,
        default=str,
    )


async def socratic_gate_resume(
    gate_id: str,
    go: bool = False,
    answers_json: str = "",
    user_id: str | None = None,
) -> str:
    """Resume a pending Socratic gate by ``gate_id`` with structured answers.

    Prefer this over free-text follow-ups when Cursor has a ``gate_id``.
    ``go=True`` accepts all proposed defaults. Otherwise pass
    ``answers_json`` as ``{"context_key": "value", ...}``.
    """
    from brain_os.interfaces import mcp_server as srv
    from brain_os.services.socratic_gate import (
        build_resume_answer_text,
        classify_socratic_resume,
        log_socratic_gate_event,
        pending_is_expired,
    )

    await srv._ensure_initialized()
    if srv._pipeline is None:
        return "Brain OS pipeline not available."

    gid = (gate_id or "").strip().lower()
    if not gid:
        return json.dumps({"error": "gate_id_required"})

    sender_id = srv._mcp_query_sender_id(user_id)
    pending = await srv._pipeline._get_clarification(gate_id=gid)
    if pending is None:
        return json.dumps({"error": "gate_not_found", "gate_id": gid}, indent=2)
    if pending_is_expired(pending):
        await srv._pipeline._pop_clarification(sender_id, gate_id=gid)
        log_socratic_gate_event(
            {"status": "expired", "reason": "ttl_exceeded", "gate_id": gid, "source": "mcp"}
        )
        return json.dumps({"error": "gate_expired", "gate_id": gid}, indent=2)

    answers: dict[str, str] = {}
    if answers_json and answers_json.strip():
        try:
            parsed = json.loads(answers_json)
            if isinstance(parsed, dict):
                answers = {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            return json.dumps({"error": "answers_json_invalid", "gate_id": gid}, indent=2)

    answer_text = build_resume_answer_text(go=go or not answers, answers=answers, gate_id=gid)
    match = await classify_socratic_resume(
        answer_text,
        pending,
        llm_client=None,
        session_id=str(pending.get("session_id") or sender_id),
        conversation_id=str(pending.get("conversation_id") or sender_id),
    )
    if match.disposition != "answer":
        from brain_os.services.socratic_gate import (
            build_socratic_cards,
            open_slots,
            questions_from_slots,
        )

        cards = []
        if match.disposition == "reask" and match.slots:
            cards = build_socratic_cards(questions_from_slots(open_slots(match.slots)))
        return json.dumps(
            {
                "error": "not_an_answer" if match.disposition != "reask" else None,
                "ok": match.disposition == "reask",
                "disposition": match.disposition,
                "reason": match.reason,
                "gate_id": gid,
                "mapped_answers": match.mapped_answers,
                "reask": match.reask_rendered,
                "cards": cards,
            },
            indent=2,
            default=str,
        )

    meta: dict[str, Any] = {
        "session_id": str(pending.get("session_id") or sender_id),
        "conversation_id": str(pending.get("conversation_id") or sender_id),
        "gate_id": gid,
        "_socratic_mapped_answers": dict(match.mapped_answers),
    }
    owner = str(pending.get("sender_id") or sender_id)

    response, agents_used, run_id = await srv._pipeline.process_request(
        raw_input=answer_text,
        channel="mcp",
        sender_id=owner,
        metadata=meta,
    )
    return json.dumps(
        {
            "ok": True,
            "gate_id": gid,
            "run_id": run_id,
            "agents": agents_used,
            "mapped_answers": match.mapped_answers,
            "response": response,
        },
        indent=2,
        default=str,
    )


def register(mcp: FastMCP) -> None:
    """Register query tools on the given FastMCP instance."""
    mcp.tool()(hardened_mcp_tool(query_brain))
    mcp.tool()(hardened_mcp_tool(discover_tools_for_query))
    mcp.tool()(hardened_mcp_tool(recommend_mcp_profile))
    mcp.tool()(hardened_mcp_tool(search_knowledge))
    mcp.tool()(hardened_mcp_tool(quick_answer))
    mcp.tool()(hardened_mcp_tool(socratic_gate_status))
    mcp.tool()(hardened_mcp_tool(socratic_gate_resume))
