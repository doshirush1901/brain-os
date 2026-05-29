"""Phase: plan. Extracted from pipeline.py 2026-05-15.

Imports shared infrastructure ONLY from brain_os.pipeline_runtime — never
from brain_os.pipeline (circular).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from brain_os.data.models import PipelineContextModel
from brain_os.exceptions import ToolExecutionError
from brain_os.services.degradation import record_degradation_event


async def maybe_router_embedding_tiebreak(
    router: Any,
    qdrant: Any,
    query: str,
    routing: dict[str, Any] | None,
    scoreboard: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Optional embedding tie-break for deterministic routing (plan / route)."""
    if routing is None:
        return None, None
    from brain_os.config import get_settings as _grt

    a = _grt().app
    if not a.router_embedding_tiebreak_enabled:
        return routing, None
    margin = float(scoreboard.get("margin") or 0.0)
    if margin > float(a.router_embedding_tiebreak_max_margin):
        return routing, {"skipped": "high_regex_margin"}

    embeddings = getattr(qdrant, "_embeddings", None)
    if embeddings is None:
        return routing, {"skipped": "no_embedding_service"}

    from brain_os.brain.deterministic_router import IntentCategory
    from brain_os.brain.router_embedding_tiebreak import pick_intent_via_embedding

    current = str(routing.get("intent") or "")
    override, emb_detail = await pick_intent_via_embedding(
        query,
        current_intent=current,
        embeddings=embeddings,
    )
    if emb_detail.get("override") and isinstance(emb_detail["override"], str):
        try:
            new_int = IntentCategory(emb_detail["override"])
            return router.get_routing(new_int), emb_detail
        except ValueError:
            return routing, emb_detail
    return routing, emb_detail


async def resolve_route_preamble(
    *,
    router: Any,
    resolved_input: str,
    bypass_cheap_exits: bool,
    procedural_memory: Any | None,
    maybe_router_embedding_tiebreak_fn: Callable[
        [str, dict[str, Any] | None, dict[str, Any]],
        Awaitable[tuple[dict[str, Any] | None, dict[str, Any] | None]],
    ],
    on_progress: Any | None,
    logger: Any,
) -> dict[str, Any]:
    """Resolve route method + agent/tool lists before 5.x execution/enrichment."""
    if on_progress:
        await on_progress({"type": "routing", "method": "checking"})

    routing_scoreboard_snap = router.routing_scoreboard(resolved_input)
    routing = router.route(resolved_input)
    routing, emb_tie_detail = await maybe_router_embedding_tiebreak_fn(
        resolved_input,
        routing,
        routing_scoreboard_snap,
    )

    route_method: str | None = None
    agent_names: list[str] = []
    optional_agent_names: list[str] = []
    required_tools: list[str] = []
    optional_requested_snap: list[str] = []
    truth_hint_response: str | None = None

    if routing is not None:
        route_method = "deterministic"
        agent_names = routing["required_agents"]
        optional_agent_names = list(routing.get("optional_agents") or [])
        optional_requested_snap = list(optional_agent_names)
        required_tools = list(routing.get("required_tools") or [])
        logger.info("ROUTE FAST | intent=%s -> %s", routing["intent"], agent_names)

    if route_method is None and not bypass_cheap_exits:
        try:
            from brain_os.brain.truth_hints import TruthHintsEngine

            engine = TruthHintsEngine()
            await engine._load()
            if not engine.is_complex_query(resolved_input):
                hint = engine.match(resolved_input)
                if hint is not None:
                    truth_hint_response = hint["answer"]
                    route_method = "truth_hint"
                    agent_names = []
                    logger.info("TRUTH HINT | matched: %s", hint.get("patterns", ["?"])[0][:60])
        except Exception:
            logger.debug("Truth hints check failed (non-critical)")

    if route_method is None and procedural_memory is not None:
        try:
            procedure = await procedural_memory.find_procedure(resolved_input)
        except Exception:
            logger.exception("ProceduralMemory lookup failed")
            procedure = None

        if procedure is not None:
            route_method = "procedural"
            agent_names = procedure.steps
            optional_agent_names = []
            required_tools = []
            logger.info(
                "ROUTE PROCEDURE | pattern=%s (used %dx)",
                procedure.trigger_pattern,
                procedure.times_used,
            )

    if route_method is None:
        route_method = "llm"
        logger.info("ROUTE LLM | delegating to Athena")

    return {
        "route_method": route_method,
        "agent_names": agent_names,
        "optional_agent_names": optional_agent_names,
        "required_tools": required_tools,
        "optional_requested_snap": optional_requested_snap,
        "truth_hint_response": truth_hint_response,
        "routing_scoreboard_snap": routing_scoreboard_snap,
        "emb_tie_detail": emb_tie_detail,
    }


async def resolve_route_with_trace(
    *,
    router: Any,
    resolved_input: str,
    bypass_cheap_exits: bool,
    procedural_memory: Any | None,
    maybe_router_embedding_tiebreak_fn: Callable[
        [str, dict[str, Any] | None, dict[str, Any]],
        Awaitable[tuple[dict[str, Any] | None, dict[str, Any] | None]],
    ],
    on_progress: Any | None,
    trace: dict[str, Any],
    record_route_stage_fn: Callable[[], None],
    logger: Any,
) -> dict[str, Any]:
    """Resolve route preamble, attach routing telemetry, and record route stage."""
    route_preamble = await resolve_route_preamble(
        router=router,
        resolved_input=resolved_input,
        bypass_cheap_exits=bypass_cheap_exits,
        procedural_memory=procedural_memory,
        maybe_router_embedding_tiebreak_fn=maybe_router_embedding_tiebreak_fn,
        on_progress=on_progress,
        logger=logger,
    )
    trace["routing_telemetry"] = {
        "pattern_scoreboard": route_preamble["routing_scoreboard_snap"],
        "embedding_tiebreak": route_preamble["emb_tie_detail"],
    }
    record_route_stage_fn()
    return route_preamble


async def build_enrichment_parts(
    *,
    resolved_input: str,
    email_scope: str,
    contact_email: str,
    history_summary: str,
    require_thread_evidence: bool,
    lineage: Any,
    format_goal_lineage_enrichment_fn: Callable[[Any], str],
    adaptive_style: Any | None,
    realtime_observer: Any | None,
    endocrine: Any | None,
    power_level_tracker: Any | None,
    pantheon: Any,
    episodic: Any | None,
    crm: Any | None,
    long_term: Any | None,
    prefetch_outreach_thread_evidence_fn: Callable[
        [str], Awaitable[tuple[str, list[dict[str, Any]]]]
    ],
    logger: Any,
    memory_blocks: Any | None = None,
    relationship_memory: Any | None = None,
    prefetched_memory_blocks_xml: str = "",
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build enrichment text chunks and prefetched tool audit rows for execute context."""
    enrichment_parts: list[str] = []
    prefetched_tool_audit: list[dict[str, Any]] = []
    blocks_injected = False

    if lineage:
        glenr = format_goal_lineage_enrichment_fn(lineage)
        if glenr:
            enrichment_parts.append(glenr)

    try:
        if adaptive_style is not None:
            style_tracker = adaptive_style
        else:
            from brain_os.brain.adaptive_style import AdaptiveStyleTracker

            style_tracker = AdaptiveStyleTracker()
            await style_tracker._load()
        await style_tracker.update_profile(contact_email, resolved_input)
        style_prompt = style_tracker.get_style_prompt(contact_email)
        if style_prompt:
            enrichment_parts.append(style_prompt)
    except Exception:
        logger.debug("AdaptiveStyle not available")

    try:
        if realtime_observer is not None:
            observer = realtime_observer
        else:
            from brain_os.brain.realtime_observer import RealTimeObserver

            observer = RealTimeObserver()
            await observer._load()
        learnings_prompt = observer.format_for_prompt(contact_email)
        if learnings_prompt:
            enrichment_parts.append(learnings_prompt)
    except Exception:
        logger.debug("RealTimeObserver not available")

    if endocrine is not None:
        try:
            status = endocrine.get_status()
            enrichment_parts.append(
                f"System state: confidence={status.get('confidence', 0.5):.2f} "
                f"energy={status.get('energy', 0.5):.2f}"
            )
        except Exception:
            logger.debug("Endocrine status not available", exc_info=True)

    try:
        if power_level_tracker is not None:
            tracker = power_level_tracker
        else:
            from brain_os.brain.power_levels import PowerLevelTracker

            tracker = PowerLevelTracker()
            await tracker._load()
        top = tracker.get_leaderboard()[:3]
        if top:
            enrichment_parts.append(
                "Top agents: " + ", ".join(f"{a['agent']}({a['tier']})" for a in top)
            )
    except Exception:
        logger.debug("PowerLevelTracker not available", exc_info=True)

    try:
        chiron = pantheon.get_agent("chiron")
        if chiron is not None and hasattr(chiron, "get_sales_guidance"):
            guidance = await chiron.get_sales_guidance()
            if guidance and len(guidance) > 20:
                enrichment_parts.append(f"Sales coaching:\n{guidance[:500]}")
    except (ToolExecutionError, Exception):
        logger.debug("Chiron sales guidance not available", exc_info=True)

    if history_summary:
        enrichment_parts.append(f"Earlier conversation summary:\n{history_summary}")

    prefetched_blocks = (prefetched_memory_blocks_xml or "").strip()
    if prefetched_blocks:
        enrichment_parts.append(prefetched_blocks)
        blocks_injected = True
    elif memory_blocks is not None and contact_email and "@" in contact_email:
        try:
            from brain_os.memory.block_sync import ensure_contact_blocks_rendered

            block_xml = await ensure_contact_blocks_rendered(
                memory_blocks,
                contact_email,
                relationship=relationship_memory,
                episodic=episodic,
                crm=crm,
            )
            if block_xml:
                enrichment_parts.append(block_xml)
                blocks_injected = True
        except Exception:
            logger.debug("Memory blocks enrichment not available", exc_info=True)

    if episodic is not None and not blocks_injected:
        try:
            episodes = await episodic.surface_relevant_episodes(
                resolved_input,
                contact_email,
            )
            if episodes:
                ep_text = "\n".join(f"- {e.get('narrative', '')[:200]}" for e in episodes[:3])
                enrichment_parts.append(f"Relevant past interactions:\n{ep_text}")
        except Exception:
            logger.debug("Episodic memory enrichment not available", exc_info=True)

    try:
        from brain_os.brain.analog_accounts import maybe_enrich_analog_accounts

        analog_blob = await maybe_enrich_analog_accounts(
            resolved_input=resolved_input,
            email_scope=email_scope,
            crm=crm,
            long_term=long_term,
            contact_email=contact_email,
        )
        if analog_blob:
            enrichment_parts.append(analog_blob)
    except Exception:
        logger.debug("Analog accounts enrichment not available", exc_info=True)

    if require_thread_evidence and email_scope in ("both", "live_email"):
        try:
            evidence_blob, audit_rows = await prefetch_outreach_thread_evidence_fn(resolved_input)
            if evidence_blob:
                enrichment_parts.append(evidence_blob)
            if audit_rows:
                prefetched_tool_audit.extend(audit_rows)
        except Exception:
            logger.warning("Outreach prefetch failed", exc_info=True)

    try:
        from brain_os.brain.honcho_user_model import fetch_enrichment_text

        honcho_blob = await fetch_enrichment_text(contact_email, resolved_input)
        if honcho_blob:
            enrichment_parts.append(f"User model (Honcho dialectic):\n{honcho_blob}")
    except Exception:
        logger.debug("Honcho user-model enrichment not available", exc_info=True)

    return enrichment_parts, prefetched_tool_audit


def resolve_email_scope_and_tool_discovery(
    *,
    pipeline: Any,
    resolved_input: str,
    trace: dict[str, Any],
    get_email_processor_fn: Callable[[Any], Any],
    logger: Any,
) -> tuple[str, bool, dict[str, Any]]:
    """Resolve email scope flags and progressive tool-discovery metadata."""
    email_scope = pipeline._resolve_email_scope(resolved_input)
    require_thread_evidence = pipeline._is_outreach_prioritization_query(resolved_input)
    tool_discovery_meta: dict[str, Any] = {}
    logger.info("EMAIL SCOPE | %s", email_scope)
    if email_scope in ("live_email", "both") and get_email_processor_fn(pipeline._pantheon) is None:
        record_degradation_event(
            trace,
            layer="live_mailbox",
            code="processor_unavailable",
            detail=email_scope,
        )

    try:
        from brain_os.brain.tool_discovery import (
            DEFAULT_MCP_TOOL_CANDIDATES,
            rank_mcp_tools,
        )
        from brain_os.config import get_settings as get_settings_tool_discovery

        td_cfg = get_settings_tool_discovery().app
        if td_cfg.progressive_tool_discovery:
            ranked_tools = rank_mcp_tools(resolved_input, DEFAULT_MCP_TOOL_CANDIDATES)
            shortlisted_tools = ranked_tools[: td_cfg.progressive_tool_discovery_top_k]
            tool_discovery_meta = {
                "shortlisted_tools": shortlisted_tools,
                "shortlisted_count": len(shortlisted_tools),
                "ranked_count": len(ranked_tools),
                "expand_on_low_confidence": td_cfg.progressive_tool_discovery_expand_on_low_confidence,
            }
            if td_cfg.tool_selection_telemetry:
                logger.info(
                    "TOOL DISCOVERY | shortlist=%d ranked=%d tools=%s",
                    len(shortlisted_tools),
                    len(ranked_tools),
                    ",".join(shortlisted_tools),
                )
    except Exception:
        logger.debug("Tool discovery metadata build failed", exc_info=True)
    return email_scope, require_thread_evidence, tool_discovery_meta


def build_execution_context(
    *,
    perception: Any,
    channel: str,
    conversation_memory: Any,
    relationship_memory: Any | None,
    goal_manager: Any | None,
    procedural_memory: Any | None,
    crm: Any | None,
    endocrine: Any | None,
    tool_stats_tracker: Any | None,
    enrichment_parts: list[str],
    contact_email: str,
    run_id: str,
    goal_lineage_payload: dict[str, Any] | None,
    governance_dict: dict[str, Any] | None,
    email_scope: str,
    prefetched_tool_audit: list[dict[str, Any]],
    tool_discovery_meta: dict[str, Any],
    agent_journal: Any | None,
    memory_block_store: Any | None = None,
) -> dict[str, Any]:
    """Build execute-context dict from pipeline services and enrichment blocks."""
    services: dict[str, Any] = {
        "_delegation_depth": 0,
        "conversation_memory": conversation_memory,
        "relationship_memory": relationship_memory,
        "goal_manager": goal_manager,
        "procedural_memory": procedural_memory,
        "crm": crm,
        "endocrine": endocrine,
    }
    if memory_block_store is not None:
        from brain_os.service_keys import ServiceKey as SK

        services[SK.MEMORY_BLOCK_STORE] = memory_block_store
    if tool_stats_tracker is not None:
        services["tool_stats_tracker"] = tool_stats_tracker

    ctx = PipelineContextModel(
        perception=perception,
        channel=channel,
        services=services,
        enrichment="\n\n".join(enrichment_parts) if enrichment_parts else "",
        mem0_user_id=contact_email,
        run_id=run_id,
        goal_lineage=goal_lineage_payload,
        governance=governance_dict,
    )
    context: dict[str, Any] = ctx.model_dump()
    context["email_scope"] = email_scope

    from brain_os.config import get_settings as _gs_req_snap

    if _gs_req_snap().app.request_prompt_snapshot:
        from brain_os.prompt_loader import snapshot_soul_preamble_from_disk

        context["_soul_preamble_snapshot"] = snapshot_soul_preamble_from_disk()
    context["_tool_audit"] = list(prefetched_tool_audit)
    if tool_discovery_meta:
        context["tool_discovery"] = tool_discovery_meta
    if agent_journal is not None:
        context["_agent_journal"] = agent_journal
    return context


def apply_llm_provider_overrides(
    *,
    context: dict[str, Any],
    resolved_input: str,
    route_method: str,
    channel: str,
    metadata: dict[str, Any],
    resolve_provider_fn: Callable[..., str | None],
    is_high_stakes_fn: Callable[[str], bool],
    uncensored_turn: bool,
    ollama_ready: bool,
    trace: dict[str, Any],
    logger: Any,
) -> None:
    """Apply provider override hints (cloud/local) to execute context."""
    llm_provider_override = resolve_provider_fn(
        resolved_input,
        route_method=route_method or "",
        channel=channel,
        metadata=metadata,
    )
    if llm_provider_override:
        context["llm_provider_override"] = llm_provider_override
        logger.info(
            "LLM ROUTE PROVIDER | method=%s provider=%s high_stakes=%s",
            route_method,
            llm_provider_override,
            is_high_stakes_fn(resolved_input),
        )
    if uncensored_turn and ollama_ready:
        context["llm_provider_override"] = "ollama"
        trace["uncensored_local_llm_mode"] = True
        logger.info("UNCENSORED LOCAL LLM | forcing llm_provider_override=ollama for this turn")


__all__ = [
    "apply_llm_provider_overrides",
    "build_enrichment_parts",
    "build_execution_context",
    "maybe_router_embedding_tiebreak",
    "resolve_email_scope_and_tool_discovery",
    "resolve_route_preamble",
    "resolve_route_with_trace",
]
