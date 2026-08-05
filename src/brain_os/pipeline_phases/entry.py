"""Phase: pipeline entry helpers (dedup / clarification / cheap early exits).

This module keeps high-churn orchestration blocks out of ``brain_os.pipeline`` while
preserving runtime behavior exactly.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from brain_os.exceptions import DatabaseError, BrainOSError, ToolExecutionError
from brain_os.pipeline_phases.compile import schedule_exit_run_record
from brain_os.pipeline_phases.error_handling import (
    complete_short_circuit_response,
    finalize_response_with_dedup_cache,
    record_dedup_hit_metadata,
    unpack_inproc_dedup_entry,
)
from brain_os.pipeline_phases.replan import (
    build_clarification_reask_payload,
    clarification_resume_answer_is_insufficient,
)
from brain_os.pipeline_runtime import (
    _decode_dedup_payload,
    _encode_dedup_payload,
    _format_pipeline_summary_md,
    _is_quick_pipeline_query,
    _should_bypass_cheap_exits,
)
from brain_os.services.degradation import record_degradation_event


def _graphe_short_circuit_log(
    pipeline: Any,
    *,
    raw_input: str,
    raw_response: str,
    agents_used: list[str],
    run_id: str,
    channel: str,
    route_method: str,
    contact_email: str,
    started_at: float,
    trace: dict[str, Any],
    early_exit: str | None = None,
    dedup_hit: str | None = None,
) -> dict[str, Any]:
    return {
        "pantheon": pipeline._pantheon,
        "query": raw_input,
        "agents_used": agents_used,
        "raw_response": raw_response,
        "run_id": run_id,
        "channel": channel,
        "route_method": route_method,
        "contact_email": contact_email,
        "started_at": started_at,
        "early_exit": early_exit or trace.get("early_exit"),
        "dedup_hit": dedup_hit,
        "email_scope": str(trace.get("email_scope") or "no_email"),
    }


@dataclass(frozen=True)
class DedupRuntimeState:
    now: float
    fingerprint: str
    redis_key: str
    bypass_cheap_exits: bool
    pending: dict[str, Any] | None


async def maybe_short_circuit_operator_deterministic(
    *,
    pipeline: Any,
    resolved_input: str,
    bypass_cheap_exits: bool,
    ambiguity_sensitive: bool,
    on_progress: Any | None,
    run_id: str,
    contact_email: str,
    channel: str,
    t0: float,
    raw_input: str,
    active_goal: str | None,
    dedup_fingerprint: str,
    dedup_now: float,
    redis_dedup_key: str,
    push_timings_fn: Any,
    record_stage_fn: Any,
    logger: Any,
    meta: dict[str, Any],
    trace: dict[str, Any],
    ts_start: float,
    sender_id: str,
) -> tuple[str, list[str], str] | None:
    """Stable import-time wrapper for operator deterministic short-circuit."""
    return await _maybe_short_circuit_operator_deterministic_impl(
        pipeline=pipeline,
        resolved_input=resolved_input,
        bypass_cheap_exits=bypass_cheap_exits,
        ambiguity_sensitive=ambiguity_sensitive,
        on_progress=on_progress,
        run_id=run_id,
        contact_email=contact_email,
        channel=channel,
        t0=t0,
        raw_input=raw_input,
        active_goal=active_goal,
        dedup_fingerprint=dedup_fingerprint,
        dedup_now=dedup_now,
        redis_dedup_key=redis_dedup_key,
        push_timings_fn=push_timings_fn,
        record_stage_fn=record_stage_fn,
        logger=logger,
        meta=meta,
        trace=trace,
        ts_start=ts_start,
        sender_id=sender_id,
    )


async def prepare_dedup_runtime_state(
    *,
    pipeline: Any,
    raw_input: str,
    sender_id: str,
    meta: dict[str, Any],
    app_meta: Any,
) -> DedupRuntimeState:
    """Build dedup fingerprint state and load pending clarification state."""
    from brain_os.services.socratic_gate import extract_gate_id

    gate_id = str(meta.get("gate_id") or "").strip() or extract_gate_id(raw_input)
    pending = await pipeline._pop_clarification(sender_id, gate_id=gate_id)
    now = time.monotonic()
    uncensored = int(bool(pipeline._uncensored_local_llm_active(meta)))
    fingerprint = hashlib.sha256(
        f"{sender_id}:{raw_input}:uncensored={uncensored}".encode()
    ).hexdigest()[:16]
    redis_key = f"{sender_id}:{fingerprint}"
    bypass_cheap_exits = _should_bypass_cheap_exits(
        raw_input,
        enabled=app_meta.cheap_exit_bypass_enabled,
        bypass_keywords=app_meta.cheap_exit_bypass_keywords,
        allowlist_keywords=app_meta.cheap_exit_allowlist_keywords,
    )
    return DedupRuntimeState(
        now=now,
        fingerprint=fingerprint,
        redis_key=redis_key,
        bypass_cheap_exits=bypass_cheap_exits,
        pending=pending,
    )


async def maybe_short_circuit_dedup_hit(
    *,
    pipeline: Any,
    state: DedupRuntimeState,
    sender_id: str,
    meta: dict[str, Any],
    trace: dict[str, Any],
    run_id: str,
    channel: str,
    ts_start: float,
    raw_input: str,
    logger: Any,
) -> tuple[str, list[str], str] | None:
    """Return cached response on Redis/in-proc dedup hit, or ``None``."""
    pending = state.pending
    if (
        not state.bypass_cheap_exits
        and pending is None
        and pipeline._redis is not None
        and pipeline._redis.available
    ):
        redis_hit = await pipeline._redis.dedup_check(state.redis_key)
        decoded = _decode_dedup_payload(redis_hit)
        if decoded is not None:
            cached_shaped, cached_agents, cached_rid = decoded
            logger.info(
                "DEDUP | Redis cache hit for %s (original_run_id=%s, agents=%s)",
                sender_id,
                cached_rid or "<legacy>",
                cached_agents or [],
            )
            record_dedup_hit_metadata(
                meta=meta,
                trace=trace,
                hit="redis",
                original_run_id=cached_rid,
                original_agents_used=list(cached_agents),
            )
            pipeline._attach_pipeline_trace(meta, trace)
            schedule_exit_run_record(
                meta=meta,
                run_id=run_id,
                channel=channel,
                sender_id=sender_id,
                ts_start=ts_start,
                trace=trace,
                agents_used=list(cached_agents),
                raw_input=raw_input,
                response_text=cached_shaped,
            )
            from brain_os.brain.graphe_instrumentation import log_graphe_pipeline_turn

            await log_graphe_pipeline_turn(
                pipeline._pantheon,
                query=raw_input,
                agents_used=list(cached_agents),
                raw_response=cached_shaped,
                run_id=run_id,
                channel=channel,
                route_method="dedup_redis",
                contact_email=sender_id,
                pipeline_ms=0.0,
                dedup_hit="redis",
                early_exit=trace.get("early_exit"),
            )
            return cached_shaped, list(cached_agents), run_id

    async with pipeline._state_lock:
        # Tuple shape is (shaped, ts, agents_used, run_id); v[1] is timestamp.
        pipeline._recent_messages = {
            k: v for k, v in pipeline._recent_messages.items() if state.now - v[1] < 300
        }
        if (
            not state.bypass_cheap_exits
            and pending is None
            and state.fingerprint in pipeline._recent_messages
        ):
            entry = pipeline._recent_messages[state.fingerprint]
            cached_resp, cached_agents_inproc, cached_rid_inproc = unpack_inproc_dedup_entry(entry)
            logger.info(
                "DEDUP | returning cached response for %s (original_run_id=%s, agents=%s)",
                sender_id,
                cached_rid_inproc or "<legacy>",
                cached_agents_inproc or [],
            )
            record_dedup_hit_metadata(
                meta=meta,
                trace=trace,
                hit="inproc",
                original_run_id=cached_rid_inproc,
                original_agents_used=cached_agents_inproc,
            )
            pipeline._attach_pipeline_trace(meta, trace)
            schedule_exit_run_record(
                meta=meta,
                run_id=run_id,
                channel=channel,
                sender_id=sender_id,
                ts_start=ts_start,
                trace=trace,
                agents_used=list(cached_agents_inproc),
                raw_input=raw_input,
                response_text=cached_resp,
            )
            from brain_os.brain.graphe_instrumentation import log_graphe_pipeline_turn

            await log_graphe_pipeline_turn(
                pipeline._pantheon,
                query=raw_input,
                agents_used=list(cached_agents_inproc),
                raw_response=cached_resp,
                run_id=run_id,
                channel=channel,
                route_method="dedup_inproc",
                contact_email=sender_id,
                pipeline_ms=0.0,
                dedup_hit="inproc",
                early_exit=trace.get("early_exit"),
            )
            return cached_resp, list(cached_agents_inproc), run_id

    return None


async def maybe_resume_pending_clarification(
    *,
    pipeline: Any,
    pending: dict[str, Any] | None,
    raw_input: str,
    sender_id: str,
    channel: str,
    trace: dict[str, Any],
    meta: dict[str, Any],
    run_id: str,
    ts_start: float,
    logger: Any,
) -> tuple[str, list[str], str] | None:
    """Handle clarification re-ask / resume short-circuits before PERCEIVE."""
    if pending is None:
        return None

    pending_question = str(pending.get("clarification_question") or "").strip()
    answer = (raw_input or "").strip()
    agent_name = str(pending.get("agent_name") or "sphinx")
    if clarification_resume_answer_is_insufficient(answer, pending_question):
        clarification_payload = build_clarification_reask_payload(
            pending_question=pending_question,
            pending=pending,
        )
        clarification_q = await pipeline._store_clarification(
            sender_id=sender_id,
            agent_name=agent_name,
            original_query=str(pending.get("original_query") or raw_input),
            payload=clarification_payload,
        )
        shaped = await pipeline._voice.shape_response(clarification_q, channel)
        trace["early_exit"] = "clarification_reask"
        trace["agents"] = [agent_name]
        pipeline._attach_pipeline_trace(meta, trace)
        schedule_exit_run_record(
            meta=meta,
            run_id=run_id,
            channel=channel,
            sender_id=sender_id,
            ts_start=ts_start,
            trace=trace,
            agents_used=[agent_name],
            raw_input=raw_input,
            response_text=shaped,
        )
        return shaped, [agent_name], run_id

    agent = pipeline._pantheon.get_agent(pending["agent_name"])
    if agent is None:
        return None
    followup_ctx = {
        "original_query": pending["original_query"],
        "clarification_answer": raw_input,
        "channel": channel,
    }
    try:
        raw_response = await agent.handle(
            pending["original_query"],
            followup_ctx,
        )
        logger.info("CLARIFY-RESUME | agent=%s", pending["agent_name"])
        shaped = await pipeline._voice.shape_response(
            raw_response,
            channel,
        )
        trace["early_exit"] = "clarification_resume"
        trace["agents"] = [pending["agent_name"]]
        pipeline._attach_pipeline_trace(meta, trace)
        schedule_exit_run_record(
            meta=meta,
            run_id=run_id,
            channel=channel,
            sender_id=sender_id,
            ts_start=ts_start,
            trace=trace,
            agents_used=[pending["agent_name"]],
            raw_input=raw_input,
            response_text=shaped,
        )
        return shaped, [pending["agent_name"]], run_id
    except (ToolExecutionError, Exception):
        logger.exception("Clarification resume failed")
        return None


async def maybe_short_circuit_fast_path(
    *,
    pipeline: Any,
    resolved_input: str,
    bypass_cheap_exits: bool,
    ambiguity_sensitive: bool,
    on_progress: Any | None,
    run_id: str,
    contact_email: str,
    channel: str,
    t0: float,
    raw_input: str,
    active_goal: str | None,
    dedup_fingerprint: str,
    dedup_now: float,
    redis_dedup_key: str,
    push_timings_fn: Any,
    record_stage_fn: Any,
    logger: Any,
    meta: dict[str, Any],
    trace: dict[str, Any],
    ts_start: float,
    sender_id: str,
) -> tuple[str, list[str], str] | None:
    """Run fast-path short-circuit and finalize with normal learn/dedup behavior."""
    from brain_os.brain.fast_path import classify as fp_classify
    from brain_os.brain.fast_path import generate as fp_generate

    fast_result = (
        fp_classify(resolved_input) if not bypass_cheap_exits and not ambiguity_sensitive else None
    )
    if fast_result is None or not fast_result.matched:
        return None
    if on_progress:
        await on_progress(
            {
                "type": "fast_path",
                "category": fast_result.category.value if fast_result.category else None,
            }
        )
    fp_response = fast_result.response
    if fp_response is None:
        fp_response = await fp_generate(
            resolved_input,
            fast_result.category,
            session_id=run_id,
            user_id=contact_email,
        )
    logger.info("FAST PATH | category=%s", fast_result.category)

    if on_progress:
        await on_progress({"type": "shaping"})
    shaped = await pipeline._voice.shape_response(fp_response, channel)
    fp_agents = ["fast_path"]
    trace["route"] = "fast_path"
    trace["agents"] = fp_agents
    schedule_exit_run_record(
        meta=meta,
        run_id=run_id,
        channel=channel,
        sender_id=sender_id,
        ts_start=ts_start,
        trace=trace,
        agents_used=fp_agents,
        raw_input=raw_input,
        response_text=shaped,
        tool_stats_tracker=getattr(pipeline, "_tool_stats_tracker", None),
    )
    return await complete_short_circuit_response(
        route_label="fast",
        contact_email=contact_email,
        started_at=t0,
        logger=logger,
        learn_fn=lambda: pipeline._learn(
            contact_email=contact_email,
            channel=channel,
            raw_input=raw_input,
            raw_response=fp_response,
            route_method="fast_path",
            agents_used=fp_agents,
            active_goal=active_goal,
            resolved_input=resolved_input,
            run_id=run_id,
        ),
        push_timings_fn=push_timings_fn,
        finalize_response_fn=lambda: finalize_response_with_dedup_cache(
            recent_messages=pipeline._recent_messages,
            fingerprint=dedup_fingerprint,
            now_epoch_s=dedup_now,
            shaped=shaped,
            agents_used=fp_agents,
            run_id=run_id,
            redis_cache=pipeline._redis,
            redis_key=redis_dedup_key,
            encode_dedup_payload=_encode_dedup_payload,
        ),
        record_stage_fn=record_stage_fn,
        graphe_log=_graphe_short_circuit_log(
            pipeline,
            raw_input=raw_input,
            raw_response=fp_response,
            agents_used=fp_agents,
            run_id=run_id,
            channel=channel,
            route_method="fast_path",
            contact_email=contact_email,
            started_at=t0,
            trace=trace,
        ),
    )


async def _maybe_short_circuit_operator_deterministic_impl(
    *,
    pipeline: Any,
    resolved_input: str,
    bypass_cheap_exits: bool,
    ambiguity_sensitive: bool,
    on_progress: Any | None,
    run_id: str,
    contact_email: str,
    channel: str,
    t0: float,
    raw_input: str,
    active_goal: str | None,
    dedup_fingerprint: str,
    dedup_now: float,
    redis_dedup_key: str,
    push_timings_fn: Any,
    record_stage_fn: Any,
    logger: Any,
    meta: dict[str, Any],
    trace: dict[str, Any],
    ts_start: float,
    sender_id: str,
) -> tuple[str, list[str], str] | None:
    """Operator desk / revenue / stale-leads queries without full pantheon routing."""
    if bypass_cheap_exits or ambiguity_sensitive:
        return None
    from brain_os.brain.operator_deterministic import match_operator_query, resolve_operator_route
    from brain_os.brain.routing_metrics import record_route_hit

    match = match_operator_query(resolved_input)
    if match is None:
        return None
    if on_progress:
        await on_progress({"type": "operator_deterministic", "kind": match.kind.value})
    try:
        op_raw = await resolve_operator_route(match.kind)
    except Exception as exc:
        logger.debug("operator deterministic route failed: %s", exc, exc_info=True)
        return None
    if not (op_raw or "").strip():
        return None
    record_route_hit("operator_deterministic")
    logger.info("OPERATOR DETERMINISTIC | kind=%s", match.kind.value)
    if on_progress:
        await on_progress({"type": "shaping"})
    shaped = await pipeline._voice.shape_response(op_raw, channel)
    op_agents = ["operator_deterministic"]
    trace["route"] = "operator_deterministic"
    trace["agents"] = op_agents
    schedule_exit_run_record(
        meta=meta,
        run_id=run_id,
        channel=channel,
        sender_id=sender_id,
        ts_start=ts_start,
        trace=trace,
        agents_used=op_agents,
        raw_input=raw_input,
        response_text=shaped,
        tool_stats_tracker=getattr(pipeline, "_tool_stats_tracker", None),
    )
    return await complete_short_circuit_response(
        route_label="operator",
        contact_email=contact_email,
        started_at=t0,
        logger=logger,
        learn_fn=lambda: pipeline._learn(
            contact_email=contact_email,
            channel=channel,
            raw_input=raw_input,
            raw_response=op_raw,
            route_method="operator_deterministic",
            agents_used=op_agents,
            active_goal=active_goal,
            resolved_input=resolved_input,
            run_id=run_id,
        ),
        push_timings_fn=push_timings_fn,
        finalize_response_fn=lambda: finalize_response_with_dedup_cache(
            recent_messages=pipeline._recent_messages,
            fingerprint=dedup_fingerprint,
            now_epoch_s=dedup_now,
            shaped=shaped,
            agents_used=op_agents,
            run_id=run_id,
            redis_cache=pipeline._redis,
            redis_key=redis_dedup_key,
            encode_dedup_payload=_encode_dedup_payload,
        ),
        record_stage_fn=record_stage_fn,
        graphe_log=_graphe_short_circuit_log(
            pipeline,
            raw_input=raw_input,
            raw_response=op_raw,
            agents_used=op_agents,
            run_id=run_id,
            channel=channel,
            route_method="operator_deterministic",
            contact_email=contact_email,
            started_at=t0,
            trace=trace,
        ),
    )


async def maybe_short_circuit_quick_pipeline(
    *,
    pipeline: Any,
    resolved_input: str,
    bypass_cheap_exits: bool,
    ambiguity_sensitive: bool,
    on_progress: Any | None,
    channel: str,
    contact_email: str,
    t0: float,
    raw_input: str,
    run_id: str,
    active_goal: str | None,
    dedup_fingerprint: str,
    dedup_now: float,
    redis_dedup_key: str,
    trace: dict[str, Any],
    push_timings_fn: Any,
    record_stage_fn: Any,
    logger: Any,
    meta: dict[str, Any],
    ts_start: float,
    sender_id: str,
) -> tuple[str, list[str], str] | None:
    """Run quick-pipeline short-circuit and finalize with learn + dedup."""
    if (
        bypass_cheap_exits
        or ambiguity_sensitive
        or pipeline._crm is None
        or not _is_quick_pipeline_query(resolved_input)
    ):
        return None
    try:
        if on_progress:
            await on_progress({"type": "quick_pipeline"})
        from brain_os.config import get_settings as qp_get_settings
        from brain_os.systems.crm_snapshot_cache import get_pipeline_summary_cached

        qp_ttl = float(qp_get_settings().app.crm_pipeline_cache_ttl_seconds)
        summary = await get_pipeline_summary_cached(
            pipeline._crm,
            filters=None,
            ttl_seconds=qp_ttl,
        )
        qp_raw = _format_pipeline_summary_md(summary)
        logger.info("QUICK PIPELINE | short-circuit")
        if on_progress:
            await on_progress({"type": "shaping"})
        qp_shaped = await pipeline._voice.shape_response(qp_raw, channel)
        qp_agents = ["quick_pipeline"]
        trace["route"] = "quick_pipeline"
        trace["agents"] = qp_agents
        schedule_exit_run_record(
            meta=meta,
            run_id=run_id,
            channel=channel,
            sender_id=sender_id,
            ts_start=ts_start,
            trace=trace,
            agents_used=qp_agents,
            raw_input=raw_input,
            response_text=qp_shaped,
            tool_stats_tracker=getattr(pipeline, "_tool_stats_tracker", None),
        )
        return await complete_short_circuit_response(
            route_label="quick_pipeline",
            contact_email=contact_email,
            started_at=t0,
            logger=logger,
            learn_fn=lambda: pipeline._learn(
                contact_email=contact_email,
                channel=channel,
                raw_input=raw_input,
                raw_response=qp_raw,
                route_method="quick_pipeline",
                agents_used=qp_agents,
                active_goal=active_goal,
                resolved_input=resolved_input,
                run_id=run_id,
            ),
            push_timings_fn=push_timings_fn,
            finalize_response_fn=lambda: finalize_response_with_dedup_cache(
                recent_messages=pipeline._recent_messages,
                fingerprint=dedup_fingerprint,
                now_epoch_s=dedup_now,
                shaped=qp_shaped,
                agents_used=qp_agents,
                run_id=run_id,
                redis_cache=pipeline._redis,
                redis_key=redis_dedup_key,
                encode_dedup_payload=_encode_dedup_payload,
            ),
            record_stage_fn=record_stage_fn,
            graphe_log=_graphe_short_circuit_log(
                pipeline,
                raw_input=raw_input,
                raw_response=qp_raw,
                agents_used=qp_agents,
                run_id=run_id,
                channel=channel,
                route_method="quick_pipeline",
                contact_email=contact_email,
                started_at=t0,
                trace=trace,
            ),
        )
    except (DatabaseError, BrainOSError, Exception):
        record_degradation_event(trace, layer="crm", code="quick_pipeline_read_failed")
        logger.debug("Quick pipeline failed (non-critical), continuing", exc_info=True)
        return None


__all__ = [
    "DedupRuntimeState",
    "maybe_resume_pending_clarification",
    "maybe_short_circuit_dedup_hit",
    "maybe_short_circuit_fast_path",
    "maybe_short_circuit_operator_deterministic",
    "maybe_short_circuit_quick_pipeline",
    "prepare_dedup_runtime_state",
]
