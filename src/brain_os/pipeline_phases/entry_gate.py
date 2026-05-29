"""Entry / post-remember graph gates (transition table → short-circuits)."""

from __future__ import annotations

from typing import Any

from brain_os.pipeline_checkpoint import (
    checkpoint_supports_pipeline_resume,
    parse_clarification_checkpoint,
    stage_clarification_pipeline_resume,
)
from brain_os.pipeline_phases.entry import (
    DedupRuntimeState,
    maybe_resume_pending_clarification,
    maybe_short_circuit_dedup_hit,
    maybe_short_circuit_fast_path,
    maybe_short_circuit_quick_pipeline,
)
from brain_os.pipeline_phases.replan import (
    build_clarification_reask_payload,
    clarification_resume_answer_is_insufficient,
)
from brain_os.pipeline_runtime import _decode_dedup_payload
from brain_os.pipeline_transitions import (
    DedupHitKind,
    PipelineNode,
    attach_transition_to_trace,
    decide_entry_pre_perceive_from_turn,
    decide_post_remember_next,
    entry_outcome_matches_decision,
)


def _clarification_resume_agent_available(pipeline: Any, pending: dict[str, Any] | None) -> bool:
    if pending is None:
        return False
    return pipeline._pantheon.get_agent(str(pending.get("agent_name") or "")) is not None


async def probe_dedup_hit(*, pipeline: Any, state: DedupRuntimeState) -> DedupHitKind:
    """Non-mutating dedup probe for the entry transition table."""
    if state.bypass_cheap_exits or state.pending is not None:
        return "none"
    if pipeline._redis is not None and pipeline._redis.available:
        redis_hit = await pipeline._redis.dedup_check(state.redis_key)
        if _decode_dedup_payload(redis_hit) is not None:
            return "redis"

    async with pipeline._state_lock:
        pipeline._recent_messages = {
            k: v for k, v in pipeline._recent_messages.items() if state.now - v[1] < 300
        }
        if state.fingerprint in pipeline._recent_messages:
            return "inproc"

    return "none"


async def run_entry_pre_perceive_gate(
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
    """Entry graph: probe dedup, decide transition, run matching short-circuit."""
    dedup_hit = await probe_dedup_hit(pipeline=pipeline, state=state)
    decision = decide_entry_pre_perceive_from_turn(
        bypass_cheap_exits=state.bypass_cheap_exits,
        pending_clarification=state.pending,
        dedup_hit=dedup_hit,
        raw_input=raw_input,
        clarification_resume_agent_available=_clarification_resume_agent_available(
            pipeline,
            state.pending,
        ),
    )
    attach_transition_to_trace(trace, decision, phase="entry")

    if decision.next_node == PipelineNode.ENTRY_DEDUP:
        out = await maybe_short_circuit_dedup_hit(
            pipeline=pipeline,
            state=state,
            sender_id=sender_id,
            meta=meta,
            trace=trace,
            run_id=run_id,
            channel=channel,
            ts_start=ts_start,
            raw_input=raw_input,
            logger=logger,
        )
        if not entry_outcome_matches_decision(decision, out):
            logger.warning(
                "ENTRY transition parity mismatch | decision=%s result=%s",
                decision.next_node.value,
                "set" if out else "none",
            )
        return out

    if decision.next_node in (
        PipelineNode.ENTRY_CLARIFY_REASK,
        PipelineNode.ENTRY_CLARIFY_RESUME,
        PipelineNode.ENTRY_CLARIFY_PIPELINE_RESUME,
    ):
        out = await maybe_resume_pending_clarification_with_checkpoint(
            pipeline=pipeline,
            pending=state.pending,
            raw_input=raw_input,
            sender_id=sender_id,
            channel=channel,
            trace=trace,
            meta=meta,
            run_id=run_id,
            ts_start=ts_start,
            logger=logger,
        )
        if not entry_outcome_matches_decision(decision, out):
            logger.warning(
                "ENTRY transition parity mismatch | decision=%s result=%s",
                decision.next_node.value,
                "set" if out else "none",
            )
        return out

    return None


async def run_post_remember_cheap_exit_gate(
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
    active_goal: Any,
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
    """Post-REMEMBER graph: record transition, then fast-path or quick-pipeline only."""
    from brain_os.brain.fast_path import classify as fp_classify

    fast_matched = False
    if not bypass_cheap_exits and not ambiguity_sensitive:
        fast_result = fp_classify(resolved_input)
        fast_matched = fast_result is not None and fast_result.matched

    decision = decide_post_remember_next(
        bypass_cheap_exits=bypass_cheap_exits,
        ambiguity_sensitive=ambiguity_sensitive,
        fast_path_matched=fast_matched,
        crm_available=pipeline._crm is not None,
        resolved_input=resolved_input,
    )
    attach_transition_to_trace(trace, decision, phase="post_remember")

    common = dict(
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
    if decision.next_node == PipelineNode.FAST_PATH:
        return await maybe_short_circuit_fast_path(**common)
    if decision.next_node == PipelineNode.QUICK_PIPELINE:
        return await maybe_short_circuit_quick_pipeline(**common)
    return None


async def maybe_resume_pending_clarification_with_checkpoint(
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
    """Wrapper that stages pipeline resume when checkpoint has contact_email."""
    if pending is None:
        return None

    pending_question = str(pending.get("clarification_question") or "").strip()
    answer = (raw_input or "").strip()
    agent_name = str(pending.get("agent_name") or "sphinx")
    checkpoint = parse_clarification_checkpoint(pending)

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
            checkpoint=checkpoint,
        )
        shaped = await pipeline._voice.shape_response(clarification_q, channel)
        trace["early_exit"] = "clarification_reask"
        trace["agents"] = [agent_name]
        pipeline._attach_pipeline_trace(meta, trace)
        from brain_os.pipeline_phases.compile import schedule_exit_run_record

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

    if checkpoint_supports_pipeline_resume(checkpoint):
        stage_clarification_pipeline_resume(
            meta=meta,
            trace=trace,
            pending=pending,
            checkpoint=checkpoint,
            raw_input=raw_input,
            run_id=run_id,
            channel=channel,
        )
        logger.info(
            "CLARIFY-RESUME | pipeline continue | checkpoint_run=%s",
            checkpoint.get("run_id"),
        )
        return None

    return await maybe_resume_pending_clarification(
        pipeline=pipeline,
        pending=pending,
        raw_input=raw_input,
        sender_id=sender_id,
        channel=channel,
        trace=trace,
        meta=meta,
        run_id=run_id,
        ts_start=ts_start,
        logger=logger,
    )


__all__ = [
    "maybe_resume_pending_clarification_with_checkpoint",
    "probe_dedup_hit",
    "run_entry_pre_perceive_gate",
    "run_post_remember_cheap_exit_gate",
]
