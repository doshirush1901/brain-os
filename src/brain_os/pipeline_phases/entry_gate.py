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
    pending = state.pending
    socratic_match: dict[str, Any] | None = None

    # Clarification resume guard: TTL for all pending; full match for Socratic.
    # Unrelated next messages abandon the gate (do NOT record as answers).
    if pending is not None:
        from brain_os.services.socratic_gate import (
            classify_socratic_resume,
            log_socratic_gate_event,
            pending_is_expired,
        )

        if pending.get("socratic"):
            llm = None
            try:
                sphinx = pipeline._pantheon.get_agent("sphinx")
                if sphinx is not None:
                    await sphinx._ensure_llm()
                    llm = getattr(sphinx, "_llm", None)
            except Exception:
                llm = None
            session_id = str(
                meta.get("session_id") or meta.get("conversation_id") or sender_id or ""
            )
            conversation_id = str(meta.get("conversation_id") or meta.get("session_id") or "")
            match = await classify_socratic_resume(
                raw_input,
                pending,
                llm_client=llm,
                session_id=session_id,
                conversation_id=conversation_id,
            )
            socratic_match = {
                "disposition": match.disposition,
                "reason": match.reason,
                "is_answer": match.is_answer,
                "mapped_answers": dict(match.mapped_answers),
                "partial_proceed": bool(match.partial_proceed),
                "open_slots": [s.context_key for s in match.slots if s.status == "open"],
                "filled_slots": [s.context_key for s in match.slots if s.status == "filled"],
            }
            trace["socratic_resume_match"] = socratic_match
            if match.disposition == "reask":
                # Partial fill — keep same gate_id, re-ask only open slots.
                from brain_os.schemas.llm_outputs import ClarificationPayload
                from brain_os.services.socratic_gate import (
                    format_slot_progress,
                    open_slots,
                    questions_from_slots,
                    slots_to_dicts,
                )

                still = open_slots(match.slots)
                gid = str(pending.get("gate_id") or "").strip().lower()
                reask_text = match.reask_rendered or format_slot_progress(match.slots, gate_id=gid)
                open_qs = questions_from_slots(still)
                open_defaults = {
                    q.context_key: q.proposed_default
                    for q in open_qs
                    if q.context_key and q.proposed_default
                }
                payload = ClarificationPayload(
                    needs_clarification=True,
                    questions=reask_text.splitlines()
                    if reask_text
                    else [
                        "IRA NEEDS INPUT (0 questions)",
                        "",
                        "Reply 'go' to proceed with all defaults.",
                    ],
                    reason=f"Socratic reask ({match.reason})",
                    missing_slots=[s.kind or s.context_key for s in still],
                    can_answer_partially=True,
                )
                # Preserve original query + created_at; refresh slots/questions.
                extra = {
                    "socratic": True,
                    "gate_id": gid,
                    "socratic_defaults": {
                        **dict(pending.get("socratic_defaults") or {}),
                        **dict(match.mapped_answers),
                        **open_defaults,
                    },
                    "socratic_questions": [q.model_dump(mode="json") for q in open_qs],
                    "socratic_slots": slots_to_dicts(match.slots),
                    "socratic_score": pending.get("socratic_score") or {},
                    "session_id": pending.get("session_id") or "",
                    "conversation_id": pending.get("conversation_id") or "",
                    "channel": channel,
                    "sender_id": sender_id,
                    "created_at": pending.get("created_at"),
                    "status": "pending",
                }
                # Record answers received so far (filled only).
                try:
                    from brain_os.services.socratic_gate import record_socratic_resume_answers

                    record_socratic_resume_answers(
                        pending=pending,
                        answer=raw_input,
                        source="pipeline_reask",
                        mapped_answers=dict(match.mapped_answers),
                    )
                except Exception:
                    logger.debug("socratic reask answer record failed", exc_info=True)

                clarification_q = await pipeline._store_clarification(
                    sender_id=sender_id,
                    agent_name="sphinx",
                    original_query=str(pending.get("original_query") or raw_input),
                    payload=payload,
                    extra=extra,
                )
                log_socratic_gate_event(
                    {
                        "status": "reask",
                        "reason": match.reason,
                        "sender_id": sender_id,
                        "channel": channel,
                        "gate_id": gid,
                        "filled_slots": socratic_match["filled_slots"],
                        "open_slots": socratic_match["open_slots"],
                    }
                )
                shaped = clarification_q or reask_text
                trace["early_exit"] = "sphinx_socratic_reask"
                trace["agents"] = ["sphinx"]
                trace["socratic_needs_input"] = True
                if gid:
                    trace["gate_id"] = gid
                    meta["gate_id"] = gid
                return shaped, ["sphinx"], run_id
            if match.disposition == "insufficient":
                shaped = str(pending.get("clarification_question") or "").strip()
                if not shaped:
                    from brain_os.services.socratic_gate import format_slot_progress, slots_from_pending

                    shaped = format_slot_progress(
                        slots_from_pending(pending),
                        gate_id=str(pending.get("gate_id") or ""),
                    )
                from brain_os.schemas.llm_outputs import ClarificationPayload

                await pipeline._store_clarification(
                    sender_id=sender_id,
                    agent_name=str(pending.get("agent_name") or "sphinx"),
                    original_query=str(pending.get("original_query") or raw_input),
                    payload=ClarificationPayload(
                        needs_clarification=True,
                        questions=shaped.splitlines(),
                        reason="Socratic insufficient answer",
                        missing_slots=list(pending.get("missing_slots") or []),
                        can_answer_partially=True,
                    ),
                    extra={
                        k: pending[k]
                        for k in (
                            "socratic",
                            "gate_id",
                            "socratic_defaults",
                            "socratic_questions",
                            "socratic_slots",
                            "socratic_score",
                            "session_id",
                            "conversation_id",
                            "channel",
                            "sender_id",
                            "created_at",
                            "status",
                        )
                        if k in pending
                    },
                )
                log_socratic_gate_event(
                    {
                        "status": "insufficient",
                        "reason": match.reason,
                        "sender_id": sender_id,
                        "channel": channel,
                        "gate_id": pending.get("gate_id"),
                    }
                )
                trace["early_exit"] = "sphinx_socratic_insufficient"
                trace["agents"] = ["sphinx"]
                trace["socratic_needs_input"] = True
                return shaped, ["sphinx"], run_id
            if match.disposition in ("abandoned", "expired"):
                log_socratic_gate_event(
                    {
                        "status": match.disposition,
                        "reason": match.reason,
                        "sender_id": sender_id,
                        "channel": channel,
                        "gate_id": pending.get("gate_id"),
                        "original_query": pending.get("original_query"),
                        "inbound": (raw_input or "")[:500],
                        "pending_created_at": pending.get("created_at"),
                    }
                )
                logger.info(
                    "SOCRATIC %s | reason=%s — continuing with new request (no answer recorded)",
                    match.disposition.upper(),
                    match.reason,
                )
                pending = None
            elif match.disposition == "answer":
                meta["_socratic_mapped_answers"] = dict(match.mapped_answers)
                if match.slots:
                    meta["_socratic_slots"] = [s.to_dict() for s in match.slots]
                if match.partial_proceed:
                    meta["_socratic_partial_proceed"] = True
                    trace["socratic_partial_proceed"] = True
        elif pending_is_expired(pending):
            log_socratic_gate_event(
                {
                    "status": "expired",
                    "reason": "ttl_exceeded",
                    "sender_id": sender_id,
                    "channel": channel,
                    "original_query": pending.get("original_query"),
                    "inbound": (raw_input or "")[:500],
                    "pending_created_at": pending.get("created_at"),
                    "socratic": False,
                }
            )
            logger.info("CLARIFY EXPIRED | continuing with new request")
            pending = None

    decision = decide_entry_pre_perceive_from_turn(
        bypass_cheap_exits=state.bypass_cheap_exits,
        pending_clarification=pending,
        dedup_hit=dedup_hit,
        raw_input=raw_input,
        clarification_resume_agent_available=_clarification_resume_agent_available(
            pipeline,
            pending,
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

    if pending.get("socratic"):
        try:
            from brain_os.services.socratic_gate import record_socratic_resume_answers

            procedural = getattr(pipeline, "_procedural", None)
            mapped = meta.get("_socratic_mapped_answers")
            mapped_dict = mapped if isinstance(mapped, dict) else None
            promos = record_socratic_resume_answers(
                pending=pending,
                answer=answer,
                source="pipeline",
                procedural_memory=procedural,
                mapped_answers=mapped_dict,
            )
            if promos:
                trace["socratic_promotions"] = promos
        except Exception:
            logger.debug("Socratic answer logging failed", exc_info=True)

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
