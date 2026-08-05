"""Inner 17-step pipeline orchestration (Phase 3 extraction).

See ``docs/PIPELINE_SPLIT_PLAN.md``. ``RequestPipeline._process_request_inner`` delegates here.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from brain_os.brain.goal_lineage import format_goal_lineage_enrichment, parse_goal_lineage_from_metadata
from brain_os.brain.learning_telemetry import build_learning_meta
from brain_os.brain.retrieval_context import mem0_user_id_var
from brain_os.pipeline_phases.compile import (
    append_post_shape_artifacts,
    apply_metis_stability_note,
    apply_source_limitation_note,
    run_reflection_step,
    shape_pipeline_response,
)
from brain_os.pipeline_phases.entry import (
    maybe_short_circuit_operator_deterministic,
    prepare_dedup_runtime_state,
)
from brain_os.pipeline_phases.entry_gate import (
    run_entry_pre_perceive_gate,
    run_post_remember_cheap_exit_gate,
)
from brain_os.pipeline_phases.error_handling import (
    finalize_pipeline_scope,
    finalize_response_with_dedup_cache,
    start_background_task_with_lifecycle,
)
from brain_os.pipeline_phases.execute import apply_execute_trace_telemetry, run_execute_dispatch
from brain_os.pipeline_phases.outreach import (
    extract_json_payload,
    get_email_processor,
    prefetch_gmail_thread_by_id,
    prefetch_outreach_thread_evidence,
    render_outreach_ranking,
)
from brain_os.pipeline_phases.plan import (
    apply_llm_provider_overrides,
    build_enrichment_parts,
    build_execution_context,
    resolve_email_scope_and_tool_discovery,
    resolve_route_with_trace,
)
from brain_os.pipeline_phases.replan import maybe_finalize_clarification_turn, maybe_run_sphinx_gate
from brain_os.pipeline_phases.validate import (
    run_assessment_step,
    run_faithfulness_gate,
    run_guardrails_checks,
    run_validate_safety_chain,
)
from brain_os.pipeline_request_bootstrap import attach_pipeline_run_id, new_pipeline_trace_dict
from brain_os.pipeline_runtime import (
    _encode_dedup_payload,
    _faithfulness_strict_intent,
)

logger = logging.getLogger(__name__)


async def run_process_request_inner(
    pipeline: Any,
    raw_input: str,
    channel: str,
    sender_id: str,
    metadata: dict[str, Any] | None = None,
    on_progress: Any | None = None,
) -> tuple[str, list[str], str]:
    """Run the full 17-step pipeline and return ``(shaped_response, agents_used, run_id)``."""
    from brain_os.pipeline import RequestPipeline

    t0 = time.monotonic()
    ts_wall_start = time.time()
    t_last = t0
    run_stages: dict[str, float] = {}
    meta, run_id = attach_pipeline_run_id(metadata)
    strict_slack_workflow = bool(meta.get("slack_require_full_workflow"))
    meta["_run_record_ts_start"] = ts_wall_start
    meta["_run_record_raw_input"] = raw_input
    meta["channel"] = channel
    lineage = parse_goal_lineage_from_metadata(meta)
    governance_dict = meta["governance"] if isinstance(meta.get("governance"), dict) else None
    trace: dict[str, Any] = new_pipeline_trace_dict(
        channel=channel, sender_id=sender_id, run_id=run_id
    )
    _learning_had_provenance = False
    _learning_had_dlp = False

    from brain_os.config import get_settings as _gs_pipe_meta

    _app_meta = _gs_pipe_meta().app
    if _app_meta.llm_monthly_token_budget > 0:
        from brain_os.systems.llm_budget import check_budget_allows, resolve_budget_bucket

        _b_scope, _b_bucket = resolve_budget_bucket(
            sender_id,
            scope_mode=_app_meta.llm_budget_scope,
        )
        _budget_msg = await check_budget_allows(
            pipeline._redis,
            limit=_app_meta.llm_monthly_token_budget,
            scope=_b_scope,
            bucket=_b_bucket,
        )
        if _budget_msg:
            from brain_os.pipeline_transitions import (
                attach_transition_to_trace,
                decide_budget_gate,
            )

            _budget_decision = decide_budget_gate(budget_blocked=True)
            if _budget_decision is not None:
                attach_transition_to_trace(trace, _budget_decision, phase="budget")
            trace["early_exit"] = "budget_gate"
            trace["agents"] = ["budget"]
            pipeline._attach_pipeline_trace(meta, trace)
            from brain_os.pipeline_phases.compile import schedule_exit_run_record

            schedule_exit_run_record(
                meta=meta,
                run_id=run_id,
                channel=channel,
                sender_id=sender_id,
                ts_start=ts_wall_start,
                trace=trace,
                agents_used=["budget"],
                raw_input=raw_input,
                response_text=_budget_msg,
            )
            return _budget_msg, ["budget"], run_id

    def _record_stage(name: str) -> None:
        nonlocal t_last
        run_stages[name] = round(time.monotonic() - t_last, 3)
        t_last = time.monotonic()

    def _push_timings() -> None:
        if run_stages:
            pipeline._stage_timings.append({"stages": dict(run_stages), "timestamp": time.time()})

    # ── 0 / 0.5 ENTRY SHORT-CIRCUITS (clarification + dedup) ────────────
    _dedup_state = await prepare_dedup_runtime_state(
        pipeline=pipeline,
        raw_input=raw_input,
        sender_id=sender_id,
        meta=meta,
        app_meta=_app_meta,
    )
    _now = _dedup_state.now
    _fingerprint = _dedup_state.fingerprint
    _redis_dedup_key = _dedup_state.redis_key
    _bypass_cheap_exits_raw = _dedup_state.bypass_cheap_exits

    _entry_early = await run_entry_pre_perceive_gate(
        pipeline=pipeline,
        state=_dedup_state,
        sender_id=sender_id,
        meta=meta,
        trace=trace,
        run_id=run_id,
        channel=channel,
        ts_start=ts_wall_start,
        raw_input=raw_input,
        logger=logger,
    )
    if _entry_early is not None:
        return _entry_early

    from brain_os.pipeline_checkpoint import CLARIFICATION_PIPELINE_RESUME_KEY

    _clarification_pipeline_resume = meta.pop(CLARIFICATION_PIPELINE_RESUME_KEY, None)

    # ── 0.9 RESET BOREDOM (living system: any user interaction resets idle) ──
    if pipeline._endocrine is not None:
        try:
            pipeline._endocrine.reset_boredom()
        except Exception:
            logger.debug("reset_boredom failed", exc_info=True)

    _history_summary = ""
    active_goal = None
    perception: dict[str, Any]
    resolved_input = raw_input
    _bypass_cheap_exits = _bypass_cheap_exits_raw or strict_slack_workflow
    _ambiguity_sensitive = False

    if _clarification_pipeline_resume is not None:
        from brain_os.pipeline_phases.resume import build_checkpoint_resume_preamble
        from brain_os.pipeline_transitions import (
            PipelineNode,
            TransitionDecision,
            attach_transition_to_trace,
        )

        if on_progress:
            await on_progress({"type": "checkpoint_resume"})
        _resume_preamble = await build_checkpoint_resume_preamble(
            pipeline,
            resume=_clarification_pipeline_resume,
            raw_input=raw_input,
            channel=channel,
            sender_id=sender_id,
            meta=meta,
            app_meta=_app_meta,
            bypass_cheap_exits_raw=_bypass_cheap_exits_raw,
        )
        contact_email = _resume_preamble.contact_email
        perception = _resume_preamble.perception
        contact_info = perception["resolved_contact"]
        _history_summary = _resume_preamble.history_summary
        active_goal = _resume_preamble.active_goal
        resolved_input = _resume_preamble.resolved_input
        _bypass_cheap_exits = _resume_preamble.bypass_cheap_exits or strict_slack_workflow
        _ambiguity_sensitive = _resume_preamble.ambiguity_sensitive
        trace["contact"] = contact_email
        _record_stage("checkpoint_resume")
        attach_transition_to_trace(
            trace,
            TransitionDecision(
                next_node=PipelineNode.ROUTE,
                terminal="continue",
                early_exit="clarification_pipeline_resume",
            ),
            phase="clarification_pipeline_resume",
        )
        logger.info(
            "CHECKPOINT RESUME | %s | prior_run=%s",
            contact_email,
            _clarification_pipeline_resume.get("checkpoint_run_id"),
        )
    else:
        from brain_os.pipeline_phases.perceive import run_perceive_step
        from brain_os.pipeline_phases.remember import run_remember_step

        _perceive_out = await run_perceive_step(
            sensory=pipeline._sensory,
            channel=channel,
            raw_input=raw_input,
            sender_id=sender_id,
            meta=meta,
            record_stage_fn=_record_stage,
            on_progress=on_progress,
        )
        perception = _perceive_out.perception
        contact_info = _perceive_out.contact_info
        contact_email = _perceive_out.contact_email
        trace["contact"] = contact_email

        _remember_out = await run_remember_step(
            conversation=pipeline._conversation,
            unified_ctx=pipeline._unified_ctx,
            goals=pipeline._goals,
            contact_email=contact_email,
            channel=channel,
            raw_input=raw_input,
            bypass_cheap_exits_raw=_bypass_cheap_exits_raw,
            app_meta=_app_meta,
            strict_slack_workflow=strict_slack_workflow,
            record_stage_fn=_record_stage,
            on_progress=on_progress,
        )
        _history_summary = _remember_out.history_summary
        active_goal = _remember_out.active_goal
        resolved_input = _remember_out.resolved_input
        _bypass_cheap_exits = _remember_out.bypass_cheap_exits
        _ambiguity_sensitive = _remember_out.ambiguity_sensitive

    _mem0_scope_tok = mem0_user_id_var.set(contact_email)
    try:
        if _clarification_pipeline_resume is None:
            _cheap_early = await run_post_remember_cheap_exit_gate(
                pipeline=pipeline,
                resolved_input=resolved_input,
                bypass_cheap_exits=_bypass_cheap_exits,
                ambiguity_sensitive=_ambiguity_sensitive,
                on_progress=on_progress,
                run_id=run_id,
                contact_email=contact_email,
                channel=channel,
                t0=t0,
                raw_input=raw_input,
                active_goal=active_goal,
                dedup_fingerprint=_fingerprint,
                dedup_now=_now,
                redis_dedup_key=_redis_dedup_key,
                push_timings_fn=_push_timings,
                record_stage_fn=_record_stage,
                logger=logger,
                meta=meta,
                trace=trace,
                ts_start=ts_wall_start,
                sender_id=sender_id,
            )
            if _cheap_early is not None:
                return _cheap_early

            _operator_early = await maybe_short_circuit_operator_deterministic(
                pipeline=pipeline,
                resolved_input=resolved_input,
                bypass_cheap_exits=_bypass_cheap_exits,
                ambiguity_sensitive=_ambiguity_sensitive,
                on_progress=on_progress,
                run_id=run_id,
                contact_email=contact_email,
                channel=channel,
                t0=t0,
                raw_input=raw_input,
                active_goal=active_goal,
                dedup_fingerprint=_fingerprint,
                dedup_now=_now,
                redis_dedup_key=_redis_dedup_key,
                push_timings_fn=_push_timings,
                record_stage_fn=_record_stage,
                logger=logger,
                meta=meta,
                trace=trace,
                ts_start=ts_wall_start,
                sender_id=sender_id,
            )
            if _operator_early is not None:
                return _operator_early

            from brain_os.pipeline_checkpoint import build_clarification_checkpoint

            _sphinx_checkpoint = build_clarification_checkpoint(
                run_id=run_id,
                channel=channel,
                raw_input=raw_input,
                resolved_input=resolved_input,
                contact_email=contact_email,
                trace=trace,
            )
            _sphinx_early = await maybe_run_sphinx_gate(
                sphinx_agent=pipeline._pantheon.get_agent("sphinx"),
                resolved_input=resolved_input,
                channel=channel,
                sender_id=sender_id,
                contact_email=contact_email,
                run_id=run_id,
                on_progress=on_progress,
                store_clarification_fn=pipeline._store_clarification,
                clarification_checkpoint=_sphinx_checkpoint,
                shape_response_fn=pipeline._voice.shape_response,
                record_route_stage_fn=lambda: _record_stage("route"),
                push_timings_fn=_push_timings,
                trace=trace,
                meta=meta,
                ts_start=ts_wall_start,
                logger=logger,
                sphinx_timeout_s=15,
                pantheon=pipeline._pantheon,
                crm=pipeline._crm,
                knowledge_graph=pipeline._graph,
                email_processor=get_email_processor(pipeline._pantheon),
            )
            if _sphinx_early is not None:
                return _sphinx_early

        # ── 3/3.5/4/5 ROUTE PREAMBLE ─────────────────────────────────
        optional_selected_snap: list[str] = []
        _route_preamble = await resolve_route_with_trace(
            router=pipeline._router,
            resolved_input=resolved_input,
            raw_input=raw_input,
            bypass_cheap_exits=_bypass_cheap_exits,
            procedural_memory=pipeline._procedural,
            maybe_router_embedding_tiebreak_fn=pipeline._maybe_router_embedding_tiebreak,
            on_progress=on_progress,
            trace=trace,
            record_route_stage_fn=lambda: _record_stage("route"),
            logger=logger,
        )
        route_method: str = str(_route_preamble["route_method"])
        agent_names: list[str] = list(_route_preamble["agent_names"])
        optional_agent_names: list[str] = list(_route_preamble["optional_agent_names"])
        required_tools: list[str] = list(_route_preamble["required_tools"])
        optional_requested_snap: list[str] = list(_route_preamble["optional_requested_snap"])
        truth_hint_response: str | None = (
            str(_route_preamble["truth_hint_response"])
            if _route_preamble["truth_hint_response"] is not None
            else None
        )

        # ── 5.1 RESOLVE EMAIL SCOPE ───────────────────────────────
        email_scope, require_thread_evidence, tool_discovery_meta = (
            resolve_email_scope_and_tool_discovery(
                pipeline=pipeline,
                resolved_input=resolved_input,
                trace=trace,
                get_email_processor_fn=get_email_processor,
                logger=logger,
            )
        )
        _shape_gate_prefixes: list[str] = []
        if _route_preamble.get("pipeline_shape_task"):
            email_scope = "no_email"
            trace["pipeline_shape_task"] = True
            trace["fast_path"] = "pipeline_shape"
            from brain_os.pipeline_phases.pipeline_shape_gate import (
                run_pipeline_shape_triangulation_gate,
            )

            (
                _shape_blocked,
                _shape_gate_prefixes,
                _shape_tri,
            ) = await run_pipeline_shape_triangulation_gate(
                resolved_input=resolved_input,
                meta=meta,
                pantheon=pipeline._pantheon,
                email_processor=get_email_processor(pipeline._pantheon),
            )
            if _shape_tri is not None:
                trace["shape_triangulation_allowed"] = _shape_tri.allowed
                trace["shape_triangulation_gaps"] = list(_shape_tri.gap_keys)
            if _shape_blocked and _shape_gate_prefixes:
                _block_msg = _shape_gate_prefixes[0]
                trace["early_exit"] = "shape_triangulation_gate"
                trace["agents"] = ["triangulation"]
                pipeline._attach_pipeline_trace(meta, trace)
                from brain_os.pipeline_phases.compile import schedule_exit_run_record

                schedule_exit_run_record(
                    meta=meta,
                    run_id=run_id,
                    channel=channel,
                    sender_id=sender_id,
                    ts_start=ts_wall_start,
                    trace=trace,
                    agents_used=["triangulation"],
                    raw_input=raw_input,
                    response_text=_block_msg,
                )
                from brain_os.brain.graphe_instrumentation import log_graphe_pipeline_turn

                await log_graphe_pipeline_turn(
                    pipeline._pantheon,
                    query=raw_input,
                    agents_used=["triangulation"],
                    raw_response=_block_msg,
                    run_id=run_id,
                    channel=channel,
                    route_method=route_method,
                    contact_email=contact_email,
                    email_scope=email_scope,
                    started_at=t0,
                    early_exit="shape_triangulation_gate",
                )
                return _block_msg, ["triangulation"], run_id
        elif _route_preamble.get("gmail_thread_read_only"):
            email_scope = "live_email"
            trace["gmail_thread_read_only"] = True
            trace["gmail_thread_ids"] = list(_route_preamble.get("gmail_thread_ids") or [])
            trace["fast_path"] = "gmail_thread_read"
        prefetched_tool_audit: list[dict[str, Any]] = []

        # ── 5.5 ENRICH CONTEXT ─────────────────────────────────────
        if on_progress:
            await on_progress({"type": "enriching"})

        enrichment_parts, prefetched_tool_audit = await build_enrichment_parts(
            resolved_input=resolved_input,
            email_scope=email_scope,
            contact_email=contact_email,
            history_summary=_history_summary,
            require_thread_evidence=require_thread_evidence,
            lineage=lineage,
            format_goal_lineage_enrichment_fn=format_goal_lineage_enrichment,
            adaptive_style=pipeline._adaptive_style,
            realtime_observer=pipeline._realtime_observer,
            endocrine=pipeline._endocrine,
            power_level_tracker=pipeline._power_level_tracker,
            pantheon=pipeline._pantheon,
            episodic=pipeline._episodic,
            crm=pipeline._crm,
            long_term=pipeline._long_term,
            prefetch_outreach_thread_evidence_fn=lambda query: prefetch_outreach_thread_evidence(
                crm=pipeline._crm,
                email_processor=get_email_processor(pipeline._pantheon),
                query=query,
                thread_id_pattern=pipeline._THREAD_ID_PATTERN,
                logger=logger,
            ),
            prefetch_gmail_thread_by_id_fn=lambda ids: prefetch_gmail_thread_by_id(
                ids,
                email_processor=get_email_processor(pipeline._pantheon),
                crm=pipeline._crm,
            ),
            gmail_thread_ids=list(_route_preamble.get("gmail_thread_ids") or []),
            logger=logger,
            memory_blocks=pipeline._memory_blocks,
            relationship_memory=pipeline._relationship,
            prefetched_memory_blocks_xml=str(perception.get("memory_blocks") or ""),
        )

        if _clarification_pipeline_resume is not None:
            _clarification_note = (
                "Prior clarification turn — original question: "
                f"{_clarification_pipeline_resume.get('original_query', '')}\n"
                "User clarification answer: "
                f"{_clarification_pipeline_resume.get('clarification_answer', '')}"
            )
            enrichment_parts = [_clarification_note, *list(enrichment_parts)]

        if _shape_gate_prefixes:
            enrichment_parts = [*list(_shape_gate_prefixes), *list(enrichment_parts)]

        _record_stage("enrich")

        # ── 6. EXECUTE ───────────────────────────────────────────────
        # Pass perception and enrichment for prompt context, plus live
        # service references so agents can query memory dynamically
        # through their ReAct tools instead of relying on static snapshots.
        context: dict[str, Any] = build_execution_context(
            perception=perception,
            channel=channel,
            conversation_memory=pipeline._conversation,
            relationship_memory=pipeline._relationship,
            goal_manager=pipeline._goals,
            procedural_memory=pipeline._procedural,
            crm=pipeline._crm,
            endocrine=pipeline._endocrine,
            tool_stats_tracker=pipeline._tool_stats_tracker,
            enrichment_parts=enrichment_parts,
            contact_email=contact_email,
            run_id=run_id,
            goal_lineage_payload=lineage.model_dump() if lineage else None,
            governance_dict=governance_dict,
            email_scope=email_scope,
            prefetched_tool_audit=prefetched_tool_audit,
            tool_discovery_meta=tool_discovery_meta,
            agent_journal=pipeline._agent_journal,
            memory_block_store=pipeline._memory_blocks,
            gmail_thread_read_only=bool(_route_preamble.get("gmail_thread_read_only")),
            gmail_thread_ids=list(_route_preamble.get("gmail_thread_ids") or []),
        )
        if _route_preamble.get("route_intent"):
            context["route_intent"] = str(_route_preamble["route_intent"])
        context["route_method"] = route_method
        apply_llm_provider_overrides(
            context=context,
            resolved_input=resolved_input,
            route_method=route_method or "",
            channel=channel,
            metadata=meta,
            resolve_provider_fn=pipeline._resolve_llm_route_provider_override,
            is_high_stakes_fn=pipeline._is_high_stakes_llm_route_query,
            uncensored_turn=RequestPipeline._uncensored_local_llm_active(meta),
            ollama_ready=RequestPipeline._ollama_openai_compat_ready(),
            trace=trace,
            logger=logger,
        )

        raw_response: str
        agents_used: list[str]
        # Pristine classifier JSON for email gold eval (before Aletheia/Metis/voice rewrite).
        _email_gold_eval_json: str | None = None

        (
            raw_response,
            agents_used,
            route_method,
            optional_selected_snap,
            _email_gold_eval_json,
        ) = await run_execute_dispatch(
            pipeline=pipeline,
            require_thread_evidence=require_thread_evidence,
            route_method=route_method,
            resolved_input=resolved_input,
            context=context,
            on_progress=on_progress,
            truth_hint_response=truth_hint_response,
            agent_names=agent_names,
            optional_agent_names=optional_agent_names,
            required_tools=required_tools,
            logger=logger,
            extract_json_payload=extract_json_payload,
            render_outreach_ranking=render_outreach_ranking,
            thread_id_pattern=pipeline._THREAD_ID_PATTERN,
        )
        apply_execute_trace_telemetry(
            trace=trace,
            route_method=route_method,
            agents_used=agents_used,
            optional_requested_snap=optional_requested_snap,
            optional_selected_snap=optional_selected_snap,
            logger=logger,
            record_execute_stage_fn=lambda: _record_stage("execute"),
        )

        _safety = await run_validate_safety_chain(
            raw_response=raw_response,
            agents_used=agents_used,
            resolved_input=resolved_input,
            channel=channel,
            strict_full_workflow=strict_slack_workflow,
            trace=trace,
            pantheon=pipeline._pantheon,
            on_progress=on_progress,
            logger=logger,
            record_compliance_stage_fn=lambda: _record_stage("compliance"),
            record_dlp_stage_fn=lambda: _record_stage("dlp"),
        )
        raw_response = _safety.raw_response
        agents_used = _safety.agents_used
        _learning_had_provenance = _learning_had_provenance or _safety.had_provenance
        _learning_had_dlp = _learning_had_dlp or _safety.had_dlp

        # ── 6.4 FAITHFULNESS GATE ─────────────────────────────────────
        shared_kb_evidence: list[dict[str, Any]] = []
        raw_response, shared_kb_evidence = await run_faithfulness_gate(
            route_method=route_method,
            resolved_input=resolved_input,
            raw_response=raw_response,
            agents_used=agents_used,
            channel=channel,
            run_id=run_id,
            retriever=pipeline._pantheon.retriever,
            trace=trace,
            on_progress=on_progress,
            uncensored_turn=RequestPipeline._uncensored_local_llm_active(meta),
            faithfulness_strict_intent_fn=_faithfulness_strict_intent,
            logger=logger,
        )
        _record_stage("faithfulness")

        from brain_os.services.speculation_mode import record_speculation_shadow_for_turn

        record_speculation_shadow_for_turn(
            metadata=meta,
            resolved_input=resolved_input,
            raw_response=raw_response,
            kb_evidence=shared_kb_evidence,
            trace=trace,
            run_id=run_id,
            channel=channel,
        )

        # ── 6.4b GUARDRAILS (all routed responses) ───────────────────
        raw_response = await run_guardrails_checks(
            route_method=route_method,
            raw_response=raw_response,
            trace=trace,
            uncensored_turn=RequestPipeline._uncensored_local_llm_active(meta),
            logger=logger,
        )

        # ── 6.5 CLARIFICATION CHECK ──────────────────────────────────
        from brain_os.pipeline_checkpoint import build_clarification_checkpoint

        _clarify_checkpoint = build_clarification_checkpoint(
            run_id=run_id,
            channel=channel,
            raw_input=raw_input,
            resolved_input=resolved_input,
            contact_email=contact_email,
            route_method=route_method,
            agents_used=agents_used,
            trace=trace,
        )
        _clarify_result = await maybe_finalize_clarification_turn(
            raw_response=raw_response,
            sender_id=sender_id,
            resolved_input=resolved_input,
            contact_email=contact_email,
            agents_used=agents_used,
            channel=channel,
            run_id=run_id,
            store_clarification_fn=pipeline._store_clarification,
            clarification_checkpoint=_clarify_checkpoint,
            shape_response_fn=pipeline._voice.shape_response,
            push_timings_fn=_push_timings,
            logger=logger,
        )
        if _clarify_result is not None:
            return _clarify_result

        # ── 7. ASSESS ────────────────────────────────────────────────
        if on_progress:
            await on_progress({"type": "assessing"})

        raw_response, confidence_prefix, _confidence_value = await run_assessment_step(
            metacognition=pipeline._metacognition,
            shared_kb_evidence=shared_kb_evidence,
            run_id=run_id,
            meta=meta,
            retriever=pipeline._pantheon.retriever,
            resolved_input=resolved_input,
            agents_used=agents_used,
            raw_response=raw_response,
            uncensored_turn=RequestPipeline._uncensored_local_llm_active(meta),
            logger=logger,
        )
        if _confidence_value is not None:
            trace["confidence"] = _confidence_value
        _record_stage("assess")

        # ── 8. REFLECT ───────────────────────────────────────────────
        if on_progress:
            await on_progress({"type": "reflecting"})

        reflection_text = await run_reflection_step(
            inner_voice=pipeline._inner_voice,
            contact_email=contact_email,
            channel=channel,
            raw_input=raw_input,
            raw_response=raw_response,
            logger=logger,
        )

        # ── 8.5 SOURCE LIMITATION NOTE ────────────────────────────────
        raw_response = apply_source_limitation_note(
            raw_response=raw_response,
            agents_used=agents_used,
            logger=logger,
        )

        # ── 9. SHAPE ─────────────────────────────────────────────────
        if on_progress:
            await on_progress({"type": "shaping"})

        shaped = await shape_pipeline_response(
            email_gold_eval_json=_email_gold_eval_json,
            confidence_prefix=confidence_prefix,
            raw_response=raw_response,
            reflection_text=reflection_text,
            contact_info=contact_info,
            contact_email=contact_email,
            channel=channel,
            voice=pipeline._voice,
            endocrine=pipeline._endocrine,
            logger=logger,
            record_shape_stage=lambda: _record_stage("shape"),
        )

        # ── 9.5 STABILITY SCORING (Metis) ────────────────────────────
        shaped, metis_result = await apply_metis_stability_note(
            shaped=shaped,
            pantheon=pipeline._pantheon,
            agents_used=agents_used,
            raw_response=raw_response,
            logger=logger,
        )

        from brain_os.pipeline_phases.triangulation_gaps_append import (
            maybe_append_triangulation_gaps_block,
        )

        _tool_audit_pre_graphe = (
            context.get("_tool_audit") if isinstance(context.get("_tool_audit"), list) else []
        )
        shaped = await maybe_append_triangulation_gaps_block(
            shaped=shaped,
            raw_input=raw_input,
            pantheon=pipeline._pantheon,
            channel=channel,
            contact_email=contact_email,
            tool_audit=_tool_audit_pre_graphe,
            metadata=meta,
            trace=trace,
            logger=logger,
        )

        _pipeline_ms_for_learning = (time.monotonic() - t0) * 1000
        _learning_meta = build_learning_meta(
            metis_result=metis_result,
            pipeline_ms=_pipeline_ms_for_learning,
            email_scope=email_scope,
            agents_used=agents_used,
            raw_response=raw_response,
            had_provenance_warning=_learning_had_provenance,
            had_dlp_flag=_learning_had_dlp,
            route_method=route_method,
        )

        # ── 9.6 LOG SESSION (Graphe) ─────────────────────────────────
        _tool_audit_rows = (
            context.get("_tool_audit") if isinstance(context.get("_tool_audit"), list) else []
        )
        shaped = await append_post_shape_artifacts(
            shaped=shaped,
            pantheon=pipeline._pantheon,
            raw_input=raw_input,
            raw_response=raw_response,
            agents_used=agents_used,
            run_id=run_id,
            channel=channel,
            route_method=route_method,
            email_scope=email_scope,
            contact_email=contact_email,
            goal_lineage_payload=lineage.model_dump() if lineage else None,
            learning_meta=_learning_meta,
            tool_audit=_tool_audit_rows,
            email_gold_eval_json=_email_gold_eval_json,
            tool_stats_tracker=pipeline._tool_stats_tracker,
            shared_kb_evidence=shared_kb_evidence,
            logger=logger,
            meta=meta,
            trace=trace,
            ts_start=ts_wall_start,
            stage_timings=dict(run_stages),
            sender_id=sender_id,
        )

        # ── 10. LEARN (background) ───────────────────────────────────
        # Run learning (Sophia, RealTimeObserver, etc.) in background so we return
        # the response immediately; avoids timeout when using CLI/Cursor.
        async def _learn_then_record() -> None:
            await pipeline._learn(
                contact_email=contact_email,
                channel=channel,
                raw_input=raw_input,
                raw_response=raw_response,
                route_method=route_method,
                agents_used=agents_used,
                active_goal=active_goal,
                resolved_input=resolved_input,
                run_id=run_id,
            )
            _record_stage("learn")
            _push_timings()

        _learn_task = start_background_task_with_lifecycle(
            run_id=run_id,
            task_coro_factory=_learn_then_record,
            record_started=pipeline._record_learn_started,
            record_finished=pipeline._record_learn_finished,
            logger=logger,
        )
        pipeline._last_learn_task = _learn_task

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "RETURN | %s | %.0fms | route=%s agents=%s",
            contact_email,
            elapsed_ms,
            route_method,
            agents_used,
        )

        # ── 11. RETURN ───────────────────────────────────────────────
        return await finalize_response_with_dedup_cache(
            recent_messages=pipeline._recent_messages,
            fingerprint=_fingerprint,
            now_epoch_s=_now,
            shaped=shaped,
            agents_used=agents_used,
            run_id=run_id,
            redis_cache=pipeline._redis,
            redis_key=_redis_dedup_key,
            encode_dedup_payload=_encode_dedup_payload,
        )
    finally:
        from brain_os.brain.run_record_assembler import RunRecordAssemblyContext

        _rr_fallback: RunRecordAssemblyContext | None = None
        if not meta.get("_run_record_persist_scheduled"):
            _rr_fallback = RunRecordAssemblyContext(
                run_id=run_id,
                channel=channel,
                sender_id=sender_id,
                ts_start=float(meta.get("_run_record_ts_start") or ts_wall_start),
                trace=trace,
                meta=meta,
                agents_used=list(trace.get("agents") or []),
                raw_input=str(meta.get("_run_record_raw_input") or raw_input),
                response_text=str(meta.get("_run_record_last_response") or ""),
                tool_stats_tracker=pipeline._tool_stats_tracker,
                stage_timings=dict(run_stages) if run_stages else None,
                email_scope=str(trace.get("email_scope") or "") or None,
            )
        finalize_pipeline_scope(
            metadata=meta,
            trace=trace,
            attach_pipeline_trace=pipeline._attach_pipeline_trace,
            mem0_scope_token=_mem0_scope_tok,
            reset_mem0_scope=mem0_user_id_var.reset,
            logger=logger,
            run_record_fallback=_rr_fallback,
        )
