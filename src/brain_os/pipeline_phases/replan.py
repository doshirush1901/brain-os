"""Phase: replan. Extracted from pipeline_loop.py / pipeline.py 2026-05-15.

Phase modules MUST NOT import from brain_os.pipeline (circular). Shared helpers
live in ira.pipeline_runtime; import from there when this slice needs them.

Contains pure clarification copy for RequestPipeline early exits and
assertion / replan bookkeeping helpers for AgentLoop (pipeline_loop).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from brain_os.schemas.llm_outputs import ClarificationPayload, parse_clarification_text

# ── Clarification copy (RequestPipeline clarification resume / re-ask) ─────

DEFAULT_CLARIFICATION_ONE_SENTENCE = "Could you clarify your request in one sentence?"
CLARIFICATION_REASK_REASON = "Clarification response did not resolve missing details"


def clarification_resume_answer_is_insufficient(
    answer: str,
    pending_question: str,
    *,
    min_len: int = 2,
) -> bool:
    """True when the user's reply is too short or only echoes the pending question."""
    answer_stripped = (answer or "").strip()
    pq = (pending_question or "").strip()
    too_short = len(answer_stripped) < min_len
    echoed = bool(pq) and answer_stripped.lower() == pq.lower()
    return too_short or echoed


def build_clarification_reask_payload(
    *,
    pending_question: str,
    pending: dict[str, Any],
) -> ClarificationPayload:
    """Payload for re-asking after a non-answer clarification turn."""
    q = (pending_question or "").strip() or DEFAULT_CLARIFICATION_ONE_SENTENCE
    return ClarificationPayload(
        needs_clarification=True,
        questions=[q],
        reason=CLARIFICATION_REASK_REASON,
        missing_slots=list(pending.get("missing_slots") or []),
        can_answer_partially=bool(pending.get("can_answer_partially")),
    )


# ── Agent loop: assertion retries, replan / clarify escalation ─────────────


def merge_assertion_retry_counts(
    counts: dict[int, int],
    failed_assertion_ids: list[int],
    *,
    retry_limit: int,
) -> tuple[dict[int, int], list[int]]:
    """Bump counts for each failed assertion; return (new_counts, ids at/over limit)."""
    out = dict(counts)
    escalations: list[int] = []
    for aid in failed_assertion_ids:
        n = out.get(aid, 0) + 1
        out[aid] = n
        if n >= retry_limit:
            escalations.append(aid)
    return out, escalations


def format_validation_contract_failure_reason(
    *,
    phase_summary: str,
    failed_assertion_ids: list[int],
    missing_evidence: list[str],
) -> str:
    """Single-line reason for REPLAN / CLARIFY after a failed phase validator."""
    reason_parts = [
        "Validation contract failure.",
        phase_summary or "Validator marked this phase as not passing.",
    ]
    if failed_assertion_ids:
        failed_ids = ", ".join(str(i) for i in failed_assertion_ids)
        reason_parts.append(f"Failed assertion IDs: {failed_ids}.")
    if missing_evidence:
        reason_parts.append(f"Missing evidence: {'; '.join(missing_evidence[:3])}.")
    return " ".join(reason_parts)


def resolve_assertion_text(assertions: list[str], assertion_id: int) -> str:
    """Human-readable assertion body, or a placeholder when index is out of range."""
    idx = assertion_id - 1
    if 0 <= idx < len(assertions):
        return assertions[idx]
    return f"Assertion {assertion_id}"


def preferred_corrective_agents(available_agent_keys: set[str] | frozenset[str]) -> list[str]:
    """Prefer Vera + Clio when Vera exists; otherwise Clio-only."""
    keys = available_agent_keys
    if "vera" in keys:
        return ["vera", "clio"]
    return ["clio"]


def build_corrective_phase_kwargs(
    *,
    assertion_id: int,
    assertion_text: str,
    phase_id: int,
    preferred_agents: list[str],
) -> dict[str, Any]:
    """Keyword args sufficient to construct a ``Phase`` for a corrective pass."""
    return {
        "id": phase_id,
        "title": f"Corrective pass for assertion {assertion_id}",
        "description": (
            "Address validator failure with concrete evidence and fix gaps for "
            f"assertion {assertion_id}: {assertion_text}"
        ),
        "agents": list(preferred_agents),
        "delegation_type": "generic",
        "expected_output": f"Assertion {assertion_id} resolved with supporting evidence",
    }


def format_assertion_lines_for_clarify(
    assertions: list[str], assertion_ids: list[int]
) -> list[str]:
    """Numbered lines for user-facing clarification after repeated assertion failures."""
    lines: list[str] = []
    for assertion_id in assertion_ids:
        idx = assertion_id - 1
        if 0 <= idx < len(assertions):
            lines.append(f"{assertion_id}) {assertions[idx]}")
        else:
            lines.append(f"{assertion_id}) (assertion text unavailable)")
    return lines


def build_assertion_escalation_clarification_question(
    assertions: list[str],
    assertion_ids: list[int],
) -> str:
    """Copy for CLARIFY when assertion retries are exhausted."""
    assertion_lines = format_assertion_lines_for_clarify(assertions, assertion_ids)
    listed = " | ".join(assertion_lines)
    return (
        "I have retried validation but these assertions still fail. "
        f"Please clarify acceptance criteria or provide missing evidence for: {listed}"
    )


def format_completed_phases_summary(
    completed_snippets: list[tuple[int, str, str]],
    *,
    max_snippet_chars: int = 300,
) -> str:
    """One line per completed phase for Athena replan prompts.

    ``max_snippet_chars`` defaults to 300 (terse, replan-sized). Callers that
    need the verifier/judge to see full evidence (citations, URLs, dates)
    should pass a larger cap — at 300 the StandingGoalJudge never sees the
    source lines and returns ``continue`` forever.
    """
    return "\n".join(
        f"Phase {phase_id} ({title}): {snippet[:max_snippet_chars]}"
        for phase_id, title, snippet in completed_snippets
    )


def format_athena_replan_user_prompt(
    *,
    goal: str,
    completed_summary: str,
    decision_reason: str,
    failed_assertion_ids: list[int],
    next_start_phase_id: int,
) -> str:
    """User-message body for the replan LLM call."""
    return f"""The execution plan needs revision.

ORIGINAL GOAL: {goal}
COMPLETED SO FAR:
{completed_summary}

REASON FOR REPLAN: {decision_reason}
FAILED ASSERTION IDS (if any): {failed_assertion_ids}

Create ONLY the remaining phases (do not repeat completed ones).
Return a JSON array of phase objects with the same structure as before.
Start phase IDs from {next_start_phase_id}.
"""


async def maybe_finalize_clarification_turn(
    *,
    raw_response: str,
    sender_id: str,
    resolved_input: str,
    contact_email: str,
    agents_used: list[str],
    channel: str,
    run_id: str,
    store_clarification_fn: Callable[..., Awaitable[str]],
    shape_response_fn: Callable[[str, str], Awaitable[str]],
    push_timings_fn: Callable[[], None],
    logger: Any,
    clarification_checkpoint: dict[str, Any] | None = None,
) -> tuple[str, list[str], str] | None:
    """Return early clarification response triple, or ``None`` when not needed."""
    clarify_payload = parse_clarification_text(raw_response)
    if not clarify_payload.needs_clarification:
        return None
    clarification_q = await store_clarification_fn(
        sender_id=sender_id,
        agent_name=agents_used[0] if agents_used else "athena",
        original_query=resolved_input,
        payload=clarify_payload,
        checkpoint=clarification_checkpoint,
    )
    logger.info("CLARIFY | stored pending for %s (sender=%s)", contact_email, sender_id)
    shaped = await shape_response_fn(clarification_q, channel)
    push_timings_fn()
    return shaped, agents_used, run_id


async def maybe_run_sphinx_gate(
    *,
    sphinx_agent: Any | None,
    resolved_input: str,
    channel: str,
    sender_id: str,
    contact_email: str,
    run_id: str,
    on_progress: Callable[[dict[str, Any]], Awaitable[None]] | None,
    store_clarification_fn: Callable[..., Awaitable[str]],
    shape_response_fn: Callable[[str, str], Awaitable[str]],
    record_route_stage_fn: Callable[[], None],
    push_timings_fn: Callable[[], None],
    trace: dict[str, Any],
    meta: dict[str, Any],
    ts_start: float,
    logger: Any,
    sphinx_timeout_s: int = 15,
    clarification_checkpoint: dict[str, Any] | None = None,
    pantheon: Any | None = None,
    crm: Any | None = None,
    email_processor: Any | None = None,
    knowledge_graph: Any | None = None,
) -> tuple[str, list[str], str] | None:
    """Run Sphinx clarity gate and return an early clarification response when needed.

    Sphinx v2: stakes × ambiguity Socratic gate runs first (fast, no agent
    execution when gated). Legacy ``[CLEAR]/[CLARIFY]`` Sphinx ReAct remains
    as a fallback when the Socratic gate is disabled or pass-through fails open.
    """
    try:
        from brain_os.config import get_settings
        from brain_os.services.socratic_gate import (
            evaluate_socratic_gate,
            gate_score_as_dict,
            gather_live_gap_context,
            query_looks_outboundish,
            socratic_extra_state,
        )

        # Organism golden / replay grade answers, not clarification theater.
        if str(meta.get("eval_channel") or "") in {"organism_golden", "replay"}:
            return None

        app = get_settings().app
        ask_first = bool(meta.get("ask_first") or meta.get("socratic_ask_first"))
        if app.socratic_gate_enabled or ask_first:
            if on_progress:
                await on_progress({"type": "sphinx_checking"})
            llm = None
            if sphinx_agent is not None:
                try:
                    await sphinx_agent._ensure_llm()
                    llm = getattr(sphinx_agent, "_llm", None)
                except Exception:
                    llm = None

            gap_kwargs: dict[str, Any] = {}
            # Meta overrides win (tests / operator-forced legs).
            for key in (
                "has_open_quote",
                "mail_touch_count",
                "kb_quote_hits",
                "crm_stage",
                "entity_candidates",
            ):
                if key in meta and meta[key] is not None:
                    gap_kwargs[key] = meta[key]

            if query_looks_outboundish(resolved_input) or ask_first:
                live = await gather_live_gap_context(
                    resolved_input,
                    crm=crm,
                    email_processor=email_processor,
                    knowledge_graph=knowledge_graph,
                    timeout_s=min(2.5, max(0.8, float(sphinx_timeout_s) * 0.2)),
                    entity_candidates=gap_kwargs.get("entity_candidates"),
                )
                for k, v in live.as_eval_kwargs().items():
                    gap_kwargs.setdefault(k, v)
                if live.sources:
                    trace["socratic_gap_sources"] = list(live.sources)
                    meta["socratic_gap_sources"] = list(live.sources)

            async def _eval(llm_client: Any | None) -> Any:
                return await evaluate_socratic_gate(
                    resolved_input,
                    llm_client=llm_client,
                    ask_first=ask_first,
                    channel=channel,
                    enabled=True if ask_first else None,
                    **gap_kwargs,
                )

            try:
                socratic = await asyncio.wait_for(_eval(llm), timeout=sphinx_timeout_s)
            except TimeoutError:
                logger.warning(
                    "Sphinx Socratic LLM path timed out after %ds — deterministic fallback",
                    sphinx_timeout_s,
                )
                socratic = await evaluate_socratic_gate(
                    resolved_input,
                    llm_client=None,
                    ask_first=ask_first,
                    channel=channel,
                    enabled=True if ask_first else None,
                    **gap_kwargs,
                )
            trace["socratic_gate"] = gate_score_as_dict(socratic.score)
            if gap_kwargs:
                trace["socratic_gap_kwargs"] = {
                    k: (list(v) if isinstance(v, list) else v) for k, v in gap_kwargs.items()
                }
            if socratic.gated and socratic.clarification is not None:
                session_id = str(
                    meta.get("session_id") or meta.get("conversation_id") or sender_id or ""
                )
                conversation_id = str(meta.get("conversation_id") or meta.get("session_id") or "")
                clarification_q = await store_clarification_fn(
                    sender_id=sender_id,
                    agent_name="sphinx",
                    original_query=resolved_input,
                    payload=socratic.clarification,
                    checkpoint=clarification_checkpoint,
                    extra=socratic_extra_state(
                        socratic,
                        session_id=session_id,
                        conversation_id=conversation_id,
                        channel=channel,
                        sender_id=sender_id,
                    ),
                )
                if on_progress:
                    await on_progress(
                        {
                            "type": "sphinx_clarifying",
                            "questions": clarification_q[:300],
                            "socratic": True,
                        }
                    )
                logger.info(
                    "SPHINX SOCRATIC | stakes=%.2f amb=%.2f gate=%.2f for %s",
                    socratic.score.stakes,
                    socratic.score.ambiguity,
                    socratic.score.gate_score,
                    contact_email,
                )
                # Pass through verbatim — voice reshape must not bury the gate.
                shaped = clarification_q
                record_route_stage_fn()
                push_timings_fn()
                trace["uncertainty_contract"] = "clarify"
                trace["early_exit"] = "sphinx_socratic"
                trace["agents"] = ["sphinx"]
                trace["socratic_needs_input"] = True
                if socratic.gate_id:
                    trace["gate_id"] = socratic.gate_id
                    meta["gate_id"] = socratic.gate_id
                from brain_os.pipeline_phases.compile import schedule_exit_run_record

                schedule_exit_run_record(
                    meta=meta,
                    run_id=run_id,
                    channel=channel,
                    sender_id=sender_id,
                    ts_start=ts_start,
                    trace=trace,
                    agents_used=["sphinx"],
                    raw_input=resolved_input,
                    response_text=shaped,
                )
                from brain_os.brain.graphe_instrumentation import log_graphe_pipeline_turn

                await log_graphe_pipeline_turn(
                    pantheon,
                    query=resolved_input,
                    agents_used=["sphinx"],
                    raw_response=clarification_q,
                    run_id=run_id,
                    channel=channel,
                    route_method="sphinx_socratic",
                    contact_email=contact_email,
                    early_exit="sphinx_socratic",
                )
                return shaped, ["sphinx"], run_id
            logger.info(
                "SPHINX SOCRATIC PASS | stakes=%.2f amb=%.2f gate=%.2f",
                socratic.score.stakes,
                socratic.score.ambiguity,
                socratic.score.gate_score,
            )
            if app.socratic_gate_enabled:
                return None
    except TimeoutError:
        logger.warning("Sphinx Socratic gate timed out after %ds — proceeding", sphinx_timeout_s)
    except Exception:
        logger.debug("Sphinx Socratic gate failed (non-critical)", exc_info=True)

    try:
        if sphinx_agent is not None:
            if on_progress:
                await on_progress({"type": "sphinx_checking"})
            sphinx_verdict = await asyncio.wait_for(
                sphinx_agent.handle(resolved_input, {"channel": channel, "sender_id": sender_id}),
                timeout=sphinx_timeout_s,
            )
            parsed_verdict = parse_clarification_text(sphinx_verdict)
            if parsed_verdict.needs_clarification:
                clarification_q = await store_clarification_fn(
                    sender_id=sender_id,
                    agent_name="sphinx",
                    original_query=resolved_input,
                    payload=parsed_verdict,
                    checkpoint=clarification_checkpoint,
                )
                if on_progress:
                    await on_progress(
                        {"type": "sphinx_clarifying", "questions": clarification_q[:300]}
                    )
                logger.info("SPHINX CLARIFY | stored pending for %s", contact_email)
                shaped = await shape_response_fn(clarification_q, channel)
                record_route_stage_fn()
                push_timings_fn()
                trace["uncertainty_contract"] = "clarify"
                trace["early_exit"] = "sphinx_clarify"
                trace["agents"] = ["sphinx"]
                from brain_os.pipeline_phases.compile import schedule_exit_run_record

                schedule_exit_run_record(
                    meta=meta,
                    run_id=run_id,
                    channel=channel,
                    sender_id=sender_id,
                    ts_start=ts_start,
                    trace=trace,
                    agents_used=["sphinx"],
                    raw_input=resolved_input,
                    response_text=shaped,
                )
                from brain_os.brain.graphe_instrumentation import log_graphe_pipeline_turn

                await log_graphe_pipeline_turn(
                    pantheon,
                    query=resolved_input,
                    agents_used=["sphinx"],
                    raw_response=clarification_q,
                    run_id=run_id,
                    channel=channel,
                    route_method="sphinx_clarify",
                    contact_email=contact_email,
                    early_exit="sphinx_clarify",
                )
                return shaped, ["sphinx"], run_id
            if str(sphinx_verdict).upper().strip().startswith("[CLEAR]"):
                logger.info("SPHINX CLEAR | query is actionable")
    except TimeoutError:
        logger.warning("Sphinx gate timed out after %ds — proceeding", sphinx_timeout_s)
    except Exception:
        logger.debug("Sphinx gate failed (non-critical)", exc_info=True)
    return None


__all__ = [
    "CLARIFICATION_REASK_REASON",
    "DEFAULT_CLARIFICATION_ONE_SENTENCE",
    "build_assertion_escalation_clarification_question",
    "build_clarification_reask_payload",
    "build_corrective_phase_kwargs",
    "clarification_resume_answer_is_insufficient",
    "format_assertion_lines_for_clarify",
    "format_athena_replan_user_prompt",
    "format_completed_phases_summary",
    "format_validation_contract_failure_reason",
    "maybe_finalize_clarification_turn",
    "maybe_run_sphinx_gate",
    "merge_assertion_retry_counts",
    "preferred_corrective_agents",
    "resolve_assertion_text",
]
