"""Phase: execute. Extracted from pipeline.py / pipeline_loop.py 2026-05-15.

Phase modules MUST NOT import from brain_os.pipeline (circular). Shared helpers
live in ira.pipeline_runtime; import from there when this slice needs them.

Pure helpers for routed multi-agent execution (progress payloads, synthesis
context), optional-agent picking, required-tool presence checks, and AgentLoop
delegation / handoff string assembly.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from brain_os.pipeline_phases.outreach import (
    build_outreach_shortlist,
    compose_outreach_workflow_query,
    has_thread_evidence,
)
from brain_os.schemas.llm_outputs import OutreachRankingOutput

EXECUTE_AGENT_PREVIEW_CHARS = 200
_HANDOFF_EVIDENCE_PREVIEW_CHARS = 160
_HANDOFF_AGENT_ERROR_SNIPPET_CHARS = 200

_DECISION_NEEDS_FAILED_OBSERVE = frozenset({"replan", "clarify", "abort"})
_DLP_BLOCK_MESSAGE = (
    "I can't share this response as it contains confidential "
    "business data. Please rephrase your request."
)
_DLP_REVIEW_NOTE = (
    "\n\n> **Content note:** This response may contain "
    "sensitive data. Review before sharing externally."
)


def execute_path_agent_started_event(agent: str, *, role: str = "") -> dict[str, Any]:
    """Progress payload when a routed agent begins (RequestPipeline execute path)."""
    return {"type": "agent_started", "agent": agent, "role": role}


def execute_path_agent_done_event(
    agent: str, done_raw: Any, *, preview_chars: int = 200
) -> dict[str, Any]:
    """Progress payload when a routed agent finishes (includes preview truncation)."""
    preview = (
        done_raw[:preview_chars] if isinstance(done_raw, str) else str(done_raw)[:preview_chars]
    )
    return {"type": "agent_done", "agent": agent, "preview": preview}


def execute_path_synthesizing_event() -> dict[str, Any]:
    return {"type": "synthesizing", "agent": "athena"}


def athena_multi_agent_synthesis_context(agent_responses: dict[str, str]) -> dict[str, Any]:
    """Context dict passed to Athena when stitching multiple agent outputs."""
    return {"agent_responses": dict(agent_responses)}


def phase_started_progress_event(
    *,
    phase_id: int,
    title: str,
    agents: list[str],
) -> dict[str, Any]:
    return {"type": "phase_started", "phase_id": phase_id, "title": title, "agents": agents}


def phase_execution_policy_progress_event(*, phase_id: int, policy: str) -> dict[str, Any]:
    return {"type": "phase_execution_policy", "phase_id": phase_id, "policy": policy}


def phase_agent_started_loop_event(
    *,
    phase_id: int,
    agent_name: str,
    role: str,
) -> dict[str, Any]:
    return {
        "type": "agent_started",
        "phase_id": phase_id,
        "agent": agent_name,
        "role": role,
    }


def phase_agent_done_loop_event(
    *,
    phase_id: int,
    agent_name: str,
    response: Any,
    preview_chars: int = EXECUTE_AGENT_PREVIEW_CHARS,
) -> dict[str, Any]:
    return {
        "type": "agent_done",
        "phase_id": phase_id,
        "agent": agent_name,
        "preview": str(response or "")[:preview_chars],
    }


def phase_completed_progress_event(*, phase_id: int, decision_value: str) -> dict[str, Any]:
    return {"type": "phase_completed", "phase_id": phase_id, "decision": decision_value}


def build_athena_delegation_prompt(
    *,
    phase_id: int,
    description: str,
    expected_output: str,
    prior_context: str,
) -> str:
    """User/delegation body for a single phase agent (AgentLoop)."""
    prompt = (
        f"## Task from Athena (Phase {phase_id})\n"
        f"**Objective:** {description}\n"
        f"**Expected Output:** {expected_output}\n"
    )
    if prior_context:
        prompt += f"\n**Context from prior phases:**\n{prior_context}\n"
    prompt += (
        "\nRespond with your findings. Be specific, cite sources, "
        "and use tables where appropriate. If you lack data, say so."
    )
    return prompt


def format_prior_phase_context_lines(
    completed_snippets: list[tuple[int, str, str]],
    *,
    current_phase_id: int,
    result_max_chars: int = 500,
) -> str:
    """Concatenate prior completed phase results for delegation context."""
    lines = [
        f"Phase {pid} ({title}): {result[:result_max_chars]}"
        for pid, title, result in completed_snippets
        if pid < current_phase_id
    ]
    return "\n".join(lines)


def collect_phase_handoff_open_issues(
    agent_responses: dict[str, str],
    *,
    decision_value: str,
    decision_reason: str,
    clarification_question: str,
    validation_passed: bool,
    validation_summary: str,
    missing_evidence: list[str],
    contradictions: list[str],
    agent_error_snippet_chars: int = _HANDOFF_AGENT_ERROR_SNIPPET_CHARS,
) -> list[str]:
    """Human-readable issues list stored on ``PhaseHandoff``."""
    open_issues = [
        f"{agent}: {response[:agent_error_snippet_chars]}"
        for agent, response in agent_responses.items()
        if response.startswith("(Error:")
        or response.startswith("(Agent")
        or "insufficient" in response.lower()
    ]
    if decision_value == "replan" and decision_reason:
        open_issues.append(f"Replan required: {decision_reason}")
    if decision_value == "clarify" and clarification_question:
        open_issues.append(f"Clarification required: {clarification_question}")
    if not validation_passed:
        open_issues.append(
            f"Validator failure: {validation_summary or 'One or more assertions failed.'}"
        )
        for issue in missing_evidence[:3]:
            open_issues.append(f"Missing evidence: {issue}")
        for issue in contradictions[:3]:
            open_issues.append(f"Contradiction: {issue}")
    return open_issues


def format_phase_handoff_evidence_refs(
    agent_responses: dict[str, str],
    *,
    preview_chars: int = _HANDOFF_EVIDENCE_PREVIEW_CHARS,
) -> list[str]:
    refs: list[str] = []
    for agent, response in agent_responses.items():
        preview = " ".join(str(response).split())[:preview_chars]
        refs.append(f"{agent}: {preview}")
    return refs


def build_phase_handoff_command_specs(
    agent_responses: dict[str, str],
    *,
    decision_value: str,
    validation_passed: bool,
) -> list[tuple[str, int]]:
    """``(command, exit_code)`` rows later wrapped as ``CommandRun`` dataclasses."""
    commands: list[tuple[str, int]] = [
        (
            f"agent:{agent}",
            0 if not str(response).startswith("(Error:") else 1,
        )
        for agent, response in agent_responses.items()
    ]
    if decision_value in _DECISION_NEEDS_FAILED_OBSERVE:
        commands.append(("observe_phase_result", 1))
    else:
        commands.append(("observe_phase_result", 0))
    commands.append(("validator:phase_contract_check", 0 if validation_passed else 1))
    return commands


def pick_optional_agents_in_order(
    ordered_candidates: list[str],
    *,
    budget: int,
    resolve_agent: Callable[[str], Any],
) -> list[str]:
    """First ``budget`` agents from ``ordered_candidates`` that resolve to a live agent."""
    picked: list[str] = []
    for name in ordered_candidates:
        if resolve_agent(name) is None:
            continue
        picked.append(name)
        if len(picked) >= budget:
            break
    return picked


def missing_required_tool_names(
    required_tools: list[str],
    *,
    has_crm: bool,
    has_retriever: bool,
    has_plutus: bool,
    has_hermes: bool,
) -> list[str]:
    """Return ordered, de-duplicated internal tool keys that are required but absent."""
    missing: list[str] = []
    for tool in required_tools:
        t = (tool or "").strip().lower()
        if not t:
            continue
        if t == "crm" and not has_crm:
            missing.append("crm")
            continue
        if t == "retriever" and not has_retriever:
            missing.append("retriever")
            continue
        if t == "pricing_engine" and not has_plutus:
            missing.append("pricing_engine")
            continue
        if t == "drip_engine" and not has_hermes:
            missing.append("drip_engine")
            continue
    out: list[str] = []
    for name in missing:
        if name not in out:
            out.append(name)
    return out


async def run_routed_execution_path(
    *,
    route_method: str,
    required_agents: list[str],
    optional_agents: list[str],
    required_tools: list[str],
    resolved_input: str,
    context: dict[str, Any],
    on_progress: Callable[[dict[str, Any]], Awaitable[None]] | None,
    rank_optional_for_query_fn: Callable[[str, list[str]], list[str]],
    resolve_agent_fn: Callable[[str], Any],
    execute_routed_fn: Callable[
        [list[str], str, dict[str, Any], Callable[[dict[str, Any]], Awaitable[None]] | None],
        Awaitable[tuple[str, list[str]]],
    ],
    has_crm: bool,
    has_retriever: bool,
    has_plutus: bool,
    has_hermes: bool,
    tool_stats_tracker: Any,
    logger: logging.Logger,
) -> tuple[str, list[str], list[str]]:
    """Execute deterministic/procedural route with tool gate + telemetry."""
    from brain_os.config import get_settings as _gs_route_exec

    app_cfg = _gs_route_exec().app
    selected_optional: list[str] = []
    if app_cfg.optional_agent_execution_enabled and app_cfg.optional_agent_budget > 0:
        ordered = (
            rank_optional_for_query_fn(resolved_input, optional_agents)
            if resolved_input
            else list(optional_agents)
        )
        selected_optional = pick_optional_agents_in_order(
            ordered,
            budget=app_cfg.optional_agent_budget,
            resolve_agent=resolve_agent_fn,
        )
    route_agents = list(required_agents) + selected_optional

    missing_tools = missing_required_tool_names(
        required_tools,
        has_crm=has_crm,
        has_retriever=has_retriever,
        has_plutus=has_plutus,
        has_hermes=has_hermes,
    )
    if app_cfg.required_tool_enforcement and missing_tools:
        logger.warning(
            "ROUTE TOOL GATE | method=%s missing=%s",
            route_method,
            ",".join(missing_tools),
        )
        raw_response = (
            "I can not safely run this routed workflow because required capabilities are "
            f"missing: {', '.join(missing_tools)}. Please restore those services and retry."
        )
        agents_used = ["route_tool_gate"]
        tracker = tool_stats_tracker
        if tracker is not None and hasattr(tracker, "record_route_event"):
            try:
                await tracker.record_route_event(
                    source=route_method,
                    required=list(required_agents),
                    optional=list(optional_agents),
                    executed=[],
                    skipped_optional=list(optional_agents),
                    required_tools=list(required_tools),
                    missing_tools=list(missing_tools),
                )
            except Exception:
                logger.debug("Route telemetry (tool gate) failed", exc_info=True)
        return raw_response, agents_used, selected_optional

    raw_response, agents_used = await execute_routed_fn(
        route_agents,
        resolved_input,
        context,
        on_progress,
    )
    tracker = tool_stats_tracker
    if tracker is not None and hasattr(tracker, "record_route_event"):
        try:
            executed_optional = [a for a in selected_optional if a in agents_used]
            skipped_optional = [a for a in optional_agents if a not in selected_optional]
            await tracker.record_route_event(
                source=route_method,
                required=list(required_agents),
                optional=list(optional_agents),
                executed=list(route_agents),
                skipped_optional=skipped_optional,
                required_tools=list(required_tools),
                missing_tools=[],
                executed_optional=executed_optional,
            )
        except Exception:
            logger.debug("Route telemetry emit failed", exc_info=True)
    return raw_response, agents_used, selected_optional


def collect_outreach_thread_ids(
    shortlist: list[dict[str, Any]] | None,
    tool_audit: Any,
) -> set[str]:
    """Collect grounded thread IDs from shortlist rows and successful tool reads."""
    allowed: set[str] = set()
    for row in shortlist or []:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("thread_id") or "").strip()
        if tid:
            allowed.add(tid)
    if isinstance(tool_audit, list):
        for item in tool_audit:
            if not isinstance(item, dict):
                continue
            if item.get("tool") != "read_email_thread" or not bool(item.get("success")):
                continue
            tid = str(item.get("thread_id") or "").strip()
            if tid:
                allowed.add(tid)
    return allowed


def normalize_outreach_workflow_output(
    raw_response: str,
    *,
    allowed_thread_ids: set[str] | None,
    extract_json_payload: Callable[[str], str | None],
    render_outreach_ranking: Callable[[OutreachRankingOutput], str],
    logger: logging.Logger,
) -> tuple[str, bool]:
    """Validate/normalize outreach workflow JSON into rendered ranking text."""
    payload = extract_json_payload(raw_response)
    if not payload:
        return (
            "I can't finalize outreach ranking because the workflow output "
            "was not structured JSON with thread-level evidence.",
            False,
        )
    try:
        parsed = OutreachRankingOutput.model_validate_json(payload)
    except Exception as exc:
        logger.warning("Outreach workflow JSON validation failed: %s", exc)
        return (
            "I can't finalize outreach ranking because the workflow JSON was invalid. "
            "Please rerun with structured output including ranked candidates and thread IDs.",
            False,
        )
    for c in parsed.ranked[:3]:
        if not (c.thread_id or "").strip() or not (c.evidence_line or "").strip():
            return (
                "I can't finalize outreach ranking because one or more ranked "
                "candidates are missing thread_id or evidence_line.",
                False,
            )
    if allowed_thread_ids:
        bad = [
            c.thread_id.strip()
            for c in parsed.ranked[:3]
            if (c.thread_id or "").strip() not in allowed_thread_ids
        ]
        rec_tid = (parsed.recommendation.thread_id or "").strip()
        if rec_tid and rec_tid not in allowed_thread_ids:
            bad.append(rec_tid)
        if bad:
            return (
                "I can't finalize outreach ranking because returned thread_id values "
                "are not grounded in prefetched/shortlisted evidence.",
                False,
            )
    return render_outreach_ranking(parsed), True


async def run_aletheia_compliance_check(
    raw_response: str,
    agents_used: list[str],
    *,
    aletheia: Any,
    logger: logging.Logger,
) -> tuple[str, list[str], bool]:
    """Run Aletheia provenance check; return updated text/agents and provenance flag."""
    had_provenance = False
    if aletheia is None:
        return raw_response, agents_used, had_provenance
    try:
        provenance = await aletheia.check_provenance(raw_response)
        if provenance.get("unverifiable"):
            unverifiable = provenance["unverifiable"]
            had_provenance = True
            note = (
                "\n\n> **Provenance note:** "
                + f"{len(unverifiable)} claim(s) could not be traced to a source: "
                + ", ".join(unverifiable[:3])
                + ("..." if len(unverifiable) > 3 else "")
                + "."
            )
            raw_response += note
            if "aletheia" not in agents_used:
                agents_used.append("aletheia")
            logger.info(
                "ALETHEIA | verdict=%s unverifiable=%d",
                provenance["verdict"],
                len(unverifiable),
            )
    except Exception:
        logger.exception("Aletheia compliance check failed")
    return raw_response, agents_used, had_provenance


async def run_aegis_dlp_check(
    raw_response: str,
    agents_used: list[str],
    *,
    aegis: Any,
    logger: logging.Logger,
    post_gapper: bool = False,
    append_agent: bool = True,
) -> tuple[str, list[str], bool]:
    """Run Aegis content scan; return updated text/agents and DLP-flag."""
    had_dlp = False
    if aegis is None:
        return raw_response, agents_used, had_dlp
    try:
        dlp_result = await aegis.check_content(raw_response)
        verdict = str(dlp_result.get("verdict") or "")
        if verdict == "BLOCK":
            had_dlp = True
            raw_response = _DLP_BLOCK_MESSAGE
            if post_gapper:
                logger.warning("AEGIS | post-Gapper BLOCK — response replaced")
            else:
                logger.warning("AEGIS | BLOCK — response replaced")
        elif verdict == "REVIEW_NEEDED":
            had_dlp = True
            raw_response += _DLP_REVIEW_NOTE
            if post_gapper:
                logger.info("AEGIS | post-Gapper REVIEW_NEEDED — caveat appended")
            else:
                logger.info("AEGIS | REVIEW_NEEDED — caveat appended")
        if append_agent and "aegis" not in agents_used:
            agents_used.append("aegis")
    except Exception:
        if post_gapper:
            logger.exception("Aegis post-Gapper DLP check failed")
        else:
            logger.exception("Aegis DLP check failed")
    return raw_response, agents_used, had_dlp


async def run_mnemon_correction_check(
    raw_response: str,
    agents_used: list[str],
    *,
    mnemon: Any,
    logger: logging.Logger,
) -> tuple[str, list[str]]:
    """Run Mnemon correction pass; return updated text/agents."""
    if mnemon is None:
        return raw_response, agents_used
    try:
        corrected = await mnemon.check_and_correct(raw_response)
        if corrected != raw_response:
            raw_response = corrected
            if "mnemon" not in agents_used:
                agents_used.append("mnemon")
            logger.info("MNEMON | corrections applied to response")
    except Exception:
        logger.exception("Mnemon correction check failed")
    return raw_response, agents_used


def is_slack_pipeline_channel(channel: str) -> bool:
    return (channel or "").strip().lower() in ("slack_dm", "slack_channel")


async def run_gapper_resolution(
    raw_response: str,
    agents_used: list[str],
    *,
    resolved_input: str,
    gapper: Any,
    on_progress: Callable[[dict[str, Any]], Awaitable[None]] | None,
    logger: logging.Logger,
    detect_gaps_fn: Callable[[str], list[Any]] | None = None,
    channel: str = "",
    strict_full_workflow: bool = False,
) -> tuple[str, list[str]]:
    """Run Gapper when gaps are detected; return updated text/agents."""
    if gapper is None:
        return raw_response, agents_used
    if is_slack_pipeline_channel(channel) and not strict_full_workflow:
        logger.debug("GAP RESOLVE | skipped for Slack channel=%s", channel)
        return raw_response, agents_used
    try:
        detector = detect_gaps_fn
        if detector is None:
            from brain_os.agents.gapper import detect_gaps as detector

        gaps = detector(raw_response)
        if gaps:
            if on_progress:
                await on_progress({"type": "gap_resolving", "gaps": len(gaps)})
            logger.info("GAP RESOLVE | %d gaps detected, invoking Gapper", len(gaps))
            resolved = await gapper.resolve_gaps(raw_response, resolved_input)
            if resolved and resolved != raw_response:
                raw_response = resolved
                agents_used.append("gapper")
    except Exception:
        logger.exception("Gapper resolution failed, continuing with original response")
    return raw_response, agents_used


async def run_execute_dispatch(
    *,
    pipeline: Any,
    require_thread_evidence: bool,
    route_method: str,
    resolved_input: str,
    context: dict[str, Any],
    on_progress: Any | None,
    truth_hint_response: str | None,
    agent_names: list[str],
    optional_agent_names: list[str],
    required_tools: list[str],
    logger: logging.Logger,
    extract_json_payload: Callable[[str], str],
    render_outreach_ranking: Callable[[OutreachRankingOutput], str],
    thread_id_pattern: Any,
) -> tuple[str, list[str], str, list[str], str | None]:
    """Execute route dispatch branch and return response + execution telemetry."""
    raw_response: str
    agents_used: list[str]
    optional_selected_snap: list[str] = []
    email_gold_eval_json: str | None = None

    if require_thread_evidence:
        route_method = "outreach_workflow"
        outreach_shortlist = await pipeline_phases_build_outreach_shortlist(
            pipeline=pipeline,
            query=resolved_input,
            thread_id_pattern=thread_id_pattern,
            logger=logger,
        )
        if outreach_shortlist:
            context["outreach_shortlist"] = outreach_shortlist
        workflow_query = pipeline_phases_compose_outreach_workflow_query(
            resolved_input, outreach_shortlist
        )
        raw_response, agents_used = await pipeline._execute_routed(
            pipeline._OUTREACH_WORKFLOW_AGENTS,
            workflow_query,
            context,
            on_progress,
        )
        allowed_thread_ids = collect_outreach_thread_ids(
            outreach_shortlist,
            context.get("_tool_audit"),
        )
        raw_response, ok = normalize_outreach_workflow_output(
            raw_response,
            allowed_thread_ids=allowed_thread_ids,
            extract_json_payload=extract_json_payload,
            render_outreach_ranking=render_outreach_ranking,
            logger=logger,
        )
        if not ok:
            logger.warning("OUTREACH WORKFLOW | invalid structured output")
    elif truth_hint_response is not None:
        raw_response = truth_hint_response
        agents_used = ["truth_hints"]
    elif route_method in ("deterministic", "procedural"):
        raw_response, agents_used, selected_optional = await run_routed_execution_path(
            route_method=route_method,
            required_agents=agent_names,
            optional_agents=optional_agent_names,
            required_tools=required_tools,
            resolved_input=resolved_input,
            context=context,
            on_progress=on_progress,
            rank_optional_for_query_fn=pipeline._rank_optional_for_query,
            resolve_agent_fn=pipeline._pantheon.get_agent,
            execute_routed_fn=pipeline._execute_routed,
            has_crm=pipeline._crm is not None,
            has_retriever=getattr(pipeline._pantheon, "retriever", None) is not None,
            has_plutus=pipeline._pantheon.get_agent("plutus") is not None,
            has_hermes=pipeline._pantheon.get_agent("hermes") is not None,
            tool_stats_tracker=pipeline._tool_stats_tracker,
            logger=logger,
        )
        optional_selected_snap = list(selected_optional)
        if "[IRA_EMAIL_GOLD_EVAL_v1]" in resolved_input:
            email_gold_eval_json = raw_response.strip()
    else:
        raw_response = await pipeline._pantheon.process(
            resolved_input,
            context,
            on_progress=on_progress,
        )
        agents_used = ["athena"]

    if require_thread_evidence and not pipeline_phases_has_thread_evidence(
        context.get("_tool_audit")
    ):
        raw_response = (
            "I can't recommend who to email next yet because this run did not read a live "
            "email thread. Please fetch the relevant thread first (search_emails + "
            "read_email_thread), or provide the thread ID so I can ground the recommendation "
            "on latest buyer intent."
        )
        logger.warning(
            "OUTREACH GATE | blocked recommendation due to missing read_email_thread evidence",
        )

    return raw_response, agents_used, route_method, optional_selected_snap, email_gold_eval_json


async def pipeline_phases_build_outreach_shortlist(
    *,
    pipeline: Any,
    query: str,
    thread_id_pattern: Any,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    return await build_outreach_shortlist(
        crm=pipeline._crm,
        query=query,
        thread_id_pattern=thread_id_pattern,
        logger=logger,
    )


def pipeline_phases_compose_outreach_workflow_query(
    resolved_input: str,
    outreach_shortlist: list[dict[str, Any]],
) -> str:
    return compose_outreach_workflow_query(resolved_input, outreach_shortlist)


def pipeline_phases_has_thread_evidence(tool_audit: Any) -> bool:
    return has_thread_evidence(tool_audit)


def apply_execute_trace_telemetry(
    *,
    trace: dict[str, Any],
    route_method: str,
    agents_used: list[str],
    optional_requested_snap: list[str],
    optional_selected_snap: list[str],
    logger: logging.Logger,
    record_execute_stage_fn: Callable[[], None],
) -> None:
    """Record execute stage and enrich routing telemetry trace fields."""
    record_execute_stage_fn()
    trace["route"] = route_method
    trace["agents"] = agents_used
    rte = trace.get("routing_telemetry")
    if isinstance(rte, dict):
        rte["route_method_resolved"] = route_method
        rte["optional_agents_requested"] = optional_requested_snap
        rte["optional_agents_selected_for_execution"] = optional_selected_snap
        rte["optional_agents_skipped_budget"] = [
            agent for agent in optional_requested_snap if agent not in optional_selected_snap
        ]
    logger.info("EXECUTE | route=%s agents=%s", route_method, agents_used)


__all__ = [
    "EXECUTE_AGENT_PREVIEW_CHARS",
    "apply_execute_trace_telemetry",
    "athena_multi_agent_synthesis_context",
    "build_athena_delegation_prompt",
    "build_phase_handoff_command_specs",
    "collect_outreach_thread_ids",
    "collect_phase_handoff_open_issues",
    "execute_path_agent_done_event",
    "execute_path_agent_started_event",
    "execute_path_synthesizing_event",
    "format_phase_handoff_evidence_refs",
    "format_prior_phase_context_lines",
    "missing_required_tool_names",
    "normalize_outreach_workflow_output",
    "phase_agent_done_loop_event",
    "phase_agent_started_loop_event",
    "phase_completed_progress_event",
    "phase_execution_policy_progress_event",
    "phase_started_progress_event",
    "pick_optional_agents_in_order",
    "run_aegis_dlp_check",
    "run_aletheia_compliance_check",
    "run_execute_dispatch",
    "run_gapper_resolution",
    "run_mnemon_correction_check",
    "run_routed_execution_path",
]
