from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from brain_os.config import get_settings
from brain_os.pipeline_phases.execute import (
    phase_agent_done_loop_event,
    phase_agent_started_loop_event,
)
from brain_os.pipeline_phases.replan import format_completed_phases_summary
from brain_os.systems.standing_goal import (
    StandingGoalStatus,
    record_standing_goal_history,
    standing_goal_defaults,
)

logger = logging.getLogger(__name__)


async def run_autonomous_loop(
    *,
    plan: Any,
    on_progress: Callable[[dict[str, Any]], Awaitable[None]] | None,
    execute_phase_fn: Callable[..., Awaitable[Any]],
    replan_fn: Callable[..., Awaitable[Any]],
    compile_fn: Callable[..., Awaitable[str]],
    standing_goal_status_active: Any,
    loop_decision_replan: Any,
    loop_decision_clarify: Any,
    loop_decision_complete: Any,
    loop_decision_abort: Any,
    max_phases: int,
) -> str:
    while not plan.is_complete:
        phase = plan.current_phase
        if phase is None:
            break

        result = await execute_phase_fn(plan, phase, on_progress)

        if result.decision == loop_decision_replan:
            plan = await replan_fn(plan, result, on_progress)
        elif result.decision == loop_decision_clarify:
            question = result.clarification_question or "I need clarification before continuing."
            logger.info("Autonomous loop paused for clarification: %s", question)
            return f"[CLARIFY] {question}"
        elif result.decision == loop_decision_complete:
            if (
                plan.standing_goal is not None
                and plan.standing_goal.status == standing_goal_status_active
            ):
                continue
            break
        elif result.decision == loop_decision_abort:
            return f"Task aborted: {result.decision_reason}"

        _max_turns, standing_goal_max_phases = standing_goal_defaults()
        phase_cap = max(max_phases, standing_goal_max_phases)
        if len(plan.completed_phases) > phase_cap:
            logger.warning("Max phases exceeded, forcing completion")
            break

    if plan.standing_goal is not None and plan.standing_goal.status == standing_goal_status_active:
        return (
            "[STANDING_GOAL] Objective still active after phase loop. "
            "Run more execute_phase calls or clear_standing_goal."
        )
    return await compile_fn(plan, on_progress=on_progress)


def phase_allows_parallel_read_only(phase: Any) -> bool:
    delegation_type = (phase.delegation_type or "").strip().lower()
    return delegation_type in {"readonly", "read_only"}


async def collect_phase_agent_responses(
    *,
    phase: Any,
    prior_context: str,
    on_progress: Callable[[dict[str, Any]], Awaitable[None]] | None,
    run_single_phase_agent_fn: Callable[..., Awaitable[tuple[str, str]]],
) -> dict[str, str]:
    if phase_allows_parallel_read_only(phase) and len(phase.agents) > 1:
        tasks = [
            run_single_phase_agent_fn(
                phase=phase,
                prior_context=prior_context,
                agent_name=agent_name,
                on_progress=on_progress,
            )
            for agent_name in phase.agents
        ]
        pairs = await asyncio.gather(*tasks)
        return {agent_name: response for agent_name, response in pairs}

    responses: dict[str, str] = {}
    for agent_name in phase.agents:
        name, response = await run_single_phase_agent_fn(
            phase=phase,
            prior_context=prior_context,
            agent_name=agent_name,
            on_progress=on_progress,
        )
        responses[name] = response
    return responses


async def run_single_phase_agent(
    *,
    pantheon: Any,
    phase: Any,
    prior_context: str,
    agent_name: str,
    on_progress: Callable[[dict[str, Any]], Awaitable[None]] | None,
    build_delegation_prompt_fn: Callable[[Any, str], str],
) -> tuple[str, str]:
    agent = pantheon.get_agent(agent_name.lower())
    if agent is None:
        return agent_name, f"(Agent '{agent_name}' not found)"

    if on_progress:
        await on_progress(
            phase_agent_started_loop_event(
                phase_id=phase.id,
                agent_name=agent_name,
                role=getattr(agent, "role", ""),
            )
        )

    delegation_prompt = build_delegation_prompt_fn(phase, prior_context)
    try:
        response = await agent.handle(delegation_prompt)
    except Exception as exc:  # pragma: no cover - preserved behavior branch
        logger.exception("Phase %d agent '%s' failed", phase.id, agent_name)
        response = f"(Error: {exc})"

    if on_progress:
        await on_progress(
            phase_agent_done_loop_event(
                phase_id=phase.id,
                agent_name=agent_name,
                response=response,
            )
        )
    return agent_name, response


def merge_standing_objective_into_contract(*, plan: Any) -> None:
    objective = plan.standing_objective.strip()
    if not objective:
        return
    if objective not in plan.validation_contract.assertions:
        plan.validation_contract.assertions.insert(0, objective)
    if plan.phases and 1 not in (
        assertion_id
        for assertion_ids in plan.validation_contract.phase_assertion_map.values()
        for assertion_id in assertion_ids
    ):
        first_phase_id = plan.phases[0].id
        existing = plan.validation_contract.phase_assertion_map.get(first_phase_id, [])
        plan.validation_contract.phase_assertion_map[first_phase_id] = [1, *existing]


def standing_goal_blocks_compile(*, plan: Any) -> str | None:
    if not plan.standing_objective.strip() or plan.standing_goal is None:
        return None
    if plan.standing_goal.status == StandingGoalStatus.ACTIVE:
        return (
            "Standing objective is still active. Complete all phases, achieve the "
            "objective, or call clear_standing_goal / pause_standing_goal before compile."
        )
    return None


async def run_standing_goal_judge(
    *,
    plan: Any,
    standing_goal_judge: Any,
    get_contract_summary_fn: Callable[[Any], dict[str, Any]],
) -> Any:
    completed_summary = format_completed_phases_summary(
        [(phase.id, phase.title, str(phase.result or "")) for phase in plan.completed_phases]
    )
    contract_summary = json.dumps(get_contract_summary_fn(plan), indent=2)[:8000]
    return await standing_goal_judge.evaluate(
        objective=plan.standing_objective,
        plan_goal=plan.goal,
        completed_phases_summary=completed_summary,
        validation_contract_summary=contract_summary,
        turn=plan.standing_goal.turn_count,
        max_turns=plan.standing_goal.max_turns,
    )


async def inject_continuation_phases(
    *,
    plan: Any,
    judge_reason: str,
    athena: Any,
    available_agents: list[str],
    phase_ctor: Callable[..., Any],
) -> int:
    start_id = max((phase.id for phase in plan.phases), default=0) + 1
    prompt = f"""The standing objective is not yet satisfied.

STANDING OBJECTIVE: {plan.standing_objective}

JUDGE REASON TO CONTINUE: {judge_reason}

COMPLETED WORK:
{format_completed_phases_summary([(phase.id, phase.title, str(phase.result or "")) for phase in plan.completed_phases])}

Add 1 to 3 new phases to finish the objective. Return a JSON array of phase objects:
[
  {{
    "id": {start_id},
    "title": "...",
    "description": "...",
    "agents": ["agent_name"],
    "delegation_type": "generic",
    "expected_output": "..."
  }}
]

Available agents: {", ".join(sorted(available_agents))}
"""
    added = 0
    try:
        raw = await athena.call_llm(
            "You are Athena adding continuation phases toward a standing objective.",
            prompt,
            temperature=0.2,
        )
        cleaned = (raw or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                cleaned,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
        parsed = json.loads(cleaned)
        items = parsed if isinstance(parsed, list) else parsed.get("phases", [])
        if not isinstance(items, list):
            items = []
        next_id = start_id
        for phase_data in items[:3]:
            if not isinstance(phase_data, dict):
                continue
            phase_id = phase_data.get("id", next_id)
            if not isinstance(phase_id, int):
                phase_id = next_id
            next_id = max(next_id + 1, phase_id + 1)
            agents = phase_data.get("agents", [])
            if not isinstance(agents, list):
                agents = []
            agents = [str(name).lower() for name in agents if str(name).strip()] or ["clio"]
            plan.phases.append(
                phase_ctor(
                    id=phase_id,
                    title=str(phase_data.get("title") or f"Continuation phase {phase_id}"),
                    description=str(phase_data.get("description") or judge_reason),
                    agents=agents,
                    delegation_type=str(phase_data.get("delegation_type") or "generic"),
                    expected_output=str(phase_data.get("expected_output") or ""),
                )
            )
            assertion_id = len(plan.validation_contract.assertions) + 1
            plan.validation_contract.assertions.append(
                str(phase_data.get("expected_output") or plan.standing_objective)
            )
            plan.validation_contract.phase_assertion_map[phase_id] = [assertion_id]
            added += 1
    except Exception:  # pragma: no cover - preserved behavior branch
        logger.exception("Continuation phase injection failed for plan %s", plan.plan_id)
        plan.phases.append(
            phase_ctor(
                id=start_id,
                title="Continuation toward standing objective",
                description=judge_reason,
                agents=["clio"],
                delegation_type="generic",
                expected_output=plan.standing_objective,
            )
        )
        assertion_id = len(plan.validation_contract.assertions) + 1
        plan.validation_contract.assertions.append(plan.standing_objective)
        plan.validation_contract.phase_assertion_map[start_id] = [assertion_id]
        added = 1
    return added


async def maybe_apply_standing_goal(
    *,
    plan: Any,
    decision: Any,
    run_standing_goal_judge_fn: Callable[[Any], Awaitable[Any]],
    inject_continuation_phases_fn: Callable[[Any, str], Awaitable[int]],
    loop_decision_continue: Any,
    loop_decision_complete: Any,
    blocked_decisions: set[Any],
) -> Any:
    if not get_settings().app.standing_goal_enabled:
        return decision
    if not plan.standing_objective.strip() or plan.standing_goal is None:
        return decision
    if not plan.standing_goal.is_active():
        return decision
    if plan.current_phase is not None:
        return decision
    if decision.decision in blocked_decisions:
        return decision

    verdict = await run_standing_goal_judge_fn(plan)
    decision.standing_goal_verdict = verdict.verdict_label
    decision.standing_goal_turn = verdict.turn
    decision.standing_goal_max_turns = verdict.max_turns
    record_standing_goal_history(plan.standing_goal, verdict=verdict)

    if verdict.done:
        plan.standing_goal.status = StandingGoalStatus.ACHIEVED
        decision.decision = loop_decision_complete
        decision.decision_reason = verdict.reason
        return decision

    _max_turns, max_phases = standing_goal_defaults()
    if plan.standing_goal.turn_count >= plan.standing_goal.max_turns:
        plan.standing_goal.status = StandingGoalStatus.BUDGET_EXHAUSTED
        decision.decision = loop_decision_complete
        decision.decision_reason = (
            f"Standing objective turn budget exhausted ({verdict.turn}/{verdict.max_turns}): "
            f"{verdict.reason}"
        )
        return decision

    if len(plan.phases) >= max_phases:
        plan.standing_goal.status = StandingGoalStatus.BUDGET_EXHAUSTED
        decision.decision = loop_decision_complete
        decision.decision_reason = (
            f"Standing objective phase cap reached ({len(plan.phases)}/{max_phases}): "
            f"{verdict.reason}"
        )
        return decision

    plan.standing_goal.turn_count += 1
    added = await inject_continuation_phases_fn(plan, verdict.reason)
    decision.continuation_phases_added = added
    plan.continuation_context = verdict.reason
    decision.decision = loop_decision_continue
    decision.decision_reason = (
        f"Continuing toward standing objective ({plan.standing_goal.turn_count}/"
        f"{plan.standing_goal.max_turns}): {verdict.reason}"
    )
    return decision
