from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def strip_fenced_json(raw_text: str) -> str:
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
    return cleaned


def parse_validation_contract(
    *,
    raw_contract: Any,
    validation_contract_ctor: type[Any],
) -> Any:
    from brain_os.systems.phase_contract import parse_validation_contract as _parse

    return _parse(raw_contract, validation_contract_ctor=validation_contract_ctor)


def ensure_validation_contract(
    *,
    contract: Any,
    phases: list[Any],
    validator_mode: str = "strict",
) -> Any:
    from brain_os.systems.phase_contract import ensure_validation_contract as _ensure

    return _ensure(contract, phases, validator_mode=validator_mode)


def parse_plan_with_fallback(
    *,
    request: str,
    raw_plan: str,
    validator_mode: str,
    max_phases: int,
    parse_validation_contract_fn: Any,
    ensure_validation_contract_fn: Any,
    phase_ctor: type[Any],
    plan_ctor: type[Any],
    validation_contract_ctor: type[Any],
) -> Any:
    data: dict[str, Any] | None = None
    cleaned = strip_fenced_json(raw_plan)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            data = parsed
    except json.JSONDecodeError:
        logger.warning("Could not parse plan JSON, using fallback plan")

    plan_id = f"plan_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}"
    if data:
        plan_id = str(data.get("plan_id", plan_id))

    goal = request
    complexity = "moderate"
    raw_phases: list[dict[str, Any]] = []
    validation_contract = validation_contract_ctor()
    if data:
        goal = str(data.get("goal") or request)
        complexity = str(data.get("complexity") or "moderate")
        validation_contract = parse_validation_contract_fn(data.get("validation_contract"))
        phases_candidate = data.get("phases", [])
        if isinstance(phases_candidate, list):
            raw_phases = [phase for phase in phases_candidate if isinstance(phase, dict)]

    phases: list[Any] = []
    for idx, phase_data in enumerate(raw_phases, start=1):
        phase_id = phase_data.get("id", idx)
        if not isinstance(phase_id, int):
            phase_id = idx
        agents = phase_data.get("agents", [])
        if not isinstance(agents, list):
            agents = []
        agents = [str(agent).lower() for agent in agents if str(agent).strip()] or ["clio"]
        depends_on = phase_data.get("depends_on", [])
        if not isinstance(depends_on, list):
            depends_on = []
        phases.append(
            phase_ctor(
                id=phase_id,
                title=str(phase_data.get("title") or f"Phase {idx}"),
                description=str(phase_data.get("description") or ""),
                agents=agents,
                delegation_type=str(phase_data.get("delegation_type") or "generic"),
                expected_output=str(phase_data.get("expected_output") or ""),
                depends_on=[dep for dep in depends_on if isinstance(dep, int)],
            )
        )

    if not phases:
        phases = [
            phase_ctor(
                id=1,
                title="Research and Analyze",
                description=request,
                agents=["clio"],
                delegation_type="generic",
                expected_output="Grounded findings for the request",
            ),
            phase_ctor(
                id=2,
                title="Synthesize Report",
                description="Compile findings into final response",
                agents=["calliope"],
                delegation_type="generic",
                expected_output="Final report",
                depends_on=[1],
            ),
        ]

    validation_contract = ensure_validation_contract_fn(
        validation_contract,
        phases,
        validator_mode=validator_mode,
    )
    return plan_ctor(
        plan_id=plan_id,
        goal=goal,
        original_request=request,
        phases=phases[:max_phases],
        complexity=complexity,
        status="created",
        validation_contract=validation_contract,
    )


async def run_phase_validator(
    *,
    plan: Any,
    phase: Any,
    responses: dict[str, str],
    athena: Any,
    phase_validation_result_ctor: type[Any],
) -> Any:
    from brain_os.systems.phase_contract import run_phase_validator as _run

    return await _run(
        plan=plan,
        phase=phase,
        responses=responses,
        athena=athena,
        phase_validation_result_ctor=phase_validation_result_ctor,
    )


async def observe_phase_decision(
    *,
    plan: Any,
    phase: Any,
    responses: dict[str, str],
    athena: Any,
    phase_result_ctor: type[Any],
    loop_decision_complete: Any,
    loop_decision_continue: Any,
    decision_map: dict[str, Any],
) -> Any:
    if plan.is_complete:
        return phase_result_ctor(
            phase_id=phase.id,
            agent_responses=responses,
            decision=loop_decision_complete,
        )

    observation_prompt = f"""You just completed Phase {phase.id}: "{phase.title}"

Agent responses:
{json.dumps(responses, indent=2)}

Remaining phases: {[phase_item.title for phase_item in plan.phases if phase_item.status.value == "pending"]}

Based on these results, what should we do next?
Return a JSON object:
{{
    "decision": "continue|replan|clarify|complete",
    "reason": "Brief explanation",
    "clarification_question": "Only if decision is 'clarify'"
}}

RULES:
- "continue" if results are sufficient and next phase should proceed
- "replan" if results reveal the remaining phases need adjustment
- "clarify" if results are ambiguous and user input is needed
- "complete" if all necessary information has been gathered
"""

    try:
        eval_json = await athena.call_llm(
            "You are Athena evaluating phase results. Be decisive.",
            observation_prompt,
            temperature=0.1,
        )
        eval_data = json.loads(eval_json)
        decision_key = str(eval_data.get("decision", "continue")).strip().lower()
        return phase_result_ctor(
            phase_id=phase.id,
            agent_responses=responses,
            decision=decision_map.get(decision_key, loop_decision_continue),
            decision_reason=eval_data.get("reason", ""),
            clarification_question=eval_data.get("clarification_question", ""),
        )
    except Exception:
        return phase_result_ctor(
            phase_id=phase.id,
            agent_responses=responses,
            decision=loop_decision_continue,
            decision_reason="Observation parse failed, continuing",
        )
