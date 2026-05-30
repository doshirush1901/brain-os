"""Shared phase validation contracts for AgentLoop and TaskOrchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from brain_os.config import get_settings
from brain_os.schemas.llm_outputs import TaskPlan, TaskPlanPhase

logger = logging.getLogger(__name__)

_PHASE_VALIDATOR_ERRORS = (
    json.JSONDecodeError,
    ValueError,
    KeyError,
    TypeError,
    asyncio.TimeoutError,
    httpx.HTTPError,
)

_VALIDATOR_MODES = frozenset({"strict", "crm_only", "relaxed_evidence"})

# Substrings that must not appear in validator prompts (builder trace anti-pattern).
_FORBIDDEN_VALIDATOR_PROMPT_MARKERS = (
    "BRAIN_TOOL_META",
    "react_observation",
    "tool_call_id",
    "<<<HANDOFF_BRIEF>>>",
)


@dataclass
class ValidationContract:
    assertions: list[str] = field(default_factory=list)
    phase_assertion_map: dict[int, list[int]] = field(default_factory=dict)
    completion_gate: str = "All critical assertions must pass before completion."
    mode: str = "strict"


@dataclass
class AssertionCoverageDelta:
    phase_id: int = 0
    assertion_ids: list[int] = field(default_factory=list)
    assertions: list[str] = field(default_factory=list)


@dataclass
class PhaseValidationResult:
    passed: bool = True
    summary: str = ""
    failed_assertion_ids: list[int] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)


@dataclass
class CommandRun:
    command: str
    exit_code: int


@dataclass
class PhaseHandoff:
    completed_work: str = ""
    commands_run: list[CommandRun] = field(default_factory=list)
    open_issues: list[str] = field(default_factory=list)
    assertions_claimed: list[int] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)


@dataclass
class TaskPhaseValidationContext:
    """Minimal phase view for :func:`run_phase_validator` from CLI tasks."""

    id: int
    title: str
    description: str
    expected_output: str
    assertion_coverage_delta: AssertionCoverageDelta


@dataclass
class TaskValidationPlan:
    validation_contract: ValidationContract


def phase_validator_fail_open() -> bool:
    """When True, validator LLM errors allow the phase to continue (legacy default)."""
    return bool(get_settings().app.phase_validator_fail_open)


def parse_validation_contract(
    raw_contract: Any,
    *,
    validation_contract_ctor: type[Any] | None = None,
) -> ValidationContract:
    ctor = validation_contract_ctor or ValidationContract
    if not isinstance(raw_contract, dict):
        return ctor()

    raw_assertions = raw_contract.get("assertions", [])
    assertions = [str(item).strip() for item in raw_assertions if str(item).strip()]

    raw_map = raw_contract.get("phase_assertion_map", {})
    phase_assertion_map: dict[int, list[int]] = {}
    if isinstance(raw_map, dict):
        for key, value in raw_map.items():
            try:
                phase_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, list):
                assertion_ids = [int(v) for v in value if isinstance(v, int) or str(v).isdigit()]
                if assertion_ids:
                    phase_assertion_map[phase_id] = assertion_ids

    completion_gate = str(raw_contract.get("completion_gate") or "").strip()
    if not completion_gate:
        completion_gate = "All critical assertions must pass before completion."
    mode = str(raw_contract.get("validator_mode") or raw_contract.get("mode") or "strict").strip()
    if mode not in _VALIDATOR_MODES:
        mode = "strict"

    return ctor(
        assertions=assertions,
        phase_assertion_map=phase_assertion_map,
        completion_gate=completion_gate,
        mode=mode,
    )


def ensure_validation_contract(
    contract: ValidationContract,
    phases: list[Any],
    *,
    validator_mode: str = "strict",
) -> ValidationContract:
    if phases and not contract.assertions:
        contract.assertions = [
            f"Phase {getattr(phase, 'id', idx)} output delivered: "
            f"{getattr(phase, 'expected_output', None) or getattr(phase, 'title', '')}"
            for idx, phase in enumerate(phases, start=1)
        ]

    if phases and not contract.phase_assertion_map:
        contract.phase_assertion_map = {}
        for index, phase in enumerate(phases):
            phase_id = int(getattr(phase, "id", index + 1))
            if index < len(contract.assertions):
                contract.phase_assertion_map[phase_id] = [index + 1]

    if not contract.completion_gate:
        contract.completion_gate = "All critical assertions must pass before completion."
    mode = str(contract.mode or "strict").strip().lower()
    incoming = str(validator_mode or "").strip().lower()
    if incoming and incoming in _VALIDATOR_MODES:
        if incoming != "strict" or mode == "strict":
            mode = incoming
    if mode not in _VALIDATOR_MODES:
        mode = "strict"
    contract.mode = mode
    return contract


def build_assertion_coverage_delta(
    contract: ValidationContract,
    phase_id: int,
) -> AssertionCoverageDelta:
    assertion_ids = list(contract.phase_assertion_map.get(phase_id, []))
    assertion_texts: list[str] = []
    for assertion_id in assertion_ids:
        index = assertion_id - 1
        if 0 <= index < len(contract.assertions):
            assertion_texts.append(contract.assertions[index])
    return AssertionCoverageDelta(
        phase_id=phase_id,
        assertion_ids=assertion_ids,
        assertions=assertion_texts,
    )


def validation_contract_from_task_plan(
    plan: TaskPlan,
    *,
    validator_mode: str = "strict",
) -> ValidationContract:
    """Derive a validation contract from a CLI :class:`TaskPlan`."""
    raw = plan.validation_contract
    stubs = _task_plan_phases_stub(plan)
    if raw is not None and raw.assertions:
        contract = ValidationContract(
            assertions=list(raw.assertions),
            phase_assertion_map={int(k): list(v) for k, v in raw.phase_assertion_map.items()},
            completion_gate=str(raw.completion_gate or "").strip()
            or "All critical assertions must pass before completion.",
            mode=str(raw.validator_mode or validator_mode).strip(),
        )
        return ensure_validation_contract(contract, stubs, validator_mode=validator_mode)

    contract = ValidationContract()
    return ensure_validation_contract(contract, stubs, validator_mode=validator_mode)


def _task_plan_phases_stub(plan: TaskPlan) -> list[TaskPhaseValidationContext]:
    stubs: list[TaskPhaseValidationContext] = []
    for idx, phase in enumerate(plan.phases):
        phase_id = idx + 1
        stubs.append(
            TaskPhaseValidationContext(
                id=phase_id,
                title=phase.title,
                description=phase.description,
                expected_output=phase.description or phase.title,
                assertion_coverage_delta=AssertionCoverageDelta(),
            )
        )
    return stubs


def task_phase_validation_context(
    phase_index: int,
    phase: TaskPlanPhase,
    contract: ValidationContract,
) -> TaskPhaseValidationContext:
    phase_id = phase_index + 1
    delta = build_assertion_coverage_delta(contract, phase_id)
    return TaskPhaseValidationContext(
        id=phase_id,
        title=phase.title,
        description=phase.description,
        expected_output=phase.description or phase.title,
        assertion_coverage_delta=delta,
    )


def contract_to_dict(contract: ValidationContract) -> dict[str, Any]:
    return {
        "assertions": list(contract.assertions),
        "phase_assertion_map": {str(k): v for k, v in contract.phase_assertion_map.items()},
        "completion_gate": contract.completion_gate,
        "validator_mode": contract.mode,
    }


def validation_result_to_dict(result: PhaseValidationResult) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "summary": result.summary,
        "failed_assertion_ids": list(result.failed_assertion_ids),
        "missing_evidence": list(result.missing_evidence),
        "contradictions": list(result.contradictions),
    }


def build_long_account_journey_assertions(
    *,
    company: str,
    contact_email: str | None = None,
) -> list[str]:
    """Canonical assertions for long account-journey loops (mail + CRM + evidence + draft prep)."""
    who = contact_email.strip() if isinstance(contact_email, str) else ""
    scope = f"{company} ({who})" if who else company
    return [
        f"Mailbox journey for {scope} covers full available history with chronological events.",
        "Timeline includes machine-model and commercial offer milestones with evidence snippets.",
        "Contradictions or fit mismatches (e.g., roll-fed vs sheet-fed) are called out explicitly.",
        "Meeting context packet includes current need, constraints, objections, and agenda.",
        "Draft recommendations remain draft-only and require explicit send approval.",
    ]


def build_validator_prompt(
    *,
    phase: Any,
    responses: dict[str, str],
    mode: str,
) -> str:
    """Build validator prompt; must contain only agent response text, not ReAct traces."""
    return f"""Validate the phase output against the contract assertions.

PHASE: {phase.id} - {phase.title}
PHASE OBJECTIVE: {phase.description}
EXPECTED OUTPUT: {phase.expected_output}

ASSERTIONS TO VALIDATE:
{
        json.dumps(
            [
                {"id": assertion_id, "assertion": assertion}
                for assertion_id, assertion in zip(
                    phase.assertion_coverage_delta.assertion_ids,
                    phase.assertion_coverage_delta.assertions,
                    strict=False,
                )
            ],
            indent=2,
        )
    }

AGENT RESPONSES:
{json.dumps(responses, indent=2)}

VALIDATOR MODE: {mode}
MODE RULES:
- strict: require robust evidence and flag contradictions.
- relaxed_evidence: allow operational summaries with caveats when evidence is partial.

Return JSON only:
{{
  "passed": true|false,
  "summary": "One-line verdict",
  "failed_assertion_ids": [1],
  "missing_evidence": ["what evidence is missing"],
  "contradictions": ["any contradictory claims"]
}}
"""


def assert_validator_prompt_excludes_traces(prompt: str) -> None:
    """Guardrail: validator must not receive builder ReAct traces."""
    lowered = prompt.lower()
    for marker in _FORBIDDEN_VALIDATOR_PROMPT_MARKERS:
        if marker.lower() in lowered:
            msg = f"Validator prompt must not include trace marker: {marker}"
            raise ValueError(msg)


async def run_phase_validator(
    *,
    plan: Any,
    phase: Any,
    responses: dict[str, str],
    athena: Any,
    phase_validation_result_ctor: type[Any] | None = None,
) -> PhaseValidationResult:
    ctor = phase_validation_result_ctor or PhaseValidationResult
    mode = str(plan.validation_contract.mode or "strict").strip().lower()
    if mode not in _VALIDATOR_MODES:
        mode = "strict"

    if not phase.assertion_coverage_delta.assertion_ids:
        return ctor(
            passed=True,
            summary="No assertions mapped to this phase.",
        )

    if mode == "crm_only":
        if not responses:
            return ctor(
                passed=False,
                summary="CRM-only validator: no agent responses available.",
                failed_assertion_ids=list(phase.assertion_coverage_delta.assertion_ids),
                missing_evidence=["No phase responses available for CRM-only validation."],
            )
        all_failed = all(str(value).startswith("(Error:") for value in responses.values())
        if all_failed:
            return ctor(
                passed=False,
                summary="CRM-only validator: all agent responses were errors.",
                failed_assertion_ids=list(phase.assertion_coverage_delta.assertion_ids),
                missing_evidence=["All delegated agent responses failed."],
            )
        return ctor(
            passed=True,
            summary=(
                "CRM-only validator mode: accepted based on available agent/tool outputs "
                "(archive cross-check skipped)."
            ),
        )

    validator_prompt = build_validator_prompt(phase=phase, responses=responses, mode=mode)
    assert_validator_prompt_excludes_traces(validator_prompt)

    try:
        validator_json = await athena.call_llm(
            "You are a strict validator with fresh context. "
            "Assess only whether required assertions are satisfied by evidence.",
            validator_prompt,
            temperature=0.0,
            model_profile="verifier",
        )
        parsed = json.loads(validator_json)
        if not isinstance(parsed, dict):
            raise ValueError("Validator response is not a JSON object")

        passed = bool(parsed.get("passed", True))
        failed_ids_raw = parsed.get("failed_assertion_ids", [])
        missing_raw = parsed.get("missing_evidence", [])
        contradictions_raw = parsed.get("contradictions", [])

        failed_ids = [
            int(value) for value in failed_ids_raw if isinstance(value, int) or str(value).isdigit()
        ]
        missing_evidence = [str(value) for value in missing_raw if str(value).strip()]
        contradictions = [str(value) for value in contradictions_raw if str(value).strip()]
        summary = str(parsed.get("summary") or "").strip()
        if mode == "relaxed_evidence" and not passed:
            has_material = any(
                str(value).strip() and not str(value).startswith("(Error:")
                for value in responses.values()
            )
            if has_material:
                caveats: list[str] = []
                if failed_ids:
                    caveats.append(f"failed_assertions={failed_ids}")
                if missing_evidence:
                    caveats.append(f"missing_evidence={len(missing_evidence)}")
                if contradictions:
                    caveats.append(f"contradictions={len(contradictions)}")
                summary = "Relaxed-evidence validator accepted phase with caveats" + (
                    f" ({', '.join(caveats)})." if caveats else "."
                )
                passed = True
                failed_ids = []

        return ctor(
            passed=passed,
            summary=summary,
            failed_assertion_ids=failed_ids,
            missing_evidence=missing_evidence,
            contradictions=contradictions,
        )
    except _PHASE_VALIDATOR_ERRORS:
        logger.exception("Phase validator failed for phase %s", getattr(phase, "id", "?"))
        if phase_validator_fail_open() and mode == "strict":
            return ctor(
                passed=True,
                summary="Validator unavailable; allowing phase to continue.",
            )
        return ctor(
            passed=False,
            summary="Validator unavailable; phase blocked (APP__PHASE_VALIDATOR_FAIL_OPEN=false).",
            failed_assertion_ids=list(phase.assertion_coverage_delta.assertion_ids),
            missing_evidence=["Validator LLM call failed."],
        )
