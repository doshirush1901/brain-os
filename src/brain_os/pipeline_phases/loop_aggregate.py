from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def record_assertion_metrics(*, runtime_metrics_module: Any, phase: Any) -> None:
    total = len(phase.assertion_coverage_delta.assertion_ids)
    if total <= 0:
        return
    runtime_metrics_module.incr("assertions_total", total)

    if phase.validation_result.passed:
        runtime_metrics_module.incr("assertions_passed", total)
        return

    failed_ids = phase.validation_result.failed_assertion_ids
    if not failed_ids:
        return
    passed = max(0, total - len(set(failed_ids)))
    if passed > 0:
        runtime_metrics_module.incr("assertions_passed", passed)


def record_decision_metrics(
    *,
    runtime_metrics_module: Any,
    decision: Any,
    loop_decision_replan: Any,
    loop_decision_clarify: Any,
) -> None:
    if decision.decision == loop_decision_replan:
        runtime_metrics_module.incr("replans_triggered")
    elif decision.decision == loop_decision_clarify:
        runtime_metrics_module.incr("clarifications_triggered")


def build_assertion_coverage_delta(
    *,
    plan: Any,
    phase: Any,
    assertion_coverage_delta_ctor: type[Any],
) -> Any:
    assertion_ids = plan.validation_contract.phase_assertion_map.get(phase.id, [])
    assertion_texts: list[str] = []
    for assertion_id in assertion_ids:
        index = assertion_id - 1
        if 0 <= index < len(plan.validation_contract.assertions):
            assertion_texts.append(plan.validation_contract.assertions[index])
    return assertion_coverage_delta_ctor(
        phase_id=phase.id,
        assertion_ids=assertion_ids,
        assertions=assertion_texts,
    )


def build_phase_handoff(
    *,
    phase: Any,
    decision: Any,
    agent_responses: dict[str, str],
    validation_result: Any,
    collect_open_issues_fn: Any,
    format_evidence_refs_fn: Any,
    build_command_specs_fn: Any,
    command_run_ctor: type[Any],
    phase_handoff_ctor: type[Any],
) -> Any:
    open_issues = collect_open_issues_fn(
        agent_responses,
        decision_value=decision.decision.value,
        decision_reason=decision.decision_reason,
        clarification_question=decision.clarification_question,
        validation_passed=validation_result.passed,
        validation_summary=validation_result.summary or "",
        missing_evidence=validation_result.missing_evidence,
        contradictions=validation_result.contradictions,
    )
    evidence_refs = format_evidence_refs_fn(agent_responses)
    command_specs = build_command_specs_fn(
        agent_responses,
        decision_value=decision.decision.value,
        validation_passed=validation_result.passed,
    )
    commands_run = [
        command_run_ctor(command=command, exit_code=exit_code)
        for command, exit_code in command_specs
    ]
    return phase_handoff_ctor(
        completed_work=phase.expected_output or phase.title,
        commands_run=commands_run,
        open_issues=open_issues,
        assertions_claimed=phase.assertion_coverage_delta.assertion_ids,
        evidence_refs=evidence_refs,
    )


def get_contract_summary(*, plan: Any) -> dict[str, Any]:
    assertions = plan.validation_contract.assertions
    matrix: list[dict[str, Any]] = []
    passed_count = 0
    failed_count = 0

    phases_by_id = {phase.id: phase for phase in plan.phases}
    for assertion_id, assertion_text in enumerate(assertions, start=1):
        mapped_phase_ids = sorted(
            phase_id
            for phase_id, ids in plan.validation_contract.phase_assertion_map.items()
            if assertion_id in ids
        )
        mapped_phases = [
            phases_by_id[phase_id] for phase_id in mapped_phase_ids if phase_id in phases_by_id
        ]
        evidence_refs: list[str] = []
        status = "unresolved"
        latest_outcome: tuple[int, str] | None = None

        for phase in mapped_phases:
            evidence_refs.extend(phase.handoff.evidence_refs[:2])
            phase_claims = assertion_id in phase.assertion_coverage_delta.assertion_ids
            if not phase_claims:
                continue
            if not phase.validation_result.passed:
                failed_ids = phase.validation_result.failed_assertion_ids
                if not failed_ids or assertion_id in failed_ids:
                    latest_outcome = (phase.id, "failed")
            elif phase.status.value == "completed":
                latest_outcome = (phase.id, "passed")

        if latest_outcome is not None:
            status = latest_outcome[1]
        elif not mapped_phase_ids:
            status = "unassigned"

        if status == "passed":
            passed_count += 1
        elif status == "failed":
            failed_count += 1

        matrix.append(
            {
                "id": assertion_id,
                "assertion": assertion_text,
                "status": status,
                "phase_ids": mapped_phase_ids,
                "evidence_refs": list(dict.fromkeys(evidence_refs))[:4],
            }
        )

    unresolved_entries = [
        entry for entry in matrix if entry["status"] in {"unresolved", "unassigned"}
    ]
    return {
        "validator_mode": plan.validation_contract.mode,
        "assertions_total": len(assertions),
        "assertions_passed": passed_count,
        "assertions_failed": failed_count,
        "assertions_unresolved": len(unresolved_entries),
        "unresolved_assertions": [
            {"id": entry["id"], "assertion": entry["assertion"], "status": entry["status"]}
            for entry in unresolved_entries
        ],
        "matrix": matrix,
    }


def build_contract_coverage_markdown(*, contract_summary: dict[str, Any]) -> str:
    matrix = contract_summary.get("matrix", [])
    lines = [
        "## Validation Contract Coverage",
        "",
        f"- Validator mode: {contract_summary.get('validator_mode', 'strict')}",
        f"- Assertions total: {contract_summary.get('assertions_total', 0)}",
        f"- Assertions passed: {contract_summary.get('assertions_passed', 0)}",
        f"- Assertions failed: {contract_summary.get('assertions_failed', 0)}",
        f"- Assertions unresolved: {contract_summary.get('assertions_unresolved', 0)}",
        "",
        "| # | Assertion | Status | Phases | Evidence refs |",
        "|---|---|---|---|---|",
    ]

    for entry in matrix:
        phase_text = ", ".join(str(phase_id) for phase_id in entry.get("phase_ids", [])) or "-"
        evidence = "; ".join(entry.get("evidence_refs", [])[:2]) or "-"
        assertion_text = str(entry.get("assertion", "")).replace("\n", " ").strip()[:160]
        lines.append(
            f"| {entry.get('id')} | {assertion_text} | {entry.get('status')} | {phase_text} | {evidence} |"
        )

    unresolved = contract_summary.get("unresolved_assertions", [])
    if unresolved:
        lines.extend(["", "### Unresolved Assertions"])
        for item in unresolved:
            lines.append(f"- [{item.get('status')}] {item.get('id')}: {item.get('assertion')}")
    return "\n".join(lines)


def plan_to_snapshot(*, plan: Any) -> dict[str, Any]:
    return {
        "goal": plan.goal,
        "original_request": plan.original_request,
        "standing_objective": plan.standing_objective,
        "continuation_context": plan.continuation_context,
        "complexity": plan.complexity,
        "status": plan.status,
        "created_at": plan.created_at,
        "revision_count": plan.revision_count,
        "standing_goal": plan.standing_goal.to_dict() if plan.standing_goal else None,
        "assertion_retry_counts": plan.assertion_retry_counts,
        "validation_contract": {
            "assertions": plan.validation_contract.assertions,
            "phase_assertion_map": {
                str(key): value
                for key, value in plan.validation_contract.phase_assertion_map.items()
            },
            "completion_gate": plan.validation_contract.completion_gate,
            "mode": plan.validation_contract.mode,
        },
        "phases": [
            {
                "id": phase.id,
                "title": phase.title,
                "description": phase.description,
                "agents": phase.agents,
                "delegation_type": phase.delegation_type,
                "expected_output": phase.expected_output,
                "depends_on": phase.depends_on,
                "status": phase.status.value,
                "result": phase.result,
            }
            for phase in plan.phases
        ],
    }


def plan_from_snapshot(
    *,
    snapshot: dict[str, Any],
    phase_ctor: type[Any],
    phase_status_enum: type[Any],
    parse_validation_contract_fn: Any,
    validation_contract_ctor: type[Any],
    standing_goal_state: type[Any],
    plan_ctor: type[Any],
) -> Any | None:
    plan_id = str(snapshot.get("plan_id") or "")
    if not plan_id:
        return None

    phases: list[Any] = []
    for phase_data in snapshot.get("phases", []):
        if not isinstance(phase_data, dict):
            continue
        status_raw = str(phase_data.get("status") or phase_status_enum.PENDING.value)
        try:
            status = phase_status_enum(status_raw)
        except ValueError:
            status = phase_status_enum.PENDING
        phases.append(
            phase_ctor(
                id=int(phase_data.get("id") or len(phases) + 1),
                title=str(phase_data.get("title") or ""),
                description=str(phase_data.get("description") or ""),
                agents=[
                    str(agent).lower()
                    for agent in phase_data.get("agents", [])
                    if str(agent).strip()
                ]
                or ["clio"],
                delegation_type=str(phase_data.get("delegation_type") or "generic"),
                expected_output=str(phase_data.get("expected_output") or ""),
                depends_on=[
                    dep for dep in phase_data.get("depends_on", []) if isinstance(dep, int)
                ],
                status=status,
                result=str(phase_data.get("result") or ""),
            )
        )

    raw_validation_contract = snapshot.get("validation_contract")
    validation_contract = (
        parse_validation_contract_fn(raw_validation_contract)
        if isinstance(raw_validation_contract, dict)
        else validation_contract_ctor()
    )

    raw_standing_goal = snapshot.get("standing_goal")
    standing_goal = (
        standing_goal_state.from_dict(raw_standing_goal)
        if isinstance(raw_standing_goal, dict)
        else None
    )

    raw_retry_counts = snapshot.get("assertion_retry_counts")
    retry_counts: dict[int, int] = {}
    if isinstance(raw_retry_counts, dict):
        for key, value in raw_retry_counts.items():
            try:
                retry_counts[int(key)] = int(value)
            except (TypeError, ValueError):
                continue

    return plan_ctor(
        plan_id=plan_id,
        goal=str(snapshot.get("goal") or ""),
        original_request=str(snapshot.get("original_request") or ""),
        phases=phases,
        complexity=str(snapshot.get("complexity") or "moderate"),
        status=str(snapshot.get("status") or "created"),
        created_at=str(snapshot.get("created_at") or datetime.now(UTC).isoformat()),
        revision_count=int(snapshot.get("revision_count") or 0),
        validation_contract=validation_contract,
        assertion_retry_counts=retry_counts,
        standing_objective=str(snapshot.get("standing_objective") or ""),
        standing_goal=standing_goal,
        continuation_context=str(snapshot.get("continuation_context") or ""),
    )
