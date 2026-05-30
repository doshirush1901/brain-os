"""
pipeline_loop.py — Iterative Agent Loop (OpenManus-inspired)

This module adds an iterative Plan-Execute-Observe loop on top of Brain OS's
existing RequestPipeline. Instead of a single fire-and-forget pass, the
loop allows Athena to:

  1. Plan: Break the request into phases
  2. Execute: Run one phase at a time via the Pantheon
  3. Observe: Evaluate the result and decide whether to continue, re-plan,
     or request clarification
  4. Compile: Synthesize all phase results into a final output

This is designed to be called from the MCP server's plan_task / execute_phase
tools, or directly from the API server for streaming use cases.

Integration:
    This does NOT replace RequestPipeline. It wraps it. The existing pipeline
    handles the per-turn processing (classify → route → respond → learn).
    This loop handles multi-turn orchestration across phases.

Usage:
    from brain_os.pipeline_loop import AgentLoop
    loop = AgentLoop(pantheon)
    plan = await loop.plan(request)
    for phase in plan.phases:
        result = await loop.execute_phase(phase)
        if result.needs_replan:
            plan = await loop.replan(plan, result)
    report = await loop.compile(plan)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from brain_os.brain import runtime_metrics
from brain_os.pipeline_phases.execute import (
    build_athena_delegation_prompt,
    build_phase_handoff_command_specs,
    collect_phase_handoff_open_issues,
    format_phase_handoff_evidence_refs,
    format_prior_phase_context_lines,
    phase_completed_progress_event,
    phase_execution_policy_progress_event,
    phase_started_progress_event,
)
from brain_os.pipeline_phases.loop_aggregate import (
    build_assertion_coverage_delta,
    build_contract_coverage_markdown,
    build_phase_handoff,
    get_contract_summary,
    plan_from_snapshot,
    plan_to_snapshot,
    record_assertion_metrics,
    record_decision_metrics,
)
from brain_os.pipeline_phases.loop_control import (
    collect_phase_agent_responses,
    inject_continuation_phases,
    maybe_apply_standing_goal,
    merge_standing_objective_into_contract,
    phase_allows_parallel_read_only,
    run_autonomous_loop,
    run_single_phase_agent,
    run_standing_goal_judge,
    standing_goal_blocks_compile,
)
from brain_os.pipeline_phases.loop_error import (
    ensure_validation_contract,
    observe_phase_decision,
    parse_plan_with_fallback,
    parse_validation_contract,
    run_phase_validator,
)
from brain_os.pipeline_phases.replan import (
    build_assertion_escalation_clarification_question,
    build_corrective_phase_kwargs,
    format_athena_replan_user_prompt,
    format_completed_phases_summary,
    format_validation_contract_failure_reason,
    merge_assertion_retry_counts,
    preferred_corrective_agents,
    resolve_assertion_text,
)
from brain_os.systems.phase_contract import (
    AssertionCoverageDelta,
    CommandRun,
    PhaseHandoff,
    PhaseValidationResult,
    ValidationContract,
)
from brain_os.systems.standing_goal import (
    StandingGoalJudge,
    StandingGoalState,
    StandingGoalStatus,
    StandingGoalStore,
    StandingGoalVerdict,
    standing_goal_defaults,
)

logger = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────────────
class PhaseStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_CLARIFICATION = "needs_clarification"
    REPLANNED = "replanned"


class LoopDecision(Enum):
    CONTINUE = "continue"  # Proceed to next phase
    REPLAN = "replan"  # Athena should revise the remaining plan
    CLARIFY = "clarify"  # Need user input before continuing
    COMPLETE = "complete"  # All phases done, compile results
    ABORT = "abort"  # Unrecoverable error


@dataclass
class Phase:
    id: int
    title: str
    description: str
    agents: list[str]
    delegation_type: str = "generic"
    expected_output: str = ""
    depends_on: list[int] = field(default_factory=list)
    status: PhaseStatus = PhaseStatus.PENDING
    result: str = ""
    error: str = ""
    raw_agent_outputs: dict[str, str] = field(default_factory=dict)
    handoff: PhaseHandoff = field(default_factory=lambda: PhaseHandoff())
    assertion_coverage_delta: AssertionCoverageDelta = field(
        default_factory=lambda: AssertionCoverageDelta()
    )
    validation_result: PhaseValidationResult = field(
        default_factory=lambda: PhaseValidationResult()
    )


@dataclass
class Plan:
    plan_id: str
    goal: str
    original_request: str
    phases: list[Phase]
    complexity: str = "moderate"
    status: str = "created"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    revision_count: int = 0
    validation_contract: ValidationContract = field(default_factory=lambda: ValidationContract())
    assertion_retry_counts: dict[int, int] = field(default_factory=dict)
    standing_objective: str = ""
    standing_goal: StandingGoalState | None = None
    continuation_context: str = ""

    @property
    def current_phase(self) -> Phase | None:
        for phase in self.phases:
            if phase.status == PhaseStatus.PENDING:
                return phase
        return None

    @property
    def completed_phases(self) -> list[Phase]:
        return [p for p in self.phases if p.status == PhaseStatus.COMPLETED]

    @property
    def is_complete(self) -> bool:
        return all(p.status == PhaseStatus.COMPLETED for p in self.phases)


@dataclass
class PhaseResult:
    phase_id: int
    agent_responses: dict[str, str]
    decision: LoopDecision
    decision_reason: str = ""
    clarification_question: str = ""
    failed_assertion_ids: list[int] = field(default_factory=list)
    standing_goal_verdict: str | None = None
    standing_goal_turn: int | None = None
    standing_goal_max_turns: int | None = None
    continuation_phases_added: int = 0


# ── Progress callback type ───────────────────────────────────────────
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]] | None


# ── Agent Loop ───────────────────────────────────────────────────────


class AgentLoop:
    """Iterative Plan-Execute-Observe loop for multi-phase task execution.

    This is the core of the Manus-like experience. It wraps the Pantheon
    and provides a structured way to plan, execute, observe, and compile
    multi-agent workflows.
    """

    MAX_PHASES = 10
    MAX_REPLANS = 3
    ASSERTION_RETRY_LIMIT = 2

    def __init__(self, pantheon: Any) -> None:
        self._pantheon = pantheon
        self._athena = pantheon.get_agent("athena")
        self._plans: dict[str, Plan] = {}
        self._standing_goal_judge = StandingGoalJudge()

    def get_plan(self, plan_id: str) -> Plan | None:
        """Return a previously created plan by id."""
        plan = self._plans.get(plan_id)
        if plan is not None:
            return plan
        snapshot = StandingGoalStore.load_plan_snapshot(plan_id)
        if snapshot is None:
            return None
        loaded = self._plan_from_snapshot(snapshot)
        if loaded is not None:
            self._plans[plan_id] = loaded
        return loaded

    def cancel_plan(self, plan_id: str) -> bool:
        """Remove a plan from memory (operator abort / stuck loop). Returns True if it existed."""
        existed = plan_id in self._plans
        self._plans.pop(plan_id, None)
        StandingGoalStore.delete(plan_id)
        return existed

    def pause_standing_goal(self, plan_id: str) -> bool:
        plan = self.get_plan(plan_id)
        if plan is None or plan.standing_goal is None:
            return False
        plan.standing_goal.status = StandingGoalStatus.PAUSED
        self._persist_plan(plan)
        return True

    def resume_standing_goal(self, plan_id: str) -> bool:
        plan = self.get_plan(plan_id)
        if plan is None or plan.standing_goal is None:
            return False
        if plan.standing_goal.status in {
            StandingGoalStatus.PAUSED,
            StandingGoalStatus.BUDGET_EXHAUSTED,
        }:
            plan.standing_goal.status = StandingGoalStatus.ACTIVE
            self._persist_plan(plan)
            return True
        return False

    def clear_standing_goal(self, plan_id: str) -> bool:
        plan = self.get_plan(plan_id)
        if plan is None:
            return False
        if plan.standing_goal is not None:
            plan.standing_goal.status = StandingGoalStatus.CLEARED
        plan.standing_objective = ""
        plan.standing_goal = None
        plan.continuation_context = ""
        self._persist_plan(plan)
        return True

    def standing_goal_blocks_compile(self, plan: Plan) -> str | None:
        """Return an error message if compile must not run yet."""
        return standing_goal_blocks_compile(plan=plan)

    # ── Step 1: Plan ─────────────────────────────────────────────────

    async def plan(
        self,
        request: str,
        complexity: str = "auto",
        validator_mode: str = "strict",
        standing_objective: str | None = None,
        max_continuation_turns: int | None = None,
        on_progress: ProgressCallback = None,
    ) -> Plan:
        """Analyze a request and create a structured execution plan.

        Uses Athena to break down the request into phases, each mapped
        to the appropriate specialist agents.
        """
        if on_progress:
            await on_progress({"type": "planning", "status": "started"})

        objective_text = (standing_objective or "").strip()
        planning_prompt = self._build_planning_prompt(
            request,
            complexity,
            standing_objective=objective_text,
        )

        plan_json = await self._athena.call_llm(
            "You are Athena, the CEO/Orchestrator. Create a precise "
            "execution plan for this business query.",
            planning_prompt,
            temperature=0.2,
        )

        plan = self._parse_plan(request, plan_json, validator_mode=validator_mode)
        if objective_text:
            max_turns, _max_phases = standing_goal_defaults()
            if max_continuation_turns is not None:
                max_turns = max(1, min(max_continuation_turns, 100))
            plan.standing_objective = objective_text
            plan.standing_goal = StandingGoalState(
                objective=objective_text,
                max_turns=max_turns,
            )
            self._merge_standing_objective_into_contract(plan)
        self._plans[plan.plan_id] = plan
        self._persist_plan(plan)

        if on_progress:
            await on_progress(
                {
                    "type": "planning",
                    "status": "completed",
                    "plan_id": plan.plan_id,
                    "phase_count": len(plan.phases),
                }
            )

        return plan

    # ── Step 2: Execute Phase ────────────────────────────────────────

    async def execute_phase(
        self,
        plan: Plan,
        phase: Phase | None = None,
        on_progress: ProgressCallback = None,
    ) -> PhaseResult:
        """Execute a single phase of the plan.

        Runs the assigned agents with structured delegation prompts,
        collects responses, and uses Athena to decide the next action.
        """
        if phase is None:
            phase = plan.current_phase
        if phase is None:
            return PhaseResult(
                phase_id=0,
                agent_responses={},
                decision=LoopDecision.COMPLETE,
                decision_reason="All phases already completed",
            )
        running_phase = next(
            (p for p in plan.phases if p.status == PhaseStatus.RUNNING and p.id != phase.id),
            None,
        )
        if running_phase is not None:
            return PhaseResult(
                phase_id=phase.id,
                agent_responses={},
                decision=LoopDecision.ABORT,
                decision_reason=(
                    "Another mutating phase is already running; "
                    f"phase {running_phase.id} must complete first."
                ),
            )

        phase.status = PhaseStatus.RUNNING

        if on_progress:
            await on_progress(
                phase_started_progress_event(
                    phase_id=phase.id,
                    title=phase.title,
                    agents=phase.agents,
                )
            )

        # Build context from completed phases
        prior_context = self._build_prior_context(plan, phase)

        if on_progress:
            await on_progress(
                phase_execution_policy_progress_event(
                    phase_id=phase.id,
                    policy=(
                        "parallel_read_only"
                        if self._phase_allows_parallel_read_only(phase)
                        else "serial_mutating"
                    ),
                )
            )

        # Run agent work in serial by default; allow parallel fan-out only
        # for explicitly read-only phases.
        agent_responses = await self._collect_phase_agent_responses(
            phase=phase,
            prior_context=prior_context,
            on_progress=on_progress,
        )

        # Store results (status decided after observe step).
        phase.result = json.dumps(agent_responses)
        phase.raw_agent_outputs = dict(agent_responses)
        phase.assertion_coverage_delta = self._build_assertion_coverage_delta(plan, phase)
        phase.validation_result = await self._run_phase_validator(plan, phase, agent_responses)
        self._record_assertion_metrics(phase)

        # ── Step 3: Observe — let Athena decide what to do next ──────
        if phase.validation_result.passed:
            for assertion_id in phase.assertion_coverage_delta.assertion_ids:
                plan.assertion_retry_counts.pop(assertion_id, None)
            decision = await self._observe(plan, phase, agent_responses)
        else:
            failed_assertion_ids = (
                phase.validation_result.failed_assertion_ids
                or phase.assertion_coverage_delta.assertion_ids
            )
            new_counts, escalation_ids = merge_assertion_retry_counts(
                plan.assertion_retry_counts,
                failed_assertion_ids,
                retry_limit=self.ASSERTION_RETRY_LIMIT,
            )
            plan.assertion_retry_counts = new_counts

            reason_text = format_validation_contract_failure_reason(
                phase_summary=phase.validation_result.summary or "",
                failed_assertion_ids=failed_assertion_ids,
                missing_evidence=phase.validation_result.missing_evidence,
            )
            if escalation_ids:
                question = build_assertion_escalation_clarification_question(
                    plan.validation_contract.assertions,
                    escalation_ids,
                )
                decision = PhaseResult(
                    phase_id=phase.id,
                    agent_responses=agent_responses,
                    decision=LoopDecision.CLARIFY,
                    decision_reason=reason_text,
                    clarification_question=question,
                    failed_assertion_ids=failed_assertion_ids,
                )
            else:
                decision = PhaseResult(
                    phase_id=phase.id,
                    agent_responses=agent_responses,
                    decision=LoopDecision.REPLAN,
                    decision_reason=reason_text,
                    failed_assertion_ids=failed_assertion_ids,
                )
        if decision.decision == LoopDecision.CLARIFY:
            phase.status = PhaseStatus.NEEDS_CLARIFICATION
        elif decision.decision != LoopDecision.ABORT:
            phase.status = PhaseStatus.COMPLETED
        self._record_decision_metrics(decision)

        phase.handoff = self._build_phase_handoff(
            phase=phase,
            decision=decision,
            agent_responses=agent_responses,
            validation_result=phase.validation_result,
        )

        if on_progress:
            await on_progress(
                phase_completed_progress_event(
                    phase_id=phase.id,
                    decision_value=decision.decision.value,
                )
            )

        decision = await self._maybe_apply_standing_goal(plan, decision)
        self._persist_plan(plan)
        return decision

    def _record_assertion_metrics(self, phase: Phase) -> None:
        record_assertion_metrics(runtime_metrics_module=runtime_metrics, phase=phase)

    def _record_decision_metrics(self, decision: PhaseResult) -> None:
        record_decision_metrics(
            runtime_metrics_module=runtime_metrics,
            decision=decision,
            loop_decision_replan=LoopDecision.REPLAN,
            loop_decision_clarify=LoopDecision.CLARIFY,
        )

    def _phase_allows_parallel_read_only(self, phase: Phase) -> bool:
        return phase_allows_parallel_read_only(phase)

    async def _collect_phase_agent_responses(
        self,
        phase: Phase,
        prior_context: str,
        on_progress: ProgressCallback,
    ) -> dict[str, str]:
        return await collect_phase_agent_responses(
            phase=phase,
            prior_context=prior_context,
            on_progress=on_progress,
            run_single_phase_agent_fn=self._run_single_phase_agent,
        )

    async def _run_single_phase_agent(
        self,
        phase: Phase,
        prior_context: str,
        agent_name: str,
        on_progress: ProgressCallback,
    ) -> tuple[str, str]:
        return await run_single_phase_agent(
            pantheon=self._pantheon,
            phase=phase,
            prior_context=prior_context,
            agent_name=agent_name,
            on_progress=on_progress,
            build_delegation_prompt_fn=self._build_delegation_prompt,
        )

    def _build_assertion_coverage_delta(
        self,
        plan: Plan,
        phase: Phase,
    ) -> AssertionCoverageDelta:
        return build_assertion_coverage_delta(
            plan=plan,
            phase=phase,
            assertion_coverage_delta_ctor=AssertionCoverageDelta,
        )

    def _build_phase_handoff(
        self,
        phase: Phase,
        decision: PhaseResult,
        agent_responses: dict[str, str],
        validation_result: PhaseValidationResult,
    ) -> PhaseHandoff:
        return build_phase_handoff(
            phase=phase,
            decision=decision,
            agent_responses=agent_responses,
            validation_result=validation_result,
            collect_open_issues_fn=collect_phase_handoff_open_issues,
            format_evidence_refs_fn=format_phase_handoff_evidence_refs,
            build_command_specs_fn=build_phase_handoff_command_specs,
            command_run_ctor=CommandRun,
            phase_handoff_ctor=PhaseHandoff,
        )

    async def _run_phase_validator(
        self,
        plan: Plan,
        phase: Phase,
        responses: dict[str, str],
    ) -> PhaseValidationResult:
        return await run_phase_validator(
            plan=plan,
            phase=phase,
            responses=responses,
            athena=self._athena,
            phase_validation_result_ctor=PhaseValidationResult,
        )

    # ── Step 3: Observe (internal) ───────────────────────────────────

    async def _observe(
        self,
        plan: Plan,
        phase: Phase,
        responses: dict[str, str],
    ) -> PhaseResult:
        """Let Athena evaluate phase results and decide the next action.

        This is the key differentiator from a linear pipeline. After each
        phase, Athena reviews the results and can:
        - Continue to the next phase
        - Re-plan if new information changes the approach
        - Request clarification from the user
        - Mark the task as complete
        """
        return await observe_phase_decision(
            plan=plan,
            phase=phase,
            responses=responses,
            athena=self._athena,
            phase_result_ctor=PhaseResult,
            loop_decision_complete=LoopDecision.COMPLETE,
            loop_decision_continue=LoopDecision.CONTINUE,
            decision_map={
                "continue": LoopDecision.CONTINUE,
                "replan": LoopDecision.REPLAN,
                "clarify": LoopDecision.CLARIFY,
                "complete": LoopDecision.COMPLETE,
            },
        )

    # ── Step 4: Replan ───────────────────────────────────────────────

    async def replan(
        self,
        plan: Plan,
        trigger_result: PhaseResult,
        on_progress: ProgressCallback = None,
    ) -> Plan:
        """Revise the remaining phases based on new information.

        Called when the observe step returns REPLAN. Athena reviews
        what has been learned so far and adjusts the remaining phases.
        """
        if plan.revision_count >= self.MAX_REPLANS:
            logger.warning("Max replans reached for plan %s", plan.plan_id)
            return plan

        if on_progress:
            await on_progress({"type": "replanning", "reason": trigger_result.decision_reason})

        completed_summary = format_completed_phases_summary(
            [(p.id, p.title, str(p.result or "")) for p in plan.completed_phases]
        )

        replan_prompt = format_athena_replan_user_prompt(
            goal=plan.goal,
            completed_summary=completed_summary,
            decision_reason=trigger_result.decision_reason,
            failed_assertion_ids=list(trigger_result.failed_assertion_ids),
            next_start_phase_id=max(p.id for p in plan.phases) + 1,
        )

        try:
            new_phases_json = await self._athena.call_llm(
                "You are Athena revising an execution plan based on new findings.",
                replan_prompt,
                temperature=0.2,
            )
            new_phases_data = json.loads(new_phases_json)

            # Remove pending phases and add new ones
            plan.phases = [p for p in plan.phases if p.status != PhaseStatus.PENDING]
            start_id = max((p.id for p in plan.phases), default=0) + 1
            available = frozenset(self._pantheon.agents.keys())
            preferred = preferred_corrective_agents(available)
            for offset, assertion_id in enumerate(trigger_result.failed_assertion_ids):
                assertion_text = resolve_assertion_text(
                    plan.validation_contract.assertions,
                    assertion_id,
                )
                phase_id = start_id + offset
                kwargs = build_corrective_phase_kwargs(
                    assertion_id=assertion_id,
                    assertion_text=assertion_text,
                    phase_id=phase_id,
                    preferred_agents=preferred,
                )
                plan.phases.append(Phase(**kwargs))
                plan.validation_contract.phase_assertion_map[phase_id] = [assertion_id]

            next_id = max((p.id for p in plan.phases), default=0) + 1
            for pd in new_phases_data:
                if not isinstance(pd, dict):
                    continue
                phase_id = pd.get("id", next_id)
                if not isinstance(phase_id, int):
                    phase_id = next_id
                next_id = max(next_id + 1, phase_id + 1)
                plan.phases.append(
                    Phase(
                        id=phase_id,
                        title=str(pd.get("title", f"Replanned Phase {phase_id}")),
                        description=str(pd.get("description", "")),
                        agents=[
                            str(agent).lower()
                            for agent in pd.get("agents", [])
                            if str(agent).strip()
                        ]
                        or ["clio"],
                        delegation_type=str(pd.get("delegation_type", "generic")),
                        expected_output=str(pd.get("expected_output", "")),
                    )
                )

            plan.revision_count += 1

        except (json.JSONDecodeError, Exception):
            logger.exception("Replan failed for plan %s", plan.plan_id)

        return plan

    def get_contract_summary(self, plan: Plan) -> dict[str, Any]:
        return get_contract_summary(plan=plan)

    def _build_contract_coverage_markdown(self, contract_summary: dict[str, Any]) -> str:
        return build_contract_coverage_markdown(contract_summary=contract_summary)

    # ── Standing objectives ───────────────────────────────────────────

    def _merge_standing_objective_into_contract(self, plan: Plan) -> None:
        merge_standing_objective_into_contract(plan=plan)

    async def _maybe_apply_standing_goal(
        self,
        plan: Plan,
        decision: PhaseResult,
    ) -> PhaseResult:
        return await maybe_apply_standing_goal(
            plan=plan,
            decision=decision,
            run_standing_goal_judge_fn=self._run_standing_goal_judge,
            inject_continuation_phases_fn=self._inject_continuation_phases,
            loop_decision_continue=LoopDecision.CONTINUE,
            loop_decision_complete=LoopDecision.COMPLETE,
            blocked_decisions={
                LoopDecision.CLARIFY,
                LoopDecision.ABORT,
                LoopDecision.REPLAN,
            },
        )

    async def _run_standing_goal_judge(self, plan: Plan) -> StandingGoalVerdict:
        return await run_standing_goal_judge(
            plan=plan,
            standing_goal_judge=self._standing_goal_judge,
            get_contract_summary_fn=self.get_contract_summary,
        )

    async def _inject_continuation_phases(self, plan: Plan, judge_reason: str) -> int:
        return await inject_continuation_phases(
            plan=plan,
            judge_reason=judge_reason,
            athena=self._athena,
            available_agents=list(self._pantheon.agents.keys()),
            phase_ctor=Phase,
        )

    def _persist_plan(self, plan: Plan) -> None:
        if not plan.standing_objective.strip() and plan.standing_goal is None:
            return
        try:
            StandingGoalStore.save_plan_snapshot(plan.plan_id, self._plan_to_snapshot(plan))
        except Exception:
            logger.exception("Failed to persist standing goal snapshot for %s", plan.plan_id)

    def _plan_to_snapshot(self, plan: Plan) -> dict[str, Any]:
        return plan_to_snapshot(plan=plan)

    def _plan_from_snapshot(self, snapshot: dict[str, Any]) -> Plan | None:
        return plan_from_snapshot(
            snapshot=snapshot,
            phase_ctor=Phase,
            phase_status_enum=PhaseStatus,
            parse_validation_contract_fn=self._parse_validation_contract,
            validation_contract_ctor=ValidationContract,
            standing_goal_state=StandingGoalState,
            plan_ctor=Plan,
        )

    # ── Step 5: Compile ──────────────────────────────────────────────

    async def compile(
        self,
        plan: Plan,
        title: str = "Brain OS Report",
        on_progress: ProgressCallback = None,
    ) -> str:
        """Synthesize all phase results into a final output.

        Uses Calliope (Chief Writer) if available, otherwise Athena.
        """
        compile_block = self.standing_goal_blocks_compile(plan)
        if compile_block:
            raise ValueError(compile_block)

        if on_progress:
            await on_progress({"type": "compiling", "status": "started"})

        all_findings = ""
        for phase in plan.completed_phases:
            all_findings += (
                f"\n## Phase {phase.id}: {phase.title}\n"
                f"Agents: {', '.join(phase.agents)}\n"
                f"Findings:\n{phase.result}\n"
            )
        contract_summary = self.get_contract_summary(plan)
        contract_coverage_markdown = self._build_contract_coverage_markdown(contract_summary)
        all_findings += f"\n{contract_coverage_markdown}\n"

        calliope = self._pantheon.get_agent("calliope")
        writer = calliope or self._athena

        compile_prompt = f"""Compile these multi-agent findings into a
professional, coherent response for Acme Corp leadership.

GOAL: {plan.goal}
DATE: {datetime.now(UTC).strftime("%Y-%m-%d")}

RAW FINDINGS:
{all_findings}

RULES:
- Start with a decisive executive summary (3-5 sentences)
- Preserve all data tables and numbers exactly
- Flag any conflicts between agent findings; when phases disagree, say which finding is weaker evidence and what would resolve it
- End with actionable recommendations
- Use Acme Corp terminology naturally
- Do NOT speculate beyond what the data shows
"""

        report = await writer.call_llm(
            "You are Calliope, Chief Writer. Produce a clear, professional "
            "synthesis of multi-agent findings.",
            compile_prompt,
            temperature=0.3,
        )
        report = f"{report.rstrip()}\n\n{contract_coverage_markdown}\n"

        if on_progress:
            await on_progress({"type": "compiling", "status": "completed"})

        return report

    # ── Full autonomous run ──────────────────────────────────────────

    async def run(
        self,
        request: str,
        on_progress: ProgressCallback = None,
    ) -> str:
        """Execute the full agent loop autonomously.

        This is the top-level method that runs the entire
        Plan → Execute → Observe → Compile cycle.
        """
        plan = await self.plan(request, on_progress=on_progress)
        return await run_autonomous_loop(
            plan=plan,
            on_progress=on_progress,
            execute_phase_fn=self.execute_phase,
            replan_fn=self.replan,
            compile_fn=self.compile,
            standing_goal_status_active=StandingGoalStatus.ACTIVE,
            loop_decision_replan=LoopDecision.REPLAN,
            loop_decision_clarify=LoopDecision.CLARIFY,
            loop_decision_complete=LoopDecision.COMPLETE,
            loop_decision_abort=LoopDecision.ABORT,
            max_phases=self.MAX_PHASES,
        )

    # ── Private helpers ──────────────────────────────────────────────

    def _build_planning_prompt(
        self,
        request: str,
        complexity: str,
        standing_objective: str = "",
    ) -> str:
        available = ", ".join(sorted(self._pantheon.agents.keys()))
        standing_block = ""
        if standing_objective.strip():
            standing_block = f"""
STANDING OBJECTIVE (must be fully satisfied before the task is done):
{standing_objective.strip()}

- Include validation_contract.assertions that verify this objective with evidence.
- Phases should collectively achieve the standing objective, not just the request summary.
"""
        return f"""Analyze this request and create a structured execution plan.

REQUEST: {request}
COMPLEXITY HINT: {complexity}
{standing_block}

Return a JSON object:
{{
    "goal": "One-sentence summary",
    "complexity": "simple|moderate|complex",
    "validation_contract": {{
        "assertions": ["Concrete claim that must be true to consider this done"],
        "phase_assertion_map": {{"1": [1]}},
        "completion_gate": "When all assertions pass with evidence",
        "validator_mode": "strict|crm_only|relaxed_evidence"
    }},
    "phases": [
        {{
            "id": 1,
            "title": "Phase title",
            "description": "What this phase accomplishes",
            "agents": ["agent_name"],
            "delegation_type": "generic|revenue|production|finance|hr|procurement",
            "expected_output": "What the phase should produce",
            "depends_on": []
        }}
    ]
}}

Available agents: {available}

RULES:
- Simple requests: 1-2 phases, 1-2 agents
- Moderate requests: 2-4 phases, 2-4 agents
- Complex requests: 4-8 phases, 4+ agents
- Always include a final compile/verify phase
- If ambiguous, first phase should be "clarify with user"
- Include a validation_contract that is implementation-agnostic.
- Every phase should map to at least one assertion id in phase_assertion_map.
- Default validator_mode to "strict" unless the request is explicitly operational/summary-only.
"""

    def _build_delegation_prompt(
        self,
        phase: Phase,
        prior_context: str,
    ) -> str:
        return build_athena_delegation_prompt(
            phase_id=phase.id,
            description=phase.description,
            expected_output=phase.expected_output,
            prior_context=prior_context,
        )

    def _build_prior_context(self, plan: Plan, current_phase: Phase) -> str:
        snippets = [(p.id, p.title, str(p.result or "")) for p in plan.completed_phases]
        prior = format_prior_phase_context_lines(
            snippets,
            current_phase_id=current_phase.id,
        )
        if plan.continuation_context.strip():
            prefix = (
                f"[Continuing toward standing objective] {plan.continuation_context.strip()}\n\n"
            )
            return prefix + prior
        return prior

    def _parse_plan(self, request: str, raw_plan: str, validator_mode: str = "strict") -> Plan:
        """Parse Athena plan JSON with safe fallbacks."""
        return parse_plan_with_fallback(
            request=request,
            raw_plan=raw_plan,
            validator_mode=validator_mode,
            max_phases=self.MAX_PHASES,
            parse_validation_contract_fn=self._parse_validation_contract,
            ensure_validation_contract_fn=self._ensure_validation_contract,
            phase_ctor=Phase,
            plan_ctor=Plan,
            validation_contract_ctor=ValidationContract,
        )

    def _parse_validation_contract(self, raw_contract: Any) -> ValidationContract:
        return parse_validation_contract(
            raw_contract=raw_contract,
            validation_contract_ctor=ValidationContract,
        )

    def _ensure_validation_contract(
        self,
        contract: ValidationContract,
        phases: list[Phase],
        validator_mode: str = "strict",
    ) -> ValidationContract:
        return ensure_validation_contract(
            contract=contract,
            phases=phases,
            validator_mode=validator_mode,
        )
