"""Task orchestrator — server-side plan-execute-report loop.

Provides :class:`TaskOrchestrator`, which drives multi-phase tasks through
the Pantheon.  The flow is:

1. **Clarity check** — Sphinx assesses whether the goal is actionable.
2. **Planning** — Athena generates a structured :class:`TaskPlan`.
3. **Execution** — Each phase runs through the assigned specialist agent.
4. **Reporting** — Calliope formats accumulated results into a document.

Task state is persisted in Redis so that clarification round-trips can
span multiple HTTP requests.  Progress events are emitted via an async
callback (same pattern as :class:`~brain_os.pipeline.RequestPipeline`).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langfuse.decorators import observe
from pydantic import ValidationError

from brain_os.config import get_settings
from brain_os.exceptions import BrainOSError
from brain_os.schemas.llm_outputs import ClarityAssessment, TaskPlan, TaskPlanPhase
from brain_os.systems import task_workspace as tw
from brain_os.systems.phase_contract import (
    TaskValidationPlan,
    contract_to_dict,
    run_phase_validator,
    task_phase_validation_context,
    validation_contract_from_task_plan,
    validation_result_to_dict,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]

_TASK_TTL_SECONDS = 3600
_TASK_INDEX_KEY = "task:index"
_TASK_EVENTS_PREFIX = "task_events:"
_PHASE_TIMEOUT = 120
_REPORTS_DIR = Path("data/reports")


@dataclass
class TaskResult:
    """Value object returned when a task completes (or pauses for clarification)."""

    task_id: str
    status: str  # "complete", "clarification_needed", "aborted", "error", "dormant"
    summary: str = ""
    file_path: str = ""
    file_format: str = ""
    clarification_questions: list[str] = field(default_factory=list)
    workspace_dir: str = ""
    continuation_phases_added: int = 0
    standing_goal_status: str | None = None
    standing_goal_verdict: str | None = None
    standing_goal_turn: int | None = None
    standing_goal_max_turns: int | None = None


class _TaskOrchestratorCore:
    """Core plan-execute-report loop (composed with long-lived mixin as TaskOrchestrator)."""

    def __init__(
        self,
        *,
        pantheon: Any,
        redis_cache: Any,
        voice: Any | None = None,
        pdfco: Any | None = None,
        tasks_root: Path | str | None = None,
    ) -> None:
        self._pantheon = pantheon
        self._redis = redis_cache
        self._voice = voice
        self._pdfco = pdfco
        self._tasks_root = Path(tasks_root) if tasks_root is not None else Path("data/tasks")
        # Fallback persistence when Redis is unavailable.
        self._local_state: dict[str, dict[str, Any]] = {}
        self._local_events: dict[str, list[dict[str, Any]]] = {}
        self._local_index: list[str] = []

    # ── public API ────────────────────────────────────────────────────────

    async def create_task(
        self,
        goal: str,
        user_id: str | None = None,
        output_format: str = "markdown",
        *,
        trigger: str = "",
        event_meta: dict[str, Any] | None = None,
    ) -> str:
        """Persist initial task state in Redis and return a task_id."""
        task_id = uuid.uuid4().hex[:12]
        workspace_dir = ""
        try:
            ws = tw.resolve_task_workspace_dir(self._tasks_root, task_id)
            workspace_dir = str(ws)
            tw.init_workspace_files(ws, task_id=task_id, goal=goal)
        except (OSError, ValueError) as exc:
            logger.warning("Task workspace init skipped for %s: %s", task_id, exc)

        state = {
            "task_id": task_id,
            "goal": goal,
            "user_id": user_id or "anonymous",
            "output_format": output_format,
            "trigger": trigger,
            "event_meta": event_meta or {},
            "status": "created",
            "plan": None,
            "phase_results": {},
            "abort_requested": False,
            "abort_reason": "",
            "workspace_dir": workspace_dir,
        }
        await self._save_state(task_id, state)
        await self._update_task_index(task_id)
        return task_id

    async def get_task_state(self, task_id: str) -> dict[str, Any] | None:
        """Return the persisted task state from Redis."""
        return await self._load_state(task_id)

    async def list_tasks(
        self,
        limit: int = 20,
        *,
        status_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent task states, newest first."""
        from brain_os.systems import task_workspace_persist as twp

        seen: dict[str, dict[str, Any]] = {}
        for row in twp.list_workspace_summaries(self._tasks_root, limit=limit * 3):
            tid = str(row.get("task_id") or "")
            if tid:
                seen[tid] = row
        task_ids: list[str] = []
        if self._redis is not None and self._redis.available:
            task_ids = await self._redis.get_json(_TASK_INDEX_KEY) or []
        else:
            task_ids = list(self._local_index)
        if not isinstance(task_ids, list):
            task_ids = []
        for task_id in reversed(task_ids):
            tid = str(task_id)
            state = await self._load_state(tid)
            if state:
                seen[tid] = twp.state_to_summary(state)
        rows = list(seen.values())
        if status_filter:
            rows = [r for r in rows if str(r.get("status") or "") == status_filter]
        rank: dict[str, int] = {str(tid): idx for idx, tid in enumerate(reversed(task_ids)) if tid}
        rows.sort(
            key=lambda row: (
                str(row.get("updated_at") or ""),
                rank.get(str(row.get("task_id") or ""), 9999),
            ),
            reverse=True,
        )
        # When ``updated_at`` is missing, lower redis rank index = more recent.
        if rows and not any(row.get("updated_at") for row in rows):
            rows.sort(key=lambda row: rank.get(str(row.get("task_id") or ""), 9999))
        return rows[:limit]

    async def append_task_event(self, task_id: str, event: dict[str, Any]) -> None:
        """Append an event to a task's event log."""
        local_events = self._local_events.get(task_id, [])
        local_events.append(event)
        self._local_events[task_id] = local_events[-1000:]

        if self._redis is None or not self._redis.available:
            return
        key = f"{_TASK_EVENTS_PREFIX}{task_id}"
        events = await self._redis.get_json(key) or []
        if not isinstance(events, list):
            events = []
        events.append(event)
        events = events[-1000:]
        await self._redis.set_json(key, events, _TASK_TTL_SECONDS)

    async def get_task_events(self, task_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Return most recent events for a task."""
        if self._redis is not None and self._redis.available:
            key = f"{_TASK_EVENTS_PREFIX}{task_id}"
            events = await self._redis.get_json(key) or []
            if isinstance(events, list):
                return events[-limit:]
        return self._local_events.get(task_id, [])[-limit:]

    async def abort_task(self, task_id: str, reason: str = "") -> bool:
        """Request abortion of a running task."""
        state = await self._load_state(task_id)
        if state is None:
            return False
        state["abort_requested"] = True
        state["abort_reason"] = reason
        state["status"] = "aborting"
        await self._save_state(task_id, state)
        return True

    @observe(name="task_orchestrator.run_task")
    async def run_task(
        self,
        task_id: str,
        on_progress: ProgressCallback | None = None,
    ) -> TaskResult:
        """Execute the full plan-execute-report loop for a task."""
        state = await self._load_state(task_id)
        if state is None:
            return TaskResult(
                task_id=task_id,
                status="error",
                summary="Task not found",
                workspace_dir="",
            )

        ws0 = str(state.get("workspace_dir") or "")

        try:
            await self._emit(on_progress, "task_created", task_id=task_id)

            # 1. Clarity check — skip for long-lived / --until tasks (standing
            # objective is already the verifiable done gate; Sphinx pausing
            # overnight research for "what does aging mean?" wastes the night).
            skip_clarity = bool(state.get("long_lived")) or bool(
                str(state.get("standing_objective") or "").strip()
            )
            if not skip_clarity:
                clarity = await self._check_clarity(state["goal"], on_progress)
                if not clarity.clear:
                    state["status"] = "awaiting_clarification"
                    state["clarification_questions"] = clarity.clarifying_questions
                    state["clarification_missing_slots"] = list(clarity.missing_slots)
                    from datetime import UTC, datetime

                    state["socratic_created_at"] = (
                        datetime.now(UTC).isoformat().replace("+00:00", "Z")
                    )
                    state["session_id"] = task_id
                    state["conversation_id"] = task_id
                    defaults_raw = (clarity.suggested_default_scope or "").strip()
                    if defaults_raw.startswith("{"):
                        try:
                            import json as _json

                            bundle = _json.loads(defaults_raw)
                            if isinstance(bundle, dict) and "defaults" in bundle:
                                state["socratic_defaults"] = bundle.get("defaults") or {}
                                state["socratic_questions"] = bundle.get("questions") or []
                                state["socratic_score"] = bundle.get("score") or {}
                                if bundle.get("gate_id"):
                                    state["gate_id"] = bundle.get("gate_id")
                            else:
                                state["socratic_defaults"] = (
                                    bundle if isinstance(bundle, dict) else {}
                                )
                            state["socratic"] = True
                        except Exception:
                            state["socratic_defaults"] = {}
                    from brain_os.services.socratic_gate import extract_gate_id

                    joined_q = "\n".join(str(q) for q in (clarity.clarifying_questions or []))
                    gid = extract_gate_id(joined_q)
                    if gid:
                        state["gate_id"] = gid
                    await self._save_state(task_id, state)
                    await self._emit(
                        on_progress,
                        "clarification_needed",
                        questions=clarity.clarifying_questions,
                        reason=clarity.ambiguity_reason,
                        missing_slots=clarity.missing_slots,
                        task_id=task_id,
                        socratic=bool(state.get("socratic")),
                        socratic_defaults=state.get("socratic_defaults") or {},
                    )
                    self._workspace_note_clarification_needed(state)
                    return TaskResult(
                        task_id=task_id,
                        status="clarification_needed",
                        clarification_questions=clarity.clarifying_questions,
                        workspace_dir=ws0,
                    )
            else:
                logger.info(
                    "Skipping Sphinx clarity check — long-lived/standing-objective task %s",
                    task_id,
                )

            # 2. Plan
            return await self._plan_and_execute(task_id, state, on_progress)

        except (
            TimeoutError,
            BrainOSError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            ValidationError,
        ) as exc:
            logger.exception("Task %s failed", task_id)
            await self._emit(on_progress, "task_error", error=str(exc))
            st = await self._load_state(task_id)
            wd = str((st or {}).get("workspace_dir") or "")
            if wd:
                self._workspace_safe_note(Path(wd), f"Status: error — {exc}", status="error")
            return TaskResult(task_id=task_id, status="error", summary=str(exc), workspace_dir=wd)

    @observe(name="task_orchestrator.resume_with_clarification")
    async def resume_with_clarification(
        self,
        task_id: str,
        answer: str,
        on_progress: ProgressCallback | None = None,
    ) -> TaskResult:
        """Resume a task that was paused for clarification."""
        state = await self._load_state(task_id)
        if state is None:
            return TaskResult(
                task_id=task_id,
                status="error",
                summary="Task not found",
                workspace_dir="",
            )
        if state.get("status") != "awaiting_clarification":
            return TaskResult(
                task_id=task_id,
                status="error",
                summary="Task is not awaiting clarification",
                workspace_dir=self._workspace_dir_from_state(state),
            )
        if not (answer or "").strip():
            return TaskResult(
                task_id=task_id,
                status="clarification_needed",
                clarification_questions=list(state.get("clarification_questions") or []),
                summary="Please answer the pending clarification question(s) before resuming.",
                workspace_dir=self._workspace_dir_from_state(state),
            )

        original_goal = state["goal"]
        clarified = (answer or "").strip()
        if state.get("socratic"):
            try:
                from brain_os.services.socratic_gate import (
                    classify_socratic_resume,
                    expand_go_answer,
                    log_socratic_gate_event,
                    record_socratic_resume_answers,
                )

                pending_socratic = {
                    "socratic": True,
                    "socratic_defaults": state.get("socratic_defaults") or {},
                    "socratic_questions": state.get("socratic_questions") or [],
                    "socratic_score": state.get("socratic_score") or {},
                    "created_at": state.get("socratic_created_at")
                    or state.get("created_at")
                    or state.get("updated_at"),
                    "session_id": str(state.get("session_id") or task_id),
                    "conversation_id": str(state.get("conversation_id") or task_id),
                }
                match = await classify_socratic_resume(
                    clarified,
                    pending_socratic,
                    llm_client=None,
                    session_id=str(task_id),
                    conversation_id=str(task_id),
                )
                if match.disposition in ("abandoned", "expired"):
                    log_socratic_gate_event(
                        {
                            "status": match.disposition,
                            "reason": match.reason,
                            "source": "task",
                            "task_id": task_id,
                            "inbound": clarified[:500],
                        }
                    )
                    return TaskResult(
                        task_id=task_id,
                        status="error",
                        summary=(
                            f"Pending Socratic gate {match.disposition} ({match.reason}). "
                            "Start a new task for the new request; answers were not recorded."
                        ),
                        workspace_dir=self._workspace_dir_from_state(state),
                    )
                clarified = expand_go_answer(
                    clarified,
                    defaults=pending_socratic["socratic_defaults"],
                    questions=pending_socratic["socratic_questions"],
                )
                record_socratic_resume_answers(
                    pending=pending_socratic,
                    answer=answer,
                    source="task",
                    mapped_answers=match.mapped_answers or None,
                )
            except Exception:
                logger.debug("task socratic resume expand failed", exc_info=True)
        state["goal"] = f"{original_goal}\n\nClarification: {clarified}"
        state["status"] = "resumed"
        await self._save_state(task_id, state)
        self._workspace_resume_note(state)

        try:
            return await self._plan_and_execute(task_id, state, on_progress)
        except (
            TimeoutError,
            BrainOSError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            ValidationError,
        ) as exc:
            logger.exception("Task %s failed after clarification", task_id)
            await self._emit(on_progress, "task_error", error=str(exc))
            wd = self._workspace_dir_from_state(await self._load_state(task_id))
            if wd:
                self._workspace_safe_note(Path(wd), f"Status: error — {exc}", status="error")
            return TaskResult(
                task_id=task_id,
                status="error",
                summary=str(exc),
                workspace_dir=wd,
            )

    @observe(name="task_orchestrator.retry_task")
    async def retry_task(
        self,
        task_id: str,
        *,
        from_phase: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> TaskResult:
        """Retry a persisted task from a specific phase index.

        If ``from_phase`` is None, retries from the first phase.
        """
        state = await self._load_state(task_id)
        if state is None:
            return TaskResult(
                task_id=task_id,
                status="error",
                summary="Task not found",
                workspace_dir="",
            )

        wdir = self._workspace_dir_from_state(state)

        plan_data = state.get("plan")
        if not isinstance(plan_data, dict):
            return TaskResult(
                task_id=task_id,
                status="error",
                summary="Task has no persisted plan to retry",
                workspace_dir=wdir,
            )

        try:
            plan = TaskPlan.model_validate(plan_data)
        except ValidationError as exc:
            return TaskResult(
                task_id=task_id,
                status="error",
                summary=f"Invalid persisted plan: {exc}",
                workspace_dir=wdir,
            )

        total_phases = len(plan.phases)
        start_phase = 0 if from_phase is None else from_phase
        if start_phase < 0 or start_phase >= total_phases:
            return TaskResult(
                task_id=task_id,
                status="error",
                summary=f"Invalid from_phase {start_phase}; valid range is 0..{max(0, total_phases - 1)}",
                workspace_dir=wdir,
            )

        # Seed accumulated with prior results if retry starts mid-plan.
        accumulated: dict[int, dict[str, Any]] = {}
        phase_results = state.get("phase_results", {})
        if not isinstance(phase_results, dict):
            phase_results = {}
        for idx in range(start_phase):
            key = str(idx)
            if key not in phase_results:
                return TaskResult(
                    task_id=task_id,
                    status="error",
                    summary=f"Cannot retry from phase {start_phase}; missing prior result for phase {idx}",
                    workspace_dir=wdir,
                )
            phase = plan.phases[idx]
            accumulated[idx] = {
                "title": phase.title,
                "agent": phase.agent,
                "result": phase_results[key],
            }

        await self._emit(
            on_progress,
            "task_retry_started",
            task_id=task_id,
            from_phase=start_phase,
            total_phases=total_phases,
        )
        state["status"] = "retrying"
        await self._save_state(task_id, state)
        self._workspace_retry_note(state, from_phase=start_phase)
        return await self._execute_plan(
            task_id=task_id,
            state=state,
            plan=plan,
            accumulated=accumulated,
            start_phase=start_phase,
            on_progress=on_progress,
            goal=state.get("goal", ""),
        )

    # ── internal orchestration ────────────────────────────────────────────

    async def _plan_and_execute(
        self,
        task_id: str,
        state: dict[str, Any],
        on_progress: ProgressCallback | None,
    ) -> TaskResult:
        """Generate a plan, execute all phases, and produce a report."""
        goal = state["goal"]

        # Plan
        plan = await self._generate_plan(goal, on_progress)
        validation_contract = validation_contract_from_task_plan(plan)
        contract_payload = contract_to_dict(validation_contract)
        state["plan"] = plan.model_dump()
        state["validation_contract"] = contract_payload
        state["status"] = "executing"
        await self._save_state(task_id, state)

        wd = state.get("workspace_dir") or ""
        if wd:
            try:
                p = Path(wd)
                tw.write_plan_file(p, plan)
                tw.write_contract_file(
                    p,
                    {
                        "validation_contract": contract_payload,
                        "phase_verdicts": {},
                    },
                )
                tw.append_memory_line(p, "Plan created and written to PLAN.md.")
                tw.update_readme_status(p, "executing")
            except OSError:
                logger.warning("Could not write PLAN.md for task %s", task_id, exc_info=True)

        len(plan.phases)
        await self._emit(
            on_progress,
            "plan_created",
            phases=[
                {
                    "id": i,
                    "title": p.title,
                    "agent": p.agent,
                    "description": p.description,
                    "depends_on": p.depends_on,
                }
                for i, p in enumerate(plan.phases)
            ],
        )

        return await self._execute_plan(
            task_id=task_id,
            state=state,
            plan=plan,
            accumulated={},
            start_phase=0,
            on_progress=on_progress,
            goal=goal,
        )

    async def _execute_plan(
        self,
        *,
        task_id: str,
        state: dict[str, Any],
        plan: TaskPlan,
        accumulated: dict[int, dict[str, Any]],
        start_phase: int,
        on_progress: ProgressCallback | None,
        goal: str,
    ) -> TaskResult:
        """Execute a plan from ``start_phase`` and compile a report."""
        total_phases = len(plan.phases)
        pending = set(range(start_phase, total_phases))
        completed: set[int] = set(accumulated.keys())
        phase_durations_ms: list[float] = []
        task_started = time.time()
        wdir = self._workspace_dir_from_state(state)

        while pending:
            runnable = [
                idx
                for idx in sorted(pending)
                if self._deps_satisfied(plan.phases[idx].depends_on, completed)
            ]
            if not runnable:
                await self._emit(
                    on_progress,
                    "task_error",
                    task_id=task_id,
                    reason="No runnable phases due to dependency cycle or invalid depends_on",
                )
                if wdir:
                    self._workspace_safe_note(
                        Path(wdir),
                        "Error: no runnable phases (dependency cycle or invalid depends_on).",
                        status="error",
                    )
                return TaskResult(
                    task_id=task_id,
                    status="error",
                    summary="Dependency cycle or invalid phase dependencies detected",
                    workspace_dir=wdir,
                )

            idx = runnable[0]
            phase = plan.phases[idx]
            latest = await self._load_state(task_id)
            if latest and latest.get("abort_requested"):
                latest["status"] = "aborted"
                await self._save_state(task_id, latest)
                reason = latest.get("abort_reason", "")
                await self._emit(
                    on_progress,
                    "task_aborted",
                    task_id=task_id,
                    reason=reason,
                )
                if wdir:
                    self._workspace_safe_note(
                        Path(wdir),
                        f"Task aborted: {reason}".strip(),
                        status="aborted",
                    )
                return TaskResult(
                    task_id=task_id,
                    status="aborted",
                    summary=f"Task aborted. {reason}".strip(),
                    workspace_dir=wdir,
                )

            context_summary = self._build_phase_context(accumulated)
            phase_started = time.time()
            result, err = await self._execute_phase_with_validation(
                task_id=task_id,
                idx=idx,
                phase=phase,
                plan=plan,
                state=state,
                context_summary=context_summary,
                on_progress=on_progress,
                goal=goal,
                total_phases=total_phases,
                wdir=wdir,
            )
            if err is not None:
                return err
            phase_elapsed_ms = (time.time() - phase_started) * 1000
            phase_durations_ms.append(phase_elapsed_ms)
            accumulated[idx] = {"title": phase.title, "agent": phase.agent, "result": result}
            state["phase_results"][str(idx)] = result
            if wdir:
                try:
                    tw.record_phase_completed(
                        Path(wdir),
                        phase_index=idx,
                        agent=phase.agent,
                        title=phase.title,
                        result_preview=str(result)[:300],
                        result_body=str(result),
                    )
                except OSError:
                    logger.warning("Phase workspace write failed", exc_info=True)
            completed.add(idx)
            pending.remove(idx)
            avg_phase_ms = sum(phase_durations_ms) / len(phase_durations_ms)
            remaining = len(pending)
            eta_ms = avg_phase_ms * remaining
            elapsed_ms = (time.time() - task_started) * 1000
            progress_pct = round((len(completed) / max(1, total_phases)) * 100, 1)
            await self._emit(
                on_progress,
                "phase_progress",
                phase_index=idx + 1,
                total_phases=total_phases,
                completed_phases=len(completed),
                progress_pct=progress_pct,
                elapsed_ms=round(elapsed_ms, 1),
                eta_ms=round(eta_ms, 1),
            )
            await self._save_state(task_id, state)

        output_format = state.get("output_format", "markdown")
        file_path = await self._generate_report(
            task_id,
            goal,
            accumulated,
            output_format,
            on_progress,
            workspace_dir=wdir,
        )

        state["status"] = "complete"
        state["file_path"] = file_path
        await self._save_state(task_id, state)

        summary = f"Completed {len(plan.phases)} phases. Report saved to {file_path}"
        await self._emit(
            on_progress,
            "task_complete",
            task_id=task_id,
            summary=summary,
        )

        return TaskResult(
            task_id=task_id,
            status="complete",
            summary=summary,
            file_path=file_path,
            file_format=output_format,
            workspace_dir=wdir,
        )

    async def _check_clarity(
        self,
        goal: str,
        on_progress: ProgressCallback | None,
    ) -> ClarityAssessment:
        """Ask Sphinx for a structured clarity assessment.

        Prefers Sphinx v2 Socratic gate (stakes × ambiguity + defaults) and
        falls back to legacy ``assess_clarity`` when Socratic pass-through.
        """
        from brain_os.services.task_phase_deterministic import goal_is_structured_multiphase

        if goal_is_structured_multiphase(goal):
            logger.info("Skipping Sphinx clarity check — structured multi-phase goal")
            return ClarityAssessment(
                clear=True,
                ambiguity_reason="",
                clarifying_questions=[],
                missing_slots=[],
                confidence=0.95,
                can_answer_partially=False,
            )

        await self._emit(on_progress, "clarity_checking")

        try:
            from brain_os.config import get_settings
            from brain_os.services.socratic_gate import (
                evaluate_socratic_gate,
                to_clarity_assessment,
            )

            if get_settings().app.socratic_gate_enabled:
                llm = None
                sphinx = self._pantheon.get_agent("sphinx")
                if sphinx is not None:
                    try:
                        await sphinx._ensure_llm()
                        llm = getattr(sphinx, "_llm", None)
                    except Exception:
                        llm = None
                socratic = await asyncio.wait_for(
                    evaluate_socratic_gate(
                        goal,
                        llm_client=llm,
                        channel="task",
                        is_multiphase_task=True,
                    ),
                    timeout=_PHASE_TIMEOUT,
                )
                if socratic.gated:
                    return to_clarity_assessment(socratic)
                # Not gated: fall through to legacy Sphinx assess_clarity for
                # task safety (mocks + thin goals still get a second opinion).
        except TimeoutError:
            logger.warning("Socratic clarity check timed out — falling back to Sphinx")
        except Exception:
            logger.debug("Socratic clarity check failed — falling back", exc_info=True)

        sphinx = self._pantheon.get_agent("sphinx")
        if sphinx is None:
            logger.warning("Sphinx not available — requesting clarification for task safety")
            return ClarityAssessment(
                clear=False,
                ambiguity_reason="Sphinx unavailable; need explicit scope before deep task execution.",
                clarifying_questions=[
                    "What exact outcome do you want from this task?",
                    "Which entity/time range should this task focus on?",
                ],
                missing_slots=["outcome", "scope"],
                confidence=0.2,
                can_answer_partially=False,
            )

        try:
            return await asyncio.wait_for(
                sphinx.assess_clarity(goal),
                timeout=_PHASE_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "Sphinx clarity check timed out — requesting clarification for task safety"
            )
            return ClarityAssessment(
                clear=False,
                ambiguity_reason="Clarity check timed out; need explicit scope for task execution.",
                clarifying_questions=[
                    "What is the primary goal for this task?",
                    "What constraints or time window should I use?",
                ],
                missing_slots=["goal", "constraints"],
                confidence=0.2,
                can_answer_partially=False,
            )
        except (
            BrainOSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
        ):
            logger.exception(
                "Sphinx clarity check failed — requesting clarification for task safety"
            )
            return ClarityAssessment(
                clear=False,
                ambiguity_reason="Clarity check failed; need explicit scope for task execution.",
                clarifying_questions=[
                    "What exact deliverable do you want?",
                    "Which company/project/time range should I use?",
                ],
                missing_slots=["deliverable", "scope"],
                confidence=0.2,
                can_answer_partially=False,
            )

    async def _generate_plan(
        self,
        goal: str,
        on_progress: ProgressCallback | None,
    ) -> TaskPlan:
        """Ask Athena to produce a structured execution plan."""
        await self._emit(on_progress, "planning")

        athena = self._pantheon.get_agent("athena")
        if athena is None:
            logger.error("Athena not available — cannot plan")
            return TaskPlan(goal=goal, phases=[])

        plan = await athena.generate_plan(goal)

        if not plan.phases:
            logger.warning("Athena returned empty plan — creating single-agent fallback")
            plan = TaskPlan(
                goal=goal,
                phases=[TaskPlanPhase(title="Research", agent="clio", description=goal)],
                reasoning="Fallback: no phases generated",
            )

        return plan

    def _task_validator_enabled(self) -> bool:
        return bool(get_settings().app.task_phase_validator_enabled)

    def _task_validator_retry_limit(self) -> int:
        return int(get_settings().app.task_phase_validator_retry)

    async def _validate_task_phase(
        self,
        *,
        plan: TaskPlan,
        contract_payload: dict[str, Any],
        phase_index: int,
        phase: TaskPlanPhase,
        result: str,
        workspace: Path | None,
    ) -> tuple[bool, str]:
        """Run shared phase validator; return (passed, summary)."""
        if not self._task_validator_enabled():
            return True, "Task phase validator disabled."

        athena = self._pantheon.get_agent("athena")
        if athena is None:
            return True, "Athena unavailable; skipping phase validation."

        from brain_os.systems.phase_contract import ValidationContract

        contract = ValidationContract(
            assertions=list(contract_payload.get("assertions") or []),
            phase_assertion_map={
                int(k): list(v)
                for k, v in (contract_payload.get("phase_assertion_map") or {}).items()
            },
            completion_gate=str(contract_payload.get("completion_gate") or ""),
            mode=str(contract_payload.get("validator_mode") or "strict"),
        )
        ctx = task_phase_validation_context(phase_index, phase, contract)
        validation_plan = TaskValidationPlan(validation_contract=contract)
        responses = {phase.agent or "agent": result}
        verdict = await run_phase_validator(
            plan=validation_plan,
            phase=ctx,
            responses=responses,
            athena=athena,
        )
        if workspace is not None:
            tw.append_contract_phase_verdict(
                workspace,
                phase_index=phase_index,
                verdict=validation_result_to_dict(verdict),
            )
        self._record_task_phase_validator_metrics(
            passed=verdict.passed,
            contract_payload=contract_payload,
            phase_index=phase_index,
        )
        return verdict.passed, verdict.summary or ""

    @staticmethod
    def _record_task_phase_validator_metrics(
        *,
        passed: bool,
        contract_payload: dict[str, Any],
        phase_index: int,
    ) -> None:
        from brain_os.brain import runtime_metrics

        raw_map = contract_payload.get("phase_assertion_map") or {}
        assertion_ids = raw_map.get(str(phase_index + 1)) or raw_map.get(phase_index + 1) or []
        total = len(assertion_ids) if assertion_ids else 1
        runtime_metrics.incr("assertions_total", total)
        if passed:
            runtime_metrics.incr("assertions_passed", total)

    async def _execute_phase_with_validation(
        self,
        *,
        task_id: str,
        idx: int,
        phase: TaskPlanPhase,
        plan: TaskPlan,
        state: dict[str, Any],
        context_summary: str,
        on_progress: ProgressCallback | None,
        goal: str,
        total_phases: int,
        wdir: str,
    ) -> tuple[str, TaskResult | None]:
        """Run specialist phase plus validator; return ``(result, error_result)``."""
        contract_payload = dict(state.get("validation_contract") or {})
        ws_path = Path(wdir) if wdir else None
        result = await self._execute_phase(
            task_id,
            idx,
            phase,
            context_summary,
            on_progress,
            goal=goal,
            total_phases=total_phases,
            workspace_dir=wdir,
        )
        passed, val_summary = await self._validate_task_phase(
            plan=plan,
            contract_payload=contract_payload,
            phase_index=idx,
            phase=phase,
            result=result,
            workspace=ws_path,
        )
        retry_limit = self._task_validator_retry_limit()
        attempt = 0
        while not passed and attempt < retry_limit:
            attempt += 1
            if ws_path is not None:
                tw.append_memory_line(
                    ws_path,
                    f"Phase {idx + 1} validator failed (attempt {attempt}): {val_summary}",
                )
            result = await self._execute_phase(
                task_id,
                idx,
                phase,
                context_summary,
                on_progress,
                goal=goal,
                total_phases=total_phases,
                workspace_dir=wdir,
            )
            passed, val_summary = await self._validate_task_phase(
                plan=plan,
                contract_payload=contract_payload,
                phase_index=idx,
                phase=phase,
                result=result,
                workspace=ws_path,
            )
        if not passed:
            msg = f"Phase {idx + 1} failed validation: {val_summary}"
            if ws_path is not None:
                tw.append_memory_line(ws_path, msg)
                tw.update_readme_status(ws_path, "error")
            await self._emit(
                on_progress,
                "task_error",
                task_id=task_id,
                reason=msg,
                phase_index=idx + 1,
            )
            state["status"] = "error"
            await self._save_state(task_id, state)
            return result, TaskResult(
                task_id=task_id,
                status="error",
                summary=msg,
                workspace_dir=wdir,
            )
        return result, None

    async def _execute_phase(
        self,
        task_id: str,
        phase_id: int,
        phase: TaskPlanPhase,
        context_from_previous: str,
        on_progress: ProgressCallback | None,
        *,
        goal: str = "",
        total_phases: int | None = None,
        workspace_dir: str = "",
    ) -> str:
        """Run a single phase through the assigned agent.

        Context dict includes ``task_workspace_dir`` / ``task_phase_index`` for
        optional agent or prompt extensions.
        """
        await self._emit(
            on_progress,
            "phase_started",
            phase_id=phase_id,
            phase_index=phase_id + 1,
            total_phases=total_phases,
            agent=phase.agent,
            title=phase.title,
        )

        from brain_os.systems.operator_inbox import require_operator_release_for_task_phase

        release_skip = require_operator_release_for_task_phase(phase)
        if release_skip:
            result = (
                "Phase skipped: operator release required "
                "(clear /api/operator/inbox and POST /api/operator/release, or "
                "poetry run brain operator release). "
                f"Reason: {release_skip}"
            )
            await self._emit(
                on_progress,
                "phase_completed",
                phase_id=phase_id,
                phase_index=phase_id + 1,
                total_phases=total_phases,
                agent=phase.agent,
                title=phase.title,
                skipped=True,
                skip_reason=release_skip,
            )
            return result

        result: str | None = None
        try:
            from brain_os.services.task_phase_deterministic import (
                run_hex_brief_phase,
                run_mail_journey_phase,
                run_outbound_draft_phase,
            )

            result = await run_mail_journey_phase(
                goal=goal,
                phase=phase,
                context_from_previous=context_from_previous,
                pantheon=self._pantheon,
            )
            if result is None:
                result = await run_hex_brief_phase(
                    goal=goal,
                    phase=phase,
                    context_from_previous=context_from_previous,
                    pantheon=self._pantheon,
                )
            if result is None:
                result = await run_outbound_draft_phase(
                    goal=goal,
                    phase=phase,
                    context_from_previous=context_from_previous,
                    pantheon=self._pantheon,
                )
        except Exception as exc:
            logger.warning("Deterministic task phase handler failed: %s", exc)

        agent_key = str(phase.agent or "").strip().lower()
        agent = self._pantheon.get_agent(agent_key) if agent_key else None
        if result is not None:
            pass
        elif agent is None:
            result = f"(Agent '{phase.agent}' not found — skipped)"
            logger.warning("Phase %d: agent '%s' not found", phase_id, phase.agent)
        else:
            task_prompt = phase.description
            if goal:
                task_prompt = (
                    f"OVERALL TASK GOAL:\n{goal}\n\nYOUR PHASE ASSIGNMENT:\n{phase.description}"
                )
            if context_from_previous:
                task_prompt += f"\n\nContext from previous phases:\n{context_from_previous}"

            if workspace_dir:
                memories_tail = tw.read_memories_tail(Path(workspace_dir))
                if memories_tail:
                    task_prompt += (
                        "\n\nRecent task memory (tail of MEMORIES.md in your workspace):\n"
                        f"{memories_tail}\n"
                    )
                task_prompt += (
                    "\n\n---\nTASK_WORKSPACE_DIR="
                    f"{workspace_dir}\n"
                    "You may read PLAN.md, MEMORIES.md, and phases/ under that directory "
                    "for continuity with prior steps.\n"
                )

            handle_ctx: dict[str, Any] = {
                "task_phase": True,
                "task_id": task_id,
                "task_workspace_dir": workspace_dir,
                "task_phase_index": phase_id,
            }

            try:
                result = await asyncio.wait_for(
                    agent.handle(task_prompt, handle_ctx),
                    timeout=_PHASE_TIMEOUT,
                )
            except TimeoutError:
                result = f"(Agent '{phase.agent}' timed out after {_PHASE_TIMEOUT}s)"
                logger.warning("Phase %d: agent '%s' timed out", phase_id, phase.agent)
            except (BrainOSError, OSError, RuntimeError, ValueError, TypeError) as exc:
                result = f"(Agent '{phase.agent}' error: {exc})"
                logger.exception("Phase %d: agent '%s' failed", phase_id, phase.agent)

        result = result or f"(Agent '{phase.agent}' returned no output)"
        await self._emit(
            on_progress,
            "phase_done",
            phase_id=phase_id,
            phase_index=phase_id + 1,
            total_phases=total_phases,
            agent=phase.agent,
            preview=str(result)[:300],
        )
        return result

    async def _generate_report(
        self,
        task_id: str,
        goal: str,
        accumulated: dict[int, dict[str, Any]],
        output_format: str,
        on_progress: ProgressCallback | None,
        workspace_dir: str = "",
    ) -> str:
        """Ask Calliope to format results and save to a file."""
        await self._emit(on_progress, "report_generating")

        compiled = "\n\n---\n\n".join(
            f"## {info['title']} ({info['agent']})\n\n{info['result']}"
            for info in accumulated.values()
        )

        calliope = self._pantheon.get_agent("calliope")
        if calliope is not None:
            try:
                markdown_report = await asyncio.wait_for(
                    calliope.handle(
                        f"Format these research results into a comprehensive, professional "
                        f"report.\n\nGoal: {goal}\n\nResults:\n\n{compiled}",
                    ),
                    timeout=_PHASE_TIMEOUT,
                )
            except (TimeoutError, BrainOSError, OSError, RuntimeError, ValueError, TypeError):
                logger.exception("Calliope report formatting failed — using raw results")
                markdown_report = f"# Report: {goal}\n\n{compiled}"
        else:
            markdown_report = f"# Report: {goal}\n\n{compiled}"

        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        md_path = _REPORTS_DIR / f"{task_id}.md"
        md_path.write_text(markdown_report, encoding="utf-8")

        file_path = str(md_path)

        if output_format == "pdf" and self._pdfco is not None and self._pdfco.available:
            try:
                html = f"<html><body><pre>{markdown_report}</pre></body></html>"
                pdf_bytes = await self._pdfco.html_to_pdf(html, name=f"{task_id}.pdf")
                pdf_path = _REPORTS_DIR / f"{task_id}.pdf"
                pdf_path.write_bytes(pdf_bytes)
                file_path = str(pdf_path)
            except (TimeoutError, OSError, ValueError, TypeError, BrainOSError):
                logger.exception("PDF generation failed — falling back to markdown")

        await self._emit(
            on_progress,
            "report_ready",
            file_path=file_path,
            format=output_format,
        )
        if workspace_dir:
            try:
                p = Path(workspace_dir)
                tw.append_memory_line(p, f"Report ready: {file_path}")
                tw.update_readme_status(p, "complete")
                tw.append_readme_artifact(p, "Report", file_path)
            except OSError:
                logger.warning("Task workspace report note failed", exc_info=True)
        return file_path

    @staticmethod
    def _workspace_dir_from_state(state: dict[str, Any] | None) -> str:
        if not state:
            return ""
        return str(state.get("workspace_dir") or "")

    def _workspace_note_clarification_needed(self, state: dict[str, Any]) -> None:
        wd = state.get("workspace_dir") or ""
        if not wd:
            return
        self._workspace_safe_note(
            Path(wd),
            "Status: awaiting_clarification (Sphinx requested user input).",
            status="awaiting_clarification",
        )

    def _workspace_resume_note(self, state: dict[str, Any]) -> None:
        wd = state.get("workspace_dir") or ""
        if not wd:
            return
        self._workspace_safe_note(
            Path(wd),
            "Resumed with user clarification (goal extended in task state).",
            status="resumed",
        )

    def _workspace_retry_note(self, state: dict[str, Any], *, from_phase: int) -> None:
        wd = state.get("workspace_dir") or ""
        if not wd:
            return
        self._workspace_safe_note(
            Path(wd),
            f"Retry started from phase index {from_phase} (0-based).",
            status="retrying",
        )

    @staticmethod
    def _workspace_safe_note(workspace: Path, message: str, *, status: str | None = None) -> None:
        try:
            tw.append_memory_line(workspace, message)
            if status:
                tw.update_readme_status(workspace, status)
        except OSError:
            logger.warning("Task workspace note failed for %s", workspace, exc_info=True)

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_phase_context(self, accumulated: dict[int, dict[str, Any]]) -> str:
        """Summarise completed phases into a context string for the next agent."""
        if not accumulated:
            return ""
        parts = []
        for info in accumulated.values():
            result = str(info.get("result") or "")
            parts.append(f"[{info['title']} — {info['agent']}]: {result[:1000]}")
        return "\n\n".join(parts)

    @staticmethod
    def _deps_satisfied(depends_on: list[int], completed: set[int]) -> bool:
        """Accept both 0-based and 1-based dependency IDs from planners."""
        for dep in depends_on:
            if dep in completed:
                continue
            if (dep - 1) in completed:
                continue
            return False
        return True

    async def _save_state(self, task_id: str, state: dict[str, Any]) -> None:
        self._local_state[task_id] = dict(state)
        ws_dir = str(state.get("workspace_dir") or "").strip()
        if ws_dir:
            try:
                from brain_os.systems.task_workspace_persist import write_state_json

                write_state_json(Path(ws_dir), state)
            except (OSError, ValueError) as exc:
                logger.warning("Task %s state.json write failed: %s", task_id, exc)
        if self._redis is not None and self._redis.available:
            await self._redis.set_json(f"task:{task_id}", state, _TASK_TTL_SECONDS)

    async def _load_state(self, task_id: str) -> dict[str, Any] | None:
        if self._redis is not None and self._redis.available:
            state = await self._redis.get_json(f"task:{task_id}")
            if isinstance(state, dict):
                self._local_state[task_id] = dict(state)
                return state
        local = self._local_state.get(task_id)
        if isinstance(local, dict):
            return dict(local)
        ws = self._tasks_root / task_id
        if ws.is_dir():
            from brain_os.systems.task_workspace_persist import read_state_json

            disk = read_state_json(ws)
            if isinstance(disk, dict):
                self._local_state[task_id] = dict(disk)
                return dict(disk)
        return None

    async def _update_task_index(self, task_id: str) -> None:
        local_index = [i for i in self._local_index if i != task_id]
        local_index.append(task_id)
        self._local_index = local_index[-500:]

        if self._redis is None or not self._redis.available:
            return
        index = await self._redis.get_json(_TASK_INDEX_KEY) or []
        if not isinstance(index, list):
            index = []
        if task_id in index:
            index = [i for i in index if i != task_id]
        index.append(task_id)
        # Keep bounded to avoid unbounded growth.
        index = index[-500:]
        await self._redis.set_json(_TASK_INDEX_KEY, index, _TASK_TTL_SECONDS)

    @staticmethod
    async def _emit(
        callback: ProgressCallback | None,
        event_type: str,
        **payload: Any,
    ) -> None:
        if callback is not None:
            await callback({"type": event_type, **payload})


from brain_os.systems.task_orchestrator_long_lived import (  # noqa: E402
    LongLivedTaskOrchestratorMixin,
)


class TaskOrchestrator(LongLivedTaskOrchestratorMixin, _TaskOrchestratorCore):
    """Public orchestrator: long-lived waves + base clarity/plan/execute."""
