"""Scheduled jobs: optional ``brain ask``-style pipeline runs or queue-only hooks.

Jobs are defined as a JSON array (see ``APP__HEARTBEAT_JOBS_PATH``).  Each
object has a ``name`` and ``interval_hours``.  Either:

* ``query`` — run :meth:`brain_os.pipeline.RequestPipeline.process_request` when due, or
* ``action`` == ``enqueue_pending_memory`` plus non-empty ``body`` — append to
  :class:`~brain_os.memory.pending_memory_queue.PendingMemoryQueue` (no LLM / no pipeline).
* ``action`` == ``hephaestion_nightly`` / ``hephaestion_weekly`` / ``hephaestion_monthly`` —
  system audit snapshots (``brain audit system``).
* ``action`` == ``prediction_reconcile`` — closed-loop hot-lead prediction reconciliation
  (optional ``skip_llm``; no ``query`` required).
* ``action`` == ``agent_journal`` — first-person agent reflections (``DreamMode.run_journal_only``;
  optional ``since_last_journal`` default true, ``lookback_hours`` default 12).
* ``action`` == ``pantheon_conference`` — weekly AI Engineering Conference (``brain conference run``);
  runs only on configured weekday (default Sunday IST) unless heartbeat ``--force``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.exc import SQLAlchemyError

from brain_os.config import get_settings
from brain_os.exceptions import BrainOSError
from brain_os.systems.llm_budget import (
    add_monthly_usage,
    check_budget_allows,
    estimate_turn_tokens,
    resolve_budget_bucket,
)

logger = logging.getLogger(__name__)

_HEARTBEAT_JOB_ERRORS = (
    BrainOSError,
    SQLAlchemyError,
    PydanticValidationError,
    OSError,
    ImportError,
    ModuleNotFoundError,
    RuntimeError,
    TypeError,
    ValueError,
    KeyError,
    json.JSONDecodeError,
    TimeoutError,
)

_LAST_RUN_KEY = "heartbeat:last_run:"


def load_heartbeat_jobs(jobs_path: str | Path) -> list[dict[str, Any]]:
    path = Path(jobs_path)
    if not path.is_file():
        logger.warning("Heartbeat jobs file missing: %s", path)
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Heartbeat jobs invalid JSON (%s): %s", path, exc)
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            out.append(item)
    return out


def _interval_seconds(job: dict[str, Any]) -> float:
    try:
        h = float(job.get("interval_hours", 24))
    except (TypeError, ValueError):
        h = 24.0
    return max(60.0, h * 3600.0)


async def _is_due(
    redis: Any,
    *,
    job_name: str,
    interval_seconds: float,
    force: bool,
) -> bool:
    if force:
        return True
    if redis is None or not getattr(redis, "available", False):
        return False
    key = f"{_LAST_RUN_KEY}{job_name}"
    raw = await redis.get(key)
    if not raw:
        return True
    try:
        last = float(raw)
    except (TypeError, ValueError):
        return True
    return (time.time() - last) >= interval_seconds


async def _mark_ran(redis: Any, job_name: str) -> None:
    if redis is None or not getattr(redis, "available", False):
        return
    key = f"{_LAST_RUN_KEY}{job_name}"
    await redis.set(key, str(time.time()))


async def _emit_heartbeat_tick(event_bus: Any, *, job_count: int) -> None:
    if event_bus is None:
        return
    from brain_os.systems.data_event_bus import EventType, SourceStore
    from brain_os.systems.event_emit import emit_data_event

    await emit_data_event(
        event_bus,
        event_type=EventType.HEARTBEAT_TICK,
        entity_type="heartbeat",
        entity_id="tick",
        payload={"job_count": job_count},
        source_store=SourceStore.CRM,
    )


async def run_heartbeat_jobs(
    *,
    redis: Any,
    jobs_path: str | Path,
    pipeline: Any,
    force: bool = False,
    only_name: str | None = None,
    task_orchestrator: Any = None,
    event_bus: Any = None,
    google_calendar: Any = None,
) -> list[dict[str, Any]]:
    """Execute due jobs; return one result dict per job definition (including skips)."""
    jobs = load_heartbeat_jobs(jobs_path)
    results: list[dict[str, Any]] = []
    if not jobs:
        return [{"job": "_", "status": "skipped", "reason": "no_jobs_file_or_empty"}]

    await _emit_heartbeat_tick(event_bus, job_count=len(jobs))

    settings = get_settings()
    budget_limit = int(settings.app.llm_monthly_token_budget or 0)
    budget_scope_mode = str(settings.app.llm_budget_scope or "user")

    for job in jobs:
        name = str(job.get("name", "unknown"))
        if only_name is not None and name != only_name:
            continue

        interval_s = _interval_seconds(job)
        if not await _is_due(redis, job_name=name, interval_seconds=interval_s, force=force):
            results.append({"job": name, "status": "skipped", "reason": "interval"})
            continue

        action = str(job.get("action") or "").strip().lower()
        if action == "enqueue_pending_memory":
            body = str(job.get("body") or "").strip()
            if not body:
                results.append({"job": name, "status": "skipped", "reason": "empty_body"})
                continue
            try:
                from brain_os.memory.pending_memory_queue import PendingMemoryQueue

                q = PendingMemoryQueue()
                await q.initialize()
                src = str(job.get("source") or "heartbeat").strip()[:80] or "heartbeat"
                await q.enqueue(body, source=src)
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "agents_consulted": [],
                        "action": action,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s enqueue failed", name)
                results.append({"job": name, "status": "error", "error": str(exc)})
            continue

        if action == "agency_morning_digest":
            try:
                from brain_os.services.agency.agency_notify import run_agency_morning_digest

                idle = job.get("followup_idle_days")
                top_n = job.get("top_n")
                digest = run_agency_morning_digest(
                    followup_idle_days=int(idle) if idle is not None else None,
                    top_n=int(top_n) if top_n is not None else None,
                    webhook=bool(job.get("webhook", settings.app.agency_digest_webhook_enabled)),
                    trigger=f"heartbeat:{name}",
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok" if digest.get("ok") else "error",
                        "agents_consulted": [],
                        "action": action,
                        "card_count": digest.get("card_count"),
                        "webhook_sent": digest.get("webhook_sent"),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s agency digest failed", name)
                results.append({"job": name, "status": "error", "error": str(exc)})
            continue

        if action == "calendar_upcoming":
            try:
                from brain_os.systems.calendar_heartbeat import emit_upcoming_calendar_events

                lookahead = float(job.get("lookahead_hours") or 24)
                count = await emit_upcoming_calendar_events(
                    event_bus,
                    google_calendar,
                    lookahead_hours=lookahead,
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "events_emitted": count,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s calendar_upcoming failed", name)
                results.append({"job": name, "status": "error", "error": str(exc)})
            continue

        if action == "standing_morning_brief":
            try:
                from brain_os.services.standing_morning_brief import run_standing_morning_brief

                idle = job.get("followup_idle_days")
                brief = await run_standing_morning_brief(
                    operator_name=str(job.get("operator_name") or "").strip() or None,
                    followup_idle_days=int(idle) if idle is not None else None,
                    webhook=bool(job.get("webhook", True)),
                    trigger=f"heartbeat:{name}",
                    redis=redis,
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok" if brief.get("ok") else "error",
                        "agents_consulted": [],
                        "action": action,
                        "synthesis": brief.get("synthesis"),
                        "webhook_sent": brief.get("webhook_sent"),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s morning brief failed", name)
                results.append({"job": name, "status": "error", "error": str(exc)})
            continue

        if action in ("hephaestion_nightly", "hephaestion_weekly", "hephaestion_monthly"):
            try:
                if action == "hephaestion_nightly":
                    from brain_os.audit.engine import run_nightly_audit

                    payload = run_nightly_audit(
                        run_pytest_cov=bool(job.get("run_pytest_cov", False)),
                        write_brief=bool(job.get("write_brief", True)),
                        scan_langfuse=bool(job.get("scan_langfuse", True)),
                    )
                elif action == "hephaestion_weekly":
                    from brain_os.audit.engine import run_weekly_report

                    payload = run_weekly_report()
                else:
                    from brain_os.audit.engine import run_monthly_report

                    payload = run_monthly_report()
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        **{k: v for k, v in payload.items() if k != "brief"},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s Hephaestion failed", name)
                results.append({"job": name, "status": "error", "error": str(exc)})
            continue

        if action == "prediction_reconcile":
            try:
                from brain_os.memory.prediction_reconciliation import run_prediction_reconciliation_cycle

                skip_llm = bool(
                    job.get("skip_llm", settings.app.prediction_reconcile_heartbeat_skip_llm)
                )
                report = await run_prediction_reconciliation_cycle(skip_llm=skip_llm)
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "reconciled": report.reconciled,
                        "correct": report.correct,
                        "incorrect": report.incorrect,
                        "snapshot_written": report.snapshot_written,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s prediction reconcile failed", name)
                results.append({"job": name, "status": "error", "error": str(exc)})
            continue

        if action == "operator_daily_dashboard":
            try:
                from brain_os.services.brain_daily_dashboard import build_and_persist_daily_snapshot

                target_date = str(job.get("local_date") or "").strip() or None
                skip_llm = bool(job.get("skip_llm", False))
                snap = await build_and_persist_daily_snapshot(
                    local_date=target_date,
                    include_llm=not skip_llm,
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "local_date": snap.local_date,
                        "has_narrative": snap.narrative is not None,
                        "emails_sent": snap.metrics.emails_sent_count,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s operator_daily_dashboard failed", name)
                results.append({"job": name, "status": "error", "error": str(exc)})
            continue

        if action == "pantheon_conference":
            try:
                from brain_os.services.pantheon_conference import run_pantheon_conference_heartbeat

                payload = await run_pantheon_conference_heartbeat(job, force=force)
                if payload.get("status") == "ok":
                    await _mark_ran(redis, name)
                results.append({"job": name, **payload})
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s pantheon_conference failed", name)
                results.append({"job": name, "status": "error", "error": str(exc)})
            continue

        if action == "agent_journal":
            try:
                from brain_os.brain.dream_bootstrap import build_dream_mode

                since_last = bool(job.get("since_last_journal", True))
                try:
                    lookback = float(job.get("lookback_hours", 12.0))
                except (TypeError, ValueError):
                    lookback = 12.0
                lookback = max(1.0, min(lookback, 168.0))

                dream_mode = await build_dream_mode()
                try:
                    journal_result = await dream_mode.run_journal_only(
                        since_last_journal=since_last,
                        lookback_hours=lookback,
                    )
                finally:
                    await dream_mode.close()

                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": list(journal_result.get("agents") or []),
                        "entries_saved": journal_result.get("entries_saved", 0),
                        "agents_processed": journal_result.get("agents_processed", 0),
                        "since_last_journal": since_last,
                        "lookback_hours": lookback,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s agent_journal failed", name)
                results.append({"job": name, "status": "error", "error": str(exc)})
            continue

        query = str(job.get("query") or "").strip()
        if action and not query:
            results.append({"job": name, "status": "skipped", "reason": f"unknown_action:{action}"})
            continue
        if not query:
            results.append({"job": name, "status": "skipped", "reason": "no_query"})
            continue

        from brain_os.systems.operator_inbox import require_operator_release_for_heartbeat

        release_skip = require_operator_release_for_heartbeat(job=job)
        if release_skip:
            results.append({"job": name, "status": "skipped", "reason": release_skip})
            continue

        sender = str(job.get("sender_id") or "heartbeat").strip() or "heartbeat"
        budget_scope, budget_bucket_id = resolve_budget_bucket(sender, scope_mode=budget_scope_mode)
        if budget_limit > 0 and redis is not None and getattr(redis, "available", False):
            block = await check_budget_allows(
                redis,
                limit=budget_limit,
                scope=budget_scope,
                bucket=budget_bucket_id,
            )
            if block:
                results.append({"job": name, "status": "skipped", "reason": "llm_budget"})
                continue

        try:
            response, agents, _run_id = await pipeline.process_request(
                query,
                channel="heartbeat",
                sender_id=sender,
                metadata={"heartbeat_job": name},
            )
            await _mark_ran(redis, name)
            results.append(
                {
                    "job": name,
                    "status": "ok",
                    "agents_consulted": agents,
                    "response_chars": len(response or ""),
                }
            )
            if budget_limit > 0 and redis is not None and getattr(redis, "available", False):
                est = estimate_turn_tokens(query, response or "", agents)
                if est > 0:
                    await add_monthly_usage(
                        redis,
                        scope=budget_scope,
                        bucket=budget_bucket_id,
                        tokens=est,
                    )
        except _HEARTBEAT_JOB_ERRORS as exc:
            logger.exception("Heartbeat job %s pipeline failed", name)
            results.append({"job": name, "status": "error", "error": str(exc)})

    try:
        from brain_os.systems.heartbeat_outcome_log import append_heartbeat_outcomes

        append_heartbeat_outcomes(results)
    except Exception:
        logger.debug("heartbeat outcome log append skipped", exc_info=True)

    return results
