"""Scheduled jobs: optional ``brain ask``-style pipeline runs or queue-only hooks.

Jobs are defined as a JSON array (see ``APP__HEARTBEAT_JOBS_PATH``).  Each
object has a ``name`` and ``interval_hours``.  Optional wall-clock fields
(``at`` ``HH:MM``, ``tz`` IANA name, ``days`` weekday list like ``["mon","fri"]``;
omit ``days`` for daily) gate jobs that set ``at``; ``interval_hours`` applies
when ``at`` is absent.  Either:

* ``query`` — run :meth:`brain_os.pipeline.RequestPipeline.process_request` when due, or
* ``action`` == ``enqueue_pending_memory`` plus non-empty ``body`` — append to
  :class:`~brain_os.memory.pending_memory_queue.PendingMemoryQueue` (no LLM / no pipeline).
* ``action`` == ``hephaestion_nightly`` / ``hephaestion_weekly`` / ``hephaestion_monthly`` —
  system audit snapshots (``brain audit system``).
* ``action`` == ``graphify_refresh`` — refresh the Graphify code graph from the curated
  corpus (Tree-sitter only, no LLM; schedule before ``hephaestion_nightly`` so the audit
  reads a fresh graph).
* ``action`` == ``graph_sync_maintenance`` — sync Neo4j Chunk→entity links into Qdrant
  payloads, then run a graph/vector drift audit (no LLM).
* ``action`` == ``graph_person_hygiene`` — clear junk Person emails, flag internals,
  merge valid email duplicates (no LLM; writes Neo4j).
* ``action`` == ``graph_analytics`` — Aura Graph Analytics job
  (``analytics_job``: ``dedup`` | ``communities`` | ``pagerank``); skip when
  ``data/operations/GRAPH_ANALYTICS_PAUSED`` exists (no LLM; GDS credit burn).
* ``action`` == ``prediction_reconcile`` — closed-loop hot-lead prediction reconciliation
  (optional ``skip_llm``; no ``query`` required).
* ``action`` == ``agent_journal`` — first-person agent reflections (``DreamMode.run_journal_only``;
  optional ``since_last_journal`` default true, ``lookback_hours`` default 12).
* ``action`` == ``pantheon_conference`` — weekly AI Engineering Conference (``brain conference run``);
  runs only on configured weekday (default Sunday IST) unless heartbeat ``--force``.
* ``action`` == ``deliverability_daily`` — sender-pool DNS checks (SPF/DKIM/DMARC,
  fail-closed) + bounce/reply reputation rollup with auto-pause (no LLM).
* ``action`` == ``compliance_monthly`` — consent suppression sweep into the
  blacklist + previous-month compliance report export (no LLM).
* ``action`` == ``calendar_upcoming`` — emit ``CALENDAR_UPCOMING`` on the data event bus.
* ``action`` == ``proactive_watch_inbox`` — stale deals + reply SLA → operator inbox
  (``data/operations/proactive_inbox_queue.jsonl``; no LLM; ``skip_llm: true``).
* ``action`` == ``morning_brain`` — unified morning action queue from dream,
  serendipity, installed base, wonder, prediction drift, and revenue desk
  (``brain_os.services.morning_brain.run_morning_brain``; optional ``force``,
  ``followup_idle_days``; respects ``APP__MORNING_BRAIN_ENABLED`` unless forced).
* ``action`` == ``wonder_research`` — curiosity research pass over
  ``data/wonder/research_queue.yml`` (findings + journal + graph + episodes)
* ``action`` == ``wonder_leash`` — propose EAT digests from recent wonder
  findings (never auto-approve); webhook ``wonder_leash`` when new proposals
* ``action`` == ``curiosity_digest`` — weekly cross-pollination of Agora notes +
  curiosity findings (max 3 cited connections → morning brief)
* ``action`` == ``serendipity_collect`` — serendipity signal collectors
  (``data/serendipity/collectors.yml``; budget-capped; respects ``PAUSED``)
* ``action`` == ``serendipity_digest`` — top queue items + SLA nudges (Slack off by default).
* ``action`` == ``serendipity_tuning`` — monthly hit-rate brief
  ``data/serendipity/tuning/YYYY-MM.md`` (no LLM; operator edits config).
* ``action`` == ``reply_learning_weekly`` — weekly markdown learning brief from
  reply-loop records + funnel snapshot (no LLM).
* ``action`` == ``email_taxonomy_drift_weekly`` — inbound class distribution vs
  trailing 4 weeks; silent/exploding classes → morning brief (no LLM).
* ``action`` == ``retrieval_eval_weekly`` — golden P@5 harness →
  ``data/brain/retrieval_p5_ledger.jsonl`` (vitals Brain line; alert if drop >0.1).
* ``action`` == ``slope_snapshot_nightly`` — organ-slope metrics →
  ``data/eval/slope_ledger.jsonl`` (no LLM; compounding proof).
* ``action`` == ``outreach_experiment_weekly`` — (angle × subject_style) scoreboard,
  bandit posterior, and FIFO winning few-shots (``prompts/outreach_winning_examples.txt``).
* ``action`` == ``message_bandit_no_reply_scan`` — aged sends with ``variant_id`` and
  no reply → Thompson failure (closes loop with reply_success from reply_loop).
* ``action`` == ``founder_voice_refresh_monthly`` — distill gold corpus → candidate
  profile/prompt; enqueue ``founder_voice`` for ``brain learning approve`` (never auto-apply).
* ``action`` == ``tone_clinic_weekly`` — mine/score outbound teachers → few-shot
  candidate; enqueue ``tone_clinic`` for ``brain learning approve`` (never auto-apply).
* ``action`` == ``installed_base_weekly`` — installed-base digest: fleet coverage,
  triggers, warm funnel, cold-vs-warm reply rates → journal + operator webhook.
* ``action`` == ``portfolio_refresh_weekly`` — re-stamp Atlas
  ``atlas_production_portfolio.json`` per-project ``last_evidence_at`` from
  production-update dirs + logbook (no phase invention; keeps staleness honest).
* ``action`` == ``production_photo_watch`` — scan ``data/production updates/**``
  for new shop photos under WO-matching dirs → logbook ``photo_drop`` + queue
  customer-update draft (never sends).
* ``action`` == ``wo_watch`` — stage ETA delay flags + overdue milestones;
  when the job sets ``queue_drafts`` / ``chase``, also run the payment-chase
  ladder (nudge / reminder / founder-call / dispatch_hold). Chase queues
  Calliope drafts + inbox tasks only — never auto-sends.
* ``action`` == ``tally_bank_book_watch`` — mtime watch on Tally bank-book
  imports dir; reminds operator to run ``brain receivables reconcile``
  (verification only; never auto-confirms).
* ``action`` == ``dream_digest_weekly`` — roll the last 7 days of
  ``data/dream_reports/*.md`` into one digest and post to the operator
  webhook/Slack (no LLM).
* ``action`` == ``learn_week_brief`` — weekly “what did we learn?” one-pager
  from EAT digests + Mnemon corrections + dream gaps
  (``data/reports/learn_week_YYYY-Www.md``; webhook event ``learn_week``).
* ``action`` == ``relationship_breath`` — end-of-day relationship gaps
  (replies / idle open quotes / follow-ups / missing next step)
  → ``data/reports/relationship_breath_YYYY-MM-DD.md``; webhook
  ``relationship_breath`` (no LLM).
* ``action`` == ``knowledge_grill_weekly`` — teach-back on recent EAT digests
  (keyword overlap vs retrieval); ``data/reports/knowledge_grill_YYYY-Www.md``;
  webhook ``knowledge_grill`` (no LLM).
* ``action`` == ``dream_stale_check`` — alert when no dream cycle in
  ``APP__DREAM_STALE_HOURS`` (default 24); fires ``dream_stale`` webhook
  (critical); Slack mirror off unless ``post_slack`` / ``APP__DREAM_STALE_SLACK_ALERT``; deduped per stale episode.
* ``action`` == ``dream_nightly`` — full dream cycle (same path as ``brain dream`` /
  ``execute_dream_cycle``); wall-clock ``at``/``tz``; skips when checkpoint
  shows another cycle running; total runtime capped (default 90 min).
* ``action`` == ``curious_harvest_nightly`` — Curious Brain OS doubt harvest (after dream; ``APP__CURIOUS_IRA_ENABLED``).
* ``action`` == ``curious_email_daily`` — Curious Brain OS daily questions email (07:15 IST; system mail).
* ``action`` == ``stomach_email_digest`` — Gmail poll + Delphi classify + DigestiveSystem ingest
  (continuous stomach; one poll cycle; rate-limited by email processor).
* ``action`` == ``gmail_oauth_preflight`` — token expiry warn (7d) + critical alert path
  when mailbox OAuth is dead (vitals + morning P0).
* ``action`` == ``write_contract_retry_drain`` — replay write-contract retry queue
  (was unbounded because enqueue ran with no scheduled drain).
* ``action`` == ``backup_nightly`` — WAL-safe SQLite fleet + Neo4j dump +
  runtime/secrets archive via ``scripts/backup_ira.sh`` (after dream);
  fail-closed outcome with bytes/duration/per-store status.
* ``action`` == ``learning_loop_watchdog`` — daily check of last-success
  timestamps for dream / prediction_reconcile / learning_promote / gepa /
  graphe_meta / backup_nightly, **plus** the universal schedule watchdog over
  every enabled job in ``heartbeat_jobs.json`` (cadence from interval/at/days;
  alert after 2 missed slots); injects CRITICAL morning-brief alerts
  (fail-closed: watchdog errors become alerts).
* ``action`` == ``learning_reconcile_weekly`` — alias of ``prediction_reconcile``
  (weekly prediction-outcome pass; optional ``skip_llm``).
* ``action`` == ``learning_promote_weekly`` — compile Graphe sessions into
  procedure candidates (optional ``compile_first``, default true) then promote
  into ProceduralMemory (respects moat/env gates; no LLM).
* ``action`` == ``gepa_overlay_weekly`` — compile GEPA strategy overlays from
  tool-invocation stats, then promote candidate → active when gated (no LLM).
* ``action`` == ``agent_peer_review`` — weekly critic rotation: sample one
  high-traffic agent's recent outputs, Vera/Metis/Sophia scorecard, GEPA
  overlay candidate into the learning approval queue (optional ``use_llm``).
* ``action`` == ``correction_ledger_weekly`` — audit stale Mnemon ledger entities;
  auto-verify rows missing ``corrected_at`` only (optional ``fix_missing_dates``,
  ``dry_run``; no LLM).
* ``action`` == ``db_hygiene`` — SQLite fleet ``PRAGMA wal_checkpoint(TRUNCATE)``
  under ``data/**/*.db`` (skip locked; Neo4j/archive excluded), embedding-cache
  byte-cap prune report, optional ``archive_stray_predictions`` (no LLM).
* ``action`` == ``mem0_forgetting_weekly`` — Mem0 hygiene sweep (archive →
  hard-delete after 30d when ``APP__DREAM_MEM0_FORGETTING_ENABLED``); audit log
  ``data/operations/mem0_forgetting_events.jsonl``.
* ``action`` == ``graphe_meta_backfill_weekly`` — backfill missing
  ``learning_meta_json`` on Graphe rows (optional ``limit``, ``dry_run``).
* ``action`` == ``session_mine_batch`` — per-session miner over unmined Graphe
  rows + drain ``session_mine_queue.jsonl`` (optional ``limit``, ``since_days``;
  uses cheap ``brain_model_fast``; small batches every few hours).
* ``action`` == ``social_learn_daily`` — read-only Moltbook learning ear
  (``brain_os.social.feed_learner``): harvest allowlisted submolts into
  ``data/social/learning_journal.jsonl``; optional ``build_digest`` (default
  true on the weekly cadence) writes ``data/social/learning_digest_YYYY-WW.md``.
  Requires only ``MOLTBOOK_API_KEY``; respects ``data/social/PAUSED``.
* ``action`` == ``social_watch_comments`` — read-only engagement ear
  (``brain_os.social.comment_watcher``): poll comments on posted outbox threads,
  journal non-noise replies tagged ``social_engagement``. Optional ``use_llm``.
* ``action`` == ``social_daily_cycle`` — queue up to 3 posts from approved
  content, optional auto-publish (``BRAIN_SOCIAL_AUTO_PUBLISH``), comment watch,
  operator digest email (``BRAIN_SOCIAL_DIGEST_EMAIL`` / ``GOOGLE_IRA_EMAIL``).
* ``action`` == ``sourcing_weekly`` — Places NA state-grid + Firecrawl discovery,
  dedupe vs CRM/Neo4j, COLD_LIST_GATE, refresh top-100 cohort (no send).
* ``action`` == ``universe_refresh_weekly`` — NA DEMO shortlist from classify CSVs;
  suppress via send ledger + CRM + ``pf1_universe_drop_domains.txt`` (no send).
  Optional ``run_pf1_global`` / ``run_phase7`` wrap legacy builder scripts.
* ``action`` == ``buyer_persona_refresh_weekly`` — Apollo people for shortlisted
  domains missing a named contact; optional NeverBounce; CRM upsert (no send).
  Requires ``APP__BUYER_PERSONA_REFRESH_ENABLED=true``.
* ``action`` == ``drip_cycle_daily`` — ``AutonomousDripEngine.run_cycle()`` (draft/send
  per drip config; respects training mode).
* ``action`` == ``top100_cycle_daily`` — ``run_top100_cycle`` (draft-only → Tinder;
  cold experiment lane, capped).
* ``action`` == ``warm_lane_daily`` — ``draft_warm_lane_batch`` (top N warm
  obligations → approval queue ``campaign_id=warm_lane``; never sends).
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import os
import sys
import tempfile
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import fcntl
except ImportError:  # pragma: no cover — POSIX-only; heartbeat runs on Linux/macOS
    fcntl = None  # type: ignore[assignment]

from brain_os.config import get_settings
from brain_os.contracts.draft_sender import DraftSender
from brain_os.systems.llm_budget import (
    add_monthly_usage,
    check_budget_allows,
    estimate_turn_tokens,
    resolve_budget_bucket,
)

logger = logging.getLogger(__name__)

# Procedural PG DDL is Alembic/server-owned; heartbeat ensures once per process.
_PROCEDURAL_TABLES_ENSURED = False


def _job_error_outcome(name: str, exc: BaseException, **extra: Any) -> dict[str, Any]:
    """Build a heartbeat error result with a traceback tail for ops hotspots."""
    import traceback

    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    flat = "".join(tb_lines).strip().splitlines()
    tail = "\n".join(flat[-20:]) if flat else ""
    out: dict[str, Any] = {
        "job": name,
        "status": "error",
        "error": str(exc),
        "traceback_tail": tail,
    }
    out.update(extra)
    return out


# Fail-closed interpreter / dep sanity for heartbeat (stale .venv caused silent
# prediction_reconcile / outreach job deaths). Versions are major.minor floors
# aligned with poetry.lock; bump when lock majors move.
_HEARTBEAT_CRITICAL_DEPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("pydantic", (2, 0)),
    ("sqlalchemy", (2, 0)),
    ("openai", (1, 0)),
)
_heartbeat_deps_ok: bool | None = None


def _parse_version_tuple(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in (raw or "").split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def verify_heartbeat_runtime_deps(*, force: bool = False) -> None:
    """Ensure this process is the in-project ``.venv`` with expected dep floors.

    Raises ``RuntimeError`` (logged CRITICAL) when the interpreter or packages
    drift — callers must refuse to run jobs rather than fail silently mid-loop.
    """
    global _heartbeat_deps_ok
    if _heartbeat_deps_ok is True and not force:
        return

    # Prefer sys.prefix — ``.venv/bin/python`` often symlinks to Homebrew's
    # real binary, so Path(sys.executable).resolve() loses the ``.venv`` marker.
    prefix = Path(sys.prefix).resolve()
    exe_raw = Path(sys.executable)
    in_project_venv = (
        prefix.name == ".venv"
        or "/.venv/" in str(prefix)
        or "/.venv/" in str(exe_raw)
        or str(exe_raw).endswith("/.venv/bin/python")
        or str(exe_raw).endswith("/.venv/bin/python3")
    )
    if not in_project_venv:
        msg = (
            f"Heartbeat refused: interpreter is not the project .venv "
            f"(executable={exe_raw}, prefix={prefix}). Use poetry run / "
            f"repo .venv/bin/python — see AGENTS.md § Runtime environment."
        )
        logger.critical(msg)
        _heartbeat_deps_ok = False
        raise RuntimeError(msg)

    problems: list[str] = []
    for dist_name, min_ver in _HEARTBEAT_CRITICAL_DEPS:
        try:
            installed = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{dist_name}: not installed")
            continue
        got = _parse_version_tuple(installed)
        if got < min_ver:
            problems.append(
                f"{dist_name}: {installed} < required {'.'.join(str(x) for x in min_ver)}"
            )
    # Smoke-import the installed package (editable .venv → src/).
    try:
        import brain_os
    except ImportError as exc:
        problems.append(f"ira import failed: {exc}")

    if problems:
        msg = (
            "Heartbeat refused: dependency sanity check failed — "
            + "; ".join(problems)
            + ". Run: poetry install --sync --with heavy,dev"
        )
        logger.critical(msg)
        _heartbeat_deps_ok = False
        raise RuntimeError(msg)

    logger.info(
        "Heartbeat dependency sanity OK (python=%s prefix=%s pydantic=%s sqlalchemy=%s openai=%s)",
        exe_raw,
        prefix,
        importlib.metadata.version("pydantic"),
        importlib.metadata.version("sqlalchemy"),
        importlib.metadata.version("openai"),
    )
    _heartbeat_deps_ok = True


# Fail-loud: ANY job exception becomes an error outcome. (Previously a narrow
# tuple let AttributeError/etc. abort the whole tick with zero error rows —
# the Jul 2026 silent-stale incident.) CancelledError is BaseException, not
# caught here.
_HEARTBEAT_JOB_ERRORS = (Exception,)

_HEARTBEAT_LOG_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    ImportError,
    json.JSONDecodeError,
)

from brain_os.systems.heartbeat_schedule import (  # noqa: E402 — after local error tuples
    _CATCH_UP_LATE_GRACE_SECONDS,
    _DREAM_BUSY_SKIP_FILE,
    _LAST_RUN_FILE,
    _LAST_RUN_KEY,
    _WEEKDAY_ALIASES,
    DEFAULT_CATCH_UP_WINDOW_HOURS,
    _atomic_write_json,
    _dream_busy_dedupe_path,
    _dream_busy_slot_key,
    _due_state,
    _fetch_last_run_unix,
    _interval_seconds,
    _is_due,
    _job_schedule_weekdays,
    _job_trailing_window_days,
    _last_run_file_path,
    _last_run_lock_path,
    _mark_ran,
    _now_in_tz,
    _parse_at_hhmm,
    _read_last_run_file,
    _should_log_dream_busy_skip,
    _wall_clock_fire_on_date,
    _wall_clock_slot_unsatisfied,
    _write_last_run_file,
    current_wall_clock_fire_time,
    is_interval_job_due,
    is_wall_clock_job_due,
    load_heartbeat_jobs,
    missed_wall_clock_fire,
    wall_clock_due_info,
)

_DREAM_BUSY_GRACE_SECONDS = 120.0
_WAKE_GAP_SECONDS = 30 * 60


def _dream_checkpoint_busy(
    *,
    grace_seconds: float = _DREAM_BUSY_GRACE_SECONDS,
    max_busy_seconds: float | None = None,
) -> bool:
    """True when another dream cycle appears to hold the checkpoint.

    ``grace_seconds`` is unused historically; ``max_busy_seconds`` caps how long a
    ``status=running`` checkpoint blocks a new cycle (abandoned runs unblock).
    """
    _ = grace_seconds
    try:
        from brain_os.memory.dream_mode_checkpoint import load_dream_checkpoint
        from brain_os.memory.dream_mode_constants import DREAM_CHECKPOINT_PATH
    except ImportError:
        return False
    ckpt = load_dream_checkpoint(DREAM_CHECKPOINT_PATH)
    if str(ckpt.get("status") or "") != "running":
        return False
    updated_raw = str(ckpt.get("updated_at") or "").strip()
    if not updated_raw:
        return True
    try:
        updated = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
    except ValueError:
        return True
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    age = (datetime.now(UTC) - updated.astimezone(UTC)).total_seconds()
    if max_busy_seconds is not None and age > max(max_busy_seconds, 0.0):
        return False
    return True


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
    dry_run: bool = False,
    now: datetime | None = None,
    catch_up_window_hours: float | None = None,
    catch_up_scan: bool = False,
    draft_sender: DraftSender | None = None,
) -> list[dict[str, Any]]:
    """Execute due jobs; return one result dict per job definition (including skips).

    Missed wall-clock (``at``) slots within ``catch_up_window_hours`` still fire
    once on the next tick (outcome tagged ``catch_up=true``). ``catch_up_scan``
    is reserved for wake/start callers (daemon / server loop); currently unused
    beyond documenting wake intent.
    """
    _ = catch_up_scan
    try:
        from brain_os.logging_setup import setup_ira_logging

        setup_ira_logging(console=False)
        logger.info("Brain OS heartbeat logging attached to data/logs/")
    except Exception:  # noqa: BLE001 — logging setup failure must not stop the heartbeat
        logger.warning("Heartbeat file logging setup failed", exc_info=True)
    try:
        from brain_os.config_learning_drift import warn_learning_critical_config_drift

        warn_learning_critical_config_drift()
    except Exception:  # noqa: BLE001 — config drift check is advisory
        logger.warning("Learning-critical config drift check failed", exc_info=True)
    global _PROCEDURAL_TABLES_ENSURED
    if not _PROCEDURAL_TABLES_ENSURED:
        try:
            settings0 = get_settings()
            if getattr(settings0.app, "pg_store_procedural_enabled", False):
                from brain_os.memory.procedural_backend import ensure_procedural_tables

                await ensure_procedural_tables()
        except Exception:  # noqa: BLE001 — missing procedural tables must not stop the heartbeat
            logger.warning(
                "Heartbeat ensure_procedural_tables failed — continuing",
                exc_info=True,
            )
        _PROCEDURAL_TABLES_ENSURED = True
    try:
        from brain_os.config_learning_drift import warn_mem0_forgetting_job_drift

        warn_mem0_forgetting_job_drift(jobs_path=jobs_path)
    except Exception:  # noqa: BLE001 — config drift check is advisory
        logger.warning("Mem0 forgetting job drift check failed", exc_info=True)
    try:
        verify_heartbeat_runtime_deps()
    except RuntimeError as exc:
        dep_fail = [_job_error_outcome("_", exc, reason="dependency_sanity_failed")]
        try:
            from brain_os.systems.heartbeat_outcome_log import append_heartbeat_outcomes

            append_heartbeat_outcomes(dep_fail)
        except _HEARTBEAT_LOG_ERRORS:
            logger.debug("heartbeat outcome log append skipped", exc_info=True)
        return dep_fail

    jobs = load_heartbeat_jobs(jobs_path)
    results: list[dict[str, Any]] = []
    if not jobs:
        return [{"job": "_", "status": "skipped", "reason": "no_jobs_file_or_empty"}]

    await _emit_heartbeat_tick(event_bus, job_count=len(jobs))

    settings = get_settings()
    budget_limit = int(settings.app.llm_monthly_token_budget or 0)
    budget_scope_mode = str(settings.app.llm_budget_scope or "user")
    if catch_up_window_hours is None:
        catch_up_window_hours = float(
            getattr(settings.app, "heartbeat_catch_up_window_hours", None)
            or DEFAULT_CATCH_UP_WINDOW_HOURS
        )
    catch_up_jobs: set[str] = set()
    max_run = int(getattr(settings.app, "heartbeat_max_jobs_per_tick", None) or 8)
    if only_name is not None or force:
        max_run = 0
    ran_this_tick = 0
    ordered_jobs = sorted(
        jobs,
        key=lambda j: (0 if j.get("at") else 1, str(j.get("name") or "")),
    )

    for job in ordered_jobs:
        name = str(job.get("name", "unknown"))
        if only_name is not None and name != only_name:
            continue
        if job.get("enabled") is False:
            results.append({"job": name, "status": "skipped", "reason": "disabled"})
            continue

        interval_s = _interval_seconds(job)
        due, is_catch_up = await _due_state(
            redis,
            job=job,
            job_name=name,
            interval_seconds=interval_s,
            force=force,
            now=now,
            catch_up_window_hours=catch_up_window_hours,
        )
        if not due:
            skip_reason = "schedule" if job.get("at") else "interval"
            results.append({"job": name, "status": "skipped", "reason": skip_reason})
            continue
        if max_run > 0 and ran_this_tick >= max_run:
            results.append({"job": name, "status": "skipped", "reason": "tick_budget"})
            continue
        if is_catch_up:
            catch_up_jobs.add(name)
        ran_this_tick += 1

        # Stamp spend attribution for every LLM call inside this job.
        try:
            from brain_os.services.llm_caller_context import (
                LlmCallerContext,
                set_llm_caller_context,
            )

            set_llm_caller_context(
                LlmCallerContext(
                    source="background",
                    job=name,
                    outcome_kind="heartbeat",
                    call_site=str(job.get("action") or name),
                )
            )
        except Exception:  # noqa: BLE001 — caller-context stamping is best-effort telemetry
            logger.debug("llm_caller_context stamp failed for job=%s", name, exc_info=True)

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
                results.append(_job_error_outcome(name, exc))
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
                results.append(_job_error_outcome(name, exc))
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
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "proactive_watch_inbox":
            try:
                from brain_os.services.proactive_watch_routines import run_proactive_watch_inbox

                watch_result = await run_proactive_watch_inbox(
                    stale_deal_days=int(job.get("stale_deal_days") or 7),
                    reply_sla_hours=int(job.get("reply_sla_hours") or 48),
                    limit=int(job.get("limit") or 15),
                    job_name=name,
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "agents_consulted": [],
                        "action": action,
                        "skip_llm": True,
                        **watch_result,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s proactive_watch_inbox failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "morning_brain":
            try:
                from brain_os.services.morning_brain import run_morning_brain

                pantheon = getattr(pipeline, "_pantheon", None) or getattr(
                    pipeline, "pantheon", None
                )
                crm = getattr(pipeline, "_crm", None) or getattr(pipeline, "crm", None)
                email_processor = getattr(pipeline, "_email_processor", None) or getattr(
                    pipeline, "email_processor", None
                )
                idle = job.get("followup_idle_days")
                mb_result = await run_morning_brain(
                    pantheon=pantheon,
                    crm=crm,
                    email_processor=email_processor,
                    trigger=f"heartbeat:{name}",
                    followup_idle_days=int(idle) if idle is not None else None,
                    force=bool(job.get("force") or force),
                )
                await _mark_ran(redis, name)
                report = (
                    mb_result.get("report") if isinstance(mb_result.get("report"), dict) else {}
                )
                actions_list = report.get("actions") or []
                delivery = (
                    mb_result.get("delivery") if isinstance(mb_result.get("delivery"), dict) else {}
                )
                # A brief that failed to reach the operator is an error, not a silent ok.
                mb_ok = bool(mb_result.get("ok"))
                if delivery and not delivery.get("ok"):
                    mb_ok = False
                # "morning latest.json stuck" law: latest.json must point at
                # today's report (operator-tz date) after every run — a stale
                # pointer here is the exact failure mode that left latest.json
                # frozen on 2026-06-23 for weeks while dated reports advanced.
                latest_stale_reason: str | None = None
                try:
                    from brain_os.services.morning_brain import (
                        _local_date_iso,
                        load_morning_brain_report,
                    )

                    today_local = _local_date_iso()
                    latest_report = load_morning_brain_report()
                    latest_local_date = (
                        latest_report.local_date if latest_report is not None else None
                    )
                    if latest_local_date != today_local:
                        latest_stale_reason = (
                            f"latest_json_stale: local_date={latest_local_date!r} "
                            f"expected={today_local!r}"
                        )
                        mb_ok = False
                except Exception as exc:  # noqa: BLE001 — staleness probe failure is reported as a stale reason
                    logger.exception("Heartbeat job %s latest.json date check failed", name)
                    latest_stale_reason = f"latest_json_check_error: {exc}"
                    mb_ok = False
                results.append(
                    {
                        "job": name,
                        "status": "ok" if mb_ok else "error",
                        "action": action,
                        "agents_consulted": [],
                        "action_count": len(actions_list),
                        "must_act_count": report.get("must_act_count"),
                        "critical_count": report.get("critical_count"),
                        "skipped": bool(mb_result.get("skipped")),
                        "reason": mb_result.get("reason")
                        or delivery.get("error")
                        or latest_stale_reason,
                        "delivery": delivery,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s morning_brain failed", name)
                results.append(_job_error_outcome(name, exc))
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
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "evidence_freshness_sweep":
            try:
                from brain_os.services.evidence_freshness import run_evidence_freshness_sweep

                sweep = await run_evidence_freshness_sweep(dispatch=bool(job.get("dispatch")))
                await _mark_ran(redis, name)
                summary = sweep.get("summary") or {}
                results.append(
                    {
                        "job": name,
                        "status": "ok" if sweep.get("ok") else "error",
                        "agents_consulted": [],
                        "action": action,
                        "summary": summary,
                        "path": sweep.get("path"),
                        "queue_items": summary.get("total_queue_items"),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s evidence_freshness_sweep failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "deliverability_daily":
            try:
                from brain_os.services.deliverability.dns_check import run_dns_checks
                from brain_os.services.deliverability.reputation import run_reputation_rollup
                from brain_os.services.deliverability.sender_pool import pool_status

                dns_results = run_dns_checks()
                rep_results = run_reputation_rollup()
                status_rows = pool_status()
                dns_failed = [r["sender_id"] for r in dns_results if r.get("ok") is False]
                auto_paused = [r["sender_id"] for r in rep_results if r.get("auto_paused")]
                summary = (
                    f"deliverability: {len(status_rows)} senders, "
                    f"{len(dns_failed)} dns-failed, {len(auto_paused)} auto-paused"
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "summary": summary,
                        "senders": len(status_rows),
                        "dns_failed": dns_failed,
                        "auto_paused": auto_paused,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s deliverability_daily failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "compliance_monthly":
            try:
                from datetime import UTC, datetime

                from brain_os.services.consent import suppression_sweep, write_compliance_report

                swept = suppression_sweep()
                now = datetime.now(UTC)
                prev_month = (
                    f"{now.year - 1}-12" if now.month == 1 else f"{now.year}-{now.month - 1:02d}"
                )
                report_path = write_compliance_report(prev_month)
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "summary": (
                            f"compliance: report {report_path.name}, "
                            f"{swept} address(es) swept into blacklist"
                        ),
                        "report_path": str(report_path),
                        "suppression_swept": swept,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s compliance_monthly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "buying_signal_weekly":
            try:
                import asyncio as _asyncio

                from brain_os.services.lead_scoring import buying_signal_sweep

                sweep = await _asyncio.to_thread(buying_signal_sweep)
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "summary": (
                            f"buying signals: {sweep.get('checked', 0)} leads checked, "
                            f"{sweep.get('hits', 0)} signal hit(s)"
                        ),
                        **{k: v for k, v in sweep.items() if k != "details"},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s buying_signal_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "wonder_research":
            try:
                from brain_os.systems.wonder import build_default_wonder

                wonder = await build_default_wonder(graph=getattr(pipeline, "_graph", None))
                summary = await wonder.run_once()
                await _mark_ran(redis, name)
                from brain_os.systems.heartbeat_writes_registry import make_counts

                topics_run = int(summary.get("topics_run") or 0)
                findings = int(summary.get("findings_written") or 0)
                errors = int(summary.get("errors") or 0)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        **summary,
                        "counts": make_counts(
                            attempted=max(topics_run, findings, 1),
                            written=findings,
                            failed=errors,
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s wonder_research failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "wonder_leash":
            try:
                from brain_os.systems.wonder_leash import run_wonder_leash

                leash = run_wonder_leash(
                    days=int(job.get("days", 14)),
                    limit=int(job.get("limit", 25)),
                    webhook=bool(job.get("webhook", True)),
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        **{k: v for k, v in leash.items() if k != "proposals"},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s wonder_leash failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "curiosity_digest":
            try:
                from brain_os.systems.curiosity_tickets import build_cross_pollination_digest

                digest = await build_cross_pollination_digest(
                    use_llm=not bool(job.get("skip_llm", False)),
                    days=int(job.get("days", 7)),
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "connections": len(digest.get("connections") or []),
                        "agora_notes": digest.get("agora_notes"),
                        "curiosity_findings": digest.get("curiosity_findings"),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s curiosity_digest failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "serendipity_collect":
            try:
                from brain_os.services.serendipity.runner import SerendipityRunner

                summary = await SerendipityRunner().run_once()
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": summary.get("status", "ok"),
                        "action": action,
                        "agents_consulted": [],
                        **{k: v for k, v in summary.items() if k != "status"},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s serendipity_collect failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "serendipity_collision":
            try:
                from brain_os.services.serendipity.collisions import run_collision_ritual

                summary = await run_collision_ritual(force=bool(job.get("force", False)))
                await _mark_ran(redis, name)
                raw_status = str(summary.get("status") or "ok").strip().lower()
                # Idle / empty-pool / cadence skips still prove the job ran.
                if raw_status in ("no_pools", "paused", "disabled") or raw_status == "skipped":
                    hb_status = "ok_idle"
                elif raw_status in ("ok", "ok_idle", "error"):
                    hb_status = raw_status
                else:
                    hb_status = "ok"
                results.append(
                    {
                        "job": name,
                        "status": hb_status,
                        "action": action,
                        "agents_consulted": [],
                        "ritual_status": raw_status,
                        **{k: v for k, v in summary.items() if k != "status"},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s serendipity_collision failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "serendipity_digest":
            try:
                from brain_os.services.serendipity.opportunity_queue import run_daily_digest

                digest = await run_daily_digest(
                    limit=int(job.get("limit", 5) or 5),
                    post_slack=bool(job.get("post_slack", False)),
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": digest.get("status", "ok"),
                        "action": action,
                        "agents_consulted": [],
                        **digest,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s serendipity_digest failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "serendipity_tuning":
            try:
                from brain_os.services.serendipity.learning import run_serendipity_tuning

                summary = run_serendipity_tuning(force=bool(job.get("force", False)))
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": summary.get("status", "ok"),
                        "action": action,
                        "agents_consulted": [],
                        **{k: v for k, v in summary.items() if k != "status"},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s serendipity_tuning failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "dream_digest_weekly":
            try:
                from brain_os.memory.dream_report_writer import run_dream_digest_weekly

                digest = run_dream_digest_weekly(days=int(job.get("days", 7)))
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        **digest,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job '%s' (dream_digest_weekly) failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "learn_week_brief":
            try:
                from brain_os.services.weekly_learn_brief import run_weekly_learn_brief

                learn = run_weekly_learn_brief(
                    days=int(job.get("days", 7)),
                    webhook=bool(job.get("webhook", True)),
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        **learn,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job '%s' (learn_week_brief) failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "teacher_daily":
            try:
                from brain_os.services.teacher_daily_lesson import run_teacher_daily_lesson

                lesson = run_teacher_daily_lesson(
                    webhook=bool(job.get("webhook", True)),
                    enqueue=bool(job.get("enqueue", True)),
                    teacher_id=str(job["teacher_id"]).strip() if job.get("teacher_id") else None,
                    weekday=str(job["weekday"]).strip() if job.get("weekday") else None,
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        **lesson,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job '%s' (teacher_daily) failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "relationship_breath":
            try:
                from brain_os.services.relationship_breath import run_relationship_breath

                breath = run_relationship_breath(
                    followup_idle_days=int(job.get("followup_idle_days", 15)),
                    quote_idle_days=int(job.get("quote_idle_days", 7)),
                    limit=int(job.get("limit", 25)),
                    webhook=bool(job.get("webhook", True)),
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        **breath,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job '%s' (relationship_breath) failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "knowledge_grill_weekly":
            try:
                from brain_os.services.knowledge_grill_weekly import run_knowledge_grill

                grill = await run_knowledge_grill(
                    days=int(job.get("days", 7)),
                    webhook=bool(job.get("webhook", True)),
                    pass_threshold=float(job.get("pass_threshold", 0.35)),
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        **grill,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job '%s' (knowledge_grill_weekly) failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "dream_stale_check":
            try:
                from brain_os.services.dream_stale import run_dream_stale_check

                stale = await run_dream_stale_check(
                    force=bool(job.get("force", False)),
                    post_slack=job.get("post_slack"),
                )
                await _mark_ran(redis, name)
                # Keep heartbeat outcome status in {ok, ok_idle, skipped, error}.
                # dream_staleness() uses green/yellow/red for vitals — do not let
                # that overwrite the schedule-watchdog success marker.
                # Job-level "skipped" (already alerted / disabled) still proves the
                # check ran — map to ok_idle so schedule watchdog is not silent.
                hb_status = str(stale.get("status_job") or "ok").strip().lower()
                if hb_status == "skipped":
                    hb_status = "ok_idle"
                elif hb_status not in ("ok", "ok_idle", "error"):
                    hb_status = "ok"
                vitals_status = stale.get("status")
                results.append(
                    {
                        "job": name,
                        "status": hb_status,
                        "action": action,
                        "agents_consulted": [],
                        "vitals_status": vitals_status,
                        **{k: v for k, v in stale.items() if k not in ("status_job", "status")},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job '%s' (dream_stale_check) failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "dream_nightly":
            try:
                max_runtime_min = float(job.get("max_runtime_minutes", 90) or 90)
                max_runtime_sec = max(60.0, max_runtime_min * 60.0)
                if _dream_checkpoint_busy(max_busy_seconds=max_runtime_sec):
                    if not force:
                        await _mark_ran(redis, name)
                    fire = missed_wall_clock_fire(
                        job,
                        None,
                        now=now,
                        catch_up_window_hours=catch_up_window_hours,
                    )
                    if _should_log_dream_busy_skip(name, fire, now):
                        results.append(
                            {
                                "job": name,
                                "status": "skipped",
                                "reason": "dream_already_running",
                                "action": action,
                                "agents_consulted": [],
                                "deduped_per_slot": True,
                            }
                        )
                    continue

                from brain_os.brain.dream_bootstrap import build_dream_mode

                dream_mode = await build_dream_mode()
                journal_last_24h = bool(job.get("journal_last_24h", False))
                try:
                    report = await asyncio.wait_for(
                        dream_mode.run_dream_cycle(journal_last_24h=journal_last_24h),
                        timeout=max_runtime_sec,
                    )
                except TimeoutError:
                    logger.error(
                        "Heartbeat job %s dream_nightly timed out after %.0fs",
                        name,
                        max_runtime_sec,
                    )
                    results.append(
                        {
                            "job": name,
                            "status": "timeout",
                            "action": action,
                            "agents_consulted": [],
                            "max_runtime_seconds": max_runtime_sec,
                            "error": f"dream_nightly exceeded {max_runtime_min:.0f}m cap",
                        }
                    )
                    continue

                stage_results = getattr(report, "stage_results", None) or {}
                if stage_results.get("distributed_mutex") == "skipped":
                    results.append(
                        {
                            "job": name,
                            "status": "skipped",
                            "reason": "dream_distributed_mutex",
                            "action": action,
                            "agents_consulted": [],
                        }
                    )
                    continue

                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "cycle_date": str(getattr(report, "cycle_date", "")),
                        "memories_consolidated": int(
                            getattr(report, "memories_consolidated", 0) or 0
                        ),
                        "max_runtime_seconds": max_runtime_sec,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s dream_nightly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "curious_harvest_nightly":
            try:
                from brain_os.services.curious_ira import run_curious_harvest_nightly

                if not force and not bool(
                    getattr(get_settings().app, "curious_ira_enabled", False)
                ):
                    await _mark_ran(redis, name)
                    results.append(
                        {
                            "job": name,
                            "status": "skipped",
                            "reason": "curious_ira_disabled",
                            "action": action,
                            "agents_consulted": [],
                        }
                    )
                    continue
                harvest = await run_curious_harvest_nightly(force=True)
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        **{k: v for k, v in harvest.items() if k != "top"},
                        "top_count": len(harvest.get("top") or []),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s curious_harvest_nightly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "curious_email_daily":
            try:
                from brain_os.services.curious_ira import run_curious_email_daily

                if not force and not bool(
                    getattr(get_settings().app, "curious_ira_enabled", False)
                ):
                    await _mark_ran(redis, name)
                    results.append(
                        {
                            "job": name,
                            "status": "skipped",
                            "reason": "curious_ira_disabled",
                            "action": action,
                            "agents_consulted": [],
                        }
                    )
                    continue
                email_processor = getattr(pipeline, "_email_processor", None) or getattr(
                    pipeline, "email_processor", None
                )
                crm = getattr(pipeline, "_crm", None) or getattr(pipeline, "crm", None)
                email_result = await run_curious_email_daily(
                    force=True,
                    email_processor=email_processor,
                    crm=crm,
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok"
                        if email_result.get("ok") or email_result.get("skipped")
                        else "error",
                        "action": action,
                        "agents_consulted": [],
                        **email_result,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s curious_email_daily failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "stomach_email_digest":
            try:
                from brain_os.runtime.cli_runtime import (
                    _build_digestive,
                    _build_email_processor,
                    _build_pantheon,
                )

                pantheon, shared = _build_pantheon()
                digestive, _ingestor, _qdrant = _build_digestive()
                email_proc = _build_email_processor(pantheon, digestive, shared)
                poll_results = await email_proc.run_single_poll_cycle()
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": ["delphi"],
                        "emails_processed": len(poll_results),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s stomach_email_digest failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "gmail_oauth_preflight":
            try:
                from brain_os.services.gmail_oauth_alert import preflight_gmail_token_expiry
                from brain_os.systems.heartbeat_writes_registry import make_counts

                warn_days = int(job.get("warn_days") or 7)
                pre = preflight_gmail_token_expiry(warn_days=warn_days)
                await _mark_ran(redis, name)
                status = "ok" if pre.get("ok") and pre.get("status") == "ok" else "critical"
                if pre.get("status") == "warn":
                    status = "warn"
                results.append(
                    {
                        "job": name,
                        "status": status,
                        "action": action,
                        "writes": True,
                        "agents_consulted": [],
                        "preflight": pre,
                        "counts": make_counts(
                            attempted=1,
                            written=1 if status != "ok" else 0,
                            failed=0 if pre.get("ok") else 1,
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s gmail_oauth_preflight failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "write_contract_retry_drain":
            try:
                from brain_os.brain.write_contract import replay_write_contract_retry_queue
                from brain_os.systems.heartbeat_writes_registry import make_counts

                max_n = int(job.get("max_to_process") or 200)
                dry = bool(job.get("dry_run", False))
                stats = await replay_write_contract_retry_queue(
                    max_to_process=max_n,
                    dry_run=dry,
                )
                await _mark_ran(redis, name)
                processed = int(stats.get("processed") or 0)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "writes": not dry,
                        "agents_consulted": [],
                        "stats": stats,
                        "counts": make_counts(
                            attempted=processed,
                            written=int(stats.get("replay_succeeded") or 0),
                            failed=int(stats.get("replay_failed") or 0),
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s write_contract_retry_drain failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "backup_nightly":
            try:
                from brain_os.systems.brain_backup import run_ira_backup

                max_runtime_min = float(job.get("max_runtime_minutes", 60) or 60)
                max_runtime_sec = max(60.0, max_runtime_min * 60.0)
                payload = await run_ira_backup(
                    dest=str(job["dest"]) if job.get("dest") else None,
                    timeout_seconds=max_runtime_sec,
                )
                ok = bool(payload.get("ok"))
                if ok:
                    await _mark_ran(redis, name)
                # Preserve partial_error from run_ira_backup (some stores
                # written, some failed) — loud, never all-or-nothing.
                status = str(payload.get("status") or ("ok" if ok else "error"))
                results.append(
                    {
                        "job": name,
                        "status": status,
                        "action": action,
                        "agents_consulted": [],
                        "bytes_written": payload.get("bytes_written"),
                        "duration_seconds": payload.get("duration_seconds"),
                        "dest": payload.get("dest"),
                        "store_ok_count": payload.get("store_ok_count"),
                        "store_fail_count": payload.get("store_fail_count"),
                        "stores": payload.get("stores"),
                        "restic": payload.get("restic"),
                        **({"error": payload.get("error")} if not ok else {}),
                        **({"failed": payload.get("failed")} if payload.get("failed") else {}),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s backup_nightly failed", name)
                results.append(_job_error_outcome(name, exc, action=action))
            continue

        if action == "learning_loop_watchdog":
            try:
                from brain_os.services.learning_loop_watchdog import run_learning_loop_watchdog

                wd = await run_learning_loop_watchdog(
                    inject_morning_brief=bool(job.get("inject_morning_brief", True)),
                    force_morning=bool(job.get("force_morning", False)),
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": wd.get("status", "ok"),
                        "action": action,
                        "agents_consulted": [],
                        **{k: v for k, v in wd.items() if k != "status"},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s learning_loop_watchdog failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "reply_learning_weekly":
            try:
                from brain_os.services.reply_loop import write_weekly_learning_brief
                from brain_os.services.revenue_mode_metrics import write_funnel_daily_snapshot

                brief_path = write_weekly_learning_brief()
                write_funnel_daily_snapshot()
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "summary": f"learning brief: {brief_path.name}",
                        "brief_path": str(brief_path),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s reply_learning_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "email_taxonomy_drift_weekly":
            try:
                from brain_os.services.email_taxonomy_routing import run_taxonomy_drift_watch

                drift = run_taxonomy_drift_watch(inject_morning=True)
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": drift.get("status", "ok"),
                        "action": action,
                        "agents_consulted": [],
                        "alerts": len(drift.get("alerts") or []),
                        "alerts_enqueued": drift.get("alerts_enqueued"),
                        "summary": (
                            f"taxonomy drift: {len(drift.get('alerts') or [])} alert(s), "
                            f"{drift.get('alerts_enqueued', 0)} enqueued"
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s email_taxonomy_drift_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "retrieval_eval_weekly":
            try:
                from brain_os.brain.retrieval_p5_harness import run_weekly_retrieval_eval

                eval_result = await run_weekly_retrieval_eval(write_ledger=True)
                await _mark_ran(redis, name)
                p5 = eval_result.get("p5_mean")
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "p5_mean": p5,
                        "n_queries": eval_result.get("n_queries"),
                        "summary": f"retrieval P@5={p5} (n={eval_result.get('n_queries')})",
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s retrieval_eval_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "slope_snapshot_nightly":
            try:
                from brain_os.services.slope_dashboard import run_slope_snapshot_nightly_job

                payload = await run_slope_snapshot_nightly_job()
                if not dry_run:
                    await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": str(payload.get("status") or "ok"),
                        "action": action,
                        "agents_consulted": [],
                        "skip_llm": True,
                        "writes": True,
                        **{k: v for k, v in payload.items() if k != "status"},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s slope_snapshot_nightly failed", name)
                results.append(_job_error_outcome(name, exc, action=action))
            continue

        if action == "outreach_experiment_weekly":
            try:
                from brain_os.services.outreach_experiment_weekly import run_outreach_experiment_weekly

                digest = run_outreach_experiment_weekly(
                    days=_job_trailing_window_days(job, default=30),
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "summary": f"outreach scoreboard: {Path(digest['report_path']).name}",
                        **digest,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s outreach_experiment_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "message_bandit_no_reply_scan":
            try:
                from brain_os.services.gtm_bandit_allocator import (
                    run_message_bandit_no_reply_scan,
                )
                from brain_os.systems.heartbeat_writes_registry import make_counts

                min_age = int(job.get("min_age_days") or 7)
                summary = run_message_bandit_no_reply_scan(min_age_days=min_age)
                await _mark_ran(redis, name)
                counts = summary.get("counts") or make_counts(
                    attempted=int(summary.get("recorded") or 0),
                    written=int(summary.get("recorded") or 0),
                    failed=0,
                )
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "summary": (f"message bandit no-reply: recorded={summary.get('recorded')}"),
                        "counts": counts,
                        **summary,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s message_bandit_no_reply_scan failed", name)
                results.append(_job_error_outcome(name, exc, action=action))
            continue

        if action == "founder_voice_refresh_monthly":
            try:
                from brain_os.services.founder_voice_refresh import run_founder_voice_refresh_monthly
                from brain_os.systems.heartbeat_writes_registry import make_counts

                summary = await run_founder_voice_refresh_monthly(
                    dry_run=bool(job.get("dry_run", False)),
                    max_train=int(job.get("max_train", 80) or 80),
                    max_examples=int(job.get("max_examples", 12) or 12),
                )
                await _mark_ran(redis, name)
                status = "ok" if summary.get("status") in {"ok", "skipped"} else "error"
                results.append(
                    {
                        "job": name,
                        "status": status,
                        "action": action,
                        "agents_consulted": [],
                        "summary": summary,
                        "counts": make_counts(
                            attempted=int(summary.get("attempted") or 0),
                            written=int(summary.get("written") or 0),
                            failed=int(summary.get("failed") or 0),
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s founder_voice_refresh_monthly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "tone_clinic_weekly":
            try:
                from brain_os.services.tone_clinic import run_tone_clinic
                from brain_os.systems.heartbeat_writes_registry import make_counts

                summary = await run_tone_clinic(
                    dry_run=bool(job.get("dry_run", False)),
                    use_llm_judge=not bool(job.get("skip_llm", False)),
                    limit=int(job.get("limit", 25) or 25),
                    enqueue=not bool(job.get("dry_run", False)),
                )
                await _mark_ran(redis, name)
                status = "ok" if summary.get("status") in {"ok", "skipped"} else "error"
                results.append(
                    {
                        "job": name,
                        "status": status,
                        "action": action,
                        "agents_consulted": [],
                        "summary": summary,
                        "counts": make_counts(
                            attempted=int(summary.get("attempted") or 0),
                            written=int(summary.get("written") or 0),
                            failed=int(summary.get("failed") or 0),
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s tone_clinic_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "installed_base_weekly":
            try:
                from brain_os.services.installed_base.funnel_metrics import (
                    run_installed_base_weekly_digest,
                )

                digest = await run_installed_base_weekly_digest(
                    days=int(job.get("days", 7) or 7),
                    webhook=bool(job.get("webhook", True)),
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "summary": f"installed-base digest: {digest.get('digest_path')}",
                        **digest,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s installed_base_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "portfolio_refresh_weekly":
            try:
                from brain_os.services.order_book import refresh_portfolio_evidence

                payload = refresh_portfolio_evidence(persist=not dry_run)
                if not dry_run:
                    await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok" if payload.get("status") == "ok" else "error",
                        "action": action,
                        "agents_consulted": [],
                        "skip_llm": True,
                        "summary": (
                            f"portfolio evidence refresh updated_at="
                            f"{payload.get('updated_at')} stamped="
                            f"{payload.get('stamped_projects')}"
                        ),
                        **payload,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s portfolio_refresh_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "production_photo_watch":
            try:
                from brain_os.services.production_photo_watcher import process_photo_drops

                payload = await process_photo_drops(
                    dry_run=dry_run,
                    queue_draft=bool(job.get("queue_draft", True)),
                )
                if not dry_run:
                    await _mark_ran(redis, name)
                drops = int(payload.get("drops_found") or 0)
                events_n = len(payload.get("events") or [])
                ok = bool(payload.get("ok"))
                # Idle scans (no new drops) still count as success for the watchdog.
                status = ("ok_idle" if drops == 0 and events_n == 0 else "ok") if ok else "error"
                results.append(
                    {
                        "job": name,
                        "status": status,
                        "action": action,
                        "agents_consulted": [],
                        "skip_llm": True,
                        "summary": (
                            f"photo watch drops={drops} "
                            f"events={events_n} "
                            f"drafts={len(payload.get('drafts') or [])}"
                        ),
                        **{k: v for k, v in payload.items() if k != "ok"},
                        "ok": ok,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s production_photo_watch failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "wo_watch":
            try:
                from brain_os.services.wo_watch import run_wo_watch

                # Delays always; chase only when job opts in (queue_drafts / chase).
                run_chase = bool(job.get("chase", False)) or bool(job.get("queue_drafts", False))
                delay_payload = await run_wo_watch(persist=not dry_run)
                chase_payload: dict[str, Any] | None = None
                if run_chase:
                    from brain_os.services.receivables_chase import run_wo_watch_chase

                    chase_payload = await run_wo_watch_chase(
                        dry_run=dry_run,
                        queue_drafts=bool(job.get("queue_drafts", True)) and not dry_run,
                    )
                ok = bool(delay_payload.get("ok")) and (
                    chase_payload is None or bool(chase_payload.get("ok"))
                )
                if not dry_run:
                    await _mark_ran(redis, name)
                delay_n = int(delay_payload.get("delay_count") or 0)
                proposed_n = int((chase_payload or {}).get("proposed_count") or 0)
                status = (
                    ("ok_idle" if delay_n == 0 and proposed_n == 0 else "ok") if ok else "error"
                )
                summary = (
                    f"wo_watch delays={delay_n} "
                    f"overdue_ms={delay_payload.get('overdue_count')} "
                    f"warranty={delay_payload.get('warranty_count')}"
                )
                if chase_payload is not None:
                    summary += (
                        f" chase_proposed={proposed_n} "
                        f"chase_skipped={chase_payload.get('skipped_count')}"
                    )
                results.append(
                    {
                        "job": name,
                        "status": status,
                        "action": action,
                        "agents_consulted": [],
                        "skip_llm": True,
                        "summary": summary,
                        "delay": delay_payload,
                        "chase": chase_payload,
                        "ok": ok,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s wo_watch failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "tally_bank_book_watch":
            try:
                from brain_os.services.tally_export_watcher import process_tally_export_watch

                payload = process_tally_export_watch(dry_run=dry_run)
                if not dry_run:
                    await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok" if payload.get("ok") else "error",
                        "action": action,
                        "agents_consulted": [],
                        "skip_llm": True,
                        "summary": (
                            f"tally bank-book watch new={payload.get('new_count')} "
                            f"reminders={len(payload.get('reminders') or [])}"
                        ),
                        **payload,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s tally_bank_book_watch failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "social_learn_daily":
            try:
                from dotenv import load_dotenv

                from brain_os.social.feed_learner import FeedLearner, build_weekly_digest
                from brain_os.social.moltbook_client import MoltbookClient

                # ira.social reads os.environ only (separate-credentials rule);
                # bridge MOLTBOOK_API_KEY from .env. Exported vars win.
                load_dotenv()

                complete = None
                if bool(job.get("use_llm", False)):
                    from brain_os.services.llm_client import LLMClient

                    complete = LLMClient().generate_text
                with MoltbookClient() as client:
                    learner = FeedLearner(client, complete=complete)
                    summary = await learner.run_daily()
                digest_path = None
                if bool(job.get("build_digest", False)):
                    digest_path = build_weekly_digest()
                await _mark_ran(redis, name)
                from brain_os.systems.heartbeat_writes_registry import make_counts

                harvested = int(summary.get("harvested") or 0)
                classified = int(summary.get("classified") or 0)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "counts": make_counts(
                            attempted=max(harvested, classified, 1),
                            written=harvested,
                            failed=0,
                        ),
                        **summary,
                        **({"digest_path": str(digest_path)} if digest_path else {}),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s social_learn_daily failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "social_watch_comments":
            try:
                from dotenv import load_dotenv

                from brain_os.social.comment_watcher import CommentWatcher
                from brain_os.social.moltbook_client import MoltbookClient

                load_dotenv()

                complete = None
                if bool(job.get("use_llm", False)):
                    from brain_os.services.llm_client import LLMClient

                    complete = LLMClient().generate_text
                with MoltbookClient() as client:
                    watcher = CommentWatcher(client, complete=complete)
                    summary = await watcher.run_daily()
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        **summary,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s social_watch_comments failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "social_daily_cycle":
            try:
                from dotenv import load_dotenv

                from brain_os.services.social_daily_cycle import run_daily_cycle

                load_dotenv()
                summary = await run_daily_cycle(
                    dry_run=bool(job.get("dry_run", False)),
                    auto_publish=job.get("auto_publish"),
                    send_digest_email=bool(job.get("send_digest_email", True)),
                    use_llm_comments=bool(job.get("use_llm", False)),
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": summary.get("status", "ok"),
                        "action": action,
                        "agents_consulted": [],
                        **{k: v for k, v in summary.items() if k != "digest_preview"},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s social_daily_cycle failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "graphify_refresh":
            if not bool(get_settings().app.graphify_scheduled_refresh_enabled):
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "skipped",
                        "action": action,
                        "agents_consulted": [],
                        "reason": "APP__GRAPHIFY_SCHEDULED_REFRESH_ENABLED=false",
                    }
                )
                continue
            try:
                from brain_os.services.code_graph import refresh_code_graph
                from brain_os.systems.heartbeat_writes_registry import make_counts

                payload = await refresh_code_graph()
                await _mark_ran(redis, name)
                ok = bool(payload.get("ok"))
                results.append(
                    {
                        "job": name,
                        "status": "ok" if ok else "error",
                        "action": action,
                        "agents_consulted": [],
                        "duration_ms": payload.get("duration_ms"),
                        "counts": make_counts(
                            attempted=1,
                            written=1 if ok else 0,
                            failed=0 if ok else 1,
                        ),
                        **({"error": payload.get("error")} if not ok else {}),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s graphify refresh failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "graph_sync_maintenance":
            try:
                from brain_os.brain.graph_sync_maintenance import run_graph_sync_maintenance

                max_runtime_min = float(job.get("max_runtime_minutes", 20) or 20)
                max_runtime_sec = max(60.0, max_runtime_min * 60.0)
                try:
                    payload = await asyncio.wait_for(
                        run_graph_sync_maintenance(
                            batch_size=int(job.get("batch_size", 500)),
                            write_concurrency=int(job.get("write_concurrency", 16)),
                            sleep_seconds=float(job.get("sleep_seconds", 0.0)),
                            audit_sample=int(job.get("audit_sample", 300)),
                            audit_random=not bool(job.get("full_audit", False)),
                            audit_full_scan=bool(job.get("full_audit", False)),
                            min_resolve_rate_pct=float(job.get("min_resolve_rate_pct", 99.5)),
                        ),
                        timeout=max_runtime_sec,
                    )
                except TimeoutError:
                    logger.error(
                        "Heartbeat job %s graph_sync_maintenance timed out after %.0fs",
                        name,
                        max_runtime_sec,
                    )
                    results.append(
                        {
                            "job": name,
                            "status": "timeout",
                            "action": action,
                            "agents_consulted": [],
                            "max_runtime_seconds": max_runtime_sec,
                            "error": (
                                f"graph_sync_maintenance exceeded {max_runtime_min:.0f}m cap"
                            ),
                        }
                    )
                    continue
                await _mark_ran(redis, name)
                sync = payload.get("sync") or {}
                audit = payload.get("audit") or {}
                points_failed = int(sync.get("points_failed") or 0)
                qdrant_errors = int(audit.get("qdrant_errors") or 0)
                # Operational ok = Neo4j reachable + sync wrote without failures.
                # Audit resolve-rate can lag (orphan residue) without meaning Aura
                # is down — keep that as audit_ok, not a schedule-watchdog failure.
                sync_ok = points_failed == 0 and qdrant_errors == 0
                results.append(
                    {
                        "job": name,
                        "status": "ok" if sync_ok else "error",
                        "action": action,
                        "agents_consulted": [],
                        "report_path": payload.get("report_path"),
                        "points_updated": sync.get("points_updated"),
                        "points_failed": sync.get("points_failed"),
                        "resolve_rate_pct": audit.get("resolve_rate_pct"),
                        "missing_in_qdrant": audit.get("missing_in_qdrant"),
                        "audit_ok": bool(payload.get("ok")),
                        "max_runtime_seconds": max_runtime_sec,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s graph sync maintenance failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "graph_person_hygiene":
            try:
                from brain_os.services.graph_schedule import run_person_hygiene_heartbeat

                max_runtime_min = float(job.get("max_runtime_minutes", 30) or 30)
                max_runtime_sec = max(60.0, max_runtime_min * 60.0)
                try:
                    payload = await asyncio.wait_for(
                        run_person_hygiene_heartbeat(job),
                        timeout=max_runtime_sec,
                    )
                except TimeoutError:
                    results.append(
                        {
                            "job": name,
                            "status": "timeout",
                            "action": action,
                            "agents_consulted": [],
                            "error": f"graph_person_hygiene exceeded {max_runtime_min:.0f}m",
                        }
                    )
                    continue
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok" if payload.get("ok") else "error",
                        "action": action,
                        "agents_consulted": [],
                        "writes": True,
                        "counts": payload.get("counts"),
                        "dry_run": payload.get("dry_run"),
                        "invalid_cleared": (payload.get("report") or {}).get("invalid_cleared"),
                        "nodes_merged_away": (payload.get("report") or {}).get("nodes_merged_away"),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s graph_person_hygiene failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "graph_analytics":
            try:
                from brain_os.services.graph_schedule import run_analytics_heartbeat

                max_runtime_min = float(job.get("max_runtime_minutes", 45) or 45)
                max_runtime_sec = max(60.0, max_runtime_min * 60.0)
                try:
                    payload = await asyncio.wait_for(
                        run_analytics_heartbeat(job),
                        timeout=max_runtime_sec,
                    )
                except TimeoutError:
                    results.append(
                        {
                            "job": name,
                            "status": "timeout",
                            "action": action,
                            "agents_consulted": [],
                            "error": f"graph_analytics exceeded {max_runtime_min:.0f}m",
                        }
                    )
                    continue
                await _mark_ran(redis, name)
                if payload.get("skipped"):
                    results.append(
                        {
                            "job": name,
                            "status": "ok",
                            "action": action,
                            "agents_consulted": [],
                            "skipped": True,
                            "reason": payload.get("reason"),
                            "writes": True,
                            "counts": payload.get("counts")
                            or {"attempted": 0, "written": 0, "failed": 0},
                        }
                    )
                    continue
                results.append(
                    {
                        "job": name,
                        "status": "ok" if payload.get("ok") else "error",
                        "action": action,
                        "agents_consulted": [],
                        "writes": True,
                        "counts": payload.get("counts"),
                        "analytics_job": payload.get("analytics_job"),
                        "cost": payload.get("cost"),
                        "report_summary": payload.get("report_summary"),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s graph_analytics failed", name)
                results.append(_job_error_outcome(name, exc))
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
                results.append(_job_error_outcome(name, exc))
            continue

        if action in ("prediction_reconcile", "learning_reconcile_weekly"):
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
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "learning_promote_weekly":
            try:
                compile_summary: dict[str, Any] | None = None
                if bool(job.get("compile_first", True)):
                    from brain_os.brain.learning_compiler import compile_learning_candidates

                    compile_summary = await compile_learning_candidates(
                        session_limit=int(job.get("session_limit", 2000))
                    )
                from brain_os.brain.learning_promotion import promote_learning_candidates

                promote_summary = await promote_learning_candidates(
                    dry_run=bool(job.get("dry_run", False))
                )
                await _mark_ran(redis, name)
                from brain_os.systems.heartbeat_writes_registry import make_counts

                promoted_n = int(promote_summary.get("promoted") or 0)
                enqueued_n = int(promote_summary.get("enqueued") or 0)
                candidates_n = int((compile_summary or {}).get("candidates") or 0)
                written_n = promoted_n + enqueued_n
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "compile": compile_summary,
                        "promote": promote_summary,
                        "counts": make_counts(
                            attempted=max(candidates_n, written_n),
                            written=written_n,
                            failed=0,
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s learning_promote_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "gepa_overlay_weekly":
            try:
                from brain_os.brain.gepa_overlay_compiler import (
                    compile_strategy_overlays_from_tool_stats,
                )
                from brain_os.brain.learning_promotion import promote_gepa_strategy_overlays

                compile_summary = await compile_strategy_overlays_from_tool_stats(
                    force=bool(job.get("force", False))
                )
                promote_summary = await promote_gepa_strategy_overlays(
                    dry_run=bool(job.get("dry_run", False)),
                    force=bool(job.get("force", False)),
                )
                await _mark_ran(redis, name)
                from brain_os.systems.heartbeat_writes_registry import make_counts

                overlays_n = int((compile_summary or {}).get("overlays") or 0)
                gepa_written = int(promote_summary.get("promoted") or 0) + int(
                    promote_summary.get("enqueued") or 0
                )
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "compile": compile_summary,
                        "promote": promote_summary,
                        "counts": make_counts(
                            attempted=max(overlays_n, gepa_written),
                            written=gepa_written,
                            failed=0,
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s gepa_overlay_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "agent_peer_review":
            try:
                from brain_os.services.agent_peer_review import run_agent_peer_review_heartbeat

                payload = await run_agent_peer_review_heartbeat(job, force=force)
                if payload.get("status") == "ok":
                    await _mark_ran(redis, name)
                results.append({"job": name, **payload})
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s agent_peer_review failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "graphe_meta_backfill_weekly":
            try:
                from brain_os.brain.graphe_learning_meta_backfill import (
                    backfill_missing_learning_meta,
                    build_graphe_learning_meta_snapshot,
                )

                summary = await backfill_missing_learning_meta(
                    limit=int(job.get("limit", 200)),
                    dry_run=bool(job.get("dry_run", False)),
                    source="heartbeat",
                )
                snap = build_graphe_learning_meta_snapshot()
                await _mark_ran(redis, name)
                from brain_os.systems.heartbeat_writes_registry import make_counts

                updated_n = int(summary.get("updated") or 0)
                failed_n = int(summary.get("failed") or 0)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "updated": updated_n,
                        "coverage_ratio": snap.get("coverage_ratio"),
                        "coverage_met": snap.get("coverage_met"),
                        "counts": make_counts(
                            attempted=int(summary.get("scanned") or summary.get("attempted") or 0)
                            or updated_n + failed_n,
                            written=updated_n,
                            failed=failed_n,
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s graphe_meta_backfill_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "session_mine_batch":
            try:
                from brain_os.memory.session_coverage import canonical_miner_coverage
                from brain_os.memory.session_miner import (
                    drain_mine_queue,
                    mine_unmined,
                )

                batch_limit = int(job.get("limit") or settings.app.session_mine_batch_limit or 25)
                since_days = job.get("since_days")
                since_val = float(since_days) if since_days is not None else None
                queue_summary = await drain_mine_queue(
                    limit=batch_limit,
                    enqueue_pending=not bool(job.get("skip_pending", False)),
                )
                mine_summary = await mine_unmined(
                    limit=batch_limit,
                    enqueue_pending=not bool(job.get("skip_pending", False)),
                    since_days=since_val,
                    sleep_seconds=float(job.get("sleep_seconds") or 0.0),
                    source="heartbeat_session_mine",
                )
                # Canonical rolling-7d snapshot (mop-up item 2) — pct_mined is
                # the operator-facing number; pct_meta stays for diagnostics
                # only, so vitals (morning_learning_card.py) and this
                # heartbeat job never report different coverage figures again.
                snap = await canonical_miner_coverage()
                q_errors = int(queue_summary.get("errors") or 0)
                m_errors = int(mine_summary.get("errors") or 0)
                if q_errors or m_errors:
                    logger.error(
                        "Heartbeat job %s session_mine_batch errors "
                        "queue_errors=%s queue_dead_lettered=%s mine_errors=%s",
                        name,
                        q_errors,
                        queue_summary.get("dead_lettered"),
                        m_errors,
                    )
                await _mark_ran(redis, name)
                # Queue drain can fail on poison rows while mine_unmined still
                # advances coverage — treat that as ok (with error detail) so the
                # schedule watchdog does not stay CRITICAL forever. Hard-fail only
                # when the primary mine batch itself errors.
                mined_ok = int(mine_summary.get("ok") or 0)
                if m_errors or (q_errors and mined_ok == 0):
                    status = "error"
                else:
                    status = "ok"
                from brain_os.systems.heartbeat_writes_registry import make_counts

                results.append(
                    {
                        "job": name,
                        "status": status,
                        "action": action,
                        "queue": queue_summary,
                        "mined": mine_summary,
                        "coverage_7d_pct_mined": snap.get("pct_mined"),
                        "coverage_7d_pct_mined_100": snap.get("pct_mined_100"),
                        "coverage_7d_pct_meta": snap.get("pct_meta"),
                        "error_count": q_errors + m_errors,
                        "counts": make_counts(
                            attempted=mined_ok + m_errors,
                            written=mined_ok,
                            failed=m_errors,
                        ),
                        "error": (
                            f"session_mine errors: queue={q_errors} mined={m_errors}"
                            if (q_errors or m_errors)
                            else None
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s session_mine_batch failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "mem0_forgetting_weekly":
            try:
                # Dream stage 5b (dream_stage_index.py) is the sole execution
                # owner for real Mem0 archive/delete (mop-up item 3). This
                # heartbeat job is report-only by default
                # (data/heartbeat_jobs.json sets "execute": false) — it still
                # calls run_mem0_forgetting_pass so operators get a fresh
                # dry-run snapshot on the heartbeat cadence, but it never
                # performs the destructive pass itself.
                if not settings.app.mem0_forgetting_heartbeat_enabled:
                    await _mark_ran(redis, name)
                    results.append(
                        {
                            "job": name,
                            "status": "skipped",
                            "action": action,
                            "reason": "mem0_forgetting_heartbeat_disabled",
                        }
                    )
                    continue
                from brain_os.memory.mem0_forgetting_ops import run_mem0_forgetting_pass

                force = job.get("execute")
                force_enabled = bool(force) if force is not None else None
                job_executes = bool(force) if force is not None else True
                summary = await run_mem0_forgetting_pass(
                    force_enabled=force_enabled,
                    source="heartbeat",
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "dry_run": summary.get("dry_run"),
                        "candidates": summary.get("candidates", 0),
                        "deleted": summary.get("deleted", 0),
                        "skipped_reason": summary.get("reason"),
                        "report_only": not job_executes,
                        "execution_owner": "dream_stage_5b",
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s mem0_forgetting_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "correction_ledger_weekly":
            try:
                if not settings.app.correction_ledger_hygiene_enabled:
                    await _mark_ran(redis, name)
                    results.append(
                        {
                            "job": name,
                            "status": "skipped",
                            "action": action,
                            "reason": "correction_ledger_hygiene_disabled",
                        }
                    )
                    continue
                from brain_os.memory.correction_ledger_hygiene import run_ledger_hygiene_pass

                crm = None
                if settings.app.correction_ledger_hygiene_crm_hints:
                    try:
                        from brain_os.data.crm import CRMDatabase

                        crm = CRMDatabase()
                    except Exception:  # noqa: BLE001 — CRM init failure degrades the job to no-CRM
                        crm = None
                summary = await run_ledger_hygiene_pass(
                    dry_run=bool(job.get("dry_run", False)),
                    fix_missing_dates=bool(
                        job.get(
                            "fix_missing_dates",
                            settings.app.correction_ledger_hygiene_fix_missing_dates,
                        )
                    ),
                    crm=crm,
                    source="heartbeat",
                )
                await _mark_ran(redis, name)
                from brain_os.systems.heartbeat_writes_registry import make_counts

                verify = summary.get("auto_verify_missing_date") or {}
                verified_n = int(verify.get("touched_count") or 0)
                skipped_n = len(verify.get("skipped") or [])
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "stale_count": summary.get("stale_count"),
                        "coverage_before": summary.get("coverage_before"),
                        "coverage_after": summary.get("coverage_after"),
                        "auto_verify": verify or None,
                        "counts": make_counts(
                            attempted=verified_n + skipped_n,
                            written=verified_n,
                            failed=0,
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s correction_ledger_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "db_hygiene":
            try:
                from brain_os.services.db_hygiene import run_db_hygiene_pass

                summary = await run_db_hygiene_pass(
                    prune_embedding_cache=bool(job.get("prune_embedding_cache", True)),
                    archive_stray_predictions=bool(job.get("archive_stray_predictions", False)),
                )
                await _mark_ran(redis, name)
                wal = summary.get("wal") or {}
                emb = summary.get("embedding_cache") or {}
                scanned = int(wal.get("databases_scanned") or 0)
                busy = int(wal.get("busy_or_locked") or 0)
                from brain_os.systems.heartbeat_writes_registry import make_counts

                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "databases_scanned": wal.get("databases_scanned"),
                        "wal_bytes_before_total": wal.get("wal_bytes_before_total"),
                        "wal_bytes_after_total": wal.get("wal_bytes_after_total"),
                        "busy_or_locked": wal.get("busy_or_locked"),
                        "embedding_cache_bytes_before": (emb.get("before") or {}).get("main_bytes"),
                        "embedding_cache_bytes_after": (emb.get("after") or {}).get("main_bytes"),
                        "embedding_cache_prune": emb.get("prune"),
                        "wal_databases": wal.get("databases"),
                        "predictions": summary.get("predictions") or {},
                        "ran_at": summary.get("ran_at"),
                        "counts": make_counts(
                            attempted=max(scanned, 1),
                            written=max(0, scanned - busy) if scanned else 1,
                            failed=busy,
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s db_hygiene failed", name)
                results.append(_job_error_outcome(name, exc))
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
                results.append(_job_error_outcome(name, exc))
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
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "board_meeting":
            try:
                from brain_os.systems.board_meeting_v2 import run_board_meeting_v2

                send_email = bool(job.get("email", True))
                topic = str(job.get("topic") or "").strip()
                email_processor = getattr(pipeline, "_email_processor", None) or getattr(
                    pipeline, "email_processor", None
                )
                crm = getattr(pipeline, "_crm", None) or getattr(pipeline, "crm", None)
                minutes = await run_board_meeting_v2(
                    topic,
                    send_email=send_email,
                    persist_ledger=True,
                    email_processor=email_processor,
                    crm=crm,
                )
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "meeting_id": minutes.meeting_id,
                        "agenda_count": len(minutes.agenda or []),
                        "decisions_count": len(minutes.decisions or []),
                        "email_delivery": minutes.email_delivery,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s board_meeting failed", name)
                results.append(_job_error_outcome(name, exc))
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
                from brain_os.systems.heartbeat_writes_registry import make_counts

                entries_saved = int(journal_result.get("entries_saved") or 0)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": list(journal_result.get("agents") or []),
                        "entries_saved": entries_saved,
                        "agents_processed": journal_result.get("agents_processed", 0),
                        "since_last_journal": since_last,
                        "lookback_hours": lookback,
                        "counts": make_counts(
                            attempted=max(entries_saved, 1),
                            written=entries_saved,
                            failed=0,
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s agent_journal failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "sourcing_weekly":
            try:
                from brain_os.services.sourcing_weekly_pipeline import run_sourcing_weekly

                job_dry = dry_run or bool(job.get("dry_run"))
                crm = getattr(pipeline, "_crm", None) or getattr(pipeline, "crm", None)
                graph = getattr(pipeline, "_graph", None) or getattr(
                    pipeline, "knowledge_graph", None
                )
                if crm is None and not job_dry:
                    from brain_os.data.crm import CRMDatabase

                    crm = CRMDatabase()
                    await crm.create_tables()
                payload = await run_sourcing_weekly(
                    dry_run=job_dry,
                    crm=crm,
                    graph=graph,
                    job=job,
                )
                if not job_dry:
                    await _mark_ran(redis, name)
                from brain_os.systems.heartbeat_writes_registry import make_counts

                upsert = payload.get("crm_upsert") if isinstance(payload, dict) else None
                upsert = upsert if isinstance(upsert, dict) else {}
                attempted = int(upsert.get("attempted") or 0)
                written = int(upsert.get("created") or 0)
                skipped = int(upsert.get("skipped_existing") or 0) + int(
                    upsert.get("skipped_no_domain") or 0
                )
                failed = max(0, attempted - written - skipped)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "skip_llm": True,
                        "counts": make_counts(
                            attempted=attempted,
                            written=written,
                            failed=failed,
                        ),
                        **payload,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s sourcing_weekly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "universe_refresh_weekly":
            try:
                from brain_os.services.universe_refresh import run_universe_refresh

                job_dry = dry_run or bool(job.get("dry_run"))
                crm = getattr(pipeline, "_crm", None) or getattr(pipeline, "crm", None)
                if crm is None and not job_dry:
                    try:
                        from brain_os.data.crm import CRMDatabase

                        crm = CRMDatabase()
                        await crm.create_tables()
                    except _HEARTBEAT_JOB_ERRORS:
                        logger.debug(
                            "universe_refresh_weekly: CRM unavailable; ledger/blocklist only",
                            exc_info=True,
                        )
                        crm = None
                payload = await run_universe_refresh(
                    dry_run=job_dry,
                    crm=crm,
                    job=job,
                )
                if not job_dry:
                    await _mark_ran(redis, name)
                from brain_os.systems.heartbeat_writes_registry import make_counts

                stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
                promote = (
                    payload.get("crm_promote")
                    if isinstance(payload.get("crm_promote"), dict)
                    else {}
                )
                universe_written = int(promote.get("promoted") or 0)
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "skip_llm": True,
                        **payload,
                        "counts": make_counts(
                            attempted=int(stats.get("written") or 0) or universe_written,
                            written=universe_written,
                            failed=0,
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s universe_refresh_weekly failed", name)
                results.append({"job": name, "status": "error", "error": str(exc)})
            continue

        if action == "buyer_persona_refresh_weekly":
            try:
                from brain_os.services.buyer_persona_refresh import run_buyer_persona_refresh

                job_dry = dry_run or bool(job.get("dry_run"))
                crm = getattr(pipeline, "_crm", None) or getattr(pipeline, "crm", None)
                if crm is None and not job_dry:
                    try:
                        from brain_os.data.crm import CRMDatabase

                        crm = CRMDatabase()
                        await crm.create_tables()
                    except _HEARTBEAT_JOB_ERRORS:
                        logger.debug(
                            "buyer_persona_refresh_weekly: CRM unavailable",
                            exc_info=True,
                        )
                        crm = None
                payload = await run_buyer_persona_refresh(
                    dry_run=job_dry,
                    crm=crm,
                    job=job,
                )
                run_status = str(payload.pop("status", "ok") or "ok")
                if not job_dry and run_status == "ok":
                    await _mark_ran(redis, name)
                from brain_os.systems.heartbeat_writes_registry import make_counts

                bp_stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
                bp_planned = len(payload.get("planned") or [])
                bp_written = int(bp_stats.get("crm_written") or bp_stats.get("personas_found") or 0)
                results.append(
                    {
                        "job": name,
                        "status": "ok" if run_status in {"ok", "planned", "skipped"} else "error",
                        "action": action,
                        "agents_consulted": [],
                        "skip_llm": True,
                        "run_status": run_status,
                        **payload,
                        "counts": make_counts(
                            attempted=int(bp_stats.get("scanned") or 0) or bp_planned,
                            written=bp_written,
                            failed=int(bp_stats.get("crm_failed") or 0),
                        ),
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s buyer_persona_refresh_weekly failed", name)
                results.append({"job": name, "status": "error", "error": str(exc)})
            continue

        if action == "drip_cycle_daily":
            try:
                from brain_os.services.drip_loop import classify_drip_outcome, run_honest_drip_cycle

                job_dry = dry_run or bool(job.get("dry_run"))
                crm = getattr(pipeline, "_crm", None) or getattr(pipeline, "crm", None)
                quotes = getattr(pipeline, "_quotes", None) or getattr(pipeline, "quotes", None)
                pantheon = getattr(pipeline, "_pantheon", None) or getattr(
                    pipeline, "pantheon", None
                )
                bus = getattr(pantheon, "bus", None) if pantheon is not None else None
                if crm is None:
                    results.append(
                        {
                            "job": name,
                            "status": "error",
                            "action": action,
                            "error": "missing_crm",
                            "idle_reason": None,
                        }
                    )
                    continue
                payload = await run_honest_drip_cycle(
                    crm=crm,
                    quotes=quotes,
                    message_bus=bus,
                    gmail=draft_sender,
                    dry_run=job_dry,
                )
                # Fail-closed: a drip outcome must always be ok_active / ok_idle
                # / error. Bare "ok" (or a missing status) is re-classified here
                # so no writer regression can reach heartbeat_outcomes.jsonl.
                if str(payload.get("status") or "") not in {"ok_active", "ok_idle", "error"}:
                    payload.update(classify_drip_outcome(payload, dry_run=job_dry))
                await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "action": action,
                        "agents_consulted": [],
                        **payload,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s drip_cycle_daily failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "top100_cycle_daily":
            try:
                from brain_os.services.top100_campaign import Top100Config, run_top100_cycle

                job_dry = dry_run or bool(job.get("dry_run"))
                pantheon = getattr(pipeline, "_pantheon", None) or getattr(
                    pipeline, "pantheon", None
                )
                crm = getattr(pipeline, "_crm", None) or getattr(pipeline, "crm", None)
                email_processor = getattr(pipeline, "_email_processor", None) or getattr(
                    pipeline, "email_processor", None
                )
                if pantheon is None or crm is None:
                    results.append(
                        {"job": name, "status": "skipped", "reason": "missing_pipeline_services"}
                    )
                    continue
                payload = await run_top100_cycle(
                    pantheon,
                    crm,
                    email_processor,
                    config=Top100Config(),
                    dry_run=job_dry,
                    limit=int(job.get("limit") or 0) or None,
                    refresh_cohort=bool(job.get("refresh_cohort", False)),
                )
                if not job_dry:
                    await _mark_ran(redis, name)
                from brain_os.systems.heartbeat_writes_registry import make_counts

                queued = 0
                if isinstance(payload, dict):
                    queued = int(payload.get("queued_count") or 0)
                    if not queued:
                        queued = len(payload.get("queued") or [])
                results.append(
                    {
                        "job": name,
                        "status": "ok",
                        "action": action,
                        "agents_consulted": [],
                        "counts": make_counts(
                            attempted=queued,
                            written=queued,
                            failed=0,
                        ),
                        **payload,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s top100_cycle_daily failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "warm_lane_daily":
            try:
                from brain_os.services.warm_lane import draft_warm_lane_batch
                from brain_os.systems.heartbeat_writes_registry import make_counts

                job_dry = dry_run or bool(job.get("dry_run"))
                try:
                    limit = int(job.get("limit") or 0) or int(
                        get_settings().app.warm_lane_daily_draft_limit
                    )
                except (TypeError, ValueError):
                    limit = 5
                try:
                    sla_hours = int(job.get("sla_hours") or 0) or int(
                        get_settings().app.warm_lane_sla_hours
                    )
                except (TypeError, ValueError):
                    sla_hours = 24
                payload = draft_warm_lane_batch(
                    limit=limit,
                    dry_run=job_dry,
                    sla_hours=sla_hours,
                )
                if not job_dry:
                    await _mark_ran(redis, name)
                counts = payload.get("counts") or make_counts(
                    attempted=int(payload.get("attempted") or 0),
                    written=int(payload.get("written") or 0),
                    failed=int(payload.get("failed") or 0),
                )
                results.append(
                    {
                        "job": name,
                        "status": str(payload.get("status") or "ok"),
                        "action": action,
                        "agents_consulted": [],
                        "skip_llm": True,
                        "writes": True,
                        "counts": counts,
                        **{k: v for k, v in payload.items() if k != "status"},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s warm_lane_daily failed", name)
                results.append(_job_error_outcome(name, exc, action=action))
            continue

        if action == "store_census_monthly":
            try:
                from brain_os.systems.store_census import run_store_census

                payload = run_store_census(dry_run=dry_run or bool(job.get("dry_run")))
                if not (dry_run or bool(job.get("dry_run"))):
                    await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": str(payload.get("status") or "ok"),
                        "action": action,
                        "agents_consulted": [],
                        "skip_llm": True,
                        **payload,
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s store_census_monthly failed", name)
                results.append(_job_error_outcome(name, exc))
            continue

        if action == "eval_golden_weekly":
            try:
                from brain_os.eval.organism_eval import run_eval_golden_weekly_job

                payload = await run_eval_golden_weekly_job()
                if not dry_run:
                    await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": str(payload.get("status") or "ok"),
                        "action": action,
                        "agents_consulted": [],
                        "skip_llm": True,
                        "writes": True,
                        **{k: v for k, v in payload.items() if k != "status"},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s eval_golden_weekly failed", name)
                results.append(_job_error_outcome(name, exc, action=action))
            continue

        if action == "goal_planner_nightly":
            try:
                from brain_os.systems.campaign_goals import replan_all_active
                from brain_os.systems.heartbeat_writes_registry import make_counts

                payload = replan_all_active(dry_run_dispatch=bool(dry_run))
                counts = make_counts(
                    attempted=int(payload.get("attempted") or 0),
                    written=int(payload.get("written") or 0),
                    failed=int(payload.get("failed") or 0),
                )
                if not dry_run:
                    await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": str(payload.get("status") or "ok"),
                        "action": action,
                        "agents_consulted": [],
                        "skip_llm": True,
                        "writes": True,
                        "counts": counts,
                        **{k: v for k, v in payload.items() if k not in ("status",)},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s goal_planner_nightly failed", name)
                results.append(_job_error_outcome(name, exc, action=action))
            continue

        if action == "memory_recall_eval_weekly":
            try:
                from brain_os.memory.memory_eval import run_memory_recall_eval_weekly_job

                payload = await run_memory_recall_eval_weekly_job()
                if not dry_run:
                    await _mark_ran(redis, name)
                results.append(
                    {
                        "job": name,
                        "status": str(payload.get("status") or "ok"),
                        "action": action,
                        "agents_consulted": [],
                        "skip_llm": True,
                        "writes": True,
                        **{k: v for k, v in payload.items() if k != "status"},
                    }
                )
            except _HEARTBEAT_JOB_ERRORS as exc:
                logger.exception("Heartbeat job %s memory_recall_eval_weekly failed", name)
                results.append(_job_error_outcome(name, exc, action=action))
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
        try:
            from brain_os.systems.llm_cost_governor import should_defer_background_job

            if should_defer_background_job(name):
                results.append(
                    {
                        "job": name,
                        "status": "skipped",
                        "reason": "llm_daily_budget_soft_or_hard",
                    }
                )
                continue
        except Exception:  # noqa: BLE001 — budget defer check is advisory
            logger.debug("daily USD budget defer check skipped", exc_info=True)
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
            results.append(_job_error_outcome(name, exc))

    for row in results:
        job_name = str(row.get("job") or "")
        if job_name not in catch_up_jobs:
            continue
        if row.get("status") == "skipped" and str(row.get("reason") or "") in (
            "schedule",
            "interval",
            "disabled",
        ):
            continue
        row["catch_up"] = True

    try:
        from brain_os.systems.heartbeat_outcome_log import append_heartbeat_outcomes

        append_heartbeat_outcomes(results)
    except _HEARTBEAT_LOG_ERRORS:
        logger.debug("heartbeat outcome log append skipped", exc_info=True)

    if results:
        try:
            from brain_os.systems.breath_log import record_breath

            statuses = [str(r.get("status", "")) for r in results]
            record_breath(
                {},
                jobs_run=sum(1 for s in statuses if s in ("ok", "ok_active", "ok_idle")),
                jobs_skipped=sum(1 for s in statuses if s == "skipped"),
                jobs_failed=sum(1 for s in statuses if s == "error"),
                status="heartbeat_jobs",
            )
        except _HEARTBEAT_LOG_ERRORS:
            logger.debug("breath log append skipped", exc_info=True)

    return results


async def run_heartbeat_server_loop(
    *,
    redis: Any,
    jobs_path: str | Path | None = None,
    pipeline: Any,
    interval_minutes: int = 15,
    initial_delay_seconds: float = 45.0,
    task_orchestrator: Any = None,
    event_bus: Any = None,
    google_calendar: Any = None,
    draft_sender: DraftSender | None = None,
) -> None:
    """Long-lived loop for ``APP__HEARTBEAT_SERVER_ENABLED`` (server process).

    Waits ``initial_delay_seconds`` so startup can finish, then runs due jobs
    and sleeps ``interval_minutes`` between ticks. Cancel the task on shutdown.
    Re-reads ``APP__HEARTBEAT_JOBS_PATH`` / interval from settings each tick so
    lean↔full profile switches do not require a full process restart for path
    (interval change still applies on the next sleep).

    Prefer ``brain daemon install`` (launchd KeepAlive) on a laptop — this in-
    process loop dies with the server/terminal and fights sleep. On start and
    after a ≥30 minute wake gap, passes ``catch_up_scan=True`` so missed
    wall-clock slots within the catch-up window still fire once.
    """
    logger.info(
        "Heartbeat server loop started (interval=%dm, initial_delay=%.0fs)",
        interval_minutes,
        initial_delay_seconds,
    )
    try:
        from brain_os.config_learning_drift import warn_learning_critical_config_drift

        warn_learning_critical_config_drift()
    except Exception:  # noqa: BLE001 — config drift check is advisory
        logger.warning("Learning-critical config drift check failed", exc_info=True)
    try:
        verify_heartbeat_runtime_deps(force=True)
    except RuntimeError:
        logger.critical("Heartbeat server loop will not start jobs until deps are repaired")
        return
    try:
        if initial_delay_seconds > 0:
            await asyncio.sleep(initial_delay_seconds)
        first = True
        last_wall: float | None = None
        while True:
            settings = get_settings()
            path = jobs_path or settings.app.heartbeat_jobs_path
            tick_interval_m = int(
                getattr(settings.app, "heartbeat_server_interval_minutes", None) or interval_minutes
            )
            interval_s = max(60, tick_interval_m * 60)
            now_wall = time.time()
            wake = bool(first)
            if not wake and last_wall is not None and (now_wall - last_wall) >= _WAKE_GAP_SECONDS:
                wake = True
                logger.info(
                    "Wake/sleep gap detected (%.0fs since last tick) — catch-up scan",
                    now_wall - last_wall,
                )
            first = False
            try:
                results = await run_heartbeat_jobs(
                    redis=redis,
                    jobs_path=path,
                    pipeline=pipeline,
                    force=False,
                    task_orchestrator=task_orchestrator,
                    event_bus=event_bus,
                    google_calendar=google_calendar,
                    catch_up_scan=wake,
                    draft_sender=draft_sender,
                )
                ok = sum(1 for r in results if r.get("status") == "ok")
                skipped = sum(1 for r in results if r.get("status") == "skipped")
                err = sum(1 for r in results if r.get("status") == "error")
                catch_ups = sum(1 for r in results if r.get("catch_up") is True)
                logger.info(
                    "Heartbeat server tick: ok=%d skipped=%d error=%d catch_up=%d jobs=%s",
                    ok,
                    skipped,
                    err,
                    catch_ups,
                    path,
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — heartbeat tick isolation: one tick must not kill the loop
                logger.exception("Heartbeat server tick failed")
            last_wall = time.time()
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        logger.info("Heartbeat server loop cancelled")
        raise
