"""Respiratory system — operational rhythm and cadence.

Manages Brain OS's heartbeat (periodic health logging), inhale cycle (morning
data ingestion), exhale cycle (nightly consolidation and reporting), and
per-request breath timing.
"""

from __future__ import annotations

import asyncio
import logging
import resource
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

# Background rhythms: tolerate expected I/O, HTTP, parsing, and domain failures without crashing loops.
_RESP_TASK_ERRORS = (
    httpx.HTTPError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    ImportError,
    ModuleNotFoundError,
)

_MEMORY_THRESHOLD_MB = 2048
_BREATH_THRESHOLD_MS = 30_000


class RespiratorySystem:
    """Manages Brain OS's operational rhythm via heartbeat, inhale/exhale cycles."""

    def __init__(
        self,
        *,
        dream_mode: Any | None = None,
        drip_engine: Any | None = None,
        immune_system: Any | None = None,
        email_processor: Any | None = None,
        goal_manager: Any | None = None,
        bus: Any | None = None,
        inhale_hour: int = 6,
        inhale_minute: int = 0,
        morning_brain_hour: int = 6,
        morning_brain_minute: int = 30,
        exhale_hour: int = 22,
        exhale_minute: int = 0,
        heartbeat_interval_seconds: int = 300,
        deep_consolidation_interval_hours: float = 6.0,
        morning_brain_enabled: bool = True,
    ) -> None:
        self._dream_mode = dream_mode
        self._drip_engine = drip_engine
        self._immune_system = immune_system
        self._email_processor = email_processor
        self._goal_manager = goal_manager
        self._bus = bus

        self._inhale_hour = inhale_hour
        self._inhale_minute = inhale_minute
        self._morning_brain_hour = morning_brain_hour
        self._morning_brain_minute = morning_brain_minute
        self._morning_brain_enabled = morning_brain_enabled
        self._exhale_hour = exhale_hour
        self._exhale_minute = exhale_minute
        self._heartbeat_interval = heartbeat_interval_seconds
        self._deep_consolidation_interval_hours = max(0.0, float(deep_consolidation_interval_hours))

        self._scheduler = AsyncIOScheduler()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._start_time: float = 0.0

        self._breath_durations: list[float] = []
        self._breath_lock = asyncio.Lock()
        self._dream_lock = asyncio.Lock()

    # ── HEARTBEAT ─────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                vitals = await self._collect_vitals()
                logger.info(
                    "HEARTBEAT | %s | vitals=%s",
                    datetime.now(UTC).isoformat(),
                    vitals,
                )
                unhealthy = self._is_unhealthy(vitals)
                try:
                    from brain_os.systems.breath_log import record_breath

                    record_breath(vitals, status="unhealthy" if unhealthy else "ok")
                except _RESP_TASK_ERRORS:
                    logger.debug("Breath log record failed", exc_info=True)
                if unhealthy and self._immune_system is not None:
                    await self._immune_system.respond(vitals)
            except _RESP_TASK_ERRORS:
                logger.exception("Heartbeat iteration failed")
            await asyncio.sleep(self._heartbeat_interval)

    async def _collect_vitals(self) -> dict[str, Any]:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        memory_mb = usage.ru_maxrss / divisor

        avg_breath: float = 0.0
        async with self._breath_lock:
            if self._breath_durations:
                avg_breath = sum(self._breath_durations) / len(self._breath_durations)

        return {
            "memory_mb": round(memory_mb, 1),
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "avg_breath_ms": round(avg_breath, 1),
        }

    @staticmethod
    def _is_unhealthy(vitals: dict[str, Any]) -> bool:
        if vitals.get("memory_mb", 0) > _MEMORY_THRESHOLD_MB:
            return True
        if vitals.get("avg_breath_ms", 0) > _BREATH_THRESHOLD_MS:
            return True
        return False

    # ── INHALE ────────────────────────────────────────────────────────────

    async def _inhale(self) -> None:
        logger.info("INHALE cycle starting")

        try:
            from brain_os.brain.eat_pending import ingest_pending_eat_digests
            from brain_os.brain.imports_metadata_index import build_index
            from brain_os.brain.ingestion_gatekeeper import run_ingestion_cycle

            # Discover new/changed drops under data/imports before gatekeeper scan.
            index_stats = await build_index(use_llm=False, force=False)
            logger.info("INHALE imports metadata index: %s", index_stats)
            result = await run_ingestion_cycle()
            logger.info("INHALE Alexandros ingestion: %s", result)
            # Catch EAT digests written in Cursor but never ingested (lock / agent stop).
            eat_sweep = await ingest_pending_eat_digests(limit=25)
            logger.info("INHALE EAT pending sweep: %s", eat_sweep)
        except _RESP_TASK_ERRORS:
            logger.exception("INHALE Alexandros ingestion failed")

        if self._email_processor is not None:
            try:
                await self._email_processor.process_inbox()
                logger.info("INHALE email processing complete")
            except _RESP_TASK_ERRORS:
                logger.exception("INHALE email processing failed")
        else:
            logger.debug("INHALE skipping email processing (no EmailProcessor)")

    # ── DREAM ─────────────────────────────────────────────────────────────

    async def _dream(self) -> dict[str, Any]:
        """Run the DreamMode consolidation cycle and return status metadata."""
        if self._dream_mode is None:
            logger.debug("DREAM skipping (DreamMode not configured)")
            return {"status": "skipped", "reason": "not_configured"}
        if self._dream_lock.locked():
            logger.info("DREAM skipping (already running)")
            return {"status": "skipped", "reason": "already_running"}

        async with self._dream_lock:
            logger.info("DREAM cycle starting (scheduled)")
            try:
                report = await self._dream_mode.run_dream_cycle()
                logger.info(
                    "DREAM cycle complete: consolidated=%d gaps=%d connections=%d",
                    report.memories_consolidated,
                    len(report.gaps_identified),
                    len(report.creative_connections),
                )
                return {
                    "status": "ok",
                    "memories_consolidated": report.memories_consolidated,
                    "gaps": len(report.gaps_identified),
                    "connections": len(report.creative_connections),
                }
            except _RESP_TASK_ERRORS:
                logger.exception("DREAM cycle failed")
                return {"status": "error"}

    # ── EXHALE ────────────────────────────────────────────────────────────

    async def _exhale(self) -> None:
        logger.info("EXHALE cycle starting")
        summary_parts: list[str] = []

        dream_status = await self._dream()
        if dream_status.get("status") == "ok":
            summary_parts.append(
                "Dream consolidation: "
                f"memories={dream_status.get('memories_consolidated', 0)}, "
                f"gaps={dream_status.get('gaps', 0)}, "
                f"connections={dream_status.get('connections', 0)}"
            )
        elif dream_status.get("status") == "error":
            summary_parts.append("Dream consolidation: failed (see logs)")

        if self._drip_engine is not None:
            try:
                drip_result = await self._drip_engine.evaluate_campaigns()
                summary_parts.append(f"Drip evaluation: {drip_result}")
            except _RESP_TASK_ERRORS:
                logger.exception("EXHALE drip evaluation failed")
        else:
            logger.debug("EXHALE skipping drip engine (not configured)")

        if self._goal_manager is not None:
            try:
                stalled = await self._goal_manager.sweep_stalled_goals()
                if stalled and self._bus is not None:
                    for goal in stalled:
                        await self._bus.publish(
                            "hermes",
                            {
                                "action": "draft_follow_up",
                                "contact_id": goal.contact_id,
                                "goal_type": goal.goal_type.value,
                                "stalled_since": goal.created_at.isoformat(),
                            },
                        )
                summary_parts.append(f"Stalled goals: {len(stalled)} flagged for follow-up")
            except _RESP_TASK_ERRORS:
                logger.exception("EXHALE goal sweep failed")
        else:
            logger.debug("EXHALE skipping goal sweep (GoalManager not configured)")

        try:
            from brain_os.services.top100_campaign import run_top100_weekly_if_due

            top100_result = await run_top100_weekly_if_due(
                email_processor=self._email_processor,
            )
            if top100_result.get("status") != "skipped":
                summary_parts.append(
                    f"Top100 campaign: queued={top100_result.get('queued_count', 0)} "
                    f"skipped={top100_result.get('skipped_count', 0)}"
                )
        except _RESP_TASK_ERRORS:
            logger.exception("EXHALE top100 campaign failed")

        summary = (
            "Brain OS Daily Exhale Report\n" + "\n".join(summary_parts)
            if summary_parts
            else "Brain OS Daily Exhale: no active subsystems"
        )
        logger.info(summary)

    # ── Public CLI entry points ─────────────────────────────────────────────

    async def run_inhale_cycle(self) -> None:
        """Run a single inhale cycle on demand (CLI / Cursor use)."""
        await self._inhale()

    async def _morning_brain(self) -> None:
        if not self._morning_brain_enabled:
            logger.debug("MORNING BRAIN skipping (disabled)")
            return
        logger.info("MORNING BRAIN cycle starting")
        try:
            from brain_os.services.morning_brain import run_morning_brain

            result = await run_morning_brain(trigger="respiratory:06:30")
            logger.info(
                "MORNING BRAIN complete: ok=%s actions=%s must_act=%s",
                result.get("ok"),
                len((result.get("report") or {}).get("actions") or []),
                (result.get("report") or {}).get("must_act_count"),
            )
        except _RESP_TASK_ERRORS:
            logger.exception("MORNING BRAIN cycle failed")

    async def run_morning_brain_cycle(self) -> None:
        """Run Morning Brain on demand (CLI / cron fallback)."""
        await self._morning_brain()

    async def run_exhale_cycle(self) -> None:
        """Run a single exhale cycle on demand (CLI / Cursor use)."""
        await self._exhale()

    # ── BREATH TIMING ─────────────────────────────────────────────────────

    @asynccontextmanager
    async def breath(self) -> AsyncIterator[None]:
        """Context manager that measures per-request processing time."""
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            async with self._breath_lock:
                self._breath_durations.append(elapsed_ms)
                if len(self._breath_durations) > 1000:
                    self._breath_durations = self._breath_durations[-500:]
            logger.debug("BREATH duration=%.1fms", elapsed_ms)

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the heartbeat task and scheduled inhale/exhale cycles."""
        self._start_time = time.monotonic()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self._scheduler.add_job(
            self._inhale,
            CronTrigger(hour=self._inhale_hour, minute=self._inhale_minute),
            id="inhale",
            replace_existing=True,
        )
        if self._morning_brain_enabled:
            self._scheduler.add_job(
                self._morning_brain,
                CronTrigger(hour=self._morning_brain_hour, minute=self._morning_brain_minute),
                id="morning_brain",
                replace_existing=True,
            )
        self._scheduler.add_job(
            self._exhale,
            CronTrigger(hour=self._exhale_hour, minute=self._exhale_minute),
            id="exhale",
            replace_existing=True,
        )
        if self._dream_mode is not None and self._deep_consolidation_interval_hours > 0:
            self._scheduler.add_job(
                self._dream,
                IntervalTrigger(hours=self._deep_consolidation_interval_hours),
                id="deep_consolidation",
                replace_existing=True,
            )
        self._scheduler.start()
        logger.info(
            "RespiratorySystem started (deep_consolidation_interval_hours=%.2f)",
            self._deep_consolidation_interval_hours,
        )

    async def stop(self) -> None:
        """Cancel the heartbeat and shut down the scheduler."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

        logger.info("RespiratorySystem stopped")
