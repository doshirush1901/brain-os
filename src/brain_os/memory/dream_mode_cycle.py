"""Dream cycle stage orchestration and timeout helpers (Phase 9 split)."""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import TYPE_CHECKING, Any

from langfuse.decorators import observe

from brain_os.data.models import DreamReport

if TYPE_CHECKING:
    from brain_os.memory.dream_mode import DreamMode

logger = logging.getLogger(__name__)


async def _run_stage12_with_timeout(
    dream: DreamMode,
    stage_log: dict[str, Any],
    stage_key: str,
    label: str,
    runner: Any,
    timeout_sec: float = 30.0,
) -> None:
    try:
        await asyncio.wait_for(runner(stage_log), timeout=timeout_sec)
    except TimeoutError:
        logger.warning(
            "Dream %s timed out after %.1fs; continuing cycle",
            label,
            timeout_sec,
        )
        stage_log.setdefault("stages", {})[stage_key] = {"status": "timeout"}


async def _run_stage_with_timeout(
    stage_log: dict[str, Any],
    stage_key: str,
    label: str,
    runner: Any,
    timeout_sec: float = 30.0,
    default: Any = None,
) -> Any:
    try:
        return await asyncio.wait_for(runner(), timeout=timeout_sec)
    except TimeoutError:
        logger.warning(
            "Dream %s timed out after %.1fs; continuing cycle",
            label,
            timeout_sec,
        )
        stage_log.setdefault("stages", {})[stage_key] = {"status": "timeout"}
        return default


@observe()
async def execute_dream_cycle(dream: DreamMode, *, journal_last_24h: bool = False) -> DreamReport:
    """Execute the full dream cycle: journaling first, then sleep phase, then stages 0..12."""
    logger.info("DREAM CYCLE starting (journal_last_24h=%s)", journal_last_24h)
    cycle_date = date.today().isoformat()
    dream._reset_dream_checkpoint()

    stage_log: dict[str, Any] = {"cycle_date": cycle_date, "stages": {}}
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    await dream._stage11_agent_journaling(stage_log, journal_last_24h=journal_last_24h)
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    await dream._sleep_phase(stage_log)
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    await dream._stage0_deferred_ingestion(stage_log)

    await dream._stage0_5_sleep_training(stage_log)
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    await dream._stage0_55_pending_memory_queue(stage_log)
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    interactions = await dream._stage1_memory_ingestion(stage_log)

    episodes, memories_consolidated = await dream._stage2_episodic_consolidation(
        interactions, stage_log
    )
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    insights, gaps, connections, campaign_insights = await dream._stage3_insight_generation(
        episodes, stage_log
    )

    await dream._stage3e_gap_resolution(gaps, stage_log)
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    await dream._stage3_6_prediction_reconciliation(stage_log)
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    await dream._stage4_procedural_learning(insights, episodes, stage_log)
    await dream._refresh_skills_index(stage_log)

    await dream._stage5_memory_pruning(stage_log)

    price_conflicts = await _run_stage_with_timeout(
        stage_log,
        "6_price_conflict",
        "Stage 6 (price conflict check)",
        lambda: dream._stage6_price_conflict_check(stage_log),
        default=[],
    )

    try:
        await asyncio.wait_for(dream._stage7_and_8_graph(stage_log), timeout=30.0)
    except TimeoutError:
        logger.warning("Dream Stage 7/8 (graph) timed out after 30.0s; continuing cycle")
        stage_log.setdefault("stages", {})["7_quality_review"] = {"status": "timeout"}
        stage_log.setdefault("stages", {})["8_graph_consolidation"] = {"status": "timeout"}

    await dream._stage9_follow_up_automation(stage_log)

    await dream._stage10_morning_summary(
        stage_log, memories_consolidated, gaps, connections, price_conflicts
    )

    await _run_stage12_with_timeout(
        dream,
        stage_log,
        "12_cursor_sessions",
        "Stage 12 (cursor session learning)",
        dream._stage12_cursor_session_learning,
    )
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    await _run_stage12_with_timeout(
        dream,
        stage_log,
        "12c_learning_compiler",
        "Stage 12c (learning compiler)",
        dream._stage12c_learning_compiler,
    )
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    await _run_stage12_with_timeout(
        dream,
        stage_log,
        "12e_gepa_overlay_compile",
        "Stage 12e (GEPA overlay compile)",
        dream._stage12e_gepa_overlay_compile,
    )
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    await _run_stage12_with_timeout(
        dream,
        stage_log,
        "12d_learning_promotion",
        "Stage 12d (learning promotion)",
        dream._stage12d_learning_promotion,
    )
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    await _run_stage12_with_timeout(
        dream,
        stage_log,
        "12f_skill_candidates",
        "Stage 12f (skill candidates)",
        dream._stage12f_skill_candidates,
    )
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    await _run_stage12_with_timeout(
        dream,
        stage_log,
        "12b_curator_lite",
        "Stage 12b (curator-lite)",
        dream._stage12b_curator_lite,
    )
    dream._save_checkpoint(cycle_date=cycle_date, status="running", stage_log=stage_log)

    stage_results = {
        name: info.get("status", "unknown") for name, info in stage_log.get("stages", {}).items()
    }
    stage_log["metrics"] = {
        "stages_total": len(stage_results),
        "stages_ok": sum(1 for v in stage_results.values() if v == "ok"),
        "stages_error": sum(1 for v in stage_results.values() if v == "error"),
        "stages_skipped": sum(1 for v in stage_results.values() if v == "skipped"),
        "stages_timeout": sum(1 for v in stage_results.values() if v == "timeout"),
    }

    report = DreamReport(
        cycle_date=date.today(),
        memories_consolidated=memories_consolidated,
        gaps_identified=[g.get("description", "") for g in gaps],
        creative_connections=[c.get("insight", "") for c in connections],
        campaign_insights=campaign_insights,
        stage_results=stage_results,
    )

    await dream._persist_report(report)
    await asyncio.to_thread(dream._write_dream_log, stage_log, report)
    dream._save_checkpoint(cycle_date=cycle_date, status="completed", stage_log=stage_log)

    logger.info(
        "DREAM CYCLE complete: consolidated=%d gaps=%d connections=%d insights=%d",
        memories_consolidated,
        len(gaps),
        len(connections),
        len(campaign_insights),
    )
    return report
