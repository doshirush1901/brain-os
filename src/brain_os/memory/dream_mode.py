"""Dream mode — nightly memory consolidation, gap detection, and creative synthesis.

Implements an 11-stage dream cycle:
  0.   Deferred Ingestion — ingest files accessed via Alexandros fallback.
  0.5  Sleep Training — run Nemesis corrections on pending items.
  0.55 Pending Memory — drain operator heartbeat / CLI hints into Mem0.
  1.   Memory Ingestion — pull recent interactions from the CRM.
  2.   Episodic Consolidation — group related interactions into narrative episodes.
  3a.  Cross-episode Insights — LLM-driven pattern/contradiction detection.
  3b.  Gap Detection — identify knowledge gaps from metacognition table.
  3c.  Creative Synthesis — connect gaps and episodes for novel insights.
  3d.  Campaign Reflection — review campaign myokines for marketing insights.
  3e.  Active Gap Resolution — research and resolve top priority gaps.
  3.6  Prediction Reconciliation — Sophia closed-loop hot-lead / must-act / forecasts.
  4.   Procedural Learning — turn repeatable successful patterns into procedures.
  5.   Memory Pruning — archive or summarise older, less-relevant memories.
  6.   Price Conflict Check — scan pricing data for inconsistencies.
  7.   Conversation Quality Review — review retrieval quality via co-access matrix.
  8.   Graph Consolidation — tune knowledge graph relationships.
  9.   Follow-up Automation — detect stale quotes for follow-up.
  10.  Morning Summary — log dream cycle results.
  11.  Agent Journaling — first-person nightly reflections for each agent with actions today.

Each cycle is logged to ``data/dream_log.json`` for auditability.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.contracts.dream_brain_hooks import DreamBrainHooks, DreamStageContext
from brain_os.data.models import DreamReport
from brain_os.exceptions import DatabaseError, BrainOSError, LLMError
from brain_os.memory.conversation import ConversationMemory
from brain_os.memory.dream_mode_checkpoint import (
    load_dream_checkpoint,
    reset_dream_checkpoint,
    save_dream_checkpoint,
)
from brain_os.memory.dream_mode_constants import (
    CAMPAIGN_SYSTEM_PROMPT,
    CREATIVE_SYSTEM_PROMPT,
    DREAM_CHECKPOINT_PATH,
    DREAM_LOG_PATH,
    DREAM_STRUCTURED_MAX_USER_CHARS,
    GAP_SYSTEM_PROMPT,
    INSIGHT_SYSTEM_PROMPT,
    JOURNAL_SYSTEM_PROMPT,
    PROCEDURAL_SYSTEM_PROMPT,
    PRUNE_SYSTEM_PROMPT,
)
from brain_os.memory.dream_mode_cycle import execute_dream_cycle
from brain_os.memory.episodic import EpisodicMemory
from brain_os.memory.long_term import LongTermMemory
from brain_os.schemas.llm_outputs import (
    DreamCampaignInsights,
    DreamCreative,
    DreamGaps,
    DreamInsight,
    DreamProcedures,
    DreamPrune,
)
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class DreamMode:
    def __init__(
        self,
        long_term: LongTermMemory,
        episodic: EpisodicMemory,
        conversation: ConversationMemory,
        musculoskeletal: Any | None = None,
        retriever: Any | None = None,
        crm: Any | None = None,
        procedural_memory: Any | None = None,
        data_event_bus: Any | None = None,
        db_path: str = "data/conversations.db",
        dream_log_path: str | Path | None = None,
        agent_journal: Any | None = None,
        immune_system: Any | None = None,
        power_level_tracker: Any | None = None,
        curiosity_runner: Any | None = None,
        brain_hooks: DreamBrainHooks | None = None,
    ) -> None:
        self._long_term = long_term
        self._episodic = episodic
        self._conversation = conversation
        self._musculoskeletal = musculoskeletal
        self._retriever = retriever
        self._crm = crm
        self._procedural = procedural_memory
        self._event_bus = data_event_bus
        self._db_path = db_path
        self._llm = get_llm_client()
        self._db: aiosqlite.Connection | None = None
        self._dream_log_path = Path(dream_log_path) if dream_log_path else DREAM_LOG_PATH
        self._dream_checkpoint_path = DREAM_CHECKPOINT_PATH
        self._agent_journal = agent_journal
        self._immune_system = immune_system
        self._power_level_tracker = power_level_tracker
        self._curiosity_runner = curiosity_runner
        self._brain_hooks = brain_hooks

    def _stage_context(self) -> DreamStageContext:
        return DreamStageContext(
            long_term=self._long_term,
            episodic=self._episodic,
            conversation=self._conversation,
            retriever=self._retriever,
            crm=self._crm,
            procedural=self._procedural,
            event_bus=self._event_bus,
            db_path=self._db_path,
            memory_block_store=getattr(self, "_memory_block_store", None),
        )

    def configure(self, **kwargs: Any) -> None:
        """Late-bind optional dependencies after construction."""
        if "procedural_memory" in kwargs:
            self._procedural = kwargs["procedural_memory"]
        if "crm" in kwargs:
            self._crm = kwargs["crm"]
        if "musculoskeletal" in kwargs:
            self._musculoskeletal = kwargs["musculoskeletal"]
        if "retriever" in kwargs:
            self._retriever = kwargs["retriever"]
        if "agent_journal" in kwargs:
            self._agent_journal = kwargs["agent_journal"]
        if "immune_system" in kwargs:
            self._immune_system = kwargs["immune_system"]
        if "power_level_tracker" in kwargs:
            self._power_level_tracker = kwargs["power_level_tracker"]
        if "curiosity_runner" in kwargs:
            self._curiosity_runner = kwargs["curiosity_runner"]
        if "memory_block_store" in kwargs:
            self._memory_block_store = kwargs["memory_block_store"]
        if "relationship_memory" in kwargs:
            self._relationship_memory = kwargs["relationship_memory"]
        if "brain_hooks" in kwargs:
            self._brain_hooks = kwargs["brain_hooks"]

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path, timeout=30.0)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS dream_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_date TEXT NOT NULL UNIQUE,
                memories_consolidated INTEGER NOT NULL,
                gaps_identified TEXT NOT NULL,
                creative_connections TEXT NOT NULL,
                campaign_insights TEXT NOT NULL,
                stage_results TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """
        )
        try:
            await self._db.execute(
                "ALTER TABLE dream_reports ADD COLUMN stage_results TEXT NOT NULL DEFAULT '{}'"
            )
        except Exception as exc:
            pass
        await self._db.commit()

    # ── public API ────────────────────────────────────────────────────────

    async def run_dream_cycle(self, *, journal_last_24h: bool = False) -> DreamReport:
        """Execute the full dream cycle (orchestration in ``dream_mode_cycle``)."""
        return await execute_dream_cycle(self, journal_last_24h=journal_last_24h)

    def _load_checkpoint(self) -> dict[str, Any]:
        return load_dream_checkpoint(self._dream_checkpoint_path)

    def _reset_dream_checkpoint(self) -> None:
        reset_dream_checkpoint(self._dream_checkpoint_path)

    def _save_checkpoint(self, *, cycle_date: str, status: str, stage_log: dict[str, Any]) -> None:
        save_dream_checkpoint(
            self._dream_checkpoint_path,
            cycle_date=cycle_date,
            status=status,
            stage_log=stage_log,
        )

    async def run_journal_only(
        self,
        *,
        since_last_journal: bool = True,
        lookback_hours: float = 168.0,
    ) -> dict[str, Any]:
        """Run only agent journaling (no dream cycle). Like a save: journal from where it left off till now.

        When since_last_journal is True, for each recently active agent we get actions since their
        last journal entry and write one new reflection. When False, we use last lookback_hours.
        """
        if self._agent_journal is None:
            return {"entries_saved": 0, "agents_processed": 0, "reason": "no agent_journal"}

        _EPOCH_UTC = datetime(1970, 1, 1, tzinfo=UTC)
        today = date.today()
        entries_saved = 0
        agents_processed: list[str] = []

        try:
            agent_names = await self._agent_journal.get_agents_with_actions_since_hours(
                lookback_hours
            )
            if not agent_names:
                return {"entries_saved": 0, "agents_processed": 0, "agents": []}

            for agent_name in agent_names:
                if since_last_journal:
                    since = await self._agent_journal.get_latest_journal_created_at(agent_name)
                    if since is None:
                        since = _EPOCH_UTC
                    actions_list = await self._agent_journal.get_actions_since_datetime(
                        agent_name, since
                    )
                    actions_intro = "Since your last journal entry:\n"
                else:
                    actions_list = await self._agent_journal.get_actions_since_hours(
                        agent_name, lookback_hours
                    )
                    actions_intro = f"Your actions in the last {int(lookback_hours)} hours:\n"

                if not actions_list:
                    continue

                agents_processed.append(agent_name)
                actions_text = actions_intro + "\n".join(
                    f"- {a.get('action_text', '')} (outcome: {a.get('outcome', '')})"
                    for a in actions_list
                )
                system_prompt = JOURNAL_SYSTEM_PROMPT.format(
                    agent_name=agent_name,
                    actions=actions_text,
                )
                try:
                    reflection = await self._llm.generate_text(
                        system_prompt,
                        "Write your journal entry now.",
                        temperature=0.3,
                        name="dream.agent_journal",
                    )
                    if reflection and reflection.strip():
                        await self._agent_journal.save_journal_entry(
                            agent_name=agent_name,
                            reflection_text=reflection.strip(),
                            mood="",
                            at_date=today,
                        )
                        entries_saved += 1
                        logger.debug("Journal only: entry saved for %s", agent_name)
                except (LLMError, Exception):
                    logger.warning("Journal only: LLM failed for %s", agent_name, exc_info=True)

            logger.info(
                "Journal only complete — %d entries saved for %d agents (since_last=%s)",
                entries_saved,
                len(agents_processed),
                since_last_journal,
            )
            return {
                "entries_saved": entries_saved,
                "agents_processed": len(agents_processed),
                "agents": agents_processed,
                "since_last_journal": since_last_journal,
            }
        except Exception as exc:
            logger.exception("Journal only failed")
            return {
                "entries_saved": entries_saved,
                "agents_processed": len(agents_processed),
                "agents": agents_processed,
                "error": True,
            }

    async def _sleep_phase(self, stage_log: dict[str, Any]) -> None:
        """Run optional sleep-phase steps: phantom limb re-check, trust reconciliation, curiosity."""
        # Phantom limb — re-check health and update sense_lost
        if self._immune_system is not None:
            try:
                report = await self._immune_system.run_startup_validation()
                healthy = [k for k, v in report.items() if v.get("status") == "healthy"]
                stage_log["stages"]["sleep_phantom_limb"] = {
                    "status": "ok",
                    "healthy": healthy,
                }
                logger.info("Sleep phase: phantom limb re-check — %s healthy", len(healthy))
            except Exception as exc:
                logger.exception("Sleep phase phantom limb re-check failed")
                stage_log["stages"]["sleep_phantom_limb"] = {"status": "error"}

        # Trust reconciliation — nudge trust toward default
        if self._power_level_tracker is not None and hasattr(
            self._power_level_tracker, "nudge_trust_toward_default"
        ):
            try:
                await self._power_level_tracker.nudge_trust_toward_default(step=0.05)
                stage_log["stages"]["sleep_trust"] = {"status": "ok"}
                logger.info("Sleep phase: trust reconciliation done")
            except Exception as exc:
                logger.exception("Sleep phase trust reconciliation failed")
                stage_log["stages"]["sleep_trust"] = {"status": "error"}

        # Curiosity — run one cycle if runner provided (timeout so dream doesn't hang)
        _CURIOSITY_TIMEOUT_SEC = 120
        if self._curiosity_runner is not None:
            try:
                runner = self._curiosity_runner
                if asyncio.iscoroutinefunction(runner):
                    await asyncio.wait_for(runner(), timeout=_CURIOSITY_TIMEOUT_SEC)
                else:
                    coro = runner()
                    if asyncio.iscoroutine(coro):
                        await asyncio.wait_for(coro, timeout=_CURIOSITY_TIMEOUT_SEC)
                stage_log["stages"]["sleep_curiosity"] = {"status": "ok"}
                logger.info("Sleep phase: curiosity cycle done")
            except TimeoutError:
                logger.warning(
                    "Sleep phase: curiosity cycle timed out after %ds", _CURIOSITY_TIMEOUT_SEC
                )
                stage_log["stages"]["sleep_curiosity"] = {"status": "timeout"}
            except Exception as exc:
                logger.exception("Sleep phase curiosity cycle failed")
                stage_log["stages"]["sleep_curiosity"] = {"status": "error"}

    # ── Stage 0: Deferred Ingestion ──────────────────────────────────────

    async def _stage0_deferred_ingestion(self, stage_log: dict[str, Any]) -> None:
        """Ingest files that were accessed via Alexandros fallback since the last dream."""
        if self._brain_hooks is None:
            stage_log["stages"]["0_deferred_ingestion"] = {
                "status": "skipped",
                "reason": "no_brain_hooks",
            }
            return
        await self._brain_hooks.stage0_deferred_ingestion(self._stage_context(), stage_log)

    # ── Stage 1: Memory Ingestion ─────────────────────────────────────────

    async def _stage1_memory_ingestion(self, stage_log: dict[str, Any]) -> list[dict[str, Any]]:
        """Pull recent interactions from the CRM (last 24 h)."""
        interactions: list[dict[str, Any]] = []
        try:
            if self._crm is not None:
                raw = await self._crm.list_interactions()
                cutoff = datetime.now(UTC) - timedelta(hours=24)
                for ix in raw:
                    d = ix.to_dict() if hasattr(ix, "to_dict") else ix
                    created = d.get("created_at", "")
                    if isinstance(created, str) and created:
                        try:
                            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=UTC)
                            if dt < cutoff:
                                continue
                        except ValueError:
                            pass
                    interactions.append(d)
                logger.info("Stage 1: ingested %d recent CRM interactions", len(interactions))
            else:
                logger.debug("Stage 1: no CRM configured, skipping")

            stage_log["stages"]["1_memory_ingestion"] = {
                "status": "ok",
                "interactions_found": len(interactions),
            }
        except (DatabaseError, Exception):
            logger.exception("Dream Stage 1 (memory ingestion) failed")
            stage_log["stages"]["1_memory_ingestion"] = {"status": "error"}

        return interactions

    # ── Stage 2: Episodic Consolidation ───────────────────────────────────

    async def _stage2_episodic_consolidation(
        self,
        interactions: list[dict[str, Any]],
        stage_log: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        """Group related interactions into narrative episodes."""
        episodes: list[dict[str, Any]] = []
        memories_consolidated = 0

        try:
            # Group interactions by contact_id to form per-contact episodes
            by_contact: dict[str, list[dict[str, Any]]] = {}
            for ix in interactions:
                cid = ix.get("contact_id", "unknown")
                by_contact.setdefault(cid, []).append(ix)

            for contact_id, contact_interactions in by_contact.items():
                if len(contact_interactions) < 1:
                    continue
                transcript = []
                for ix in contact_interactions:
                    direction = ix.get("direction", "OUTBOUND")
                    role = "assistant" if direction == "OUTBOUND" else "user"
                    content = ix.get("content") or ix.get("subject") or "(no content)"
                    transcript.append({"role": role, "content": content})

                episode = await self._episodic.consolidate_episode(transcript, contact_id)
                episodes.append(episode)
                memories_consolidated += 1

            # Also consolidate conversation-memory sessions (original behaviour)
            try:
                if self._conversation._db is not None:
                    cutoff = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
                    cursor = await self._conversation._db.execute(
                        "SELECT DISTINCT user_id, channel FROM conversations WHERE last_message_at >= ?",
                        (cutoff,),
                    )
                    rows = await cursor.fetchall()
                    await cursor.close()
                    for user_id, channel in rows:
                        history = await self._conversation.get_history(user_id, channel)
                        if len(history) >= 2:
                            ep = await self._episodic.consolidate_episode(history, user_id)
                            episodes.append(ep)
                            memories_consolidated += 1
            except (LLMError, Exception):
                logger.exception("Stage 2: conversation-memory consolidation failed")

            logger.info(
                "Stage 2: consolidated %d episodes from %d contact groups + conversations",
                memories_consolidated,
                len(by_contact),
            )
            stage_log["stages"]["2_episodic_consolidation"] = {
                "status": "ok",
                "episodes_created": len(episodes),
                "memories_consolidated": memories_consolidated,
            }
        except (BrainOSError, Exception):
            logger.exception("Dream Stage 2 (episodic consolidation) failed")
            stage_log["stages"]["2_episodic_consolidation"] = {"status": "error"}

        return episodes, memories_consolidated

    # ── Stage 3: Insight Generation ───────────────────────────────────────

    async def _stage3_insight_generation(
        self,
        episodes: list[dict[str, Any]],
        stage_log: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict], list[dict], list[str]]:
        """Use LLM to find patterns, contradictions, and insights across episodes."""
        insights: dict[str, Any] = {}
        gaps: list[dict[str, Any]] = []
        connections: list[dict[str, Any]] = []
        campaign_insights: list[str] = []

        # 3a — Cross-episode insight analysis
        try:
            if episodes:
                episode_text = "\n\n".join(
                    f"Episode ({e.get('user_id', 'unknown')}): {e.get('narrative', '')}"
                    for e in episodes
                )
                result = await self._llm.generate_structured(
                    INSIGHT_SYSTEM_PROMPT,
                    episode_text,
                    DreamInsight,
                    temperature=0.2,
                    max_user_chars=DREAM_STRUCTURED_MAX_USER_CHARS,
                    name="dream.insight",
                )
                insights = result.model_dump()
            stage_log.setdefault("stages", {})["3a_cross_episode_insights"] = {
                "status": "ok",
                "patterns": len(insights.get("patterns", [])),
                "contradictions": len(insights.get("contradictions", [])),
                "recommendations": len(insights.get("recommendations", [])),
            }
        except (LLMError, Exception):
            logger.exception("Dream Stage 3a (cross-episode insights) failed")
            stage_log.setdefault("stages", {})["3a_cross_episode_insights"] = {"status": "error"}

        # 3b — Knowledge gap detection (from metacognition table)
        try:
            if self._db is not None:
                await self._db.execute(
                    """CREATE TABLE IF NOT EXISTS knowledge_gaps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL, state TEXT NOT NULL,
                        gaps TEXT NOT NULL, created_at TEXT NOT NULL
                    )"""
                )
                cutoff = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
                cursor = await self._db.execute(
                    "SELECT query, gaps FROM knowledge_gaps WHERE created_at >= ?",
                    (cutoff,),
                )
                rows = await cursor.fetchall()
                await cursor.close()
                if rows:
                    gap_text = "\n".join(f"Query: {r[0]}\nGaps: {r[1]}" for r in rows)
                    result = await self._llm.generate_structured(
                        GAP_SYSTEM_PROMPT,
                        gap_text,
                        DreamGaps,
                        max_user_chars=DREAM_STRUCTURED_MAX_USER_CHARS,
                        name="dream.gaps",
                    )
                    gaps = [g.model_dump() for g in result.gaps]
            stage_log["stages"]["3b_gap_detection"] = {
                "status": "ok",
                "gaps_found": len(gaps),
            }
        except (DatabaseError, Exception):
            logger.exception("Dream Stage 3b (gap detection) failed")
            stage_log["stages"]["3b_gap_detection"] = {"status": "error"}

        # 3c — Creative synthesis
        try:
            top_gaps = gaps[:5]
            recent_episodes: list[dict[str, Any]] = []
            ep_db = getattr(self._episodic, "_db", None) or self._db
            if ep_db is not None:
                try:
                    cursor = await ep_db.execute(
                        "SELECT narrative, created_at FROM episodes ORDER BY created_at DESC LIMIT 5"
                    )
                    ep_rows = await cursor.fetchall()
                    await cursor.close()
                    recent_episodes = [{"narrative": r[0], "created_at": r[1]} for r in ep_rows]
                except Exception as exc:
                    logger.debug("Stage 3c: episodes table not available")

            context_parts = []
            if top_gaps:
                context_parts.append("Knowledge gaps:\n" + json.dumps(top_gaps, indent=2))
            if recent_episodes:
                context_parts.append(
                    "Recent episodes:\n"
                    + "\n".join(f"[{e['created_at']}] {e['narrative']}" for e in recent_episodes)
                )
            context = "\n\n".join(context_parts) if context_parts else "No gaps or episodes."
            result = await self._llm.generate_structured(
                CREATIVE_SYSTEM_PROMPT,
                context,
                DreamCreative,
                temperature=0.3,
                max_user_chars=DREAM_STRUCTURED_MAX_USER_CHARS,
                name="dream.creative",
            )
            connections = [c.model_dump() for c in result.connections]
            stage_log["stages"]["3c_creative_synthesis"] = {
                "status": "ok",
                "connections_found": len(connections),
            }
        except (LLMError, Exception):
            logger.exception("Dream Stage 3c (creative synthesis) failed")
            stage_log["stages"]["3c_creative_synthesis"] = {"status": "error"}

        # 3d — Campaign reflection (via musculoskeletal myokines)
        try:
            if self._musculoskeletal is not None:
                myokines = await self._musculoskeletal.extract_myokines(period_days=7)
                myokines_text = json.dumps(myokines, indent=2)
                result = await self._llm.generate_structured(
                    CAMPAIGN_SYSTEM_PROMPT,
                    myokines_text,
                    DreamCampaignInsights,
                    max_user_chars=DREAM_STRUCTURED_MAX_USER_CHARS,
                    name="dream.campaign",
                )
                campaign_insights = result.insights
            stage_log["stages"]["3d_campaign_reflection"] = {
                "status": "ok",
                "campaign_insights": len(campaign_insights),
            }
        except (LLMError, Exception):
            logger.exception("Dream Stage 3d (campaign reflection) failed")
            stage_log["stages"]["3d_campaign_reflection"] = {"status": "error"}

        return insights, gaps, connections, campaign_insights

    # ── Stage 3e: Active Gap Resolution ───────────────────────────────────

    async def _stage3e_gap_resolution(
        self,
        gaps: list[dict[str, Any]],
        stage_log: dict[str, Any],
    ) -> None:
        """Research and resolve the highest-priority knowledge gaps."""
        if self._brain_hooks is None:
            stage_log["stages"]["3e_gap_resolution"] = {
                "status": "skipped",
                "reason": "no_brain_hooks",
            }
            return
        await self._brain_hooks.stage3e_gap_resolution(self._stage_context(), stage_log, gaps)

    # ── Stage 3.6: Prediction reconciliation ─────────────────────────────

    async def _stage3_6_prediction_reconciliation(self, stage_log: dict[str, Any]) -> None:
        """Reconcile yesterday's predictions; optional Sophia reflection → procedural memory."""
        from brain_os.config import get_settings

        cfg = get_settings().app
        if not cfg.prediction_reconciliation_enabled:
            stage_log["stages"]["3_6_prediction_reconciliation"] = {
                "status": "skipped",
                "reason": "disabled",
            }
            return
        try:
            from brain_os.memory.prediction_reconciliation import run_prediction_reconciliation_cycle

            report = await run_prediction_reconciliation_cycle(
                crm=self._crm,
                procedural=self._procedural,
                long_term=self._long_term,
                skip_llm=False,
            )
            stage_log["stages"]["3_6_prediction_reconciliation"] = {
                "status": "ok",
                "reconciled": report.reconciled,
                "correct": report.correct,
                "incorrect": report.incorrect,
                "must_act_reconciled": report.must_act_reconciled,
                "tyche_reconciled": report.tyche_reconciled,
                "predictions_recorded": report.predictions_recorded,
                "snapshot_written": report.snapshot_written,
                "procedures_merged": report.procedures_merged,
                "failures_recorded": report.failures_recorded,
            }
            logger.info(
                "Stage 3.6: reconciled %d predictions (%d correct, %d incorrect)",
                report.reconciled,
                report.correct,
                report.incorrect,
            )
        except Exception:
            logger.exception("Dream Stage 3.6 (prediction reconciliation) failed")
            stage_log["stages"]["3_6_prediction_reconciliation"] = {"status": "error"}

    # ── Stage 4: Procedural Learning ──────────────────────────────────────

    async def _stage4_procedural_learning(
        self,
        insights: dict[str, Any],
        episodes: list[dict[str, Any]],
        stage_log: dict[str, Any],
    ) -> None:
        """Turn high-confidence insights into repeatable procedures."""
        procedures_created = 0
        try:
            if self._procedural is None:
                logger.debug("Stage 4: no ProceduralMemory configured, skipping")
                stage_log["stages"]["4_procedural_learning"] = {"status": "skipped"}
                return

            recommendations = insights.get("recommendations", [])
            high_confidence = [
                r
                for r in recommendations
                if (r.get("priority") if isinstance(r, dict) else getattr(r, "priority", ""))
                == "HIGH"
            ]

            if not high_confidence and not episodes:
                stage_log["stages"]["4_procedural_learning"] = {
                    "status": "ok",
                    "procedures_created": 0,
                }
                return

            context_parts = []
            if high_confidence:
                serializable = [
                    r.model_dump() if hasattr(r, "model_dump") else r for r in high_confidence
                ]
                context_parts.append(
                    "High-priority recommendations:\n" + json.dumps(serializable, indent=2)
                )
            if episodes:
                episode_summaries = [
                    {"user_id": e.get("user_id", ""), "narrative": e.get("narrative", "")}
                    for e in episodes[:10]
                ]
                context_parts.append("Recent episodes:\n" + json.dumps(episode_summaries, indent=2))

            context = "\n\n".join(context_parts)
            result = await self._llm.generate_structured(
                PROCEDURAL_SYSTEM_PROMPT,
                context,
                DreamProcedures,
                max_user_chars=DREAM_STRUCTURED_MAX_USER_CHARS,
                name="dream.procedural",
            )

            for proc in result.procedures:
                trigger = proc.trigger
                steps = proc.steps
                if trigger and steps:
                    await self._procedural.learn_procedure(trigger, steps)
                    procedures_created += 1

            logger.info("Stage 4: created %d new procedures", procedures_created)
            stage_log["stages"]["4_procedural_learning"] = {
                "status": "ok",
                "procedures_created": procedures_created,
            }
        except (BrainOSError, Exception):
            logger.exception("Dream Stage 4 (procedural learning) failed")
            stage_log["stages"]["4_procedural_learning"] = {"status": "error"}

    async def _refresh_skills_index(self, stage_log: dict[str, Any]) -> None:
        """Regenerate ``data/brain/skills_index.json`` from :data:`SKILL_MATRIX` (deterministic)."""
        try:
            from brain_os.skills.index_catalog import clear_catalog_cache
            from brain_os.skills.index_generator import write_skills_index

            path = write_skills_index()
            clear_catalog_cache()
            stage_log["stages"]["skills_index_refresh"] = {"status": "ok", "path": str(path)}
            logger.info("Dream: refreshed skills index at %s", path)
        except Exception as exc:
            logger.exception("Dream: skills index refresh failed")
            stage_log["stages"]["skills_index_refresh"] = {"status": "error"}

    # ── Stage 5: Memory Pruning ───────────────────────────────────────────

    async def _stage5_memory_pruning(self, stage_log: dict[str, Any]) -> None:
        """Archive or summarise older, less-relevant memories."""
        archived = 0
        summarised = 0
        try:
            if self._db is None:
                stage_log["stages"]["5_memory_pruning"] = {"status": "skipped"}
                return

            cutoff = (datetime.now(UTC) - timedelta(days=30)).isoformat()
            cursor = await self._db.execute(
                "SELECT id, narrative, key_topics, created_at FROM episodes WHERE created_at < ? ORDER BY created_at ASC LIMIT 50",
                (cutoff,),
            )
            old_episodes = await cursor.fetchall()
            await cursor.close()

            if not old_episodes:
                logger.debug("Stage 5: no old episodes to prune")
                stage_log["stages"]["5_memory_pruning"] = {
                    "status": "ok",
                    "archived": 0,
                    "summarised": 0,
                }
                return

            episodes_text = json.dumps(
                [
                    {"id": r[0], "narrative": r[1], "topics": r[2], "date": r[3]}
                    for r in old_episodes
                ],
                indent=2,
            )
            try:
                from brain_os.memory.dream_triggers import aggregate_signals

                sig = aggregate_signals()
                if sig.get("pruning_aggressive"):
                    episodes_text = (
                        "**Operator signals:** elevated verification churn — prefer archiving "
                        "episodes that encode stale commercial assertions or contradictory claims.\n\n"
                        + episodes_text
                    )
            except Exception as exc:
                logger.debug("Dream stage 5 signals skipped", exc_info=True)

            result = await self._llm.generate_structured(
                PRUNE_SYSTEM_PROMPT,
                episodes_text,
                DreamPrune,
                max_user_chars=DREAM_STRUCTURED_MAX_USER_CHARS,
                name="dream.prune",
            )
            decisions = result.model_dump()

            archive_ids = decisions.get("archive", [])
            if archive_ids:
                placeholders = ",".join("?" for _ in archive_ids)
                await self._db.execute(
                    f"DELETE FROM episodes WHERE id IN ({placeholders})",
                    archive_ids,
                )
                archived = len(archive_ids)

            for group in decisions.get("summarise", []):
                ids = group.get("ids", [])
                summary = group.get("summary", "")
                if ids and summary:
                    placeholders = ",".join("?" for _ in ids)
                    await self._db.execute(
                        f"UPDATE episodes SET narrative = ? WHERE id IN ({placeholders})",
                        [summary, *ids],
                    )
                    summarised += len(ids)

            await self._db.commit()
            logger.info("Stage 5: archived %d, summarised %d episodes", archived, summarised)
            stage_log["stages"]["5_memory_pruning"] = {
                "status": "ok",
                "archived": archived,
                "summarised": summarised,
            }
        except (DatabaseError, Exception):
            logger.exception("Dream Stage 5 (memory pruning) failed")
            stage_log["stages"]["5_memory_pruning"] = {"status": "error"}

    # ── persistence helpers ───────────────────────────────────────────────

    async def _persist_report(self, report: DreamReport) -> None:
        """Write the dream report to the SQLite database. Retries on database lock."""
        if self._db is None:
            return
        now = datetime.now(UTC).isoformat()
        params = (
            report.cycle_date.isoformat(),
            report.memories_consolidated,
            json.dumps(report.gaps_identified),
            json.dumps(report.creative_connections),
            json.dumps(report.campaign_insights),
            json.dumps(report.stage_results),
            now,
        )
        sql = """
            INSERT OR REPLACE INTO dream_reports
            (cycle_date, memories_consolidated, gaps_identified, creative_connections,
             campaign_insights, stage_results, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
        last_error: BaseException | None = None
        for attempt in range(5):
            try:
                await self._db.execute(sql, params)
                await self._db.commit()
                return
            except sqlite3.OperationalError as e:
                last_error = e
                if "locked" not in str(e).lower() and "busy" not in str(e).lower():
                    raise
                delay = 1.0 * (attempt + 1)
                logger.warning(
                    "Dream report persist failed (database locked), retry %d/5 in %.1fs: %s",
                    attempt + 1,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
        logger.error("Dream report could not be persisted after 5 retries: %s", last_error)
        raise last_error  # type: ignore[misc]

    # ── Stage 0.5: Sleep Training ────────────────────────────────────────

    async def _stage0_5_sleep_training(self, stage_log: dict[str, Any]) -> None:
        """Run Nemesis sleep trainer on pending corrections."""
        if self._brain_hooks is None:
            stage_log["stages"]["0_5_sleep_training"] = {
                "status": "skipped",
                "reason": "no_brain_hooks",
            }
            return
        await self._brain_hooks.stage0_5_sleep_training(self._stage_context(), stage_log)

    async def _stage0_55_pending_memory_queue(self, stage_log: dict[str, Any]) -> None:
        """Drain ``PendingMemoryQueue`` rows into long-term memory (Mem0)."""
        try:
            from brain_os.memory.pending_memory_queue import PendingMemoryQueue

            q = PendingMemoryQueue()
            await q.initialize()
            stats = await q.drain_to_long_term(self._long_term)
            st = stats.get("status", "ok")
            stage_log["stages"]["0_55_pending_memory"] = {"status": st, **stats}
            logger.info("Stage 0.55: pending memory queue — %s", stats)
        except Exception as exc:
            logger.exception("Dream Stage 0.55 (pending memory queue) failed")
            stage_log["stages"]["0_55_pending_memory"] = {"status": "error"}

    # ── Stage 6: Price Conflict Check ──────────────────────────────────

    async def _stage6_price_conflict_check(self, stage_log: dict[str, Any]) -> list[dict[str, Any]]:
        """Scan pricing data for inconsistencies."""
        if self._brain_hooks is None:
            stage_log["stages"]["6_price_conflict"] = {
                "status": "skipped",
                "reason": "no_brain_hooks",
            }
            return []
        return await self._brain_hooks.stage6_price_conflict_check(self._stage_context(), stage_log)

    # ── Stages 7 & 8: Quality Review + Graph Consolidation (shared graph) ──

    async def _stage7_and_8_graph(self, stage_log: dict[str, Any]) -> None:
        """Quality review and graph consolidation using a single KnowledgeGraph."""
        if self._brain_hooks is None:
            stage_log["stages"]["7_quality_review"] = {
                "status": "skipped",
                "reason": "no_brain_hooks",
            }
            stage_log["stages"]["8_graph_consolidation"] = {
                "status": "skipped",
                "reason": "no_brain_hooks",
            }
            return
        await self._brain_hooks.stage7_and_8_graph(self._stage_context(), stage_log)

    # ── Stage 9: Follow-up Automation ──────────────────────────────────

    async def _stage9_follow_up_automation(self, stage_log: dict[str, Any]) -> None:
        """Detect stale quotes and suggest follow-ups."""
        try:
            if self._crm is None:
                stage_log["stages"]["9_follow_up"] = {"status": "skipped", "reason": "no CRM"}
                return

            stale_deals = []
            deals = await self._crm.list_deals()
            cutoff = datetime.now(UTC) - timedelta(days=14)
            for deal in deals:
                updated = getattr(deal, "updated_at", None)
                if updated and isinstance(updated, str):
                    try:
                        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                        if dt < cutoff:
                            stale_deals.append(deal)
                    except ValueError:
                        pass

            stage_log["stages"]["9_follow_up"] = {"status": "ok", "stale_deals": len(stale_deals)}
            logger.info("Stage 9: %d stale deals found for follow-up", len(stale_deals))
        except (DatabaseError, Exception):
            logger.exception("Dream Stage 9 (follow-up) failed")
            stage_log["stages"]["9_follow_up"] = {"status": "error"}

    # ── Stage 10: Morning Summary ──────────────────────────────────────

    async def _stage10_morning_summary(
        self,
        stage_log: dict[str, Any],
        memories_consolidated: int,
        gaps: list[dict[str, Any]],
        connections: list[dict[str, Any]],
        price_conflicts: list[dict[str, Any]],
    ) -> None:
        """Log a morning summary of the dream cycle."""
        try:
            deferred = (
                stage_log.get("stages", {}).get("0_deferred_ingestion", {}).get("files_ingested", 0)
            )
            sleep_training = (
                stage_log.get("stages", {}).get("0_5_sleep_training", {}).get("corrections", 0)
            )

            failed_stages = [
                name
                for name, info in stage_log.get("stages", {}).items()
                if info.get("status") == "error"
            ]

            lines = [
                "Dream cycle complete.",
                f"- Memories consolidated: {memories_consolidated}",
                f"- Knowledge gaps found: {len(gaps)}",
                f"- Creative connections: {len(connections)}",
                f"- Price conflicts: {len(price_conflicts)}",
                f"- Files ingested (deferred): {deferred}",
                f"- Corrections trained: {sleep_training}",
            ]
            if failed_stages:
                lines.append(f"- FAILED stages: {', '.join(failed_stages)}")

            logger.info("Stage 10: %s", "\n".join(lines))
            stage_log["stages"]["10_morning_summary"] = {"status": "ok"}
        except (BrainOSError, Exception):
            logger.exception("Dream Stage 10 (morning summary) failed")
            stage_log["stages"]["10_morning_summary"] = {"status": "error"}

    # ── Stage 12: Cursor Session Learning ────────────────────────────────

    async def _stage12_cursor_session_learning(self, stage_log: dict[str, Any]) -> None:
        """Learn from recent Cursor chat sessions logged by Graphe."""
        try:
            from brain_os.agents.graphe import _DB_PATH as _SESSION_DB

            if not _SESSION_DB.exists():
                stage_log["stages"]["12_cursor_sessions"] = {
                    "status": "skipped",
                    "reason": "no session db",
                }
                return

            import aiosqlite

            async with aiosqlite.connect(str(_SESSION_DB), timeout=30.0) as db:
                await db.execute("PRAGMA busy_timeout=30000")
                cursor = await db.execute(
                    "SELECT query, agents_used, response_summary FROM cursor_sessions "
                    "ORDER BY timestamp DESC LIMIT 50"
                )
                rows = await cursor.fetchall()

            if not rows:
                stage_log["stages"]["12_cursor_sessions"] = {"status": "ok", "sessions_learned": 0}
                return

            patterns: list[str] = []
            for query, agents_json, summary in rows:
                patterns.append(
                    f"Q: {(query or '')[:100]} → Agents: {agents_json} → {(summary or '')[:100]}"
                )

            learning_prompt = (
                "Analyze these recent Cursor chat sessions and extract 3-5 learnings "
                "about query patterns, optimal agent routing, and common user needs:\n\n"
                + "\n".join(patterns[:30])
            )

            try:
                learnings = await self._llm.generate_text(
                    "You extract learnings from Cursor session logs for Brain OS.",
                    learning_prompt,
                    name="dream.cursor_session_learning",
                )
                if self._long_term is not None and learnings:
                    await self._long_term.store(
                        learnings,
                        user_id="global",
                        metadata={"type": "cursor_session_learning", "source": "dream_stage_12"},
                    )
                stage_log["stages"]["12_cursor_sessions"] = {
                    "status": "ok",
                    "sessions_learned": len(rows),
                }
                logger.info("Stage 12: learned from %d cursor sessions", len(rows))
            except Exception as exc:
                logger.exception("Stage 12: LLM learning failed")
                stage_log["stages"]["12_cursor_sessions"] = {"status": "error"}

        except Exception as exc:
            logger.exception("Dream Stage 12 (cursor session learning) failed")
            stage_log["stages"]["12_cursor_sessions"] = {"status": "error"}

    async def _stage12c_learning_compiler(self, stage_log: dict[str, Any]) -> None:
        """Aggregate Graphe rows into ``candidate_procedures.json`` / nudge stubs."""
        if self._brain_hooks is None:
            stage_log.setdefault("stages", {})["12c_learning_compiler"] = {
                "status": "skipped",
                "reason": "no_brain_hooks",
            }
            return
        await self._brain_hooks.stage12c_learning_compiler(self._stage_context(), stage_log)

    async def _stage12e_gepa_overlay_compile(self, stage_log: dict[str, Any]) -> None:
        """Persisted tool stats → ``strategy_overlays_candidate.json`` when enabled."""
        if self._brain_hooks is None:
            stage_log.setdefault("stages", {})["12e_gepa_overlay_compile"] = {
                "status": "skipped",
                "reason": "no_brain_hooks",
            }
            return
        await self._brain_hooks.stage12e_gepa_overlay_compile(self._stage_context(), stage_log)

    async def _stage12d_learning_promotion(self, stage_log: dict[str, Any]) -> None:
        """Optional promotion into ProceduralMemory when ``APP__LEARNED_PROCEDURE_PROMOTION_ENABLED``."""
        if self._brain_hooks is None:
            stage_log.setdefault("stages", {})["12d_learning_promotion"] = {
                "status": "skipped",
                "reason": "no_brain_hooks",
            }
            return
        await self._brain_hooks.stage12d_learning_promotion(self._stage_context(), stage_log)

    async def _stage12f_skill_candidates(self, stage_log: dict[str, Any]) -> None:
        """Compile skill candidates from praise_events.jsonl when enabled."""
        if self._brain_hooks is None:
            stage_log.setdefault("stages", {})["12f_skill_candidates"] = {
                "status": "skipped",
                "reason": "no_brain_hooks",
            }
            return
        await self._brain_hooks.stage12f_skill_candidates(self._stage_context(), stage_log)

    async def _stage12b_curator_lite(self, stage_log: dict[str, Any]) -> None:
        """Hermes-style SKILL/prompt hygiene report (no LLM). Opt-in via BRAIN_DREAM_CURATOR."""
        if self._brain_hooks is None:
            stage_log["stages"]["12b_curator_lite"] = {
                "status": "skipped",
                "reason": "no_brain_hooks",
            }
            return
        await self._brain_hooks.stage12b_curator_lite(self._stage_context(), stage_log)

    # ── Stage 11: Agent Journaling ───────────────────────────────────────

    async def _stage11_agent_journaling(
        self, stage_log: dict[str, Any], *, journal_last_24h: bool = False
    ) -> None:
        """For each agent with actions (today or last 24h), generate and save a first-person journal entry."""
        entries_saved = 0
        try:
            if self._agent_journal is None:
                stage_log["stages"]["11_agent_journaling"] = {
                    "status": "skipped",
                    "reason": "no agent_journal",
                }
                return

            today = date.today()
            if journal_last_24h:
                agent_names = await self._agent_journal.get_agents_with_actions_since_hours(24.0)
            else:
                agent_names = await self._agent_journal.get_agents_with_actions_for_date(today)
            if not agent_names:
                stage_log["stages"]["11_agent_journaling"] = {"status": "ok", "entries_saved": 0}
                return

            for agent_name in agent_names:
                if journal_last_24h:
                    actions_list = await self._agent_journal.get_actions_since_hours(
                        agent_name, 24.0
                    )
                else:
                    actions_list = await self._agent_journal.get_actions_for_date(agent_name, today)
                if not actions_list:
                    continue
                actions_text = "\n".join(
                    f"- {a.get('action_text', '')} (outcome: {a.get('outcome', '')})"
                    for a in actions_list
                )
                system_prompt = JOURNAL_SYSTEM_PROMPT.format(
                    agent_name=agent_name,
                    actions=actions_text,
                )
                try:
                    reflection = await self._llm.generate_text(
                        system_prompt,
                        "Write your journal entry now.",
                        temperature=0.3,
                        name="dream.agent_journal",
                    )
                    if reflection and reflection.strip():
                        await self._agent_journal.save_journal_entry(
                            agent_name=agent_name,
                            reflection_text=reflection.strip(),
                            mood="",
                            at_date=today,
                        )
                        entries_saved += 1
                        logger.debug("Stage 11: journal entry saved for %s", agent_name)
                except (LLMError, Exception):
                    logger.warning("Stage 11: LLM journal failed for %s", agent_name, exc_info=True)

            stage_log["stages"]["11_agent_journaling"] = {
                "status": "ok",
                "entries_saved": entries_saved,
                "agents_processed": len(agent_names),
                "window": "last_24h" if journal_last_24h else "today",
            }
            logger.info(
                "Stage 11: agent journaling — %d entries saved for %d agents (window=%s)",
                entries_saved,
                len(agent_names),
                "last_24h" if journal_last_24h else "today",
            )
        except (BrainOSError, Exception):
            logger.exception("Dream Stage 11 (agent journaling) failed")
            stage_log["stages"]["11_agent_journaling"] = {"status": "error"}

    def _write_dream_log(self, stage_log: dict[str, Any], report: DreamReport) -> None:
        """Append the cycle results to dream_log.json."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "cycle_date": report.cycle_date.isoformat(),
            "memories_consolidated": report.memories_consolidated,
            "gaps_identified": report.gaps_identified,
            "creative_connections": report.creative_connections,
            "campaign_insights": report.campaign_insights,
            "stage_results": report.stage_results,
            "stages": stage_log.get("stages", {}),
        }

        existing: list[dict[str, Any]] = []
        if self._dream_log_path.exists():
            try:
                existing = json.loads(self._dream_log_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = [existing]
            except (json.JSONDecodeError, OSError):
                existing = []

        existing.append(entry)
        if len(existing) > 450:
            logger.warning(
                "Dream log has %d entries (cap=500). Oldest entries will be dropped.",
                len(existing),
            )
        existing = existing[-500:]
        try:
            self._dream_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._dream_log_path.write_text(
                json.dumps(existing, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        except OSError:
            logger.exception("Failed to write dream log to %s", self._dream_log_path)

    # ── query helpers ─────────────────────────────────────────────────────

    async def get_dream_reports(self, limit: int = 7) -> list[DreamReport]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT cycle_date, memories_consolidated, gaps_identified,
                   creative_connections, campaign_insights, stage_results
            FROM dream_reports
            ORDER BY cycle_date DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        reports = []
        for r in rows:
            cycle_date_str = r[0]
            cycle_dt = date.fromisoformat(cycle_date_str.replace("Z", "").split("T")[0])
            gaps_val = json.loads(r[2]) if isinstance(r[2], str) else r[2] or []
            creative = json.loads(r[3]) if isinstance(r[3], str) else r[3] or []
            campaign = json.loads(r[4]) if isinstance(r[4], str) else r[4] or []
            stages = json.loads(r[5]) if isinstance(r[5], str) and r[5] else {}
            reports.append(
                DreamReport(
                    cycle_date=cycle_dt,
                    memories_consolidated=r[1],
                    gaps_identified=gaps_val,
                    creative_connections=creative,
                    campaign_insights=campaign,
                    stage_results=stages,
                )
            )
        return reports

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> DreamMode:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
