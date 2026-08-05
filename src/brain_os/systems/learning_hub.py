"""Learning hub — feedback processing, gap analysis, and procedure suggestion.

Closes the feedback loop for the Brain OS system.  After every interaction the
LearningHub can:

* Record and analyse user feedback (scores + optional corrections).
* Identify skill or knowledge gaps from poorly-rated interactions.
* Suggest new procedures for ProceduralMemory based on successful or
  corrected interactions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from langfuse.decorators import observe
from pydantic import ValidationError

from brain_os.config import LLMConfig
from brain_os.data.crm import CRMDatabase
from brain_os.memory.procedural import ProceduralMemory, Procedure
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import (
    CorrectionAnalysis,
    GapAnalysis,
    ProcedureSteps,
    SuccessPlaybookExtraction,
)
from brain_os.services.llm_client import get_llm_client
from brain_os.skills import SKILL_MATRIX

logger = logging.getLogger(__name__)

_FEEDBACK_DISK_ERRORS = (
    json.JSONDecodeError,
    sqlite3.Error,
    OSError,
    ValueError,
    TypeError,
    KeyError,
)
_MICRO_LEARNING_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    sqlite3.Error,
    asyncio.TimeoutError,
    httpx.HTTPError,
    json.JSONDecodeError,
    UnicodeDecodeError,
)
_LLM_STRUCT_ERRORS = (
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    OSError,
    asyncio.TimeoutError,
    httpx.HTTPError,
    json.JSONDecodeError,
    UnicodeDecodeError,
    ValidationError,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FEEDBACK_DB_PATH = _PROJECT_ROOT / "data" / "brain" / "feedback.db"

_FEEDBACK_THRESHOLD_POOR = 3
_FEEDBACK_THRESHOLD_GOOD = 7

_GAP_ANALYSIS_PROMPT = load_prompt("gap_analysis")

_CORRECTION_ANALYSIS_PROMPT = load_prompt("correction_analysis")

_PROCEDURE_EXTRACTION_PROMPT = load_prompt("procedure_extraction")

_SUCCESS_PLAYBOOK_PROMPT = load_prompt("success_playbook_extraction")


@dataclass
class FeedbackRecord:
    interaction_id: str
    feedback_score: int
    correction: str | None = None
    correction_analysis: dict[str, Any] = field(default_factory=dict)
    gap_analysis: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class LearningHub:
    """Processes feedback, identifies gaps, and suggests procedures."""

    def __init__(
        self,
        crm: CRMDatabase,
        procedural_memory: ProceduralMemory,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self._crm = crm
        self._procedural = procedural_memory
        self._llm = get_llm_client()

        self._recent_feedback: list[FeedbackRecord] = []
        self._init_db()
        self._load_feedback()

    # ── public API ────────────────────────────────────────────────────────

    async def process_feedback(
        self,
        interaction_id: str,
        feedback_score: int,
        correction: str | None = None,
    ) -> FeedbackRecord:
        """Log feedback against an interaction and analyse any correction.

        Parameters
        ----------
        interaction_id:
            UUID of the CRM interaction to annotate.
        feedback_score:
            1-10 rating (1 = terrible, 10 = perfect).
        correction:
            Optional free-text showing what the response *should* have been.

        Returns
        -------
        FeedbackRecord with analysis results attached.
        """
        interaction = await self._crm.get_interaction(interaction_id)
        if interaction is None:
            logger.warning("Interaction %s not found in CRM", interaction_id)

        record = FeedbackRecord(
            interaction_id=interaction_id,
            feedback_score=feedback_score,
            correction=correction,
        )

        original_content = interaction.content if interaction is not None else None

        if correction and original_content:
            record.correction_analysis = await self._analyse_correction(
                original_content,
                correction,
            )

        if feedback_score <= _FEEDBACK_THRESHOLD_POOR:
            query = interaction.subject if interaction else "(unknown query)"
            response = original_content or "(no response recorded)"
            record.gap_analysis = await self._analyse_gap(
                query,
                response,
                feedback_score,
                correction,
            )
            await self._procedural.record_failure(
                query if isinstance(query, str) else str(query),
            )

        self._recent_feedback.append(record)
        await self._save_feedback(record)
        logger.info(
            "Feedback recorded: interaction=%s score=%d has_correction=%s",
            interaction_id,
            feedback_score,
            correction is not None,
        )
        return record

    async def process_feedback_for_pipeline_run(
        self,
        run_id: str,
        feedback_score: int,
        correction: str | None = None,
    ) -> FeedbackRecord:
        """Annotate feedback using Graphe session rows (``run_id`` from execution summary)."""
        from brain_os.agents.graphe import fetch_cursor_turn_by_run_id

        row = await fetch_cursor_turn_by_run_id(run_id)
        synthetic_id = f"cursor:{run_id.strip()}"
        query = "(unknown query)"
        original_content: str | None = None
        if row:
            query = (row.get("query") or "").strip() or query
            rs = row.get("response_summary")
            if rs:
                original_content = str(rs).strip()

        record = FeedbackRecord(
            interaction_id=synthetic_id,
            feedback_score=feedback_score,
            correction=correction,
        )

        if correction and original_content:
            record.correction_analysis = await self._analyse_correction(
                original_content,
                correction,
            )

        if feedback_score <= _FEEDBACK_THRESHOLD_POOR:
            response = original_content or "(no response recorded)"
            record.gap_analysis = await self._analyse_gap(
                query,
                response,
                feedback_score,
                correction,
            )
            await self._procedural.record_failure(query)

        self._recent_feedback.append(record)
        await self._save_feedback(record)
        logger.info(
            "Feedback recorded (pipeline run): run_id=%s score=%d has_correction=%s",
            run_id,
            feedback_score,
            correction is not None,
        )
        return record

    async def suggest_procedure_for_pipeline_run(
        self,
        run_id: str,
        *,
        method_trace: dict[str, Any] | None = None,
    ) -> Procedure | None:
        """Mint/update a procedure using the logged turn for ``run_id``."""
        from brain_os.agents.graphe import fetch_cursor_turn_by_run_id

        row = await fetch_cursor_turn_by_run_id(run_id)
        if row is None:
            logger.warning("Cannot suggest procedure — no Graphe session for run_id=%s", run_id)
            return None

        query = row.get("query") or "(unknown query)"
        response = row.get("response_summary") or ""
        synthetic = f"cursor:{run_id.strip()}"
        corrected = self._find_correction_for(synthetic)
        if corrected and corrected.correction:
            response = corrected.correction

        steps, _skill_hint = await self._extract_steps_for_success(
            str(query),
            str(response),
            method_trace=method_trace,
        )
        if not steps:
            return None

        procedure = await self._procedural.learn_procedure(str(query), steps)
        logger.info(
            "Procedure suggested from pipeline run %s: pattern='%s' (%d steps)",
            run_id,
            procedure.trigger_pattern,
            len(procedure.steps),
        )
        return procedure

    async def reinforce_from_praise(
        self,
        run_id: str | None,
        *,
        query: str,
        response: str,
        agents_used: list[str],
        method_trace: dict[str, Any],
    ) -> tuple[Procedure | None, str | None]:
        """Reinforce procedural memory from operator praise and optional run evidence."""
        rid = (run_id or "").strip()
        skill_hint: str | None = None
        procedure: Procedure | None = None

        if rid:
            from brain_os.agents.graphe import fetch_cursor_turn_by_run_id

            row = await fetch_cursor_turn_by_run_id(rid)
            if row is not None:
                query = str(row.get("query") or query)
                response = str(row.get("response_summary") or response)

        steps, skill_hint = await self._extract_steps_for_success(
            query,
            response,
            method_trace=method_trace,
        )
        if not steps and agents_used:
            steps = [f"Route via agent: {a}" for a in agents_used]
        if not steps:
            return None, skill_hint

        procedure = await self._procedural.learn_procedure(query, steps)
        return procedure, skill_hint

    async def identify_skill_gap(
        self,
        interaction_id: str,
        feedback_score: int | None = None,
    ) -> dict[str, Any]:
        """Analyse a poorly-rated interaction and suggest a new skill or procedure.

        Returns a dict with ``gap_type``, ``description``, and either a
        ``suggested_skill_name`` or ``suggested_knowledge_source``.
        """
        interaction = await self._crm.get_interaction(interaction_id)
        if interaction is None:
            return {"error": f"Interaction {interaction_id} not found"}

        query = interaction.subject or "(unknown query)"
        response = interaction.content or "(no response recorded)"

        score = (
            feedback_score
            if feedback_score is not None
            else getattr(interaction, "feedback_score", 1)
        )
        gap = await self._analyse_gap(query, response, feedback_score=score)

        skill_name = gap.get("suggested_skill_name")
        if skill_name and skill_name in SKILL_MATRIX:
            gap["skill_already_exists"] = True
            gap["existing_description"] = SKILL_MATRIX[skill_name]

        return gap

    async def suggest_procedure(
        self,
        interaction_id: str,
    ) -> Procedure | None:
        """Create a new ProceduralMemory entry from a successful interaction.

        Works for both organically successful interactions and corrected
        ones (where the correction becomes the canonical response path).
        """
        interaction = await self._crm.get_interaction(interaction_id)
        if interaction is None:
            logger.warning(
                "Cannot suggest procedure — interaction %s not found",
                interaction_id,
            )
            return None

        query = interaction.subject or "(unknown query)"
        response = interaction.content or ""

        corrected = self._find_correction_for(interaction_id)
        if corrected and corrected.correction:
            response = corrected.correction

        steps = await self._extract_procedure_steps(str(query), response)
        if not steps:
            return None

        procedure = await self._procedural.learn_procedure(
            str(query),
            steps,
        )
        logger.info(
            "Procedure suggested from interaction %s: pattern='%s' (%d steps)",
            interaction_id,
            procedure.trigger_pattern,
            len(procedure.steps),
        )
        return procedure

    def get_weak_areas(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return the most problematic areas based on recent feedback.

        Each entry contains the interaction_id, score, gap_analysis, and
        correction_analysis from a poorly-rated interaction.  Results are
        sorted worst-first.
        """
        poor = [r for r in self._recent_feedback if r.feedback_score <= _FEEDBACK_THRESHOLD_POOR]
        poor.sort(key=lambda r: r.feedback_score)
        return [
            {
                "interaction_id": r.interaction_id,
                "score": r.feedback_score,
                "gap_analysis": r.gap_analysis,
                "correction_analysis": r.correction_analysis,
            }
            for r in poor[:limit]
        ]

    @observe()
    async def trigger_micro_learning_cycle(self) -> dict[str, Any]:
        """Run a targeted SleepTrainer cycle on recent high-priority corrections.

        Unlike the full dream-mode sleep training which processes all pending
        corrections in batch, this processes only the most recent HIGH/CRITICAL
        corrections for just-in-time learning during an active chat session.
        """
        try:
            from brain_os.brain.correction_store import CorrectionStore
            from brain_os.brain.embeddings import EmbeddingService
            from brain_os.brain.qdrant_manager import QdrantManager
            from brain_os.brain.sleep_trainer import SleepTrainer

            correction_store = CorrectionStore()
            await correction_store.initialize()

            pending = await correction_store.get_pending_corrections(limit=5)
            high_priority = [c for c in pending if c.get("severity") in ("HIGH", "CRITICAL")]

            if not high_priority:
                await correction_store.close()
                return {"status": "no_high_priority_corrections", "processed": 0}

            embedding = EmbeddingService()
            qdrant = QdrantManager(embedding_service=embedding)
            try:
                from brain_os.config import get_settings
                from brain_os.memory.long_term import LongTermMemory

                long_term = None
                try:
                    mem0_key = get_settings().memory.api_key.get_secret_value()
                    if mem0_key:
                        long_term = LongTermMemory()
                except (
                    ImportError,
                    OSError,
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                ):
                    pass

                trainer = SleepTrainer(
                    correction_store=correction_store,
                    qdrant_manager=qdrant,
                    embedding_service=embedding,
                    long_term=long_term,
                )

                stats = await trainer.run_training()
            finally:
                await correction_store.close()
                await qdrant.close()
                await embedding.close()

            logger.info(
                "Micro-learning cycle complete: processed %d corrections",
                stats.get("corrections_count", 0),
            )
            return stats

        except _MICRO_LEARNING_ERRORS:
            logger.exception("Micro-learning cycle failed")
            return {"status": "error", "processed": 0}

    async def notify_new_feedback(
        self,
        polarity: str,
        query: str,
        correction: str | None = None,
    ) -> None:
        """Called by FeedbackHandler when new feedback arrives.

        Triggers micro-learning for negative feedback with corrections.
        """
        if polarity == "negative" and correction:
            await self.trigger_micro_learning_cycle()

    def get_all_feedback(self) -> list[FeedbackRecord]:
        """Return all recorded feedback (most recent last)."""
        return list(self._recent_feedback)

    def get_average_score(self) -> float | None:
        """Return the mean feedback score, or ``None`` if no feedback exists."""
        if not self._recent_feedback:
            return None
        total = sum(r.feedback_score for r in self._recent_feedback)
        return total / len(self._recent_feedback)

    # ── persistence ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        _FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._init_db_sync()

    def _init_db_sync(self) -> None:
        conn = sqlite3.connect(str(_FEEDBACK_DB_PATH), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT NOT NULL,
                feedback_score INTEGER NOT NULL,
                correction TEXT,
                correction_analysis TEXT DEFAULT '{}',
                gap_analysis TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _load_feedback(self) -> None:
        try:
            self._load_feedback_sync()
        except _FEEDBACK_DISK_ERRORS:
            logger.warning("Failed to load feedback from disk", exc_info=True)

    def _load_feedback_sync(self) -> None:
        conn = sqlite3.connect(str(_FEEDBACK_DB_PATH), timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000")
        rows = conn.execute(
            "SELECT interaction_id, feedback_score, correction, "
            "correction_analysis, gap_analysis, created_at "
            "FROM feedback ORDER BY id"
        ).fetchall()
        conn.close()
        for row in rows:
            self._recent_feedback.append(
                FeedbackRecord(
                    interaction_id=row[0],
                    feedback_score=row[1],
                    correction=row[2],
                    correction_analysis=json.loads(row[3]) if row[3] else {},
                    gap_analysis=json.loads(row[4]) if row[4] else {},
                    created_at=datetime.fromisoformat(row[5]),
                )
            )
        if self._recent_feedback:
            logger.info("Loaded %d feedback records from disk", len(self._recent_feedback))

    def _save_feedback_sync(self, record: FeedbackRecord) -> None:
        conn = sqlite3.connect(str(_FEEDBACK_DB_PATH), timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute(
            "INSERT INTO feedback "
            "(interaction_id, feedback_score, correction, correction_analysis, gap_analysis, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.interaction_id,
                record.feedback_score,
                record.correction,
                json.dumps(record.correction_analysis),
                json.dumps(record.gap_analysis),
                record.created_at.isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    async def _save_feedback(self, record: FeedbackRecord) -> None:
        try:
            await asyncio.to_thread(self._save_feedback_sync, record)
        except _FEEDBACK_DISK_ERRORS:
            logger.warning("Failed to persist feedback record", exc_info=True)

    # ── internal helpers ──────────────────────────────────────────────────

    def _find_correction_for(self, interaction_id: str) -> FeedbackRecord | None:
        for record in reversed(self._recent_feedback):
            if record.interaction_id == interaction_id and record.correction:
                return record
        return None

    async def _analyse_correction(
        self,
        original: str,
        correction: str,
    ) -> dict[str, Any]:
        user = f"ORIGINAL RESPONSE:\n{original[:4000]}\n\nUSER CORRECTION:\n{correction[:4000]}"
        result = await self._llm.generate_structured(
            _CORRECTION_ANALYSIS_PROMPT,
            user,
            CorrectionAnalysis,
            name="learning_hub.correction",
        )
        return result.model_dump()

    async def _analyse_gap(
        self,
        query: str,
        response: str,
        feedback_score: int,
        correction: str | None = None,
    ) -> dict[str, Any]:
        parts = [
            f"QUERY: {query[:2000]}",
            f"RESPONSE: {response[:4000]}",
            f"FEEDBACK SCORE: {feedback_score}/10",
        ]
        if correction:
            parts.append(f"USER CORRECTION: {correction[:2000]}")
        user = "\n\n".join(parts)
        result = await self._llm.generate_structured(
            _GAP_ANALYSIS_PROMPT,
            user,
            GapAnalysis,
            name="learning_hub.gap",
        )
        return result.model_dump()

    async def _extract_procedure_steps(
        self,
        query: str,
        response: str,
    ) -> list[str]:
        user = f"QUERY: {query[:2000]}\n\nRESPONSE: {response[:6000]}"
        result = await self._llm.generate_structured(
            _PROCEDURE_EXTRACTION_PROMPT,
            user,
            ProcedureSteps,
            name="learning_hub.procedure",
        )
        return result.steps

    async def _extract_steps_for_success(
        self,
        query: str,
        response: str,
        *,
        method_trace: dict[str, Any] | None = None,
    ) -> tuple[list[str], str | None]:
        """Prefer playbook extraction when method trace includes tools."""
        trace = method_trace or {}
        tools = trace.get("tools") or []
        if tools:
            trace_json = json.dumps(trace, default=str)[:8000]
            user = (
                f"QUERY: {query[:2000]}\n\nRESPONSE: {response[:4000]}\n\n"
                f"METHOD_TRACE:\n{trace_json}"
            )
            try:
                result = await self._llm.generate_structured(
                    _SUCCESS_PLAYBOOK_PROMPT,
                    user,
                    SuccessPlaybookExtraction,
                    name="learning_hub.success_playbook",
                )
                steps = [s for s in result.steps if s and str(s).strip()]
                hint = (result.skill_hint or "").strip() or None
                if steps:
                    return steps, hint
            except _LLM_STRUCT_ERRORS:
                logger.exception("Success playbook extraction failed; falling back")

        steps = await self._extract_procedure_steps(query, response)
        return steps, None
