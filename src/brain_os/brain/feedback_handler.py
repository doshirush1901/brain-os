"""Real-time feedback processing for all Brain OS interfaces.

Detects positive, negative, and ambiguous feedback in user messages,
tracks per-agent success/failure scores, stores corrections in the
:class:`~brain_os.brain.correction_store.CorrectionStore`, and notifies the
:class:`~brain_os.systems.learning_hub.LearningHub` to trigger micro-learning.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
from langfuse.decorators import observe

from brain_os.brain.untrusted_input import sanitize_for_llm_embedding
from brain_os.exceptions import DatabaseError, BrainOSError, LLMError
from brain_os.schemas.llm_outputs import FeedbackClassification
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_SCORES_PATH = Path("data/brain/agent_scores.json")

_POSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bthanks?\b", re.IGNORECASE),
    re.compile(r"\bperfect\b", re.IGNORECASE),
    re.compile(r"\bexactly\b", re.IGNORECASE),
    re.compile(r"\bgreat\b", re.IGNORECASE),
    re.compile(r"\bcorrect\b", re.IGNORECASE),
    re.compile(r"\bgood\s+job\b", re.IGNORECASE),
    re.compile(r"\bwell\s+done\b", re.IGNORECASE),
    re.compile(r"\bbravo\b", re.IGNORECASE),
    re.compile(r"\U0001F44D"),  # thumbs up
]

_NEGATIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bthat'?s\s+not\s+right\b", re.IGNORECASE),
    re.compile(r"\bactually\s+it'?s\b", re.IGNORECASE),
    re.compile(r"\bthat'?s\s+incorrect\b", re.IGNORECASE),
    re.compile(r"\bnot\s+correct\b", re.IGNORECASE),
    re.compile(r"\bincorrect\b", re.IGNORECASE),
    re.compile(r"\bwrong\b", re.IGNORECASE),
    re.compile(r"^no[.,!?\s]", re.IGNORECASE),
    re.compile(r"^no$", re.IGNORECASE),
]

_DISAMBIGUATION_SYSTEM = (
    "You are a feedback-classification assistant. Given a user message, the "
    "previous query, and the previous response, determine whether the user is "
    "giving positive feedback, negative feedback (a correction), or something "
    "neutral (unrelated to feedback). Respond with ONLY a JSON object: "
    '{"polarity": "positive"|"negative"|"neutral", "confidence": 0.0-1.0, '
    '"extracted_correction": "<correction text or null>"}.'
)


class FeedbackHandler:
    """Detects and processes user feedback on agent responses."""

    def __init__(
        self,
        learning_hub: Any | None = None,
        correction_store: Any | None = None,
        long_term: Any | None = None,
        mem0_client: Any | None = None,
        procedural_memory: Any | None = None,
        data_event_bus: Any | None = None,
        power_level_tracker: Any | None = None,
        memory_block_store: Any | None = None,
    ) -> None:
        self._learning_hub = learning_hub
        self._correction_store = correction_store
        if long_term is not None:
            self._long_term = long_term
        elif mem0_client is not None:
            logger.warning(
                "FeedbackHandler mem0_client is deprecated; pass long_term=LongTermMemory()"
            )
            self._long_term = None
        else:
            self._long_term = None
        self._procedural_memory = procedural_memory
        self._event_bus = data_event_bus
        self._power_level_tracker = power_level_tracker
        self._memory_block_store = memory_block_store
        self._llm = get_llm_client()
        self._agent_scores: dict[str, dict[str, int]] = {}

    # ── public API ───────────────────────────────────────────────────────

    @observe()
    async def detect_feedback(
        self,
        message: str,
        previous_query: str,
        previous_response: str,
    ) -> dict[str, Any]:
        """Classify *message* as positive, negative, neutral, or ambiguous.

        Returns a dict with keys ``polarity``, ``confidence``, and
        ``extracted_correction``.
        """
        pos_hits = sum(1 for p in _POSITIVE_PATTERNS if p.search(message))
        neg_hits = sum(1 for p in _NEGATIVE_PATTERNS if p.search(message))

        if neg_hits > 0 and pos_hits == 0:
            correction = self._extract_correction(message)
            return {
                "polarity": "negative",
                "confidence": min(0.6 + neg_hits * 0.1, 1.0),
                "extracted_correction": correction,
            }

        if pos_hits > 0 and neg_hits == 0:
            return {
                "polarity": "positive",
                "confidence": min(0.6 + pos_hits * 0.1, 1.0),
                "extracted_correction": None,
            }

        if pos_hits > 0 and neg_hits > 0:
            return await self._disambiguate_with_llm(message, previous_query, previous_response)

        if self._looks_like_feedback(message):
            return await self._disambiguate_with_llm(message, previous_query, previous_response)

        return {
            "polarity": "neutral",
            "confidence": 0.8,
            "extracted_correction": None,
        }

    async def load_scores(self) -> None:
        """Public wrapper for loading persisted agent scores."""
        await self._load_scores()

    async def process_feedback(
        self,
        message: str,
        previous_query: str,
        previous_response: str,
        agents_used: list[str],
        *,
        user_id: str = "global",
        severity: str = "",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Full feedback pipeline: detect, score agents, store corrections, trigger learning."""
        if severity and severity.upper() in ("HIGH", "CRITICAL"):
            result = {
                "polarity": "negative",
                "confidence": 1.0,
                "extracted_correction": message,
            }
        else:
            result = await self.detect_feedback(message, previous_query, previous_response)
        polarity = result["polarity"]

        for agent in agents_used:
            self._update_agent_score(agent, polarity)

        if (
            polarity == "negative"
            and len(agents_used) >= 2
            and self._power_level_tracker is not None
        ):
            try:
                for i in range(len(agents_used) - 1):
                    await self._power_level_tracker.record_trust_decrease(
                        agents_used[i], agents_used[i + 1]
                    )
            except (TimeoutError, DatabaseError, BrainOSError, httpx.HTTPError):
                logger.debug("Trust decrease recording failed", exc_info=True)

        if polarity == "negative" and result.get("extracted_correction"):
            correction_text = result["extracted_correction"]

            if self._correction_store is not None:
                try:
                    from brain_os.brain.correction_store import CorrectionSeverity

                    correction_id = await self._correction_store.add_correction(
                        entity=previous_query[:200],
                        new_value=correction_text,
                        severity=CorrectionSeverity.HIGH,
                        old_value=previous_response[:500],
                        source=f"feedback:{user_id}",
                    )
                    result["correction_id"] = correction_id

                    if self._event_bus is not None:
                        try:
                            from brain_os.systems.data_event_bus import (
                                DataEvent,
                                EventType,
                                SourceStore,
                            )

                            await self._event_bus.emit(
                                DataEvent(
                                    event_type=EventType.KNOWLEDGE_CORRECTED,
                                    entity_type="correction",
                                    entity_id=str(correction_id),
                                    payload={
                                        "entity": previous_query[:200],
                                        "new_value": correction_text,
                                        "source": f"feedback:{user_id}",
                                    },
                                    source_store=SourceStore.QDRANT,
                                )
                            )
                        except (
                            TimeoutError,
                            BrainOSError,
                            OSError,
                            TypeError,
                            ValueError,
                            AttributeError,
                            KeyError,
                        ):
                            logger.warning("Failed to emit correction event", exc_info=True)
                except (
                    TimeoutError,
                    DatabaseError,
                    OSError,
                    httpx.HTTPError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    KeyError,
                ):
                    logger.exception("Correction store failed")

            if self._long_term is not None:
                try:
                    await self._long_term.store_correction(
                        original=previous_response[:500],
                        corrected=correction_text,
                        context=f"feedback:{user_id}; query={previous_query[:200]}",
                    )
                except (
                    TimeoutError,
                    DatabaseError,
                    OSError,
                    httpx.HTTPError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    KeyError,
                ):
                    logger.exception("Mem0 correction storage failed")

            if self._procedural_memory is not None:
                try:
                    await self._procedural_memory.record_failure(previous_query)
                except (
                    TimeoutError,
                    DatabaseError,
                    OSError,
                    httpx.HTTPError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    KeyError,
                ):
                    logger.exception("ProceduralMemory failure recording failed")

            if self._learning_hub is not None:
                try:
                    await self._learning_hub.trigger_micro_learning_cycle()
                    result["micro_learning_triggered"] = True
                except (
                    TimeoutError,
                    DatabaseError,
                    OSError,
                    httpx.HTTPError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    KeyError,
                ):
                    logger.exception("Micro-learning trigger failed")

            try:
                from brain_os.agents.mnemon import apply_ledger_correction

                entity_key = previous_query[:100].strip().lower()
                words = entity_key.split()
                if len(words) > 5:
                    entity_key = " ".join(words[:5])
                stale: list[str] = []
                if previous_response and len(previous_response) > 10:
                    stale.append(previous_response[:200])
                ledger_result = await apply_ledger_correction(
                    entity=entity_key,
                    current_status=correction_text[:500],
                    stale_values=stale or None,
                    source="feedback",
                )
                result["mnemon_ledger_updated"] = True
                result["override_receipt"] = ledger_result.get("override_receipt")
                result["override_entity"] = ledger_result.get("entity")
                result["override_version"] = ledger_result.get("version")
                logger.info(
                    "Mnemon ledger updated for entity '%s'",
                    ledger_result.get("entity"),
                )
            except (
                OSError,
                json.JSONDecodeError,
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
            ):
                logger.warning("Failed to update Mnemon ledger from feedback", exc_info=True)

            if self._memory_block_store is not None:
                try:
                    from brain_os.memory.block_sync import apply_corrections_to_blocks

                    applied = await apply_corrections_to_blocks(
                        self._memory_block_store,
                        [
                            {
                                "id": result.get("correction_id", 0),
                                "entity": user_id if "@" in user_id else previous_query[:200],
                                "source": f"feedback:{user_id}",
                                "new_value": correction_text,
                                "category": "CUSTOMER",
                            }
                        ],
                    )
                    result["memory_blocks_updated"] = applied
                except (
                    TimeoutError,
                    DatabaseError,
                    BrainOSError,
                    OSError,
                    httpx.HTTPError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    KeyError,
                ):
                    logger.exception("Memory block update from feedback failed")

        elif polarity == "positive":
            await self._process_positive_praise(
                result,
                message=message,
                previous_query=previous_query,
                previous_response=previous_response,
                agents_used=agents_used,
                run_id=run_id,
            )

        await self._persist_scores()
        return result

    async def _process_positive_praise(
        self,
        result: dict[str, Any],
        *,
        message: str,
        previous_query: str,
        previous_response: str,
        agents_used: list[str],
        run_id: str | None,
    ) -> None:
        from brain_os.config import get_settings

        if not get_settings().app.praise_learning_enabled:
            if self._procedural_memory is not None and agents_used:
                try:
                    await self._procedural_memory.learn_procedure(
                        previous_query,
                        agents_used,
                    )
                    result["procedure_reinforced"] = True
                except (
                    TimeoutError,
                    DatabaseError,
                    OSError,
                    httpx.HTTPError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    KeyError,
                ):
                    logger.exception("ProceduralMemory reinforcement failed")
            return

        from brain_os.brain.cursor_learning_meta import patch_learning_meta_by_run_id
        from brain_os.brain.method_trace import build_method_trace
        from brain_os.brain.praise_store import PraiseStore
        from brain_os.brain.run_record_link import append_run_link

        rid = (run_id or "").strip()
        method_trace = await build_method_trace(
            rid or None,
            query=previous_query,
            response=previous_response,
            agents_used=agents_used,
        )
        procedure_id: int | None = None
        skill_hint: str | None = None

        if rid:
            patched = await patch_learning_meta_by_run_id(
                rid,
                {
                    "operator_praise": True,
                    "praise_text": (message or "")[:500],
                },
            )
            result["graphe_meta_patched"] = patched

        procedure = None
        if self._learning_hub is not None:
            try:
                procedure, skill_hint = await self._learning_hub.reinforce_from_praise(
                    rid or None,
                    query=previous_query,
                    response=previous_response,
                    agents_used=agents_used,
                    method_trace=method_trace,
                )
                if procedure is not None:
                    procedure_id = procedure.id
                    result["procedure_reinforced"] = True
                    result["procedure_trigger"] = procedure.trigger_pattern
            except (
                TimeoutError,
                DatabaseError,
                OSError,
                httpx.HTTPError,
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
            ):
                logger.exception("LearningHub praise reinforcement failed")

        if procedure is None and self._procedural_memory is not None and agents_used:
            try:
                proc = await self._procedural_memory.learn_procedure(
                    previous_query,
                    agents_used,
                )
                procedure_id = proc.id
                result["procedure_reinforced"] = True
            except (
                TimeoutError,
                DatabaseError,
                OSError,
                httpx.HTTPError,
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
            ):
                logger.exception("ProceduralMemory reinforcement failed")

        try:
            await PraiseStore().record_praise(
                run_id=rid or None,
                praise_text=message,
                polarity="positive",
                method_trace=method_trace,
                procedure_id=procedure_id,
                skill_hint=skill_hint,
            )
            result["praise_recorded"] = True
        except (
            OSError,
            json.JSONDecodeError,
            DatabaseError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ):
            logger.exception("PraiseStore.record_praise failed")

        if rid:
            try:
                await append_run_link(
                    rid,
                    kind="praise",
                    detail={
                        "praise_text": (message or "")[:200],
                        "procedure_id": procedure_id,
                        "skill_hint": skill_hint,
                    },
                )
            except (
                TimeoutError,
                OSError,
                DatabaseError,
                BrainOSError,
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
            ):
                logger.debug("append_run_link praise failed", exc_info=True)

        try:
            from brain_os.services.operator_context import mark_operator_context_from_feedback

            n = await mark_operator_context_from_feedback(
                run_id=rid or None,
                query=previous_query,
            )
            if n:
                result["operator_context_marked"] = n
        except (
            TimeoutError,
            OSError,
            DatabaseError,
            BrainOSError,
            httpx.HTTPError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ):
            logger.debug("operator context praise mark failed", exc_info=True)

    def get_agent_scores(self) -> dict[str, dict[str, int]]:
        return dict(self._agent_scores)

    # ── pattern helpers ──────────────────────────────────────────────────

    @staticmethod
    def _extract_correction(message: str) -> str | None:
        for pattern in (
            r"actually\s+it'?s\s+(.+)",
            r"no[,.]?\s+(?:it'?s|the answer is)\s+(.+)",
            r"that'?s\s+not\s+right[,.]?\s+(.+)",
        ):
            m = re.search(pattern, message, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".")
        return message

    @staticmethod
    def _looks_like_feedback(message: str) -> bool:
        """Heuristic: short messages after a response are likely feedback."""
        return len(message.split()) <= 12

    # ── LLM disambiguation ───────────────────────────────────────────────

    async def _disambiguate_with_llm(
        self,
        message: str,
        previous_query: str,
        previous_response: str,
    ) -> dict[str, Any]:
        user_content = (
            f"Previous query: {sanitize_for_llm_embedding(previous_query, max_chars=8000)}\n"
            f"Previous response: {sanitize_for_llm_embedding(previous_response, max_chars=2000)}\n"
            f"User message: {sanitize_for_llm_embedding(message, max_chars=2000)}"
        )
        try:
            result = await self._llm.generate_structured(
                _DISAMBIGUATION_SYSTEM,
                user_content,
                FeedbackClassification,
                temperature=0.1,
                name="feedback.disambiguate",
            )
            return result.model_dump()
        except (
            TimeoutError,
            LLMError,
            httpx.HTTPError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ):
            logger.exception("LLM disambiguation failed")
            return {
                "polarity": "ambiguous",
                "confidence": 0.3,
                "extracted_correction": None,
            }

    # ── agent scoring ────────────────────────────────────────────────────

    def _update_agent_score(self, agent: str, polarity: str) -> None:
        if agent not in self._agent_scores:
            self._agent_scores[agent] = {"success": 0, "failure": 0}
        if polarity == "positive":
            self._agent_scores[agent]["success"] += 1
        elif polarity == "negative":
            self._agent_scores[agent]["failure"] += 1

    # ── persistence ──────────────────────────────────────────────────────

    async def _load_scores(self) -> None:
        if _SCORES_PATH.exists():
            try:
                raw = await asyncio.to_thread(_SCORES_PATH.read_text)
                self._agent_scores = json.loads(raw)
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not load agent scores; starting fresh")
                self._agent_scores = {}

    async def _persist_scores(self) -> None:
        try:
            _SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self._agent_scores, indent=2)
            await asyncio.to_thread(_SCORES_PATH.write_text, payload)
        except OSError:
            logger.exception("Failed to persist agent scores")
