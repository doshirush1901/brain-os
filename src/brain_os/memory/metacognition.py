"""Meta-cognition — self-awareness about knowledge quality and gaps."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import aiosqlite
from langfuse.decorators import observe

from brain_os.data.models import KnowledgeState
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import KnowledgeAssessment
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

# Allowlisted columns for idempotent knowledge_gaps migrations (no dynamic SQL identifiers).
_KNOWLEDGE_GAPS_COLUMN_DEFS: dict[str, str] = {
    "resolved_at": "TEXT DEFAULT NULL",
    "resolution": "TEXT DEFAULT NULL",
}

_ASSESS_SYSTEM_PROMPT = load_prompt("assess_knowledge")


class Metacognition:
    def __init__(
        self,
        db_path: str = "data/conversations.db",
    ) -> None:
        self._db_path = db_path
        self._llm = get_llm_client()
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path, timeout=30.0)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                state TEXT NOT NULL,
                gaps TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resolved_at TEXT DEFAULT NULL,
                resolution TEXT DEFAULT NULL
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_date ON knowledge_gaps(created_at)"
        )
        # Migrate existing tables that lack the resolution columns
        for col, col_def in _KNOWLEDGE_GAPS_COLUMN_DEFS.items():
            try:
                await self._db.execute(f"ALTER TABLE knowledge_gaps ADD COLUMN {col} {col_def}")
            except Exception as exc:
                logger.debug(
                    "knowledge_gaps column %s already exists or migration skipped: %s",
                    col,
                    exc,
                )
        await self._db.commit()

    @observe()
    async def assess_knowledge(self, query: str, retrieved_context: list[dict]) -> dict:
        has_results = len(retrieved_context) > 0
        if not has_results:
            return {
                "state": KnowledgeState.UNKNOWN,
                "confidence": 0.0,
                "sources": [],
                "conflicts": [],
                "gaps": [],
            }

        avg_score = sum(r.get("score", 0) for r in retrieved_context) / len(retrieved_context)
        has_high_confidence = any(r.get("score", 0) >= 0.7 for r in retrieved_context)
        sources = [
            {
                "source": r.get("source", ""),
                "source_type": r.get("source_type", ""),
                "score": r.get("score", 0),
            }
            for r in retrieved_context
        ]

        context_lines = []
        for i, r in enumerate(retrieved_context, 1):
            content = r.get("content", r.get("text", r.get("snippet", "")))
            context_lines.append(
                f"Chunk {i}: source={r.get('source', '')}, score={r.get('score', 0)}\n{content}"
            )
        user_msg = f"Query: {query}\n\nContext:\n" + "\n\n".join(context_lines)

        try:
            result = await self._llm.generate_structured(
                _ASSESS_SYSTEM_PROMPT,
                user_msg,
                KnowledgeAssessment,
                name="metacognition.assess",
                model_tier="cheap",
            )
            state_str = result.state
            confidence = result.confidence
            conflicts = result.conflicts
            gaps = result.gaps
        except Exception:
            logger.exception("Structured LLM call failed in Metacognition")
            if avg_score >= 0.7 and has_high_confidence:
                state_str = "KNOW_UNVERIFIED"
                confidence = avg_score
            elif avg_score >= 0.4:
                state_str = "PARTIAL"
                confidence = avg_score
            else:
                state_str = "UNCERTAIN"
                confidence = avg_score
            conflicts = []
            gaps = []

        try:
            state = KnowledgeState(state_str)
        except ValueError:
            state = KnowledgeState.UNCERTAIN

        confidence = max(0.0, min(1.0, confidence))

        return {
            "state": state,
            "confidence": confidence,
            "sources": sources,
            "conflicts": conflicts,
            "gaps": gaps,
        }

    def generate_confidence_prefix(self, state: KnowledgeState, confidence: float) -> str:
        if state == KnowledgeState.KNOW_VERIFIED:
            if confidence >= 0.8:
                return "Based on our verified documentation, "
            return "According to our records, "
        if state == KnowledgeState.KNOW_UNVERIFIED:
            return "Based on available information (not yet independently verified), "
        if state == KnowledgeState.PARTIAL:
            return "I have partial information on this. "
        if state == KnowledgeState.UNCERTAIN:
            return "I'm not entirely certain, but based on what I've found, "
        if state == KnowledgeState.CONFLICTING:
            return "I found conflicting information on this topic. "
        if state == KnowledgeState.UNKNOWN:
            return "I don't have reliable information on this. I'd recommend checking with the relevant team. "
        return ""

    async def get_unresolved_gaps(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return knowledge gaps that have not yet been resolved."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT id, query, state, gaps, created_at
            FROM knowledge_gaps
            WHERE resolved_at IS NULL
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {
                "id": r[0],
                "query": r[1],
                "state": r[2],
                "gaps": json.loads(r[3]) if isinstance(r[3], str) else r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    async def mark_gap_resolved(self, gap_id: int, resolution: str) -> None:
        """Mark a knowledge gap as resolved with the given resolution text."""
        assert self._db is not None
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            "UPDATE knowledge_gaps SET resolved_at = ?, resolution = ? WHERE id = ?",
            (now, resolution, gap_id),
        )
        await self._db.commit()
        logger.info("Knowledge gap #%d marked resolved", gap_id)

    async def log_knowledge_gap(
        self,
        query: str,
        state: KnowledgeState,
        gaps: list[str] | None = None,
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO knowledge_gaps (query, state, gaps, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                query,
                state.value,
                json.dumps(gaps or []),
                datetime.now(UTC).isoformat(),
            ),
        )
        await self._db.commit()

    async def get_gap_metrics(self) -> dict[str, Any]:
        """Return high-level gap counters for health/observability surfaces."""
        assert self._db is not None
        metrics = {
            "total": 0,
            "unresolved": 0,
            "resolved": 0,
            "latest_created_at": None,
        }
        cursor = await self._db.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN resolved_at IS NULL THEN 1 ELSE 0 END) AS unresolved,
              SUM(CASE WHEN resolved_at IS NOT NULL THEN 1 ELSE 0 END) AS resolved,
              MAX(created_at) AS latest_created_at
            FROM knowledge_gaps
            """
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row:
            metrics["total"] = int(row[0] or 0)
            metrics["unresolved"] = int(row[1] or 0)
            metrics["resolved"] = int(row[2] or 0)
            metrics["latest_created_at"] = row[3]
        return metrics

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> Metacognition:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
