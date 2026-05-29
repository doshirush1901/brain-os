"""Emotional intelligence — detect and respond to emotional states."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import aiosqlite
from langfuse.decorators import observe

from brain_os.data.models import EmotionalState
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import EmotionDetection
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_EMOTION_PATTERNS: list[tuple[re.Pattern[str], EmotionalState, float]] = [
    (re.compile(r"\burgent\b", re.IGNORECASE), EmotionalState.URGENT, 2.0),
    (re.compile(r"\basap\b", re.IGNORECASE), EmotionalState.URGENT, 2.0),
    (re.compile(r"\bimmediately\b", re.IGNORECASE), EmotionalState.URGENT, 2.0),
    (re.compile(r"\bright now\b", re.IGNORECASE), EmotionalState.URGENT, 2.0),
    (re.compile(r"\bdeadline\b", re.IGNORECASE), EmotionalState.URGENT, 2.0),
    (re.compile(r"\bcritical\b", re.IGNORECASE), EmotionalState.URGENT, 2.0),
    (re.compile(r"\bfrustrat", re.IGNORECASE), EmotionalState.FRUSTRATED, 2.0),
    (re.compile(r"\bannoy", re.IGNORECASE), EmotionalState.FRUSTRATED, 2.0),
    (re.compile(r"\bunacceptable\b", re.IGNORECASE), EmotionalState.FRUSTRATED, 2.0),
    (re.compile(r"\bridiculous\b", re.IGNORECASE), EmotionalState.FRUSTRATED, 2.0),
    (re.compile(r"\bwaste of time\b", re.IGNORECASE), EmotionalState.FRUSTRATED, 2.0),
    (re.compile(r"\bstill not\b", re.IGNORECASE), EmotionalState.FRUSTRATED, 2.0),
    (re.compile(r"\bstress", re.IGNORECASE), EmotionalState.STRESSED, 1.5),
    (re.compile(r"\boverwhelm", re.IGNORECASE), EmotionalState.STRESSED, 1.5),
    (re.compile(r"\bpressure\b", re.IGNORECASE), EmotionalState.STRESSED, 1.5),
    (re.compile(r"\bworried\b", re.IGNORECASE), EmotionalState.STRESSED, 1.5),
    (re.compile(r"\bconcerned\b", re.IGNORECASE), EmotionalState.STRESSED, 1.5),
    (re.compile(r"\bthank", re.IGNORECASE), EmotionalState.GRATEFUL, 2.0),
    (re.compile(r"\bappreciate\b", re.IGNORECASE), EmotionalState.GRATEFUL, 2.0),
    (re.compile(r"\bgrateful\b", re.IGNORECASE), EmotionalState.GRATEFUL, 2.0),
    (re.compile(r"\bhelpful\b", re.IGNORECASE), EmotionalState.GRATEFUL, 2.0),
    (re.compile(r"\bgreat job\b", re.IGNORECASE), EmotionalState.GRATEFUL, 2.0),
    (re.compile(r"\bwell done\b", re.IGNORECASE), EmotionalState.GRATEFUL, 2.0),
    (re.compile(r"\bhow does\b", re.IGNORECASE), EmotionalState.CURIOUS, 1.5),
    (re.compile(r"\bwhy does\b", re.IGNORECASE), EmotionalState.CURIOUS, 1.5),
    (re.compile(r"\bcan you explain\b", re.IGNORECASE), EmotionalState.CURIOUS, 1.5),
    (re.compile(r"\btell me more\b", re.IGNORECASE), EmotionalState.CURIOUS, 1.5),
    (re.compile(r"\bwhat if\b", re.IGNORECASE), EmotionalState.CURIOUS, 1.5),
    (re.compile(r"\binteresting\b", re.IGNORECASE), EmotionalState.CURIOUS, 1.5),
    (re.compile(r"\bexcited\b", re.IGNORECASE), EmotionalState.POSITIVE, 1.5),
    (re.compile(r"\bhappy\b", re.IGNORECASE), EmotionalState.POSITIVE, 1.5),
    (re.compile(r"\bgreat\b", re.IGNORECASE), EmotionalState.POSITIVE, 1.5),
    (re.compile(r"\bexcellent\b", re.IGNORECASE), EmotionalState.POSITIVE, 1.5),
    (re.compile(r"\bperfect\b", re.IGNORECASE), EmotionalState.POSITIVE, 1.5),
    (re.compile(r"\blove\b", re.IGNORECASE), EmotionalState.POSITIVE, 1.5),
    (re.compile(r"\bnot sure\b", re.IGNORECASE), EmotionalState.UNCERTAIN, 1.5),
    (re.compile(r"\bmaybe\b", re.IGNORECASE), EmotionalState.UNCERTAIN, 1.5),
    (re.compile(r"\bconfused\b", re.IGNORECASE), EmotionalState.UNCERTAIN, 1.5),
    (re.compile(r"\bdon't understand\b", re.IGNORECASE), EmotionalState.UNCERTAIN, 1.5),
    (re.compile(r"\bunclear\b", re.IGNORECASE), EmotionalState.UNCERTAIN, 1.5),
]

_DETECT_SYSTEM_PROMPT = load_prompt("detect_emotion")

_ADJUSTMENTS: dict[tuple[EmotionalState, str], dict[str, Any]] = {
    (EmotionalState.STRESSED, "MILD"): {
        "tone": "empathetic",
        "style_instructions": "Acknowledge the difficulty. Be reassuring. Offer concrete next steps.",
        "priority_boost": False,
    },
    (EmotionalState.STRESSED, "MODERATE"): {
        "tone": "empathetic",
        "style_instructions": "Acknowledge the difficulty. Be reassuring. Offer concrete next steps.",
        "priority_boost": False,
    },
    (EmotionalState.STRESSED, "STRONG"): {
        "tone": "empathetic",
        "style_instructions": "Acknowledge the difficulty. Be reassuring. Offer concrete next steps.",
        "priority_boost": True,
    },
    (EmotionalState.FRUSTRATED, "MILD"): {
        "tone": "empathetic",
        "style_instructions": "Focus on solving the issue.",
        "priority_boost": False,
    },
    (EmotionalState.FRUSTRATED, "MODERATE"): {
        "tone": "empathetic",
        "style_instructions": "Apologize if appropriate.",
        "priority_boost": True,
    },
    (EmotionalState.FRUSTRATED, "STRONG"): {
        "tone": "empathetic",
        "style_instructions": "Apologize if appropriate.",
        "priority_boost": True,
    },
    (EmotionalState.URGENT, "MILD"): {
        "tone": "direct",
        "style_instructions": "Be concise and action-oriented. Skip pleasantries. Lead with the answer.",
        "priority_boost": True,
    },
    (EmotionalState.URGENT, "MODERATE"): {
        "tone": "direct",
        "style_instructions": "Be concise and action-oriented. Skip pleasantries. Lead with the answer.",
        "priority_boost": True,
    },
    (EmotionalState.URGENT, "STRONG"): {
        "tone": "direct",
        "style_instructions": "Be concise and action-oriented. Skip pleasantries. Lead with the answer.",
        "priority_boost": True,
    },
    (EmotionalState.GRATEFUL, "MILD"): {
        "tone": "warm",
        "style_instructions": "Acknowledge thanks.",
        "priority_boost": False,
    },
    (EmotionalState.GRATEFUL, "MODERATE"): {
        "tone": "warm",
        "style_instructions": "Acknowledge thanks.",
        "priority_boost": False,
    },
    (EmotionalState.GRATEFUL, "STRONG"): {
        "tone": "warm",
        "style_instructions": "Acknowledge thanks.",
        "priority_boost": False,
    },
    (EmotionalState.CURIOUS, "MILD"): {
        "tone": "detailed",
        "style_instructions": "Explore further.",
        "priority_boost": False,
    },
    (EmotionalState.CURIOUS, "MODERATE"): {
        "tone": "detailed",
        "style_instructions": "Explore further.",
        "priority_boost": False,
    },
    (EmotionalState.CURIOUS, "STRONG"): {
        "tone": "detailed",
        "style_instructions": "Explore further.",
        "priority_boost": False,
    },
    (EmotionalState.POSITIVE, "MILD"): {
        "tone": "warm",
        "style_instructions": "Match energy.",
        "priority_boost": False,
    },
    (EmotionalState.POSITIVE, "MODERATE"): {
        "tone": "warm",
        "style_instructions": "Match energy.",
        "priority_boost": False,
    },
    (EmotionalState.POSITIVE, "STRONG"): {
        "tone": "warm",
        "style_instructions": "Match energy.",
        "priority_boost": False,
    },
    (EmotionalState.UNCERTAIN, "MILD"): {
        "tone": "reassuring",
        "style_instructions": "Be clear and structured.",
        "priority_boost": False,
    },
    (EmotionalState.UNCERTAIN, "MODERATE"): {
        "tone": "reassuring",
        "style_instructions": "Be clear and structured.",
        "priority_boost": False,
    },
    (EmotionalState.UNCERTAIN, "STRONG"): {
        "tone": "reassuring",
        "style_instructions": "Be clear and structured.",
        "priority_boost": False,
    },
    (EmotionalState.NEUTRAL, "MILD"): {
        "tone": "professional",
        "style_instructions": "Standard tone.",
        "priority_boost": False,
    },
    (EmotionalState.NEUTRAL, "MODERATE"): {
        "tone": "professional",
        "style_instructions": "Standard tone.",
        "priority_boost": False,
    },
    (EmotionalState.NEUTRAL, "STRONG"): {
        "tone": "professional",
        "style_instructions": "Standard tone.",
        "priority_boost": False,
    },
}


class EmotionalIntelligence:
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
            CREATE TABLE IF NOT EXISTS emotion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                state TEXT NOT NULL,
                intensity TEXT NOT NULL,
                indicators TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_emotion_log_user_created "
            "ON emotion_log(user_id, created_at)"
        )
        await self._db.commit()

    @observe()
    async def detect_emotion(self, text: str) -> dict:
        scores: dict[EmotionalState, float] = {s: 0.0 for s in EmotionalState}
        matched_indicators: dict[EmotionalState, list[str]] = {s: [] for s in EmotionalState}

        for pattern, state, weight in _EMOTION_PATTERNS:
            for m in pattern.finditer(text):
                scores[state] += weight
                phrase = m.group(0)
                if phrase not in matched_indicators[state]:
                    matched_indicators[state].append(phrase)

        excl_count = text.count("!")
        if excl_count >= 3:
            scores[EmotionalState.FRUSTRATED] += 1.0

        words = text.split()
        consecutive_caps = 0
        for w in words:
            if len(w) >= 3 and w.isupper():
                consecutive_caps += 1
            else:
                if consecutive_caps >= 3:
                    scores[EmotionalState.FRUSTRATED] += 1.0
                    scores[EmotionalState.URGENT] += 1.0
                consecutive_caps = 0
        if consecutive_caps >= 3:
            scores[EmotionalState.FRUSTRATED] += 1.0
            scores[EmotionalState.URGENT] += 1.0

        q_count = text.count("?")
        if q_count >= 3:
            scores[EmotionalState.CURIOUS] += 0.5

        top_state = max(scores, key=scores.get)
        top_score = scores[top_state]

        if top_score >= 3.0:
            if top_score >= 5.0:
                intensity = "STRONG"
            elif top_score >= 3.0:
                intensity = "MODERATE"
            else:
                intensity = "MILD"
            indicators = matched_indicators.get(top_state, [])
            return {
                "state": top_state,
                "intensity": intensity,
                "indicators": indicators,
            }

        try:
            result = await self._llm.generate_structured(
                _DETECT_SYSTEM_PROMPT,
                text,
                EmotionDetection,
                name="emotional_intelligence.detect",
            )
            state = EmotionalState(result.state)
            return {
                "state": state,
                "intensity": result.intensity,
                "indicators": result.indicators,
            }
        except (ValueError, TypeError):
            return {
                "state": EmotionalState.NEUTRAL,
                "intensity": "MILD",
                "indicators": [],
            }

    def get_response_adjustment(self, emotional_state: EmotionalState, intensity: str) -> dict:
        key = (emotional_state, intensity)
        if key in _ADJUSTMENTS:
            return _ADJUSTMENTS[key].copy()
        fallback = (emotional_state, "MILD")
        if fallback in _ADJUSTMENTS:
            return _ADJUSTMENTS[fallback].copy()
        return {
            "tone": "professional",
            "style_instructions": "Standard tone.",
            "priority_boost": False,
        }

    async def log_emotion(
        self,
        user_id: str,
        state: EmotionalState,
        intensity: str,
        indicators: list[str],
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO emotion_log (user_id, state, intensity, indicators, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                state.value,
                intensity,
                json.dumps(indicators),
                datetime.now(UTC).isoformat(),
            ),
        )
        await self._db.commit()

    async def get_emotional_profile(self, user_id: str) -> dict:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT state, intensity FROM emotion_log
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()

        if not rows:
            return {
                "dominant_state": EmotionalState.NEUTRAL,
                "state_distribution": {},
                "avg_intensity": "MILD",
                "recent_trend": "stable",
                "interaction_count": 0,
            }

        states = [r[0] for r in rows]
        intensities = [r[1] for r in rows]
        counts = Counter(states)
        dominant_state = EmotionalState(counts.most_common(1)[0][0])
        total = len(rows)
        state_distribution = {s: round(100 * c / total, 1) for s, c in counts.items()}
        intensity_counts = Counter(intensities)
        avg_intensity = intensity_counts.most_common(1)[0][0]

        positive_states = {EmotionalState.POSITIVE, EmotionalState.GRATEFUL, EmotionalState.CURIOUS}
        negative_states = {EmotionalState.STRESSED, EmotionalState.FRUSTRATED}

        recent = rows[:5]
        previous = rows[5:10]
        recent_pos = sum(1 for r in recent if r[0] in positive_states)
        recent_neg = sum(1 for r in recent if r[0] in negative_states)
        prev_pos = sum(1 for r in previous if r[0] in positive_states)
        prev_neg = sum(1 for r in previous if r[0] in negative_states)

        if recent_pos > prev_pos or recent_neg < prev_neg:
            recent_trend = "improving"
        elif recent_neg > prev_neg or recent_pos < prev_pos:
            recent_trend = "declining"
        else:
            recent_trend = "stable"

        return {
            "dominant_state": dominant_state,
            "state_distribution": state_distribution,
            "avg_intensity": avg_intensity,
            "recent_trend": recent_trend,
            "interaction_count": total,
        }

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> EmotionalIntelligence:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
