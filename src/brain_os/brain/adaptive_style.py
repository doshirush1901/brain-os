"""Adaptive style — learns each contact's communication preferences.

Analyzes incoming messages for signals about preferred formality, detail
level, technical depth, and pace, then provides style instructions for
the LLM to match the contact's expectations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from brain_os.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_PROFILES_PATH = Path("data/brain/style_profiles.json")

_LEARNING_RATE_INITIAL = 0.15
_LEARNING_RATE_MIN = 0.03
_CONFIDENCE_DECAY = 50


@dataclass
class StyleProfile:
    formality: float = 0.5
    detail: float = 0.5
    technical: float = 0.5
    pace: float = 0.5
    interactions: int = 0


_FORMAL_SIGNALS = re.compile(
    r"\b(dear|regards|sincerely|kindly|please\s+find|herewith|pursuant)\b", re.IGNORECASE
)
_INFORMAL_SIGNALS = re.compile(
    r"\b(hey|hi|thanks|cool|awesome|gonna|wanna|btw|lol)\b", re.IGNORECASE
)
_TECHNICAL_SIGNALS = re.compile(
    r"\b(specification|tolerance|micron|kN|PSI|RPM|PLC|servo|hydraulic|pneumatic|thermoform)\b",
    re.IGNORECASE,
)
_DETAIL_SIGNALS = re.compile(
    r"\b(detail|explain|elaborate|breakdown|step.by.step|thorough|comprehensive)\b",
    re.IGNORECASE,
)
_BRIEF_SIGNALS = re.compile(
    r"\b(brief|short|quick|summary|tl;?dr|bottom.line|just\s+tell\s+me)\b", re.IGNORECASE
)


class AdaptiveStyleTracker:
    """Tracks and adapts to each contact's communication style."""

    def __init__(self) -> None:
        self._profiles: dict[str, StyleProfile] = {}
        self._lock = asyncio.Lock()

    async def _load(self) -> None:
        if not _PROFILES_PATH.exists():
            return
        try:
            raw = await asyncio.to_thread(_PROFILES_PATH.read_text)
            data = json.loads(raw)
            for cid, vals in data.items():
                self._profiles[cid] = StyleProfile(**vals)
        except ConfigurationError:
            logger.debug("Failed to load style profiles")

    async def _save(self) -> None:
        _PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {cid: asdict(p) for cid, p in self._profiles.items()}
        payload = json.dumps(data, indent=2)
        await asyncio.to_thread(_PROFILES_PATH.write_text, payload)

    def analyze_message(self, text: str) -> dict[str, float]:
        """Extract style signals from a message. Returns deltas for each dimension."""
        deltas: dict[str, float] = {}

        formal_hits = len(_FORMAL_SIGNALS.findall(text))
        informal_hits = len(_INFORMAL_SIGNALS.findall(text))
        if formal_hits > informal_hits:
            deltas["formality"] = 0.1 * (formal_hits - informal_hits)
        elif informal_hits > formal_hits:
            deltas["formality"] = -0.1 * (informal_hits - formal_hits)

        tech_hits = len(_TECHNICAL_SIGNALS.findall(text))
        if tech_hits > 0:
            deltas["technical"] = min(0.1 * tech_hits, 0.3)

        detail_hits = len(_DETAIL_SIGNALS.findall(text))
        brief_hits = len(_BRIEF_SIGNALS.findall(text))
        if detail_hits > brief_hits:
            deltas["detail"] = 0.15
        elif brief_hits > detail_hits:
            deltas["detail"] = -0.15

        word_count = len(text.split())
        if word_count > 100:
            deltas["pace"] = 0.1
        elif word_count < 20:
            deltas["pace"] = -0.1

        return deltas

    async def update_profile(self, contact_id: str, message: str) -> StyleProfile:
        """Analyze a message and update the contact's style profile."""
        deltas = self.analyze_message(message)

        async with self._lock:
            profile = self._profiles.get(contact_id, StyleProfile())

            lr = max(
                _LEARNING_RATE_MIN,
                _LEARNING_RATE_INITIAL / (1 + profile.interactions / _CONFIDENCE_DECAY),
            )

            for dim, delta in deltas.items():
                current = getattr(profile, dim)
                updated = max(0.0, min(1.0, current + delta * lr))
                setattr(profile, dim, round(updated, 3))

            profile.interactions += 1
            self._profiles[contact_id] = profile
            await self._save()

        return profile

    def get_style_prompt(self, contact_id: str) -> str:
        """Return style instructions for the LLM based on learned preferences."""
        profile = self._profiles.get(contact_id)
        if profile is None or profile.interactions < 2:
            return ""

        instructions: list[str] = []

        if profile.formality > 0.65:
            instructions.append("Use formal, professional language.")
        elif profile.formality < 0.35:
            instructions.append("Use a casual, friendly tone.")

        if profile.detail > 0.65:
            instructions.append("Provide detailed, comprehensive answers with explanations.")
        elif profile.detail < 0.35:
            instructions.append("Keep responses brief and to the point.")

        if profile.technical > 0.65:
            instructions.append(
                "Use technical terminology — this contact is technically sophisticated."
            )
        elif profile.technical < 0.35:
            instructions.append("Avoid jargon — explain in simple, non-technical terms.")

        if not instructions:
            return ""

        return "Communication style for this contact: " + " ".join(instructions)

    def get_profile(self, contact_id: str) -> dict[str, Any] | None:
        profile = self._profiles.get(contact_id)
        if profile is None:
            return None
        return asdict(profile)
