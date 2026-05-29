"""Endocrine system — hormone-like state modulation.

Maintains floating-point "hormone" levels that influence Ira's behavioral
modifiers: confidence, energy, growth_signal, stress, and caution.  These
drift toward a baseline over time and are nudged by agent successes/failures,
tool outcomes, and system events.  The :class:`VoiceSystem` reads the
modifiers to shape tone, and the :class:`RequestPipeline` logs them for
observability.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_BASELINE = 0.5
_DECAY_RATE = 0.02
_MIN = 0.0
_MAX = 1.0

_HORMONES = ("confidence", "energy", "growth_signal", "stress", "caution", "boredom")


def _clamp(value: float) -> float:
    return max(_MIN, min(_MAX, value))


class EndocrineSystem:
    """Hormone-like state that modulates Ira's behavior across the pipeline."""

    def __init__(self) -> None:
        self._levels: dict[str, float] = {h: _BASELINE for h in _HORMONES}
        self._last_decay = time.monotonic()

    def boost(self, hormone: str, amount: float = 0.05) -> None:
        if hormone in self._levels:
            self._levels[hormone] = _clamp(self._levels[hormone] + amount)

    def dampen(self, hormone: str, amount: float = 0.05) -> None:
        if hormone in self._levels:
            self._levels[hormone] = _clamp(self._levels[hormone] - amount)

    def signal_success(self, agent: str) -> None:
        self.boost("confidence", 0.03)
        self.boost("energy", 0.02)
        self.dampen("stress", 0.02)
        self.dampen("caution", 0.01)

    def signal_failure(self, agent: str) -> None:
        self.dampen("confidence", 0.04)
        self.boost("stress", 0.05)
        self.boost("caution", 0.03)
        self.dampen("energy", 0.02)

    def tick_idle(self, amount: float = 0.05) -> None:
        """Increase boredom when the system has been idle (call periodically, e.g. hourly)."""
        self.boost("boredom", amount)

    def reset_boredom(self) -> None:
        """Reset boredom to zero after user interaction or curiosity cycle."""
        if "boredom" in self._levels:
            self._levels["boredom"] = _MIN

    def get_behavioral_modifiers(self) -> dict[str, Any]:
        """Return modifiers consumed by VoiceSystem and the pipeline."""
        self._apply_decay()
        lvl = self._levels

        assertiveness = (
            "high" if lvl["confidence"] > 0.65 else "low" if lvl["confidence"] < 0.35 else "normal"
        )
        verbosity = (
            "concise" if lvl["energy"] < 0.35 else "detailed" if lvl["energy"] > 0.65 else "normal"
        )
        hedging = (
            "heavy" if lvl["caution"] > 0.65 else "light" if lvl["caution"] < 0.35 else "normal"
        )

        prompt_addendum = ""
        if lvl["stress"] > 0.8:
            prompt_addendum = (
                "The system is under significant stress (e.g. a capability is degraded). "
                "Be highly cautious and explicit about any limitations or uncertainties. "
                "Do not overstate confidence."
            )

        return {
            "assertiveness": assertiveness,
            "verbosity": verbosity,
            "hedging": hedging,
            "confidence_level": round(lvl["confidence"], 3),
            "energy_level": round(lvl["energy"], 3),
            "stress_level": round(lvl["stress"], 3),
            "prompt_addendum": prompt_addendum,
        }

    def get_status(self) -> dict[str, float]:
        self._apply_decay()
        return {k: round(v, 3) for k, v in self._levels.items()}

    def _apply_decay(self) -> None:
        """Drift all hormones toward baseline proportionally to elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_decay
        if elapsed < 5.0:
            return
        self._last_decay = now

        factor = min(elapsed * _DECAY_RATE / 60, 0.1)
        for h in _HORMONES:
            diff = _BASELINE - self._levels[h]
            self._levels[h] += diff * factor
