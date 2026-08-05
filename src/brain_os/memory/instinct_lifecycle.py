"""Derived instinct (procedural-memory) lifecycle: confidence + TTL.

Pure, side-effect-free functions over a ``Procedure``-like object (anything with
``success_rate``, ``times_used``, ``last_used``). Confidence and age are computed
at read time, so this adds **no** database column to the ``procedures`` table.

This module deliberately does **not** import :mod:`brain_os.memory.procedural`, so
``procedural`` can import these helpers without an import cycle.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

#: Number of successful uses at which the evidence factor saturates to 1.0.
EVIDENCE_SATURATION = 5

#: Default idle window (days) before an instinct is considered expired.
DEFAULT_INSTINCT_TTL_DAYS = 30


class InstinctLike(Protocol):
    """Structural type for confidence/TTL math (a ``Procedure`` satisfies this)."""

    success_rate: float
    times_used: int
    last_used: datetime | None


def instinct_age_days(p: InstinctLike, *, now: datetime | None = None) -> float | None:
    """Days since ``last_used``; ``None`` when the instinct has no timestamp."""
    last = p.last_used
    if last is None:
        return None
    ref = now or datetime.now(UTC)
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=UTC)
    return max(0.0, (ref - last).total_seconds() / 86400.0)


def instinct_expired(
    p: InstinctLike,
    *,
    now: datetime | None = None,
    ttl_days: int = DEFAULT_INSTINCT_TTL_DAYS,
) -> bool:
    """True when the instinct has been idle for longer than ``ttl_days``.

    A ``ttl_days <= 0`` disables expiry, and an instinct without a ``last_used``
    timestamp never expires (there is no clock to measure against).
    """
    if ttl_days <= 0:
        return False
    age = instinct_age_days(p, now=now)
    if age is None:
        return False
    return age > float(ttl_days)


def instinct_confidence(
    p: InstinctLike,
    *,
    now: datetime | None = None,
    ttl_days: int = DEFAULT_INSTINCT_TTL_DAYS,
) -> float:
    """Derived 0..1 confidence: ``success_rate × evidence × recency``.

    - ``evidence_factor`` ramps linearly with ``times_used`` and saturates at
      :data:`EVIDENCE_SATURATION` uses.
    - ``recency_factor`` decays linearly to 0 across the TTL window; it is 1.0
      when the instinct is fresh, has no timestamp, or TTL is disabled.
    """
    rate = max(0.0, min(1.0, float(p.success_rate)))
    evidence_factor = min(1.0, max(0, int(p.times_used)) / float(EVIDENCE_SATURATION))
    age = instinct_age_days(p, now=now)
    if age is None or ttl_days <= 0:
        recency_factor = 1.0
    else:
        recency_factor = max(0.0, 1.0 - age / float(ttl_days))
    return round(rate * evidence_factor * recency_factor, 4)
