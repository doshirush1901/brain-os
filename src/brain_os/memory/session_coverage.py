"""Canonical miner coverage metric — single source of truth for vitals/heartbeat.

Before this module existed, ``learning_organ_vitals`` (vitals) computed miner
coverage over a rolling **1-day** window (falling back to 7d only when
yesterday was empty) while heartbeat's ``session_mine_batch`` outcome reported
a rolling **7-day** window under a differently-named field
(``coverage_7d_pct_meta``). Both numbers are 0-1 fractions but the field name
said "pct", and operators comparing the two consumers were comparing
different windows and sometimes different numerators (``pct_meta`` — rows with
*any* learning metadata — vs ``pct_mined`` — rows with ``session_mine.mined_at``
actually set).

This module is the single canonical entry point every consumer should call:

* **Window:** rolling 7.0 days (matches heartbeat's cadence).
* **Numerator:** ``pct_mined`` (0-1 fraction of rows actually mined), not
  ``pct_meta``.
* **Display:** also returns ``pct_mined_100`` (0-100, rounded) for
  operator-facing surfaces that print a percentage.

Does not duplicate the underlying scan logic — that stays in
:mod:`brain_os.memory.session_miner` (``coverage_snapshot`` / ``is_mined_meta``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from brain_os.memory import session_miner
from brain_os.memory.session_miner import is_mined_meta

#: Rolling window (days) every miner-coverage consumer should use.
CANONICAL_COVERAGE_WINDOW_DAYS = 7.0

__all__ = [
    "CANONICAL_COVERAGE_WINDOW_DAYS",
    "canonical_miner_coverage",
    "is_mined_meta",
]


async def canonical_miner_coverage(*, db_path: Path | None = None) -> dict[str, Any]:
    """Return the canonical miner-coverage snapshot (7d window, ``pct_mined`` numerator).

    Same shape as :func:`brain_os.memory.session_miner.coverage_snapshot` plus
    ``pct_mined_100`` (0-100, rounded to 2dp) for display. Vitals, heartbeat,
    and CLI reporting should all call this instead of calling
    ``coverage_snapshot`` directly with ad-hoc ``since_days`` values, so the
    numbers never drift apart again.

    Calls through the ``session_miner`` module object (not a direct
    ``from ... import coverage_snapshot`` binding) so tests that monkeypatch
    ``brain_os.memory.session_miner.coverage_snapshot`` keep working unchanged.
    """
    snap = await session_miner.coverage_snapshot(
        since_days=CANONICAL_COVERAGE_WINDOW_DAYS, db_path=db_path
    )
    pct_mined = snap.get("pct_mined")
    snap["pct_mined_100"] = round(float(pct_mined) * 100, 2) if pct_mined is not None else None
    return snap
