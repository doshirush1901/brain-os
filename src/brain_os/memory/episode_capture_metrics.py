"""Episode capture audit (Phase 3 biology upgrade).

Counts how many interactions become episodes vs get dropped, and why.
Two instrumentation points feed the same counter file:

* ``learn.py`` (pipeline step 10) — every turn bumps ``record_seen``; turns
  that fail the eligibility gate also bump ``record_drop("not_eligible_turn")``.
* ``dream_mode`` stage 2 — each consolidated episode bumps ``record_capture``;
  conversations below the history threshold bump
  ``record_drop("below_min_history")``.

State lives in ``data/metrics/episode_capture.json``::

    {"interactions_total": N, "episodes_captured_total": N,
     "drop_reasons": {"not_eligible_turn": N, "below_min_history": N, ...}}

All writes are best-effort and never raise into the pipeline or dream cycle.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_METRICS_PATH = Path("data/metrics/episode_capture.json")

_lock = threading.Lock()

_EPISODE_METRIC_WRITE_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    json.JSONDecodeError,
)


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _bump(path: Path, mutate: Any) -> None:
    try:
        with _lock:
            state = _load(path)
            mutate(state)
            _save(path, state)
    except _EPISODE_METRIC_WRITE_ERRORS:
        logger.debug("Episode capture metric write failed", exc_info=True)


def record_seen(count: int = 1, metrics_path: Path | str = DEFAULT_METRICS_PATH) -> None:
    """Count *count* pipeline interactions observed (learn step)."""

    def mutate(state: dict[str, Any]) -> None:
        state["interactions_total"] = int(state.get("interactions_total", 0)) + count

    _bump(Path(metrics_path), mutate)


def record_drop(
    reason: str,
    count: int = 1,
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
) -> None:
    """Count interactions dropped before becoming episodes, by reason slug."""

    def mutate(state: dict[str, Any]) -> None:
        reasons = dict(state.get("drop_reasons", {}))
        reasons[reason] = int(reasons.get(reason, 0)) + count
        state["drop_reasons"] = reasons

    logger.debug("Episode capture drop: %s (+%d)", reason, count)
    _bump(Path(metrics_path), mutate)


def record_capture(count: int = 1, metrics_path: Path | str = DEFAULT_METRICS_PATH) -> None:
    """Count episodes actually written during consolidation (dream stage 2)."""

    def mutate(state: dict[str, Any]) -> None:
        state["episodes_captured_total"] = int(state.get("episodes_captured_total", 0)) + count

    _bump(Path(metrics_path), mutate)


def capture_snapshot(metrics_path: Path | str = DEFAULT_METRICS_PATH) -> dict[str, Any]:
    """Current counters plus derived capture rate (for vitals / tests)."""
    state = _load(Path(metrics_path))
    total = int(state.get("interactions_total", 0))
    captured = int(state.get("episodes_captured_total", 0))
    return {
        "interactions_total": total,
        "episodes_captured_total": captured,
        "capture_rate": round(captured / total, 3) if total else 0.0,
        "drop_reasons": dict(state.get("drop_reasons", {})),
    }
