"""Accuracy-gated policy for must-act forecast recording (P3 noise reduction)."""

from __future__ import annotations

from typing import Any

from brain_os.config import get_settings
from brain_os.memory.prediction_log import MUST_ACT_TODAY_PATTERN, PredictionLog

_MUST_ACT_BUCKETS = frozenset({"reply", "rewrite_draft", "followup"})


async def must_act_recording_policy(log: PredictionLog) -> dict[str, Any]:
    """Return whether new must-act predictions should be recorded this cycle."""
    cfg = get_settings().app
    if not cfg.prediction_record_must_act:
        return {
            "record": False,
            "reason": "disabled",
            "accuracy": None,
            "reconciled": 0,
            "narrow": False,
            "max_per_cycle": 0,
            "blocking_only": False,
        }

    reconciled = await log.reconciled_count(pattern_id=MUST_ACT_TODAY_PATTERN)
    by_pattern = await log.accuracy_by_pattern(min_predictions=1)
    accuracy = by_pattern.get(MUST_ACT_TODAY_PATTERN)

    min_reconciled = int(cfg.prediction_must_act_min_reconciled)
    floor = float(cfg.prediction_must_act_accuracy_floor)
    target = float(cfg.prediction_must_act_accuracy_target)

    base: dict[str, Any] = {
        "accuracy": accuracy,
        "reconciled": reconciled,
        "accuracy_floor": floor,
        "accuracy_target": target,
        "min_reconciled": min_reconciled,
    }

    if reconciled < min_reconciled:
        return {
            **base,
            "record": True,
            "reason": "bootstrap",
            "narrow": False,
            "blocking_only": False,
            "max_per_cycle": int(cfg.prediction_must_act_max_per_cycle),
        }

    if accuracy is not None and accuracy < floor:
        return {
            **base,
            "record": False,
            "reason": "accuracy_below_floor",
            "narrow": False,
            "blocking_only": False,
            "max_per_cycle": 0,
        }

    narrow = bool(cfg.prediction_must_act_narrow_when_below_target) and (
        accuracy is None or accuracy < target
    )
    return {
        **base,
        "record": True,
        "reason": "narrowed" if narrow else "ok",
        "narrow": narrow,
        "blocking_only": narrow and bool(cfg.prediction_must_act_blocking_only),
        "max_per_cycle": int(cfg.prediction_must_act_max_per_cycle) if narrow else 40,
    }


def must_act_procedural_learning_enabled(*, accuracy: float | None, reconciled: int) -> bool:
    """Whether must-act reconciliation failures may feed procedural memory."""
    cfg = get_settings().app
    if not cfg.prediction_must_act_procedural_learning:
        return False
    if reconciled < int(cfg.prediction_must_act_min_reconciled):
        return False
    if accuracy is None:
        return False
    return accuracy >= float(cfg.prediction_must_act_accuracy_target)


def is_must_act_reconciliation_row(row: Any) -> bool:
    """Heuristic: must-act rows use desk bucket names, not lead-board buckets."""
    bucket = str(getattr(row, "predicted_bucket", "") or "").strip().lower()
    return bucket in _MUST_ACT_BUCKETS
