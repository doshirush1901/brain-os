"""Dream / prediction reconciliation telemetry (Letta month-1 metric #2)."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WARN_RESOLUTION_RATE = 0.75
_CRIT_RESOLUTION_RATE = 0.50
_TARGET_RESOLUTION_RATE = 0.85


def _data_root() -> Path:
    return Path(os.environ.get("BRAIN_DATA_DIR", "data")).expanduser().resolve()


def dream_reconcile_events_path() -> Path:
    path = _data_root() / "operations" / "dream_reconcile_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_dream_reconcile_event(
    *,
    source: str,
    reconciled: int,
    correct: int,
    incorrect: int,
    skipped: int,
    pending_before: int,
    sophia_lessons: int = 0,
    operator_tickets: int = 0,
) -> None:
    """Record one reconciliation cycle (dream stage 3.6 or ``brain learning reconcile``)."""
    denom = reconciled + skipped
    resolution_rate = (reconciled / denom) if denom > 0 else None
    row = {
        "ts": datetime.now(UTC).isoformat(),
        "source": source[:64],
        "reconciled": reconciled,
        "correct": correct,
        "incorrect": incorrect,
        "skipped": skipped,
        "pending_before": pending_before,
        "sophia_lessons": sophia_lessons,
        "operator_tickets": operator_tickets,
        "resolution_rate": round(resolution_rate, 4) if resolution_rate is not None else None,
    }
    path = dream_reconcile_events_path()
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("append_dream_reconcile_event failed path=%s", path)


def _load_events(*, since_ts: float) -> list[dict[str, Any]]:
    path = dream_reconcile_events_path()
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _event_ts(row)
            if ts >= since_ts:
                out.append(row)
    except OSError:
        logger.exception("load dream reconcile events failed")
    return out


def _event_ts(event: dict[str, Any]) -> float:
    ts_raw = event.get("ts")
    if not ts_raw:
        return 0.0
    try:
        return datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _dream_reconcile_alerts(
    *,
    resolution_rate_7d: float | None,
    low_resolution_days: int,
    operator_tickets_7d: int,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    if resolution_rate_7d is not None and low_resolution_days >= 2:
        if resolution_rate_7d < _CRIT_RESOLUTION_RATE or operator_tickets_7d >= 10:
            alerts.append(
                {
                    "name": "dream_reconcile_backlog",
                    "metric": "resolution_rate_7d",
                    "value": round(resolution_rate_7d, 4),
                    "threshold": _CRIT_RESOLUTION_RATE,
                    "severity": "critical",
                    "latency_note": "reconcile delays may affect live system; watch p95",
                }
            )
        elif resolution_rate_7d < _WARN_RESOLUTION_RATE:
            alerts.append(
                {
                    "name": "dream_reconcile_slipping",
                    "metric": "resolution_rate_7d",
                    "value": round(resolution_rate_7d, 4),
                    "threshold": _WARN_RESOLUTION_RATE,
                    "severity": "warning",
                    "latency_note": "backlog risk; slight p95 uptick possible",
                }
            )
    return alerts


def _low_resolution_day_count(events: list[dict[str, Any]]) -> int:
    """Days in the event window with resolution_rate below warning threshold."""
    by_day: dict[str, list[float]] = {}
    for e in events:
        rate = e.get("resolution_rate")
        if rate is None:
            denom = int(e.get("reconciled") or 0) + int(e.get("skipped") or 0)
            if denom > 0:
                rate = int(e.get("reconciled") or 0) / denom
            else:
                continue
        ts_raw = e.get("ts")
        if not ts_raw:
            continue
        day = str(ts_raw)[:10]
        by_day.setdefault(day, []).append(float(rate))
    low_days = 0
    for rates in by_day.values():
        if sum(rates) / len(rates) < _WARN_RESOLUTION_RATE:
            low_days += 1
    return low_days


async def _prediction_log_pending_count() -> int:
    from brain_os.memory.prediction_log import PredictionLog

    log = PredictionLog()
    await log.initialize()
    try:
        return len(await log.get_unreconciled())
    finally:
        await log.close()


async def build_dream_reconcile_snapshot() -> dict[str, Any]:
    """Roll up dream reconcile events + prediction log backlog."""
    now = time.time()
    since_7d = now - 7 * 86400
    events_7d = _load_events(since_ts=since_7d)

    reconciled_7d = sum(int(e.get("reconciled") or 0) for e in events_7d)
    skipped_7d = sum(int(e.get("skipped") or 0) for e in events_7d)
    operator_7d = sum(int(e.get("operator_tickets") or 0) for e in events_7d)
    denom_7d = reconciled_7d + skipped_7d
    resolution_7d = (reconciled_7d / denom_7d) if denom_7d > 0 else None

    pending_now = await _prediction_log_pending_count()
    low_days = _low_resolution_day_count(events_7d)

    return {
        "enabled": True,
        "spec": "data/knowledge/letta_self_improvement_metrics.yaml",
        "events_path": str(dream_reconcile_events_path()),
        "cycles_7d": len(events_7d),
        "reconciled_7d": reconciled_7d,
        "skipped_7d": skipped_7d,
        "resolution_rate_7d": round(resolution_7d, 4) if resolution_7d is not None else None,
        "target": f">= {_TARGET_RESOLUTION_RATE}",
        "predictions_pending": pending_now,
        "operator_tickets_7d": operator_7d,
        "low_resolution_days_7d": low_days,
        "alerts": _dream_reconcile_alerts(
            resolution_rate_7d=resolution_7d,
            low_resolution_days=low_days,
            operator_tickets_7d=operator_7d,
        ),
        "computed_at": now,
    }


async def pending_before_reconciliation() -> int:
    """Unreconciled prediction count at cycle start (for event logging)."""
    return await _prediction_log_pending_count()
