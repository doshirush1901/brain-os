"""Procedural decay telemetry for Letta month-1 self-improvement (turns 8–9).

Append-only event log under ``data/operations/procedural_decay_events.jsonl``.
Rollup + alerts align with ``data/knowledge/letta_self_improvement_metrics.yaml``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

DecayAction = Literal["pruned", "demoted", "expired"]

_WARN_PRUNE_RATE_FLOOR = 0.05
_CRIT_PRUNE_RATE_CEIL = 0.30
_CRIT_WINDOW_DAYS = 3


def _data_root() -> Path:
    return Path(os.environ.get("BRAIN_DATA_DIR", "data")).expanduser().resolve()


def decay_events_path() -> Path:
    path = _data_root() / "operations" / "procedural_decay_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_decay_event(
    *,
    action: DecayAction,
    procedure_id: int,
    trigger_pattern: str,
    success_rate_before: float,
    success_rate_after: float | None = None,
    query_snippet: str = "",
) -> None:
    """Record a prune or demotion from :meth:`ProceduralMemory.record_failure`."""
    row = {
        "ts": datetime.now(UTC).isoformat(),
        "action": action,
        "procedure_id": procedure_id,
        "trigger_pattern": (trigger_pattern or "")[:200],
        "success_rate_before": round(success_rate_before, 4),
        "success_rate_after": (
            round(success_rate_after, 4) if success_rate_after is not None else None
        ),
        "query_snippet": (query_snippet or "")[:120],
    }
    path = decay_events_path()
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("append_decay_event failed path=%s", path)


def _load_events(*, since_ts: float) -> list[dict[str, Any]]:
    path = decay_events_path()
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
            ts_raw = row.get("ts")
            if not ts_raw:
                continue
            try:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            if ts >= since_ts:
                out.append(row)
    except OSError:
        logger.exception("load procedural decay events failed")
    return out


def _procedural_decay_alerts(
    *,
    total_procedures: int,
    pruned_7d: int,
    pruned_3d: int,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    if total_procedures <= 0:
        return alerts

    rate_7d = pruned_7d / total_procedures
    rate_3d = pruned_3d / total_procedures

    if rate_7d < _WARN_PRUNE_RATE_FLOOR and total_procedures >= 10:
        alerts.append(
            {
                "name": "procedural_decay_stall",
                "metric": "prune_rate_7d",
                "value": round(rate_7d, 4),
                "threshold": _WARN_PRUNE_RATE_FLOOR,
                "severity": "warning",
                "latency_note": "bloat risk; no p95 impact expected yet",
            }
        )
    if rate_3d > _CRIT_PRUNE_RATE_CEIL:
        alerts.append(
            {
                "name": "procedural_decay_aggressive",
                "metric": "prune_rate_3d",
                "value": round(rate_3d, 4),
                "threshold": _CRIT_PRUNE_RATE_CEIL,
                "severity": "critical",
                "latency_note": "route loss risk; watch p95 > baseline +10%",
            }
        )
    return alerts


async def build_procedural_decay_snapshot(
    *,
    total_procedures: int | None = None,
) -> dict[str, Any]:
    """Roll up decay events + store size for ops dashboards."""
    now = time.time()
    since_7d = now - 7 * 86400
    since_3d = now - _CRIT_WINDOW_DAYS * 86400
    since_30d = now - 30 * 86400

    events_7d = _load_events(since_ts=since_7d)
    events_3d = [e for e in events_7d if _event_ts(e) >= since_3d]
    events_30d = _load_events(since_ts=since_30d)

    # TTL-expired instincts are a form of pruning; count them in the prune totals
    # so the decay-rate metric stays meaningful, but also surface them separately.
    _prune_actions = ("pruned", "expired")
    pruned_7d = sum(1 for e in events_7d if e.get("action") in _prune_actions)
    demoted_7d = sum(1 for e in events_7d if e.get("action") == "demoted")
    pruned_3d = sum(1 for e in events_3d if e.get("action") in _prune_actions)
    pruned_30d = sum(1 for e in events_30d if e.get("action") in _prune_actions)
    demoted_30d = sum(1 for e in events_30d if e.get("action") == "demoted")
    expired_7d = sum(1 for e in events_7d if e.get("action") == "expired")
    expired_30d = sum(1 for e in events_30d if e.get("action") == "expired")

    if total_procedures is None:
        from brain_os.memory.procedural import ProceduralMemory

        pm = ProceduralMemory()
        await pm.initialize()
        try:
            total_procedures = await pm.count_procedures()
        finally:
            await pm.close()

    total = int(total_procedures or 0)
    decay_rate_7d = (pruned_7d / total) if total else None

    return {
        "enabled": True,
        "spec": "data/knowledge/letta_self_improvement_metrics.yaml",
        "events_path": str(decay_events_path()),
        "total_procedures": total,
        "pruned_7d": pruned_7d,
        "demoted_7d": demoted_7d,
        "pruned_3d": pruned_3d,
        "pruned_30d": pruned_30d,
        "demoted_30d": demoted_30d,
        "expired_7d": expired_7d,
        "expired_30d": expired_30d,
        "decay_rate_7d": round(decay_rate_7d, 4) if decay_rate_7d is not None else None,
        "target_monthly": "0.10–0.20",
        "alerts": _procedural_decay_alerts(
            total_procedures=total,
            pruned_7d=pruned_7d,
            pruned_3d=pruned_3d,
        ),
        "computed_at": now,
    }


def _event_ts(event: dict[str, Any]) -> float:
    ts_raw = event.get("ts")
    if not ts_raw:
        return 0.0
    try:
        return datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
