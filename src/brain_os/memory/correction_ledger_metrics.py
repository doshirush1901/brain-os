"""Correction ledger freshness telemetry (Letta month-1 metric #3)."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WARN_COVERAGE = 0.85
_CRIT_COVERAGE = 0.70
_TARGET_COVERAGE = 0.90
_FRESHNESS_DAYS = 30


def _data_root() -> Path:
    return Path(os.environ.get("BRAIN_DATA_DIR", "data")).expanduser().resolve()


def correction_ledger_path() -> Path:
    return _data_root() / "brain" / "correction_ledger.json"


def _parse_corrected_at(value: Any) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        if len(raw) == 10 and raw[4] == "-":
            return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=UTC)
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_ledger_entities() -> dict[str, Any]:
    path = correction_ledger_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("correction ledger read failed")
        return {}
    entities = data.get("entities")
    return entities if isinstance(entities, dict) else {}


def append_ledger_verify_event(
    *,
    verified: int,
    stale: int,
    total: int,
    source: str = "snapshot",
) -> None:
    """Optional audit line when an operator runs ledger verify (future job)."""
    path = _data_root() / "operations" / "correction_ledger_verify_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    coverage = (verified / total) if total > 0 else None
    row = {
        "ts": datetime.now(UTC).isoformat(),
        "source": source[:64],
        "total_entities": total,
        "verified_30d": verified,
        "stale_entities": stale,
        "coverage": round(coverage, 4) if coverage is not None else None,
    }
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("append_ledger_verify_event failed")


def _ledger_coverage_alerts(coverage: float | None, total: int) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    if coverage is None or total < 1:
        return alerts
    if coverage < _CRIT_COVERAGE:
        alerts.append(
            {
                "name": "correction_ledger_stale",
                "metric": "coverage_30d",
                "value": round(coverage, 4),
                "threshold": _CRIT_COVERAGE,
                "severity": "critical",
                "latency_note": "if paired with p95 > baseline +10%, correction path degraded",
            }
        )
    elif coverage < _WARN_COVERAGE:
        alerts.append(
            {
                "name": "correction_ledger_drift",
                "metric": "coverage_30d",
                "value": round(coverage, 4),
                "threshold": _WARN_COVERAGE,
                "severity": "warning",
                "latency_note": "ledger inconsistency; monitor latency",
            }
        )
    return alerts


async def build_correction_ledger_snapshot() -> dict[str, Any]:
    """Coverage proxy: entities with ``corrected_at`` within 30 days / total entities."""
    entities = load_ledger_entities()
    cutoff = datetime.now(UTC) - timedelta(days=_FRESHNESS_DAYS)
    total = len(entities)
    verified = 0
    stale = 0
    missing_date = 0

    for _key, raw in entities.items():
        if not isinstance(raw, dict):
            stale += 1
            continue
        entry = raw
        corrected = _parse_corrected_at(entry.get("corrected_at") or entry.get("effective_from"))
        value = str(entry.get("current_status") or entry.get("correct_value") or "").strip()
        if not value:
            stale += 1
            continue
        if corrected is None:
            missing_date += 1
            stale += 1
            continue
        if corrected >= cutoff:
            verified += 1
        else:
            stale += 1

    coverage = (verified / total) if total > 0 else None

    return {
        "enabled": True,
        "spec": "data/knowledge/letta_self_improvement_metrics.yaml",
        "ledger_path": str(correction_ledger_path()),
        "total_entities": total,
        "verified_30d": verified,
        "stale_entities": stale,
        "missing_corrected_at": missing_date,
        "coverage_30d": round(coverage, 4) if coverage is not None else None,
        "target": f"> {_TARGET_COVERAGE}",
        "freshness_days": _FRESHNESS_DAYS,
        "alerts": _ledger_coverage_alerts(coverage, total),
        "computed_at": time.time(),
    }
