"""Rolling operator signals consumed by Dream mode (memory pruning aggression)."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()


def _path() -> Path:
    return get_data_dir() / "brain" / "dream_triggers.json"


def read_triggers() -> dict[str, Any]:
    p = _path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.debug("dream_triggers read failed", exc_info=True)
        return {}


def _write_triggers(payload: dict[str, Any]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def bump_event(event: str, delta: int = 1, *, ttl_hours: int = 168) -> None:
    """Increment a counter with simple time-based decay buckets (weekly window)."""
    now = datetime.now(UTC)
    cutoff = (now - timedelta(hours=max(1, ttl_hours))).isoformat()
    key = event.strip().lower().replace(" ", "_") or "unknown"
    with _LOCK:
        blob = read_triggers()
        events = blob.get("events")
        if not isinstance(events, list):
            events = []
        # Drop stale entries.
        fresh: list[dict[str, Any]] = []
        for row in events:
            if isinstance(row, dict) and isinstance(row.get("at"), str) and row["at"] >= cutoff:
                fresh.append(row)
        fresh.append({"name": key, "at": now.isoformat(), "delta": int(delta)})
        blob["events"] = fresh[-2000:]  # cap size

        totals: dict[str, int] = {}
        for row in fresh:
            n = row.get("name") if isinstance(row.get("name"), str) else "unknown"
            totals[n] = totals.get(n, 0) + int(row.get("delta", 1) or 1)
        blob["totals_window"] = totals
        blob["updated_at"] = now.isoformat()
        try:
            _write_triggers(blob)
        except Exception:
            logger.warning("dream_triggers write failed", exc_info=True)


def aggregate_signals() -> dict[str, Any]:
    """Totals in rolling window plus convenience flags."""
    blob = read_triggers()
    totals = blob.get("totals_window")
    if not isinstance(totals, dict):
        totals = {}
    faith_blocks = int(totals.get("faithfulness_block", 0) or 0)
    return {
        "faithfulness_block": faith_blocks,
        "pruning_aggressive": faith_blocks >= 3,
        "updated_at": blob.get("updated_at"),
    }
