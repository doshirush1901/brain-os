"""Operator snapshot and bootstrap for Mem0 forgetting (P5 rollout)."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from brain_os.config import get_settings
from brain_os.memory.mem0_forgetting import forgetting_events_path, run_mem0_forgetting

logger = logging.getLogger(__name__)


def _parse_event_ts(raw: Any) -> float:
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def load_forgetting_events(*, since_ts: float) -> list[dict[str, Any]]:
    path = forgetting_events_path()
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
            if _parse_event_ts(row.get("timestamp")) >= since_ts:
                out.append(row)
    except OSError:
        logger.exception("load mem0 forgetting events failed")
    return out


def build_mem0_forgetting_policy() -> dict[str, Any]:
    from brain_os.memory.forgetting_exclusions import exclusion_policy_doc

    app = get_settings().app
    enabled = bool(app.dream_mem0_forgetting_enabled)
    return {
        "execute_archive": enabled,
        "execute_deletions": enabled,  # stage-2 hard-delete when enabled
        "dry_run_default": not enabled,
        "heartbeat_enabled": bool(app.mem0_forgetting_heartbeat_enabled),
        "min_age_days": int(app.mem0_forget_min_age_days),
        "max_confidence": float(app.mem0_forget_max_confidence),
        "run_cap": int(app.mem0_forget_run_cap),
        "hard_delete_after_days": int(getattr(app, "mem0_forget_hard_delete_after_days", 30)),
        "salience_protect": float(getattr(app, "mem0_forget_salience_protect", 0.35)),
        "protected_types": ["correction", "preference"],
        "stages": ["archive", "hard_delete"],
        "exclusion_policy": exclusion_policy_doc(),
        "rollout_env": "APP__DREAM_MEM0_FORGETTING_ENABLED=true",
        "events_path": str(forgetting_events_path()),
    }


def build_mem0_forgetting_snapshot(*, days: int = 7) -> dict[str, Any]:
    """Recent candidate/delete counts from the forgetting audit log."""
    now = time.time()
    since = now - days * 86400
    events = load_forgetting_events(since_ts=since)
    by_reason: Counter[str] = Counter()
    deleted = 0
    dry_logged = 0
    for event in events:
        by_reason[str(event.get("reason") or "unknown")] += 1
        if event.get("deleted"):
            deleted += 1
        elif event.get("dry_run"):
            dry_logged += 1

    last_run_ts = max((_parse_event_ts(e.get("run_started")) for e in events), default=0.0)
    policy = build_mem0_forgetting_policy()
    return {
        "enabled": True,
        "policy": policy,
        "window_days": days,
        "events_in_window": len(events),
        "candidates_logged": len(events),
        "deleted_in_window": deleted,
        "dry_run_logged": dry_logged,
        "by_reason": dict(by_reason),
        "last_run_at": (
            datetime.fromtimestamp(last_run_ts, tz=UTC).isoformat() if last_run_ts else None
        ),
        "ready_for_execute": (
            not policy["execute_deletions"] and len(events) >= 1 and deleted == 0 and dry_logged > 0
        ),
        "computed_at": now,
    }


async def run_mem0_forgetting_pass(
    *,
    force_enabled: bool | None = None,
    source: str = "operator",
) -> dict[str, Any]:
    """Run one forgetting sweep; returns summary or error when Mem0 unavailable."""
    from brain_os.config import get_settings

    if not get_settings().memory.api_key.get_secret_value():
        return {"status": "skipped", "reason": "mem0_api_key_missing", "source": source}

    from brain_os.memory.long_term import LongTermMemory

    long_term = LongTermMemory()
    summary = await run_mem0_forgetting(long_term, force_enabled=force_enabled)
    summary["source"] = source[:64]
    return summary
