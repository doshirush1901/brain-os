"""Letta month-1 self-improvement KPI rollup (turns 7–9)."""

from __future__ import annotations

import time
from typing import Any

from brain_os.memory.correction_ledger_metrics import build_correction_ledger_snapshot
from brain_os.memory.dream_reconcile_metrics import build_dream_reconcile_snapshot
from brain_os.memory.procedural_decay_metrics import build_procedural_decay_snapshot


async def build_self_improvement_metrics_snapshot() -> dict[str, Any]:
    """Aggregate procedural decay, dream reconcile, and correction ledger KPIs."""
    procedural, dream, ledger = await _gather_snapshots()
    return {
        "source": "lettie-20260602-62678ab3 turns 7–9",
        "procedural_decay": procedural,
        "dream_reconcile": dream,
        "correction_ledger": ledger,
        "latency_guardrail": {
            "metric": "p95_request_latency_ms",
            "note": "Compare branch_telemetry + pipeline timings; batch reconcile off-peak",
        },
        "computed_at": time.time(),
    }


async def _gather_snapshots() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        await build_procedural_decay_snapshot(),
        await build_dream_reconcile_snapshot(),
        await build_correction_ledger_snapshot(),
    )
