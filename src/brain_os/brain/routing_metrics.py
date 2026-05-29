"""In-process routing hit counters (fast_path, operator_deterministic, deterministic, full_pipeline)."""

from __future__ import annotations

from brain_os.brain.runtime_metrics import incr, snapshot


def record_route_hit(route_kind: str) -> None:
    key = route_kind.strip().lower().replace(" ", "_")
    if not key:
        return
    incr(f"route_hit_{key}")


def routing_metrics_snapshot() -> dict[str, int | float]:
    raw = snapshot()
    hits = {k: v for k, v in raw.items() if k.startswith("route_hit_")}
    total = sum(hits.values())
    deterministic = (
        hits.get("route_hit_fast_path", 0)
        + hits.get("route_hit_operator_deterministic", 0)
        + hits.get("route_hit_truth_hint", 0)
        + hits.get("route_hit_procedural", 0)
        + hits.get("route_hit_deterministic", 0)
    )
    ratio = round(deterministic / total, 4) if total else 0.0
    return {
        **hits,
        "route_total": total,
        "route_deterministic_total": deterministic,
        "route_deterministic_ratio": ratio,
    }
