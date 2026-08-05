"""Read-only summary stats for the Mnemon correction ledger."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime, timedelta
from typing import Any

from brain_os.memory.correction_ledger_metrics import load_ledger_entities

_WEEKS = 8
_TOP_N = 5


def _parse_corrected_at(value: Any) -> datetime | None:
    """Parse ledger date fields (``corrected_at`` / ``effective_from``)."""
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


def _iso_week_key(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _entity_category(entry: dict[str, Any]) -> str:
    raw = entry.get("category") or entry.get("fact_type") or "general"
    text = str(raw).strip()
    return text or "general"


def _entity_correction_count(entry: dict[str, Any]) -> int:
    """Prefer ``version`` (Mnemon bump count); else 1 + len(stale_values)."""
    version = entry.get("version")
    if version is not None:
        try:
            n = int(version)
            if n >= 1:
                return n
        except (TypeError, ValueError):
            pass
    stale = entry.get("stale_values") or []
    if isinstance(stale, str):
        stale = [stale] if stale.strip() else []
    if isinstance(stale, list):
        return 1 + sum(1 for v in stale if str(v).strip())
    return 1


def _monday_of_iso_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


def build_corrections_stats(
    entities: dict[str, Any] | None = None,
    *,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Summarize ledger entities for CLI / API consumers.

    Returns empty-shaped payload when there are no entities (caller decides
    friendly messaging). Does not read or write disk when ``entities`` is passed.
    """
    if entities is None:
        entities = load_ledger_entities()

    now = as_of if as_of is not None else datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    total = 0
    by_category: Counter[str] = Counter()
    entity_counts: list[tuple[str, int]] = []
    week_hits: Counter[str] = Counter()

    for key, raw in entities.items():
        if not isinstance(raw, dict):
            continue
        total += 1
        by_category[_entity_category(raw)] += 1
        entity_counts.append((str(key), _entity_correction_count(raw)))
        corrected = _parse_corrected_at(raw.get("corrected_at") or raw.get("effective_from"))
        if corrected is not None:
            week_hits[_iso_week_key(corrected)] += 1

    entity_counts.sort(key=lambda row: (-row[1], row[0].lower()))
    top_entities = [
        {"entity": name, "corrections": count} for name, count in entity_counts[:_TOP_N]
    ]

    monday = _monday_of_iso_week(now.date())
    per_week: list[dict[str, Any]] = []
    for i in range(_WEEKS - 1, -1, -1):
        week_monday = monday - timedelta(weeks=i)
        anchor = datetime(
            week_monday.year,
            week_monday.month,
            week_monday.day,
            tzinfo=UTC,
        ) + timedelta(days=3)
        key = _iso_week_key(anchor)
        per_week.append({"week": key, "count": int(week_hits.get(key, 0))})

    return {
        "total": total,
        "empty": total == 0,
        "per_iso_week": per_week,
        "top_entities": top_entities,
        "by_category": dict(sorted(by_category.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
