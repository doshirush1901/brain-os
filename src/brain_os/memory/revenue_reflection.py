"""Dream stage 3f — daily revenue funnel reflection (Phase 3 biology upgrade).

Reads the day's funnel events from ``data/revenue_mode/`` (sends, outcomes,
reply analyses) and distills at least one short insight memory per day, e.g.
"3 sends, 1 positive reply (Test Thermoforming Co — active_project)". The
insight is deterministic — counts and named standouts, no LLM call — so the
stage is cheap and always produces material for episodic consolidation.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REVENUE_DIR = Path("data/revenue_mode")

_FUNNEL_FILES = ("activity.jsonl", "send_ledger.jsonl", "reply_analysis.jsonl")


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def load_funnel_events(
    revenue_dir: Path | str = DEFAULT_REVENUE_DIR,
    *,
    lookback_hours: int = 24,
    now: datetime | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Read recent funnel events from the revenue-mode jsonl ledgers.

    Returns ``{"activity": [...], "sends": [...], "replies": [...]}`` with
    only events whose ``timestamp`` falls within the lookback window.
    Missing files and malformed lines are skipped silently — this stage must
    never break the dream cycle.
    """
    revenue_dir = Path(revenue_dir)
    cutoff = (now or datetime.now(UTC)) - timedelta(hours=lookback_hours)
    buckets: dict[str, list[dict[str, Any]]] = {"activity": [], "sends": [], "replies": []}
    keys = ("activity", "sends", "replies")
    for key, fname in zip(keys, _FUNNEL_FILES, strict=True):
        path = revenue_dir / fname
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_ts(event.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            buckets[key].append(event)
    return buckets


def summarize_funnel(events: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Deterministic counts over the day's funnel events."""
    sends = events.get("sends", [])
    replies = events.get("replies", [])
    activity = events.get("activity", [])

    sent_companies = sorted(
        {str(e.get("company") or e.get("company_name") or "").strip() for e in sends} - {""}
    )
    positive = [r for r in replies if r.get("reply_sentiment") == "positive"]
    negative = [r for r in replies if r.get("reply_sentiment") == "negative"]
    bounces = [
        e
        for e in activity
        if "bounce" in str(e.get("outcome", "")).lower()
        or "bounce" in str(e.get("type", "")).lower()
    ]
    return {
        "sends": len(sends),
        "send_companies": sent_companies,
        "replies": len(replies),
        "positive_replies": len(positive),
        "negative_replies": len(negative),
        "positive_companies": sorted(
            {str(r.get("company_name") or "").strip() for r in positive} - {""}
        ),
        "bounces": len(bounces),
        "activity_events": len(activity),
    }


def build_insight(summary: dict[str, Any]) -> str:
    """One short, human-readable insight line for the day. Never empty."""
    parts: list[str] = []
    if summary["sends"]:
        companies = summary["send_companies"]
        suffix = f" ({', '.join(companies[:3])})" if companies else ""
        parts.append(f"{summary['sends']} send(s){suffix}")
    if summary["replies"]:
        parts.append(
            f"{summary['replies']} repl(ies): {summary['positive_replies']} positive"
            + (
                f" ({', '.join(summary['positive_companies'][:3])})"
                if summary["positive_companies"]
                else ""
            )
            + (f", {summary['negative_replies']} negative" if summary["negative_replies"] else "")
        )
    if summary["bounces"]:
        parts.append(f"{summary['bounces']} bounce(s) — check sender/domain health")
    if not parts:
        return (
            "Revenue funnel quiet today: no sends, replies, or bounces in the last "
            "24h — pipeline needs new outbound activity."
        )
    return "Revenue funnel today: " + "; ".join(parts) + "."


async def run_revenue_reflection(
    ep_db: Any,
    *,
    revenue_dir: Path | str = DEFAULT_REVENUE_DIR,
    lookback_hours: int = 24,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Execute stage 3f: read funnel, write one insight episode.

    ``ep_db`` is an aiosqlite-style connection exposing the ``episodes`` table
    (same one stage 2 reads). Returns the stage-log payload.
    """
    events = load_funnel_events(revenue_dir, lookback_hours=lookback_hours, now=now)
    summary = summarize_funnel(events)
    insight = build_insight(summary)

    stored = False
    if ep_db is not None:
        created_at = (now or datetime.now(UTC)).isoformat()
        await ep_db.execute(
            "INSERT INTO episodes (user_id, narrative, key_topics, decisions, "
            "commitments, emotional_tone, relationship_impact, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "dream_revenue",
                insight,
                json.dumps(["revenue", "funnel", "reflection"]),
                "[]",
                "[]",
                "neutral",
                "none",
                created_at,
            ),
        )
        await ep_db.commit()
        stored = True

    logger.info("Stage 3f (revenue reflection): %s", insight)
    return {
        "status": "ok",
        "insight": insight,
        "stored": stored,
        **{k: v for k, v in summary.items() if isinstance(v, int)},
    }
