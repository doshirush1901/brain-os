"""Dream report writer — human-readable cycle summaries (Phase 3 biology upgrade).

Each dream cycle appends a ~10-line markdown summary to
``data/dream_reports/YYYY-MM-DD.md`` so the operator can read what Brain OS
dreamt without parsing ``data/dream_log.json``. A weekly heartbeat action
(``dream_digest_weekly``) rolls the last 7 days into one digest and posts it
to the operator webhook (Slack).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DREAM_WEBHOOK_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    ImportError,
    httpx.HTTPError,
)

DEFAULT_REPORTS_DIR = Path("data/dream_reports")


def write_dream_report(
    stage_log: dict[str, Any],
    report: Any,
    *,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    now: datetime | None = None,
) -> Path:
    """Append a short human-readable cycle summary to today's report file."""
    now = now or datetime.now(UTC)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{now.date().isoformat()}.md"

    metrics = stage_log.get("metrics", {})
    stages = stage_log.get("stages", {})
    gaps = list(getattr(report, "gaps_identified", []) or [])
    connections = list(getattr(report, "creative_connections", []) or [])
    campaign = list(getattr(report, "campaign_insights", []) or [])
    revenue = stages.get("3f_revenue_reflection", {})

    lines = [
        f"## Dream cycle — {now.strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Stages: {metrics.get('stages_ok', 0)} ok / "
        f"{metrics.get('stages_error', 0)} error / "
        f"{metrics.get('stages_skipped', 0)} skipped / "
        f"{metrics.get('stages_timeout', 0)} timeout "
        f"(of {metrics.get('stages_total', len(stages))})",
        f"- Memories consolidated: {getattr(report, 'memories_consolidated', 0)}",
        f"- Knowledge gaps: {len(gaps)}" + (f" — top: {gaps[0][:120]}" if gaps and gaps[0] else ""),
        f"- Creative connections: {len(connections)}"
        + (f" — top: {connections[0][:120]}" if connections and connections[0] else ""),
        f"- Campaign insights: {len(campaign)}"
        + (f" — top: {campaign[0][:120]}" if campaign and campaign[0] else ""),
    ]
    if revenue.get("insight"):
        lines.append(f"- Revenue reflection: {str(revenue['insight'])[:160]}")
    errored = [k for k, v in stages.items() if isinstance(v, dict) and v.get("status") == "error"]
    if errored:
        lines.append(f"- Errored stages: {', '.join(errored[:6])}")
    lines.append("")

    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info("Dream report appended to %s", path)
    return path


def build_weekly_digest(
    *,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    days: int = 7,
    now: datetime | None = None,
) -> str:
    """Concatenate the last *days* of dream reports into one digest string."""
    now = now or datetime.now(UTC)
    reports_dir = Path(reports_dir)
    sections: list[str] = []
    for offset in range(days - 1, -1, -1):
        day = (now - timedelta(days=offset)).date().isoformat()
        path = reports_dir / f"{day}.md"
        if path.exists():
            try:
                sections.append(path.read_text(encoding="utf-8").strip())
            except OSError:
                continue
    if not sections:
        return "No dream reports in the last week — dream cycle may not be running."
    return "\n\n".join(sections)


def run_dream_digest_weekly(
    *,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    days: int = 7,
    webhook: bool = True,
) -> dict[str, Any]:
    """Build the weekly dream digest and post it to the operator webhook."""
    digest = build_weekly_digest(reports_dir=reports_dir, days=days)
    sent = False
    if webhook:
        try:
            from brain_os.config import get_settings
            from brain_os.services.operator_webhook import fire_operator_webhook

            url = get_settings().app.operator_webhook_url.get_secret_value().strip()
            if url:
                fire_operator_webhook(
                    "dream_digest_weekly",
                    {"text": digest[:8000], "days": days},
                )
                sent = True
            else:
                logger.debug("Dream digest webhook skipped (no OPERATOR_WEBHOOK_URL)")
        except _DREAM_WEBHOOK_ERRORS:
            logger.exception("Dream digest webhook failed")
    return {"digest_chars": len(digest), "webhook_sent": sent, "days": days}
