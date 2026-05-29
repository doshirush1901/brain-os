"""Lightweight deterministic answers for common operator queries (no full pipeline)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OperatorRouteKind(str, Enum):
    REVENUE_BRIEF = "revenue_brief"
    REVENUE_DESK = "revenue_desk"
    STALE_LEADS = "stale_leads"
    HOT_BOARD = "hot_board"
    LEAD_TIME_DEMO = "lead_time_pf1"


@dataclass(frozen=True, slots=True)
class OperatorRouteMatch:
    kind: OperatorRouteKind


_REVENUE_BRIEF = re.compile(
    r"\b(revenue\s+brief|must[\s-]?act\s+today|who\s+should\s+i\s+(call|email)\s+today)\b",
    re.I,
)
_REVENUE_DESK = re.compile(
    r"\b(revenue\s+desk|desk\s+status|operating\s+desk)\b",
    re.I,
)
_STALE_LEADS = re.compile(
    r"\b(stale\s+leads?|leads?\s+idle|follow[\s-]?ups?\s+due)\b",
    re.I,
)
_HOT_BOARD = re.compile(
    r"\b(hot\s+leads?\s+board|hot\s+board|what(?:'s|\s+is)\s+on\s+the\s+board)\b",
    re.I,
)
_LEAD_TIME = re.compile(
    r"\b(lead\s*time|delivery\s+time).{0,40}\bPF1\b|\bPF1\b.{0,40}\b(lead\s*time|delivery)\b",
    re.I,
)


def match_operator_query(query: str) -> OperatorRouteMatch | None:
    text = (query or "").strip()
    if not text or len(text) > 280:
        return None
    if _REVENUE_BRIEF.search(text):
        return OperatorRouteMatch(OperatorRouteKind.REVENUE_BRIEF)
    if _REVENUE_DESK.search(text):
        return OperatorRouteMatch(OperatorRouteKind.REVENUE_DESK)
    if _HOT_BOARD.search(text):
        return OperatorRouteMatch(OperatorRouteKind.HOT_BOARD)
    if _STALE_LEADS.search(text):
        return OperatorRouteMatch(OperatorRouteKind.STALE_LEADS)
    if _LEAD_TIME.search(text):
        return OperatorRouteMatch(OperatorRouteKind.LEAD_TIME_DEMO)
    return None


async def resolve_operator_route(kind: OperatorRouteKind) -> str:
    if kind == OperatorRouteKind.REVENUE_BRIEF:
        return await _resolve_revenue_brief()
    if kind == OperatorRouteKind.REVENUE_DESK:
        return _resolve_revenue_desk()
    if kind == OperatorRouteKind.STALE_LEADS:
        return await _resolve_stale_leads()
    if kind == OperatorRouteKind.HOT_BOARD:
        return _resolve_hot_board()
    if kind == OperatorRouteKind.LEAD_TIME_PF1:
        return await _resolve_lead_time_pf1()
    return ""


async def _resolve_revenue_brief() -> str:
    from brain_os.config import get_settings
    from brain_os.services.revenue_mode_desk import format_revenue_brief_text

    mode = get_settings().ira_revenue_mode
    return format_revenue_brief_text(revenue_mode_label=mode)


def _resolve_revenue_desk() -> str:
    from brain_os.config import get_settings
    from brain_os.services.revenue_mode_desk import revenue_brief_payload

    mode = get_settings().ira_revenue_mode
    payload = revenue_brief_payload(revenue_mode_label=mode)
    must = len(payload.get("must_act_today") or [])
    drafts = len(payload.get("drafts_awaiting_approval") or [])
    return (
        f"Revenue desk snapshot ({mode}): {must} must-act today, "
        f"{drafts} drafts awaiting approval. "
        "Open: poetry run brain revenue desk — or data/revenue_mode/revenue_desk.html"
    )


async def _resolve_stale_leads() -> str:
    try:
        from brain_os.data.crm import CRMDatabase

        crm = CRMDatabase()
        await crm.create_tables()
        stale = await crm.get_stale_leads(days=14)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning("stale leads lookup failed: %s", exc)
        return "Stale leads: CRM unavailable offline. Try: poetry run brain crm deals --json"
    if not stale:
        return "No stale leads in CRM (14-day window)."
    lines = ["Stale leads (14d, top 10):"]
    for row in stale[:10]:
        name = row.get("name") or row.get("email") or "?"
        company = row.get("company_name") or row.get("company") or ""
        lines.append(f"- {name} ({company})".strip())
    return "\n".join(lines)


def _resolve_hot_board() -> str:
    from brain_os.services.hot_leads_board_sync import load_board_rows_raw

    rows = load_board_rows_raw()
    if not rows:
        return (
            "Hot leads board is empty. Sync with: poetry run brain leads board sync "
            "(see docs/PERSUASION_SPRINT.md)."
        )
    lines = ["Hot leads board (top 12):"]
    for row in rows[:12]:
        company = row.get("company_name") or "?"
        bucket = row.get("bucket") or ""
        action = (row.get("recommended_action") or "")[:80]
        lines.append(f"- {company} | {bucket} | {action}")
    return "\n".join(lines)


async def _resolve_lead_time_pf1() -> str:
    try:
        from brain_os.brain.truth_hints import TruthHintsEngine

        engine = TruthHintsEngine()
        await engine._load()
        hint = engine.match("What is the lead time for a PF1?")
        if hint is not None:
            answer = str(hint.get("answer") or "").strip()
            if answer:
                return answer
    except (ImportError, OSError, AttributeError, ValueError, TypeError) as exc:
        logger.debug("truth hint DEMO lead time failed: %s", exc)
    return (
        "DEMO lead time varies by configuration and current production load. "
        "Check the latest quote or production schedule in CRM; for a precise "
        "number, ask Hephaestus with a specific DEMO model and region."
    )
