"""Operator unified approval inbox MCP tools."""

from __future__ import annotations

import json
import logging

from fastapi import HTTPException
from mcp.server.fastmcp import FastMCP

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool
from brain_os.interfaces.server_runtime import _svc
from brain_os.service_keys import ServiceKey as SK
from brain_os.services.ira_daily_dashboard import get_dashboard_today
from brain_os.services.operator_activity_today import (
    format_operator_activity_today_text,
    gather_operator_activity_today,
)
from brain_os.systems.data_dir_lock import get_data_dir
from brain_os.systems.operator_inbox import OperatorInboxService
from brain_os.systems.tinder_email_mode import TinderEmailModeService

logger = logging.getLogger(__name__)


def _maybe_svc(name: str):
    try:
        return _svc(name)
    except HTTPException:
        return None


def _inbox() -> OperatorInboxService:
    svc = _svc(SK.OPERATOR_INBOX)
    if svc is None:
        return OperatorInboxService()
    return svc


def _tinder() -> TinderEmailModeService:
    return TinderEmailModeService(data_root=get_data_dir())


async def operator_inbox_summary(as_json: bool = True) -> str:
    """Pending counts: outbound emails, lead reviews, quote drafts (unified operator queue)."""
    inbox = _inbox()
    payload = await inbox.collect_inbox(
        outbound_approvals=_svc(SK.OUTBOUND_APPROVALS),
        quotes=_svc(SK.QUOTES),
        tinder_service=_tinder(),
    )
    data = payload.model_dump()
    if as_json:
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
    s = data.get("summary") or {}
    return (
        f"{s.get('outbound_email', 0)} outbound · "
        f"{s.get('lead_review', 0)} leads · "
        f"{s.get('quote_draft', 0)} quotes · "
        f"{s.get('external_pending', 0)} external pending"
    )


async def operator_inbox_decide(
    item_id: str,
    decision: str,
    actor: str = "mcp_operator",
    snooze_days: int = 7,
    to_address: str = "",
) -> str:
    """Approve (Gmail draft), reject, or snooze one inbox item. Never auto-sends."""
    dec = decision.strip().lower()
    if dec not in ("approve", "reject", "snooze"):
        return "Error: decision must be approve, reject, or snooze"
    inbox = _inbox()
    result = await inbox.decide(
        item_id=item_id,
        decision=dec,  # type: ignore[arg-type]
        actor=actor,
        snooze_days=snooze_days,
        to_address=to_address.strip() or None,
        outbound_approvals=_svc(SK.OUTBOUND_APPROVALS),
        email_processor=_svc(SK.EMAIL_PROCESSOR),
        tinder_service=_tinder(),
        quotes=_svc(SK.QUOTES),
    )
    return json.dumps(result, indent=2, ensure_ascii=False, default=str)


async def operator_release(
    actor: str = "mcp_operator",
    ttl_hours: float = 4.0,
    force: bool = False,
) -> str:
    """Turn Ira loose — time-boxed release for heartbeat when APP__OPERATOR_RELEASE_REQUIRED=true."""
    from brain_os.config import get_settings

    inbox = _inbox()
    payload = await inbox.collect_inbox(
        outbound_approvals=_svc(SK.OUTBOUND_APPROVALS),
        quotes=_svc(SK.QUOTES),
        tinder_service=_tinder(),
    )
    ext = payload.summary.external_pending
    if ext > 0 and not force:
        return json.dumps(
            {
                "ok": False,
                "error": f"{ext} external_visible pending",
                "summary": payload.summary.model_dump(),
            },
            indent=2,
        )
    session = inbox.save_release(
        actor=actor,
        ttl_hours=ttl_hours or float(get_settings().app.operator_release_ttl_hours),
        pending_at_release=ext,
    )
    return json.dumps({"ok": True, "session": session.model_dump()}, indent=2)


async def operator_dashboard_today(as_json: bool = True) -> str:
    """Daily control room: facts, metrics, and narrative snapshot for operator today."""
    inbox = _inbox()
    inbox_payload = await inbox.collect_inbox(
        outbound_approvals=_maybe_svc(SK.OUTBOUND_APPROVALS),
        quotes=_maybe_svc(SK.QUOTES),
        tinder_service=_tinder(),
    )
    payload = await get_dashboard_today(inbox_payload=inbox_payload.model_dump())
    if as_json:
        return json.dumps(payload.model_dump(), indent=2, ensure_ascii=False, default=str)
    m = payload.metrics
    headline = payload.narrative.big_win if payload.narrative else "(no narrative yet)"
    return (
        f"{payload.local_date} ({payload.timezone}) — "
        f"emails={m.emails_sent_count} agents={m.agents_active_count} "
        f"Metis={m.brain_score} — {headline}"
    )


async def operator_activity_today(limit: int = 40, as_json: bool = True) -> str:
    """Read-only operator activity feed: dream, heartbeat, run records, journal, briefs, inbox."""
    inbox = _inbox()
    inbox_payload = await inbox.collect_inbox(
        outbound_approvals=_maybe_svc(SK.OUTBOUND_APPROVALS),
        quotes=_maybe_svc(SK.QUOTES),
        tinder_service=_tinder(),
    )
    payload = await gather_operator_activity_today(
        limit=max(1, min(int(limit), 200)),
        inbox_payload=inbox_payload.model_dump(),
    )
    if as_json:
        return json.dumps(payload.model_dump(), indent=2, ensure_ascii=False, default=str)
    return format_operator_activity_today_text(payload)


def register(mcp: FastMCP) -> None:
    """Register operator inbox tools on the given FastMCP instance."""
    mcp.tool()(hardened_mcp_tool(operator_inbox_summary))
    mcp.tool()(hardened_mcp_tool(operator_inbox_decide))
    mcp.tool()(hardened_mcp_tool(operator_release))
    mcp.tool()(hardened_mcp_tool(operator_dashboard_today))
    mcp.tool()(hardened_mcp_tool(operator_activity_today))
