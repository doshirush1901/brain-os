"""Email + outbound MCP tools.

Three groups of tools fronting Brain OS's outbound email surface:

- **Search / thread** — ``search_emails``, ``read_email_thread``
  (read ``srv._email_processor`` for Gmail introspection)
- **Drafting** — ``draft_email`` (reads ``srv._pantheon`` → Calliope agent)
- **Send** — ``send_email`` (``confirm=True`` gate; mirrors ``execute_user_initiated_outbound_send``)
- **Cadence** — ``email_touch_audit`` (dreamer cooldown / outbound policy)
- **Tinder mode** — ``tinder_mode_start``, ``tinder_mode_status``,
  ``tinder_mode_left``, ``tinder_mode_right_draft``, ``tinder_mode_send``
  (read ``srv._tinder_email_mode`` + ``srv._crm`` + ``srv._email_processor`` +
  ``srv._pantheon``)

``tinder_mode_send`` requires ``confirm=True`` to actually send via Gmail.
The gate lives in the underlying ``_tinder_email_mode.send()`` body.

All handlers go through the ``mcp_server`` lazy facade so test monkeypatches
(``mcp_mod._crm = mock`` etc.) keep working.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool


def _mcp_shared_services(srv: Any) -> dict[str, Any]:
    """Runtime service bag (``mcp_runtime._shared_services``), not legacy ``_services``."""
    return dict(getattr(srv, "_shared_services", None) or {})


logger = logging.getLogger(__name__)


async def search_emails(
    from_address: str = "",
    subject: str = "",
    label: str = "",
    query: str = "",
    after: str = "",
    before: str = "",
    max_results: int = 10,
    mailbox_scope: str = "",
) -> str:
    """Search Acme's Gmail inbox.

    Filter by sender address, subject keywords, Gmail label/folder (e.g. HR, Recruitment CVs),
    free-form query, and date range (YYYY/MM/DD format). Returns matching emails
    with id, from, to, subject, date, and thread_id.
    ``mailbox_scope`` controls mailbox fanout: primary, secondary, or both.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._email_processor is None:
        return "Email processor not available."

    try:
        from brain_os.config import get_settings

        max_results = min(25, max(1, int(max_results)))
        default_scope = str(get_settings().app.mcp_email_default_scope or "primary").strip().lower()
        scope = (mailbox_scope or default_scope).strip().lower()
        if scope not in {"primary", "secondary", "both"}:
            return "mailbox_scope must be one of: primary, secondary, both."
        emails = await srv._email_processor.search_emails(
            from_address=from_address,
            subject=subject,
            label=label,
            query=query,
            after=after,
            before=before,
            max_results=max_results,
            mailbox_scope=scope,
        )
        # Keep MCP payloads compact: full bodies can exceed stdio transport budgets
        # and cause connection drops on broad queries.
        compact_rows: list[dict[str, Any]] = []
        for e in emails:
            row = e.model_dump()
            compact_rows.append(
                {
                    "id": row.get("id"),
                    "from_address": row.get("from_address"),
                    "to_address": row.get("to_address"),
                    "subject": row.get("subject"),
                    "received_at": row.get("received_at"),
                    "thread_id": row.get("thread_id"),
                    "labels": row.get("labels") or [],
                    "source_mailbox": row.get("source_mailbox"),
                }
            )
        return json.dumps(
            compact_rows,
            indent=2,
            default=str,
        )
    except asyncio.CancelledError:
        # Keep MCP process alive when a client aborts a long Gmail call.
        logger.warning("MCP search_emails cancelled by client")
        return "Error: Gmail search was cancelled by client."
    except Exception as exc:
        logger.exception("MCP search_emails failed")
        return f"Error: {exc}"


async def read_email_thread(thread_id: str, mailbox: str = "") -> str:
    """Fetch the full email thread by Gmail thread ID.

    Returns all messages in the thread with sender, recipient,
    subject, body, and timestamps. Pass mailbox (e.g. secondary@example.com)
    when the thread is from the secondary/vendor mailbox.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._email_processor is None:
        return "Email processor not available."

    try:
        emails = await srv._email_processor.get_thread(thread_id, mailbox=mailbox.strip() or None)
        return json.dumps(
            {
                "thread_id": thread_id,
                "message_count": len(emails),
                "messages": [e.model_dump() for e in emails],
            },
            indent=2,
            default=str,
        )
    except asyncio.CancelledError:
        logger.warning("MCP read_email_thread cancelled by client")
        return "Error: Gmail thread read was cancelled by client."
    except Exception as exc:
        logger.exception("MCP read_email_thread failed")
        return f"Error: {exc}"


async def get_account_mail_journey(
    company: str,
    contact_email: str = "",
    domain: str = "",
    after: str = "",
    before: str = "",
    mailbox: str = "",
    include_attachments: bool = True,
    resume: bool = False,
) -> str:
    """Build a full account journey from mailbox history and attachments."""
    from brain_os.config import get_settings
    from brain_os.interfaces import mcp_server as srv
    from brain_os.services.account_mailbox_journey import build_account_mailbox_journey

    await srv._ensure_initialized()
    if srv._email_processor is None:
        return "Email processor not available."
    try:
        cfg = get_settings().app
        journey = await build_account_mailbox_journey(
            email_processor=srv._email_processor,
            company=company,
            contact_email=(contact_email or "").strip() or None,
            domain=(domain or "").strip() or None,
            after=after,
            before=before,
            mailbox=(mailbox or "").strip() or None,
            include_attachments=bool(include_attachments),
            resume=bool(resume),
            checkpoint_path=None,
            max_messages=cfg.account_journey_max_messages,
            max_threads=cfg.account_journey_max_threads,
            max_attachment_chars=cfg.account_journey_max_attachment_chars,
            page_size=cfg.account_journey_page_size,
        )
        return json.dumps(journey.model_dump(mode="json"), indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP get_account_mail_journey failed")
        return f"Error: {exc}"


async def draft_email(
    to: str,
    subject: str,
    context: str,
    as_json: bool = False,
) -> str:
    """Draft a professional email using Brain OS's writing agent (Calliope).

    Provide the recipient, subject, and context/instructions for the email.
    Returns a formatted email draft.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._pantheon is None:
        return "Pantheon not available."

    try:
        from brain_os.services.outbound_touch_audit import audit_outbound_touch, format_touch_audit_block
        from brain_os.services.triangulation_enforcement import (
            check_outbound_triangulation,
            resolve_outbound_target,
        )

        touch = await audit_outbound_touch(recipient_email=to)
        if not touch.allowed:
            return f"Draft blocked (outbound policy): {touch.reason}"
        audit_block = format_touch_audit_block(touch)

        ctx_lower = (context or "").lower()
        skip_tri = "skip_triangulation" in ctx_lower
        force_tri = "force_triangulation" in ctx_lower
        company, contact, should_enforce = resolve_outbound_target(to_email=to, context=context)
        if should_enforce and company:
            email_processor = getattr(srv, "_email_processor", None)
            from brain_os.services.triangulation_enforcement import (
                format_triangulation_block_json,
            )

            tri = await check_outbound_triangulation(
                company=company,
                contact_email=contact,
                pantheon=srv._pantheon,
                email_processor=email_processor,
                skip_requested=skip_tri,
                force_rebuild=force_tri,
            )
            if not tri.allowed:
                if as_json:
                    return format_triangulation_block_json(tri)
                return tri.block_message or "Triangulation gate blocked draft."

        calliope = srv._pantheon.get_agent("calliope")
        if calliope is None:
            return "Calliope agent not found."
        prompt = f"Draft an email to {to} with subject '{subject}'. Context: {context}"
        if audit_block:
            prompt += f"\n\n{audit_block}"
        return await calliope.handle(prompt)
    except Exception as exc:
        logger.exception("MCP draft_email failed")
        return f"Error: {exc}"


async def tinder_mode_start(
    mode: str = "crm_oldest_touch",
    max_cards: int = 200,
    mailbox_max_fetch: int = 120,
    exclude_domains_extra: str = "",
    resume: bool = False,
    no_icp_gate: bool = False,
    include_batch_spray: bool = False,
) -> str:
    """Start Tinder-style outbound review (stable mode 7): queue cards, oldest-touch CRM or sent-mail sample.

    ``mode``: ``crm_oldest_touch`` (default) builds one card per external contact domain from CRM deals,
    stalest ``updated_at`` first. ``mailbox_oldest`` samples recent ``in:sent``, oldest message first,
    one card per recipient domain (not a full archive crawl).

    Set ``resume=true`` to continue a previous session: same queue and cursor, optional pending draft
    preserved (no rebuild). Use ``resume=false`` (default) for a fresh queue.

    Checkpoint path: ``data/conversation/tinder_email_mode.json`` (under ``BRAIN_DATA_DIR``).
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._tinder_email_mode is None:
        return "Tinder email mode service not available."
    try:
        extra = exclude_domains_extra.strip() or None
        result = await srv._tinder_email_mode.start(
            mode=mode,
            crm=srv._crm,
            email_processor=srv._email_processor,
            max_cards=max_cards,
            mailbox_max_fetch=mailbox_max_fetch,
            exclude_domains_extra=extra,
            resume=resume,
            include_batch_spray=include_batch_spray,
            icp_gate=not no_icp_gate,
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP tinder_mode_start failed")
        return f"Error: {exc}"


async def tinder_mode_status() -> str:
    """Return Tinder email mode cursor, queue length, pending draft summary, and current card."""
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._tinder_email_mode is None:
        return "Tinder email mode service not available."
    try:
        return json.dumps(srv._tinder_email_mode.status(), indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP tinder_mode_status failed")
        return f"Error: {exc}"


async def tinder_mode_left(note: str = "") -> str:
    """Skip the current account card (left swipe) and advance. Optional note stored with skip."""
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._tinder_email_mode is None:
        return "Tinder email mode service not available."
    try:
        return json.dumps(srv._tinder_email_mode.left(note=note), indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP tinder_mode_left failed")
        return f"Error: {exc}"


async def tinder_mode_company_card(deep: bool = False, force: bool = False) -> str:
    """Build operator company intel for the current Tinder card (who/where/what). Run before ``tinder_mode_right_draft``."""
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._tinder_email_mode is None:
        return "Tinder email mode service not available."
    if srv._pantheon is None:
        return "Pantheon not available."
    try:
        result = await srv._tinder_email_mode.ensure_company_intel_card(
            pantheon=srv._pantheon,
            email_processor=srv._email_processor,
            shared_services=_mcp_shared_services(srv),
            deep=deep,
            force=force,
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP tinder_mode_company_card failed")
        return f"Error: {exc}"


async def tinder_mode_right_draft(
    edit_instructions: str = "",
    triangulate: bool = True,
    deep: bool = False,
    force_triangulate: bool = False,
) -> str:
    """Draft or revise outbound email for the current card (Calliope). Stores pending draft for send.

    By default runs account-brief **triangulation** (CRM + KB + mail + proof) before drafting.
    Set ``triangulate=false`` to skip (card context only). ``deep=true`` adds Argus dossier.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._tinder_email_mode is None:
        return "Tinder email mode service not available."
    if srv._pantheon is None:
        return "Pantheon not available."
    try:
        result = await srv._tinder_email_mode.right_draft(
            pantheon=srv._pantheon,
            edit_instructions=edit_instructions,
            email_processor=srv._email_processor,
            shared_services=_mcp_shared_services(srv),
            triangulate=triangulate,
            triangulate_deep=deep,
            force_triangulate=force_triangulate,
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP tinder_mode_right_draft failed")
        return f"Error: {exc}"


async def send_email(
    to: str,
    subject: str,
    body: str,
    confirm: bool = False,
    cc: str = "",
    thread_id: str = "",
    from_mailbox: str = "",
    campaign_id: str = "",
    idempotency_key: str = "",
    pipeline_company_name: str = "",
    run_id: str = "",
    dry_run: bool = False,
) -> str:
    """Send one email via Gmail (explicit operator approval only).

    Requires ``confirm=true`` after the user has approved the exact To/Subject/Body in chat.
    Mirrors ``POST /api/email/send`` / ``ira email send`` (blacklist, dedupe, NeverBounce, CRM log).
    Set ``dry_run=true`` to return the payload without calling Gmail.
    """
    from brain_os.interfaces import mcp_server as srv

    to_addr = (to or "").strip()
    subj = (subject or "").strip()
    text = (body or "").strip()
    if not to_addr or not subj or not text:
        return json.dumps(
            {
                "sent": False,
                "error": "missing_fields",
                "message": "to, subject, and body are required.",
            },
            indent=2,
        )

    if dry_run:
        return json.dumps(
            {
                "dry_run": True,
                "to": to_addr,
                "subject": subj,
                "body_chars": len(text),
                "cc": (cc or "").strip() or None,
                "thread_id": (thread_id or "").strip() or None,
                "from_mailbox": (from_mailbox or "").strip() or None,
                "campaign_id": (campaign_id or "").strip() or None,
                "idempotency_key": (idempotency_key or "").strip() or None,
            },
            indent=2,
            default=str,
        )

    if not confirm:
        return json.dumps(
            {
                "sent": False,
                "error": "confirm_required",
                "message": "Set confirm=true only after the operator approved this exact draft in chat.",
            },
            indent=2,
        )

    await srv._ensure_initialized()
    if srv._email_processor is None:
        return "Email processor not available."

    try:
        from brain_os.services.outbound_send import execute_user_initiated_outbound_send

        crm = srv._crm
        if crm is not None:
            await crm.create_tables()
        from brain_os.service_keys import ServiceKey as SK

        interconnections = _mcp_shared_services(srv).get(SK.EMAIL_INTERCONNECTIONS)
        result = await execute_user_initiated_outbound_send(
            srv._email_processor,
            crm,
            interconnections,
            to=to_addr,
            subject=subj,
            body=text,
            cc=(cc or "").strip() or None,
            thread_id=(thread_id or "").strip() or None,
            from_mailbox=(from_mailbox or "").strip() or None,
            campaign_id=(campaign_id or "").strip() or None,
            idempotency_key=(idempotency_key or "").strip() or None,
            revenue_pipeline_company=(pipeline_company_name or "").strip() or None,
            pipeline_run_id=(run_id or "").strip() or None,
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP send_email failed")
        return f"Error: {exc}"


async def email_touch_audit(
    email: str,
    company: str = "",
) -> str:
    """Gmail-backed outbound cadence audit (stable mode 15). Blocks dreamer cooldown violations."""
    from brain_os.interfaces import mcp_server as srv
    from brain_os.services.outbound_touch_audit import audit_outbound_touch

    await srv._ensure_initialized()
    if srv._email_processor is None:
        return "Email processor not available."
    try:
        result = await audit_outbound_touch(
            recipient_email=email,
            company=(company or "").strip(),
            email_processor=srv._email_processor,
        )
        return json.dumps(result.as_dict(), indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP email_touch_audit failed")
        return f"Error: {exc}"


async def tinder_mode_send(confirm: bool = False) -> str:
    """Send the pending Tinder draft via Gmail. Requires ``confirm=true`` (explicit operator approval)."""
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._tinder_email_mode is None:
        return "Tinder email mode service not available."
    try:
        result = await srv._tinder_email_mode.send(
            email_processor=srv._email_processor,
            confirm=confirm,
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        logger.exception("MCP tinder_mode_send failed")
        return f"Error: {exc}"


def register(mcp: FastMCP) -> None:
    """Register email tools on the given FastMCP instance."""
    mcp.tool()(hardened_mcp_tool(search_emails))
    mcp.tool()(hardened_mcp_tool(read_email_thread))
    mcp.tool()(hardened_mcp_tool(get_account_mail_journey))
    mcp.tool()(hardened_mcp_tool(draft_email))
    mcp.tool()(hardened_mcp_tool(send_email))
    mcp.tool()(hardened_mcp_tool(email_touch_audit))
    mcp.tool()(hardened_mcp_tool(tinder_mode_start))
    mcp.tool()(hardened_mcp_tool(tinder_mode_status))
    mcp.tool()(hardened_mcp_tool(tinder_mode_left))
    mcp.tool()(hardened_mcp_tool(tinder_mode_company_card))
    mcp.tool()(hardened_mcp_tool(tinder_mode_right_draft))
    mcp.tool()(hardened_mcp_tool(tinder_mode_send))
