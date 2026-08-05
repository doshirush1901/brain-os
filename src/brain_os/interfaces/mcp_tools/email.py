"""Email + outbound MCP tools.

Three groups of tools fronting Brain OS's outbound email surface:

- **Search / thread** — ``search_emails``, ``read_email_thread``
  (read ``srv._email_processor`` for Gmail introspection)
- **Drafting** — ``draft_email`` (reads ``srv._pantheon`` → Calliope agent)
- **Send** — ``send_email`` (operator ``confirm_nonce`` gate; mirrors
  ``execute_user_initiated_outbound_send``, including ``attachment_paths`` /
  ``archive_from_inbox`` / ``brief_id`` / ``skip_touch_audit``)
- **Cadence** — ``email_touch_audit`` (dreamer cooldown / outbound policy)
- **Tinder mode** — ``tinder_mode_start``, ``tinder_mode_status``,
  ``tinder_mode_left``, ``tinder_mode_right_draft``, ``tinder_mode_send``
  (read ``srv._tinder_email_mode`` + ``srv._crm`` + ``srv._email_processor`` +
  ``srv._pantheon``)

``send_email`` / ``tinder_mode_send`` require a server-issued one-time
``confirm_nonce`` (``confirm=true`` alone is model-suppliable and insufficient).

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

        app = get_settings().app
        result_cap = max(1, min(25, int(app.mcp_email_search_max_results)))
        max_results = min(result_cap, max(1, int(max_results)))
        default_scope = str(app.mcp_email_default_scope or "primary").strip().lower()
        scope = (mailbox_scope or default_scope).strip().lower()
        if scope not in {"primary", "secondary", "both"}:
            return "mailbox_scope must be one of: primary, secondary, both."
        tool_timeout = float(getattr(app, "mcp_tool_timeout_seconds", 60.0))
        search_timeout = min(
            float(app.mcp_email_search_timeout_seconds),
            max(1.0, tool_timeout - 5.0),
        )
        emails = await asyncio.wait_for(
            srv._email_processor.search_emails(
                from_address=from_address,
                subject=subject,
                label=label,
                query=query,
                after=after,
                before=before,
                max_results=max_results,
                mailbox_scope=scope,
            ),
            timeout=search_timeout,
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
    except TimeoutError:
        return (
            f"Error: Gmail search timed out after {search_timeout:.0f}s before Cursor's MCP "
            "transport timeout. Narrow the query with from/to/subject/date filters, reduce "
            "max_results, or raise APP__MCP_EMAIL_SEARCH_TIMEOUT_SECONDS."
        )
    except Exception as exc:  # noqa: BLE001 — MCP tool boundary: failures are returned to the client as text
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
    except Exception as exc:  # noqa: BLE001 — MCP tool boundary: failures are returned to the client as text
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
    except Exception as exc:  # noqa: BLE001 — MCP tool boundary: failures are returned to the client as text
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
        from brain_os.services.outbound_operational_gates import (
            operational_triangulation_skip_block_payload,
            refuse_triangulation_skip,
        )

        skip_msg = refuse_triangulation_skip(skip_tri)
        if skip_msg is not None:
            if as_json:
                import json

                return json.dumps(
                    operational_triangulation_skip_block_payload(),
                    ensure_ascii=False,
                )
            return skip_msg
        company, contact, should_enforce = resolve_outbound_target(to_email=to, context=context)
        outbound_brief = None
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
            outbound_brief = tri.brief

        try:
            from brain_os.config import get_settings
            from brain_os.services.employment_before_mail import (
                accept_unverified_employment_requested,
                enforce_employment_before_mail,
                format_employment_before_mail_block,
                skip_employment_before_mail_requested,
            )

            if (
                get_settings().app.employment_before_mail_enabled
                and not skip_employment_before_mail_requested(context)
            ):
                claimed = (company or "").strip() or ""
                await enforce_employment_before_mail(
                    to_email=to,
                    claimed_company_or_domain=claimed,
                    accept_unverified=accept_unverified_employment_requested(context),
                )
        except Exception as emp_exc:  # noqa: BLE001 — gate errors are re-raised by type; other failures are non-fatal
            from brain_os.services.employment_before_mail import EmploymentBeforeMailError

            if isinstance(emp_exc, EmploymentBeforeMailError):
                if as_json:
                    import json

                    return json.dumps(
                        {
                            "blocked": True,
                            "reason": "employment_before_mail",
                            **emp_exc.result.as_dict(),
                        },
                        ensure_ascii=False,
                    )
                return format_employment_before_mail_block(emp_exc.result)
            logger.debug("employment_before_mail skipped on draft_email", exc_info=True)

        try:
            from brain_os.config import get_settings as _gs_sf
            from brain_os.services.shopfloor_before_customer import (
                ShopfloorBeforeCustomerError,
                accept_unverified_shopfloor_requested,
                enforce_shopfloor_before_customer,
                format_shopfloor_before_customer_block,
                skip_shopfloor_before_customer_requested,
            )

            if (
                _gs_sf().app.shopfloor_before_customer_enabled
                and not skip_shopfloor_before_customer_requested(context)
                and outbound_brief is not None
            ):
                enforce_shopfloor_before_customer(
                    brief=outbound_brief,
                    accept_unverified=accept_unverified_shopfloor_requested(context),
                )
        except Exception as sf_exc:  # noqa: BLE001 — gate errors are re-raised by type; other failures are non-fatal
            from brain_os.services.shopfloor_before_customer import ShopfloorBeforeCustomerError

            if isinstance(sf_exc, ShopfloorBeforeCustomerError):
                if as_json:
                    import json

                    return json.dumps(
                        {
                            "blocked": True,
                            "reason": "shopfloor_before_customer",
                            **sf_exc.result.as_dict(),
                        },
                        ensure_ascii=False,
                    )
                return format_shopfloor_before_customer_block(sf_exc.result)
            logger.debug("shopfloor_before_customer skipped on draft_email", exc_info=True)

        calliope = srv._pantheon.get_agent("calliope")
        if calliope is None:
            return "Calliope agent not found."
        prompt = f"Draft an email to {to} with subject '{subject}'. Context: {context}"
        warmth = None
        try:
            from brain_os.systems.warmth_tone import build_tone_instruction, resolve_contact_warmth

            warmth = await resolve_contact_warmth(
                to, relationship_memory=getattr(srv, "_relationship_memory", None)
            )
            prompt += f"\n\n{build_tone_instruction(warmth)}"
        except Exception:  # noqa: BLE001 — warmth tone injection is best-effort
            logger.debug("Warmth tone injection skipped", exc_info=True)
        try:
            from brain_os.services.founder_voice import build_email_voice_prompt_block_async
            from brain_os.services.gtm_bandit_allocator import compose_message_experiment

            seed = hash(f"{to}|{subject}") & 0xFFFFFFFF
            experiment = compose_message_experiment(seed=seed)
            voice_block = await build_email_voice_prompt_block_async(
                company=str(to or "").split("@")[-1],
                warmth=warmth,
                purpose=str(context or "")[:200],
                experiment_overlay=str(experiment.get("overlay") or ""),
                query_text=f"{to} {subject} {str(context or '')[:200]}",
            )
            if voice_block:
                prompt += f"\n\n{voice_block}"
        except Exception:  # noqa: BLE001 — founder voice injection is best-effort
            logger.debug("Founder voice injection skipped", exc_info=True)
            experiment = {}
        if audit_block:
            prompt += f"\n\n{audit_block}"
        result = await calliope.handle(prompt)
        try:
            from brain_os.config import get_settings
            from brain_os.services.proof_before_prose import (
                check_proof_before_prose,
                format_proof_before_prose_block,
                skip_proof_before_prose_requested,
            )

            if (
                get_settings().app.proof_before_prose_enabled
                and not skip_proof_before_prose_requested(context, prompt)
            ):
                proof = check_proof_before_prose(str(result or ""), subject=subject)
                if proof.blocked:
                    msg = format_proof_before_prose_block(proof)
                    if as_json:
                        import json

                        return json.dumps(
                            {"blocked": True, "reason": "proof_before_prose", **proof.as_dict()},
                            ensure_ascii=False,
                        )
                    return msg
        except Exception:  # noqa: BLE001 — proof-before-prose check is best-effort on draft
            logger.debug("proof_before_prose skipped on draft_email", exc_info=True)
        from brain_os.services.outbound_voice_finalize import finalize_outbound_draft

        try:
            finalized = await finalize_outbound_draft(
                subject=subject,
                body=str(result or ""),
                rewrite=True,
                purpose=str(context or "")[:200],
                warmth=warmth,
            )
        except Exception as fin_exc:  # noqa: BLE001 — finalize failure blocks the draft instead of crashing the tool
            logger.warning("Tim Urban finalize failed on draft_email", exc_info=True)
            if as_json:
                import json

                return json.dumps(
                    {
                        "blocked": True,
                        "reason": "finalize_exception",
                        "send_ready": False,
                        "error": f"{type(fin_exc).__name__}: {fin_exc}",
                        "subject": subject,
                    },
                    ensure_ascii=False,
                    default=str,
                )
            return (
                f"Draft blocked (finalize_exception): {type(fin_exc).__name__}: {fin_exc}. "
                "Re-run draft_email; do not treat the Calliope raw body as send-ready."
            )

        body_out = str(finalized.get("body") or "")
        status = str(finalized.get("status_line") or "")
        lint_block = str(finalized.get("lint_block") or "")
        send_ready = bool(finalized.get("send_ready"))
        chrome = "\n".join(x for x in (status, lint_block) if x)
        if as_json:
            import json

            payload: dict[str, Any] = {
                "subject": finalized.get("subject") or subject,
                "body": body_out,
                "send_ready": send_ready,
                "rewrote": bool(finalized.get("rewrote")),
                "lint": finalized.get("lint"),
                "status_line": status,
            }
            if experiment:
                payload["message_experiment"] = {
                    k: experiment.get(k)
                    for k in (
                        "variant_id",
                        "angle",
                        "subject_style",
                        "send_hour",
                        "bandit_theta",
                    )
                    if experiment.get(k) is not None
                }
            if not send_ready:
                payload["blocked"] = True
                payload["reason"] = "tim_urban_voice"
                payload["status"] = "lint_failed"
            return json.dumps(payload, ensure_ascii=False, default=str)
        if not send_ready:
            return (
                f"Draft blocked (tim_urban_voice / lint_failed).\n{chrome}\n\n"
                f"--- preview (not send-ready) ---\n{body_out}"
            )
        if chrome:
            return f"{body_out}\n\n---\n{chrome}"
        return body_out
    except Exception as exc:  # noqa: BLE001 — MCP tool boundary: failures are returned to the client as text
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
    from_csv: str = "",
    sort: str = "matchmaker_total",
    min_tier: str = "",
    min_score: float = 0.0,
    require_thermoformer: bool = False,
    gauge_tier: str = "",
    allow_repeat: bool = False,
) -> str:
    """Start Tinder-style outbound review (stable mode 7): queue cards, oldest-touch CRM or sent-mail sample.

    ``mode``: ``crm_oldest_touch`` (default) builds one card per external contact domain from CRM deals,
    stalest ``updated_at`` first. ``mailbox_oldest`` samples recent ``in:sent``, oldest message first,
    one card per recipient domain (not a full archive crawl).

    ``from_csv``: path to matchmaker-scored lead CSV (Places → enrich → classify → score pipeline).
    Queue is sorted by ``sort`` (default ``matchmaker_total`` desc) — highest Acme Corp fit first.
    By default only **new companies** (no prior Gmail outbound, no prior Tinder send) and **no domain
    touched in the last 24 hours** or **outbound in the last 7 days**. Pass ``allow_repeat=true`` to
    disable recency gates (testing only).

    Optional ``min_tier`` (A/B/C), ``min_score``, ``require_thermoformer``, ``gauge_tier``.

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
            from_csv=from_csv.strip() or None,
            sort=sort,
            min_tier=min_tier,
            min_score=min_score,
            require_thermoformer=require_thermoformer,
            gauge_tier=gauge_tier,
            allow_repeat=allow_repeat,
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001 — MCP tool boundary: failures are returned to the client as text
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
    except Exception as exc:  # noqa: BLE001 — MCP tool boundary: failures are returned to the client as text
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
    except Exception as exc:  # noqa: BLE001 — MCP tool boundary: failures are returned to the client as text
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
    except Exception as exc:  # noqa: BLE001 — MCP tool boundary: failures are returned to the client as text
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
    Set ``triangulate=false`` to skip (card context only; TRAINING only).
    ``deep=true`` adds Argus dossier. OPERATIONAL refuses triangulation skip.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()
    if srv._tinder_email_mode is None:
        return "Tinder email mode service not available."
    if srv._pantheon is None:
        return "Pantheon not available."
    try:
        from brain_os.services.outbound_operational_gates import (
            operational_triangulation_skip_block_payload,
            refuse_triangulation_skip,
        )

        if refuse_triangulation_skip(not triangulate) is not None:
            return json.dumps(
                {"ok": False, **operational_triangulation_skip_block_payload()},
                indent=2,
                default=str,
            )
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
    except Exception as exc:  # noqa: BLE001 — MCP tool boundary: failures are returned to the client as text
        logger.exception("MCP tinder_mode_right_draft failed")
        return f"Error: {exc}"


def _normalize_attachment_paths(raw: list[str] | str | None) -> list[str] | None:
    """Accept FastMCP list or comma/newline-separated string; drop empties."""
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    else:
        parts = [str(p).strip() for p in raw]
    paths = [p for p in parts if p]
    return paths or None


async def send_email(
    to: str,
    subject: str,
    body: str,
    confirm: bool = False,
    confirm_nonce: str = "",
    cc: str = "",
    thread_id: str = "",
    from_mailbox: str = "",
    campaign_id: str = "",
    idempotency_key: str = "",
    pipeline_company_name: str = "",
    run_id: str = "",
    dry_run: bool = False,
    attachment_paths: list[str] | str | None = None,
    archive_from_inbox: bool | None = None,
    brief_id: str = "",
    skip_touch_audit: bool = False,
    voice_purpose: str = "",
    voice_length_dial: str = "",
) -> str:
    """Send one email via Gmail (explicit operator approval only).

    Two-step gate: first call (no nonce) returns a one-time ``confirm_nonce`` plus
    a human-readable summary; second call must echo that exact nonce. ``confirm=true``
    alone is not enough (models can supply it). Mirrors ``POST /api/email/send`` /
    ``brain email send`` (blacklist, dedupe, CRM log). Set ``dry_run=true`` to return
    the payload without calling Gmail.

    Optional kwargs aligned with the golden path / HTTP send:
    ``attachment_paths`` (repo-relative or absolute paths on the host),
    ``archive_from_inbox``, ``brief_id``, ``skip_touch_audit``.

    Pass ``voice_purpose`` / ``voice_length_dial`` matching the dials the draft was
    finalized with (e.g. ``cold`` / ``intro_long``); otherwise the send gate measures
    the body against the ``soft_short`` default and a long cold intro is refused.
    """
    from brain_os.interfaces import mcp_server as srv
    from brain_os.interfaces.mcp_irreversibility import enforce_mcp_confirm_if_irreversible

    to_addr = (to or "").strip()
    subj = (subject or "").strip()
    text = (body or "").strip()
    attachments = _normalize_attachment_paths(attachment_paths)
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
                "attachment_paths": attachments,
                "archive_from_inbox": archive_from_inbox,
                "brief_id": (brief_id or "").strip() or None,
                "skip_touch_audit": bool(skip_touch_audit),
            },
            indent=2,
            default=str,
        )

    payload = {
        "to": to_addr,
        "subject": subj,
        "body": text,
        "cc": (cc or "").strip(),
        "thread_id": (thread_id or "").strip(),
        "from_mailbox": (from_mailbox or "").strip(),
        "campaign_id": (campaign_id or "").strip(),
        "idempotency_key": (idempotency_key or "").strip(),
        "pipeline_company_name": (pipeline_company_name or "").strip(),
        "run_id": (run_id or "").strip(),
        "attachment_paths": attachments or [],
        "archive_from_inbox": archive_from_inbox,
        "brief_id": (brief_id or "").strip(),
        "skip_touch_audit": bool(skip_touch_audit),
    }
    summary = {
        "action": "send_email",
        "to": to_addr,
        "subject": subj,
        "body_preview": text[:500],
        "body_chars": len(text),
        "cc": payload["cc"] or None,
        "thread_id": payload["thread_id"] or None,
        "from_mailbox": payload["from_mailbox"] or None,
        "attachment_paths": attachments or [],
        "confirm_bool_ignored": bool(confirm),
    }
    gate = enforce_mcp_confirm_if_irreversible(
        "send_email",
        confirm_nonce=confirm_nonce,
        payload=payload,
        summary=summary,
    )
    if gate is not None:
        gate.setdefault("sent", False)
        return json.dumps(gate, indent=2, default=str)

    await srv._ensure_initialized()
    if srv._email_processor is None:
        return "Email processor not available."

    try:
        from brain_os.services.gtm_attribution import default_operator_campaign_id
        from brain_os.services.outbound_send import execute_user_initiated_outbound_send
        from brain_os.services.triangulation_enforcement import obtain_send_triangulation_proof

        crm = srv._crm
        if crm is not None:
            await crm.create_tables()
        from brain_os.service_keys import ServiceKey as SK

        proof, tri_block = await obtain_send_triangulation_proof(
            to_email=to_addr,
            context=f"{pipeline_company_name}\n{subj}",
            company_hint=(pipeline_company_name or "").strip(),
            pantheon=getattr(srv, "_pantheon", None),
            email_processor=srv._email_processor,
        )
        if tri_block is not None:
            tri_block.setdefault("sent", False)
            return json.dumps(tri_block, indent=2, default=str)

        interconnections = _mcp_shared_services(srv).get(SK.EMAIL_INTERCONNECTIONS)
        # Never leave campaign_id empty for a non-system operator send — GTM
        # funnel math needs a greppable id, not ``uncategorized_<caller>``.
        resolved_campaign_id = (campaign_id or "").strip() or default_operator_campaign_id(subj)
        result = await execute_user_initiated_outbound_send(
            srv._email_processor,
            crm,
            interconnections,
            to=to_addr,
            subject=subj,
            body=text,
            triangulation_proof=proof,
            cc=(cc or "").strip() or None,
            thread_id=(thread_id or "").strip() or None,
            from_mailbox=(from_mailbox or "").strip() or None,
            archive_from_inbox=archive_from_inbox,
            attachment_paths=attachments,
            campaign_id=resolved_campaign_id,
            idempotency_key=(idempotency_key or "").strip() or None,
            brief_id=(brief_id or "").strip() or None,
            revenue_pipeline_company=(pipeline_company_name or "").strip() or None,
            pipeline_run_id=(run_id or "").strip() or None,
            skip_touch_audit=bool(skip_touch_audit),
            voice_purpose=(voice_purpose or "").strip(),
            voice_length_dial=(voice_length_dial or "").strip() or None,
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001 — MCP tool boundary: failures are returned to the client as text
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
    except Exception as exc:  # noqa: BLE001 — MCP tool boundary: failures are returned to the client as text
        logger.exception("MCP email_touch_audit failed")
        return f"Error: {exc}"


async def tinder_mode_send(confirm: bool = False, confirm_nonce: str = "") -> str:
    """Send the pending Tinder draft via Gmail.

    Requires a server-issued ``confirm_nonce`` (``confirm=true`` alone is insufficient).
    """
    from brain_os.interfaces import mcp_server as srv
    from brain_os.interfaces.mcp_irreversibility import enforce_mcp_confirm_if_irreversible

    # Refuse before MCP/DB init — confirm failures must not open connections.
    if not confirm:
        return json.dumps(
            {
                "sent": False,
                "error": "confirm_required",
                "message": "Refusing to send without confirm=true (explicit operator approval).",
            },
            indent=2,
        )

    await srv._ensure_initialized()
    if srv._tinder_email_mode is None:
        return "Tinder email mode service not available."
    try:
        from brain_os.service_keys import ServiceKey as SK

        mode = srv._tinder_email_mode
        state = mode._load() if hasattr(mode, "_load") else {}
        pending = (state or {}).get("pending_draft") or {}
        to_addr = str(pending.get("to") or "").strip()
        subj = str(pending.get("subject") or "").strip()
        text = str(pending.get("body") or "").strip()
        payload = {"to": to_addr, "subject": subj, "body": text}
        summary = {
            "action": "tinder_mode_send",
            "to": to_addr or None,
            "subject": subj or None,
            "body_preview": text[:500] if text else None,
            "body_chars": len(text),
            "confirm_bool_ignored": bool(confirm),
        }
        gate = enforce_mcp_confirm_if_irreversible(
            "tinder_mode_send",
            confirm_nonce=confirm_nonce,
            payload=payload,
            summary=summary,
        )
        if gate is not None:
            gate.setdefault("ok", False)
            return json.dumps(gate, indent=2, default=str)

        interconnections = _mcp_shared_services(srv).get(SK.EMAIL_INTERCONNECTIONS)
        if srv._crm is not None:
            await srv._crm.create_tables()
        result = await mode.send(
            email_processor=srv._email_processor,
            confirm=True,
            crm=srv._crm,
            interconnections_service=interconnections,
        )
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001 — MCP tool boundary: failures are returned to the client as text
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
