"""Outbound triangulation gates (SOUL / Anekantavada) before draft and persuasion."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from brain_os.brain.operator_context_store import build_operator_context_store
from brain_os.brain.triangulation_query_scope import (
    _domain_to_company_hint,
    extract_company_and_contact,
)
from brain_os.config import get_settings
from brain_os.schemas.account_brief import AccountBrief
from brain_os.services.operator_context import company_context_key, record_operator_context_from_brief
from brain_os.services.triangulation_gaps import (
    format_gaps_only_block,
    triangulation_gaps,
)

logger = logging.getLogger(__name__)

TRIANGLE_LEG_KEYS = frozenset({"intent_kb", "relationship_mail", "identity_contact"})

_CONSUMER_DOMAINS = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com",
        "live.com",
        "icloud.com",
        "proton.me",
        "protonmail.com",
    }
)


@dataclass(frozen=True)
class TriangulationCheckResult:
    allowed: bool
    company: str
    contact_email: str | None
    gap_keys: list[str]
    brief: AccountBrief | None
    block_message: str | None
    gaps_block: str | None
    skipped: bool
    skip_reason: str | None = None
    reused_brief: bool = False
    operator_context_run_id: str | None = None


def triangulation_enforce_before_outbound() -> bool:
    return bool(get_settings().app.triangulation_enforce_before_outbound_draft)


def triangulation_block_on_gaps() -> bool:
    return bool(get_settings().app.triangulation_block_on_gaps)


def triangulation_hex_for_outbound() -> bool:
    return bool(get_settings().app.triangulation_hex_for_outbound_draft)


def _consumer_email(email: str) -> bool:
    dom = (email or "").strip().lower().split("@")[-1]
    return dom in _CONSUMER_DOMAINS


def resolve_outbound_target(
    *,
    to_email: str,
    context: str = "",
    company_hint: str = "",
) -> tuple[str | None, str | None, bool]:
    """Return (company, contact_email, should_enforce).

    Skips enforcement for consumer inboxes with no company hint (personal one-offs).
    Explicit ``company_hint`` (CRM / ``--pipeline-company``) wins over domain
    title-case (e.g. ``tri-aster.com`` → ``Tri Aster``).
    """
    contact = (to_email or "").strip().lower() or None
    hint = (company_hint or "").strip()
    company, contact_from_ctx = extract_company_and_contact(
        f"{context}\n{to_email}",
        fallback_contact=contact or "",
        fallback_company=hint,
    )
    if contact_from_ctx:
        contact = contact_from_ctx
    # Operator/CRM company name always beats domain-derived title-case.
    if hint:
        company = hint
    if company and contact and _consumer_email(contact):
        pass
    elif not company and contact and _consumer_email(contact):
        return None, contact, False
    if not company and contact:
        dom = contact.split("@", 1)[-1]
        if dom and dom not in _CONSUMER_DOMAINS:
            company = _domain_to_company_hint(dom)
    should = bool(contact) and bool(company or (contact and not _consumer_email(contact)))
    return company, contact, should


def _blocking_gap_keys(gap_keys: list[str], *, hex_mode: bool) -> list[str]:
    if hex_mode:
        return list(gap_keys)
    return [k for k in gap_keys if k in TRIANGLE_LEG_KEYS]


def format_triangulation_block_message(
    *,
    company: str,
    contact_email: str | None,
    gap_keys: list[str],
    hex_mode: bool,
    brief: AccountBrief | None,
) -> str:
    mode = "hex" if hex_mode else "triangle"
    gaps_block = format_gaps_only_block(
        company=company,
        contact_email=contact_email,
        mode=mode,
        brief=brief,
        gap_keys=gap_keys,
    )
    lines = [
        "Triangulation gate (SOUL / Anekantavada) — draft blocked until legs are filled.",
        f"Company: {company} | Contact: {contact_email or '—'} | Mode: {mode}",
        f"Missing legs: {', '.join(gap_keys)}",
        "",
        "Run: poetry run brain brief "
        f'"{company}"' + (f" --contact {contact_email}" if contact_email else "") + " --json",
        "",
        "Override (operator only): skip_triangulation=true, or force_triangulation=true to rebuild brief.",
    ]
    if gaps_block:
        lines.append(gaps_block)
    return "\n".join(lines)


def triangulation_block_payload(tri: TriangulationCheckResult) -> dict[str, Any]:
    """Structured block envelope for MCP/API consumers."""
    return {
        "blocked": True,
        "triangulation_blocked": True,
        "allowed": False,
        "company": tri.company,
        "contact_email": tri.contact_email,
        "gap_keys": tri.gap_keys,
        "block_message": tri.block_message,
        "gaps_block": tri.gaps_block,
        "reused_brief": tri.reused_brief,
        "operator_context_run_id": tri.operator_context_run_id,
        "skip_reason": tri.skip_reason,
    }


def format_triangulation_block_json(tri: TriangulationCheckResult) -> str:
    return json.dumps(triangulation_block_payload(tri), indent=2, ensure_ascii=False, default=str)


async def _load_reused_brief(
    company: str,
    contact_email: str | None,
    *,
    max_age_hours: float,
) -> tuple[AccountBrief | None, str | None]:
    """Return hydrated brief + operator_context run_id from recent snapshot."""
    ck = company_context_key(company)
    store = build_operator_context_store()
    try:
        rec = await store.get_recent_account_brief(
            ck,
            max_age_hours=max_age_hours,
            contact_email=contact_email,
        )
    except Exception as exc:
        logger.debug("get_recent_account_brief failed: %s", exc)
        return None, None
    if rec is None or not rec.brief_snapshot:
        return None, None
    try:
        brief = AccountBrief.model_validate(rec.brief_snapshot)
        return brief, rec.run_id
    except (ValueError, TypeError) as exc:
        logger.debug("brief_snapshot hydrate failed: %s", exc)
        return None, None


async def check_outbound_triangulation(
    *,
    company: str,
    contact_email: str | None = None,
    pantheon: Any | None = None,
    shared_services: dict[str, Any] | None = None,
    email_processor: Any | None = None,
    hex: bool | None = None,
    deep: bool = False,
    skip_requested: bool = False,
    force_rebuild: bool = False,
    enforce: bool | None = None,
    block_on_gaps: bool | None = None,
) -> TriangulationCheckResult:
    """Run account brief + gap detection; optionally block outbound drafts."""
    co = (company or "").strip()
    contact = (contact_email or "").strip().lower() or None
    cfg = get_settings().app

    do_enforce = triangulation_enforce_before_outbound() if enforce is None else bool(enforce)
    do_block = triangulation_block_on_gaps() if block_on_gaps is None else bool(block_on_gaps)
    use_hex = triangulation_hex_for_outbound() if hex is None else bool(hex)

    if skip_requested:
        return TriangulationCheckResult(
            allowed=True,
            company=co,
            contact_email=contact,
            gap_keys=[],
            brief=None,
            block_message=None,
            gaps_block=None,
            skipped=True,
            skip_reason="skip_requested",
        )

    if not do_enforce or not co:
        return TriangulationCheckResult(
            allowed=True,
            company=co,
            contact_email=contact,
            gap_keys=[],
            brief=None,
            block_message=None,
            gaps_block=None,
            skipped=True,
            skip_reason="not_enforced" if not do_enforce else "no_company",
        )

    brief: AccountBrief | None = None
    gap_keys: list[str] = []
    err: str | None = None
    reused = False
    op_ctx_id: str | None = None
    timeout_s = float(cfg.triangulation_outbound_brief_timeout_s)
    reuse_hours = float(cfg.triangulation_reuse_recent_brief_hours)

    if not force_rebuild and reuse_hours > 0:
        brief, op_ctx_id = await _load_reused_brief(co, contact, max_age_hours=reuse_hours)
        if brief is not None:
            gap_keys = triangulation_gaps(brief, hex=use_hex)
            # Invalidate reuse only when triangle legs are broken on the snapshot.
            # Hex legs may be partial on cache hits; rebuild fresh brief for those.
            blocking_reused = _blocking_gap_keys(gap_keys, hex_mode=False)
            if blocking_reused:
                logger.info(
                    "TRIANGULATION_REUSE_STALE | company=%s gaps=%s — rebuilding brief",
                    co[:40],
                    blocking_reused,
                )
                brief = None
                op_ctx_id = None
                gap_keys = []
            else:
                reused = True
                logger.info(
                    "TRIANGULATION_REUSE | company=%s run_id=%s age_hours<=%s",
                    co[:40],
                    op_ctx_id,
                    reuse_hours,
                )

    if brief is None:
        try:
            from brain_os.services.account_brief import build_account_brief

            brief = await asyncio.wait_for(
                build_account_brief(
                    co,
                    contact_email=contact,
                    deep=deep,
                    use_llm=False,
                    skip_kb=False,
                    skip_mail=False,
                    pantheon=pantheon,
                    shared_services=shared_services,
                    email_processor=email_processor,
                    record_context=False,
                ),
                timeout=timeout_s,
            )
            gap_keys = triangulation_gaps(brief, hex=use_hex)
            op_ctx_id = await record_operator_context_from_brief(
                brief,
                kind="account_brief",
                extra_summary="triangulation_gate",
            )
        except TimeoutError:
            err = "timeout"
        except Exception as exc:
            err = str(exc)[:200]
            logger.warning("outbound triangulation check failed company=%s: %s", co, exc)

    mode_label = "hex" if use_hex else "triangle"
    gaps_block = format_gaps_only_block(
        company=co,
        contact_email=contact,
        mode=mode_label,
        brief=brief if err is None else None,
        gap_keys=gap_keys if brief else None,
    )

    blocking = _blocking_gap_keys(gap_keys, hex_mode=use_hex)
    if err is not None:
        blocking = list(TRIANGLE_LEG_KEYS) if not use_hex else gap_keys or list(TRIANGLE_LEG_KEYS)

    allowed = True
    block_message: str | None = None
    if do_block and blocking:
        allowed = False
        block_message = format_triangulation_block_message(
            company=co,
            contact_email=contact,
            gap_keys=blocking,
            hex_mode=use_hex,
            brief=brief,
        )

    logger.info(
        "TRIANGULATION_ENFORCE | company=%s allowed=%s gaps=%s hex=%s err=%s reused=%s",
        co[:40],
        allowed,
        gap_keys,
        use_hex,
        err,
        reused,
    )

    return TriangulationCheckResult(
        allowed=allowed,
        company=co,
        contact_email=contact,
        gap_keys=gap_keys,
        brief=brief,
        block_message=block_message,
        gaps_block=gaps_block,
        skipped=False,
        skip_reason=err,
        reused_brief=reused,
        operator_context_run_id=op_ctx_id,
    )


# ── Send-primitive triangulation proof (defense in depth) ──────────────────

# Named override tokens for system mail that must never go through account brief.
# Every use is loudly logged. Sales / GTM outbound must NOT use these.
SYSTEM_MAIL_OVERRIDE_TOKENS = frozenset(
    {
        "system_opt_out_confirmation",
        "system_notification",
        "system_recruitment",
        "system_takeout",
        "system_social_digest",
        "system_installed_base",
        "system_consent_ack",
        "system_board_brief",
        "system_board_meeting",
        "system_morning_brief",
        "system_curious_ira",
        "system_operator_review",
        "system_rfq_alert",
        "system_service_alert",
        "system_purchase_order",
    }
)


@dataclass(frozen=True)
class TriangulationSendProof:
    """Capability token required by ``execute_user_initiated_outbound_send``.

    Mint via ``mint_send_proof_from_check`` / ``mint_send_proof_skipped`` after a
    real triangulation pass, or ``mint_system_mail_override`` for named system mail.
    """

    recipient: str
    kind: str  # "checked" | "skipped" | "system_override"
    company: str = ""
    skip_reason: str | None = None
    override_token: str | None = None
    operator_context_run_id: str | None = None
    issued_at: float = 0.0

    def matches_recipient(self, recipient: str) -> bool:
        return (self.recipient or "").strip().lower() == (recipient or "").strip().lower()


def mint_send_proof_from_check(
    tri: TriangulationCheckResult,
    *,
    recipient: str,
) -> TriangulationSendProof | None:
    """Mint a send proof from an allowed triangulation check. Returns None if blocked."""
    if not tri.allowed:
        return None
    recip = (recipient or "").strip().lower()
    if not recip:
        return None
    kind = "skipped" if tri.skipped else "checked"
    return TriangulationSendProof(
        recipient=recip,
        kind=kind,
        company=(tri.company or "").strip(),
        skip_reason=tri.skip_reason,
        operator_context_run_id=tri.operator_context_run_id,
        issued_at=time.time(),
    )


def mint_send_proof_skipped(
    *,
    recipient: str,
    reason: str = "not_enforced",
    company: str = "",
) -> TriangulationSendProof:
    """Mint proof when triangulation is intentionally not required (e.g. consumer inbox)."""
    recip = (recipient or "").strip().lower()
    if not recip:
        raise ValueError("recipient required for triangulation send proof")
    return TriangulationSendProof(
        recipient=recip,
        kind="skipped",
        company=(company or "").strip(),
        skip_reason=(reason or "not_enforced"),
        issued_at=time.time(),
    )


def mint_system_mail_override(
    *,
    token: str,
    recipient: str,
    reason: str,
) -> TriangulationSendProof:
    """Named override for system mail (opt-out ack, digests, recruitment). Loudly logged."""
    tok = (token or "").strip()
    recip = (recipient or "").strip().lower()
    if tok not in SYSTEM_MAIL_OVERRIDE_TOKENS:
        raise ValueError(
            f"Unknown system mail override token {tok!r}. "
            f"Allowed: {sorted(SYSTEM_MAIL_OVERRIDE_TOKENS)}"
        )
    if not recip:
        raise ValueError("recipient required for system mail override")
    logger.warning(
        "TRIANGULATION_SYSTEM_OVERRIDE | token=%s recipient=%s reason=%s",
        tok,
        recip,
        (reason or "")[:200],
    )
    return TriangulationSendProof(
        recipient=recip,
        kind="system_override",
        override_token=tok,
        skip_reason=(reason or "system_override")[:300],
        issued_at=time.time(),
    )


def validate_triangulation_send_proof(
    proof: TriangulationSendProof | None,
    *,
    recipient: str,
) -> dict[str, Any] | None:
    """Return a block payload if proof is missing/invalid; else None (may send)."""
    recip = (recipient or "").strip().lower()
    if proof is None:
        logger.error(
            "OUTBOUND_SEND_BLOCKED | triangulation_proof_required recipient=%s",
            recip,
        )
        return {
            "blocked": True,
            "triangulation_proof_required": True,
            "error": "triangulation_proof_required",
            "recipient": recip,
            "message": (
                "Send blocked at the outbound primitive: triangulation_proof is required. "
                "Call obtain_send_triangulation_proof(...) or mint_system_mail_override(...) "
                "and pass triangulation_proof= into execute_user_initiated_outbound_send."
            ),
        }
    if not proof.matches_recipient(recip):
        logger.error(
            "OUTBOUND_SEND_BLOCKED | triangulation_proof_recipient_mismatch proof=%s send=%s",
            proof.recipient,
            recip,
        )
        return {
            "blocked": True,
            "triangulation_proof_required": True,
            "error": "triangulation_proof_recipient_mismatch",
            "recipient": recip,
            "proof_recipient": proof.recipient,
            "message": ("Send blocked: triangulation_proof recipient does not match To address."),
        }
    if proof.kind == "system_override":
        if proof.override_token not in SYSTEM_MAIL_OVERRIDE_TOKENS:
            return {
                "blocked": True,
                "triangulation_proof_required": True,
                "error": "invalid_system_override_token",
                "recipient": recip,
                "message": "Send blocked: invalid system mail override token on proof.",
            }
        return None
    if proof.kind not in ("checked", "skipped"):
        return {
            "blocked": True,
            "triangulation_proof_required": True,
            "error": "invalid_triangulation_proof_kind",
            "recipient": recip,
            "message": f"Send blocked: unknown triangulation proof kind {proof.kind!r}.",
        }
    return None


async def obtain_send_triangulation_proof(
    *,
    to_email: str,
    context: str = "",
    company_hint: str = "",
    pantheon: Any | None = None,
    shared_services: dict[str, Any] | None = None,
    email_processor: Any | None = None,
    skip_requested: bool = False,
    force_rebuild: bool = False,
) -> tuple[TriangulationSendProof | None, dict[str, Any] | None]:
    """Resolve target, run triangulation when needed, mint send proof.

    Returns ``(proof, None)`` when send may proceed, or ``(None, block_dict)``.
    """
    company, contact, should_enforce = resolve_outbound_target(
        to_email=to_email,
        context=context,
        company_hint=company_hint,
    )
    recip = (to_email or "").strip().lower()
    if not recip:
        return None, {
            "blocked": True,
            "triangulation_proof_required": True,
            "error": "invalid_recipient",
            "message": "Cannot mint triangulation proof without a recipient.",
        }

    if not should_enforce or not company:
        return (
            mint_send_proof_skipped(
                recipient=recip,
                reason="consumer_or_no_company" if not company else "not_required",
                company=company or "",
            ),
            None,
        )

    from brain_os.services.outbound_operational_gates import refuse_triangulation_skip

    skip_msg = refuse_triangulation_skip(bool(skip_requested))
    if skip_msg is not None:
        return None, {
            "blocked": True,
            "triangulation_blocked": True,
            "error": "operational_triangulation_skip_refused",
            "message": skip_msg,
            "recipient": recip,
        }

    tri = await check_outbound_triangulation(
        company=company,
        contact_email=contact,
        pantheon=pantheon,
        shared_services=shared_services,
        email_processor=email_processor,
        skip_requested=bool(skip_requested),
        force_rebuild=bool(force_rebuild),
    )
    proof = mint_send_proof_from_check(tri, recipient=recip)
    if proof is None:
        return None, triangulation_block_payload(tri)
    return proof, None
