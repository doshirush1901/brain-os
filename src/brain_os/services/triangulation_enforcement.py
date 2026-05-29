"""Outbound triangulation gates (SOUL / Anekantavada) before draft and persuasion."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from brain_os.brain.operator_context_store import OperatorContextStore
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
    """
    contact = (to_email or "").strip().lower() or None
    company, contact_from_ctx = extract_company_and_contact(
        f"{context}\n{to_email}",
        fallback_contact=contact or "",
        fallback_company=(company_hint or "").strip(),
    )
    if contact_from_ctx:
        contact = contact_from_ctx
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
    store = OperatorContextStore()
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
            reused = True
            gap_keys = triangulation_gaps(brief, hex=use_hex)
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
