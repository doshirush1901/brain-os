"""Append gaps-only triangulation block after pipeline shape (optional)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from brain_os.brain.triangulation_query_scope import (
    TriangulationScope,
    classify_triangulation_scope,
    extract_company_and_contact,
)
from brain_os.config import get_settings
from brain_os.services.triangulation_gaps import format_gaps_only_block, triangulation_gaps

logger = logging.getLogger(__name__)

_BRIEF_TOOL_MARKERS = (
    "get_account_brief",
    "build_account_brief",
    "account_brief",
)


def _channel_allowed(channel: str) -> bool:
    cfg = get_settings().app
    raw = (cfg.pipeline_triangulation_gaps_channel_allowlist or "cli,api,cursor").strip()
    allowed = {c.strip().lower() for c in raw.split(",") if c.strip()}
    return (channel or "cli").strip().lower() in allowed


def _agent_already_ran_brief(tool_audit: list[dict[str, Any]]) -> bool:
    for row in tool_audit:
        tool = str(row.get("tool") or row.get("name") or "").lower()
        if any(m in tool for m in _BRIEF_TOOL_MARKERS):
            return True
    return False


async def maybe_append_triangulation_gaps_block(
    *,
    shaped: str,
    raw_input: str,
    pantheon: Any,
    channel: str,
    contact_email: str,
    tool_audit: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
    trace: dict[str, Any] | None,
    logger: logging.Logger,
) -> str:
    """Append gaps-only table when config + scope match; fail-open."""
    cfg = get_settings().app
    meta = metadata if isinstance(metadata, dict) else {}
    trace_out = trace if isinstance(trace, dict) else {}

    if not cfg.pipeline_triangulation_gaps_enabled:
        trace_out["triangulation_gaps"] = {"applied": False, "reason": "disabled"}
        return shaped

    if not _channel_allowed(channel):
        trace_out["triangulation_gaps"] = {"applied": False, "reason": "channel_not_allowed"}
        return shaped

    if meta.get("skip_triangulation_gaps") is True:
        trace_out["triangulation_gaps"] = {"applied": False, "reason": "metadata_skip"}
        return shaped

    if cfg.pipeline_triangulation_gaps_skip_if_agent_ran_brief and _agent_already_ran_brief(
        tool_audit
    ):
        trace_out["triangulation_gaps"] = {"applied": False, "reason": "agent_ran_brief"}
        return shaped

    scope: TriangulationScope = classify_triangulation_scope(
        raw_input,
        hex_enabled=bool(cfg.pipeline_triangulation_gaps_hex),
        skip_requested=bool(meta.get("skip_triangulation_gaps")),
    )

    if not scope.applies:
        trace_out["triangulation_gaps"] = {"applied": False, "reason": scope.reason}
        return shaped

    company, contact = extract_company_and_contact(
        raw_input,
        fallback_contact=contact_email,
        fallback_company=str(meta.get("pipeline_company_name") or ""),
    )
    if scope.company_hint:
        company = scope.company_hint
    if scope.contact_email_hint:
        contact = scope.contact_email_hint

    if not company:
        block = format_gaps_only_block(
            company="(unknown)",
            contact_email=contact,
            mode=scope.mode,
            brief=None,
            unresolved_company=True,
        )
        if block:
            shaped += block
        trace_out["triangulation_gaps"] = {
            "applied": True,
            "reason": "unresolved_company",
            "mode": scope.mode,
        }
        return shaped

    t0 = time.monotonic()
    brief = None
    gap_keys: list[str] = []
    err: str | None = None
    try:
        from brain_os.pipeline_phases.outreach import get_email_processor
        from brain_os.services.account_brief import build_account_brief

        proc = None
        try:
            proc = get_email_processor(pantheon)
        except Exception:
            logger.debug("triangulation gaps: no email processor", exc_info=True)

        pipeline_run_id = str(meta.get("pipeline_run_id") or "").strip() or None
        brief = await asyncio.wait_for(
            build_account_brief(
                company,
                contact_email=contact,
                deep=False,
                use_llm=False,
                skip_kb=False,
                skip_mail=False,
                pantheon=pantheon,
                email_processor=proc,
                pipeline_run_id=pipeline_run_id,
            ),
            timeout=float(cfg.pipeline_triangulation_gaps_timeout_s),
        )
        gap_keys = triangulation_gaps(brief, hex=scope.mode == "hex")
    except TimeoutError:
        err = "timeout"
    except Exception as exc:
        err = str(exc)[:200]
        logger.warning("pipeline triangulation gaps failed company=%s: %s", company, exc)

    block = format_gaps_only_block(
        company=company,
        contact_email=contact,
        mode=scope.mode,
        brief=brief if err is None else None,
        gap_keys=gap_keys if brief else None,
    )
    if block:
        shaped += block

    trace_out["triangulation_gaps"] = {
        "applied": bool(block),
        "company": company,
        "contact": contact,
        "mode": scope.mode,
        "gaps": gap_keys,
        "ms": round((time.monotonic() - t0) * 1000, 1),
        "error": err,
        "classifier_reason": scope.reason,
    }
    logger.info(
        "TRIANGULATION_GAPS | applied=%s company=%s gaps=%s ms=%.0f",
        bool(block),
        company[:40],
        gap_keys,
        trace_out["triangulation_gaps"].get("ms", 0),
    )
    return shaped
