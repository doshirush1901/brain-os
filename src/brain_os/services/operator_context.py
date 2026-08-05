"""Record and query operator context runs (context-graph precedents)."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from brain_os.brain.context_graph_sync import sync_operator_context_run
from brain_os.brain.knowledge_graph import normalize_entity_name
from brain_os.brain.operator_context_store import (
    OperatorContextStore,
    build_operator_context_store,
    new_operator_context_run_id,
    operator_context_enabled,
)
from brain_os.brain.run_record_link import append_run_link
from brain_os.config import get_settings
from brain_os.schemas.account_brief import AccountBrief
from brain_os.schemas.operator_context import (
    ContextPrecedent,
    OperatorContextKind,
    OperatorContextOutcome,
    OperatorContextRun,
)

_PRECEDENT_KIND_LABEL = {
    "account_brief": "account brief",
    "tinder_draft": "Tinder draft",
    "quote_prep": "quote prep",
    "persuasion_sprint": "persuasion sprint",
}

logger = logging.getLogger(__name__)


def company_context_key(company_name: str) -> str:
    return normalize_entity_name((company_name or "").strip())


def _redact_email(email: str | None) -> str | None:
    raw = (email or "").strip()
    if not raw:
        return None
    if not get_settings().app.run_record_redact_pii:
        return raw[:256]
    digest = hashlib.sha256(f"email:{raw}".encode()).hexdigest()[:12]
    return f"redacted:{digest}"


def _brief_summary(brief: AccountBrief) -> str:
    parts: list[str] = ["brief"]
    if brief.crm and brief.crm.stage:
        parts.append(f"stage={brief.crm.stage}")
    if brief.crm and brief.crm.machine_model:
        parts.append(f"machine={brief.crm.machine_model}")
    if brief.mail and brief.mail.last_received:
        parts.append("mail_in")
    if brief.risks_gaps:
        parts.append(f"gaps={len(brief.risks_gaps)}")
    return ": ".join(parts)[:400]


async def precedents_block_for_company(
    company_name: str,
    *,
    domain: str | None = None,
    machine_model: str | None = None,
    exclude_run_id: str | None = None,
) -> str:
    """Plain-text precedent block for Calliope / persuasion prompts."""
    from brain_os.schemas.account_brief import AccountBrief, CrmSnapshot

    name = (company_name or "").strip()
    if not name:
        return ""
    crm = None
    if machine_model:
        crm = CrmSnapshot(company_name=name, machine_model=machine_model.strip())
    brief = AccountBrief(company_name=name, domain=domain, crm=crm)
    precedents = await fetch_context_precedents(brief, exclude_run_id=exclude_run_id)
    return format_precedents_for_calliope(precedents)


async def fetch_context_precedents(
    brief: AccountBrief,
    *,
    exclude_run_id: str | None = None,
    limit: int | None = None,
    store: OperatorContextStore | None = None,
) -> list[ContextPrecedent]:
    """Prior operator runs for this account (fail-open empty list)."""
    if not operator_context_enabled():
        return []
    lim = limit if limit is not None else int(get_settings().app.operator_context_precedent_limit)
    try:
        st = store or build_operator_context_store()
        return await st.list_precedents(
            company_key=company_context_key(brief.company_name),
            domain=brief.domain,
            machine_model=brief.crm.machine_model if brief.crm else None,
            exclude_run_id=exclude_run_id,
            limit=lim,
        )
    except Exception:
        logger.debug("fetch_context_precedents failed", exc_info=True)
        return []


async def record_operator_context_from_brief(
    brief: AccountBrief,
    *,
    kind: OperatorContextKind = "account_brief",
    outcome: OperatorContextOutcome = "ok",
    pipeline_run_id: str | None = None,
    extra_summary: str = "",
    store: OperatorContextStore | None = None,
) -> str | None:
    """Persist a context run after brief assembly; returns run_id or None."""
    if not operator_context_enabled():
        return None
    summary = _brief_summary(brief)
    if extra_summary.strip():
        summary = f"{summary}; {extra_summary.strip()}"[:400]
    return await record_operator_context(
        company_name=brief.company_name,
        kind=kind,
        domain=brief.domain,
        contact_email=brief.contact_email,
        machine_model=brief.crm.machine_model if brief.crm else None,
        crm_stage=brief.crm.stage if brief.crm else None,
        outcome=outcome,
        summary=summary,
        pipeline_run_id=pipeline_run_id,
        brief_snapshot=brief.model_dump(mode="json"),
        store=store,
    )


async def record_operator_context(
    *,
    company_name: str,
    kind: OperatorContextKind,
    domain: str | None = None,
    contact_email: str | None = None,
    machine_model: str | None = None,
    crm_stage: str | None = None,
    outcome: OperatorContextOutcome = "ok",
    success: bool | None = None,
    summary: str = "",
    pipeline_run_id: str | None = None,
    brief_snapshot: dict[str, Any] | None = None,
    store: OperatorContextStore | None = None,
) -> str | None:
    """Write one operator context run (fail-open)."""
    if not operator_context_enabled():
        return None
    name = (company_name or "").strip()
    if not name:
        return None
    run_id = new_operator_context_run_id()
    record = OperatorContextRun(
        run_id=run_id,
        ts=time.time(),
        kind=kind,
        company_key=company_context_key(name),
        company_name=name,
        domain=(domain or "").strip().lower() or None,
        contact_email=_redact_email(contact_email),
        machine_model=(machine_model or "").strip() or None,
        crm_stage=(crm_stage or "").strip() or None,
        outcome=outcome,
        success=success,
        pipeline_run_id=(pipeline_run_id or "").strip() or None,
        summary=(summary or "")[:400],
        brief_snapshot=brief_snapshot,
    )
    try:
        st = store or build_operator_context_store()
        await st.save(record)
        await sync_operator_context_run(record)
        rid = (pipeline_run_id or "").strip()
        if rid:
            await append_run_link(
                rid,
                kind="operator_context",
                detail={
                    "operator_context_run_id": run_id,
                    "company": name,
                    "operator_kind": kind,
                },
            )
        return run_id
    except Exception:
        logger.debug("record_operator_context failed company=%s", name[:80], exc_info=True)
        return None


async def mark_operator_context_sent(
    company_name: str,
    *,
    kind: OperatorContextKind = "tinder_draft",
    store: OperatorContextStore | None = None,
) -> None:
    """Mark latest draft context run for company as sent/success (fail-open)."""
    if not operator_context_enabled():
        return
    ck = company_context_key(company_name)
    try:
        st = store or build_operator_context_store()
        run_id = await st.latest_run_id_for_company(ck, kind=kind)
        if run_id:
            await st.mark_success(run_id, outcome="sent", success=True)
            updated = await st.get(run_id)
            if updated is not None:
                await sync_operator_context_run(updated)
    except Exception:
        logger.debug("mark_operator_context_sent failed", exc_info=True)


def precedents_from_intel(intel: dict[str, Any]) -> list[ContextPrecedent]:
    """Deserialize precedents stored on a Tinder company_intel card."""
    out: list[ContextPrecedent] = []
    for row in intel.get("precedents") or []:
        if not isinstance(row, dict):
            continue
        try:
            out.append(ContextPrecedent.model_validate(row))
        except Exception:
            continue
    return out


def format_precedents_for_calliope(precedents: list[ContextPrecedent]) -> str:
    """Compact block for Calliope / Tinder prompts (plain text, no markdown)."""
    if not precedents:
        return ""
    lines = [
        "PRIOR OPERATOR PRECEDENTS (reuse approach when similar; do not invent facts):",
    ]
    for p in precedents[:3]:
        kind = _PRECEDENT_KIND_LABEL.get(p.kind, p.kind.replace("_", " "))
        ok = " [sent/ok]" if p.success is True else ""
        when = ""
        if p.ts:
            from datetime import UTC, datetime

            when = datetime.fromtimestamp(p.ts, tz=UTC).strftime("%Y-%m-%d")
        lines.append(
            f"- {kind}{ok} ({p.match_reason or 'prior'}{', ' + when if when else ''}): "
            f"{p.summary or 'no summary'}"
        )
    return "\n".join(lines)


async def mark_operator_context_from_feedback(
    *,
    run_id: str | None = None,
    query: str = "",
    company_name: str = "",
) -> int:
    """After praise/positive feedback, mark linked operator context runs successful."""
    if not operator_context_enabled():
        return 0
    marked = 0
    op_ids: list[str] = []

    rid = (run_id or "").strip()
    if rid:
        try:
            from brain_os.brain.run_record_access import fetch_run_record

            record = await fetch_run_record(rid)
            if record is not None:
                for link in record.artifacts.links or []:
                    if not isinstance(link, dict):
                        continue
                    if str(link.get("kind") or "") == "operator_context":
                        oid = str(link.get("operator_context_run_id") or "").strip()
                        if oid:
                            op_ids.append(oid)
        except Exception:
            logger.debug("mark_operator_context: run record load failed", exc_info=True)

    name = (company_name or "").strip()
    if not name and query.strip():
        try:
            from brain_os.brain.triangulation_query_scope import extract_company_and_contact

            hint, _contact = extract_company_and_contact(query)
            if hint:
                name = hint.strip()
        except Exception:
            logger.debug("mark_operator_context: company extract failed", exc_info=True)

    try:
        st = build_operator_context_store()
        seen: set[str] = set()
        for oid in op_ids:
            if oid in seen:
                continue
            seen.add(oid)
            if await st.mark_success(oid, outcome="ok", success=True):
                marked += 1
                rec = await st.get(oid)
                if rec is not None:
                    await sync_operator_context_run(rec)

        if marked == 0 and name:
            for kind in (
                "tinder_draft",
                "persuasion_sprint",
                "account_brief",
                "quote_prep",
            ):
                latest = await st.latest_run_id_for_company(
                    company_context_key(name),
                    kind=kind,  # type: ignore[arg-type]
                )
                if latest and latest not in seen:
                    seen.add(latest)
                    if await st.mark_success(latest, outcome="ok", success=True):
                        marked += 1
                        rec = await st.get(latest)
                        if rec is not None:
                            await sync_operator_context_run(rec)
                        break
    except Exception:
        logger.debug("mark_operator_context_from_feedback failed", exc_info=True)
    return marked


async def attach_precedents_to_brief(
    brief: AccountBrief,
    *,
    current_run_id: str | None = None,
) -> AccountBrief:
    precedents = await fetch_context_precedents(brief, exclude_run_id=current_run_id)
    if not precedents:
        return brief
    return brief.model_copy(update={"precedents": precedents})
