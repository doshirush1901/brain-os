"""Account brief orchestration — CRM, mail, KB, proof, optional Argus."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from brain_os.brain.company_similarity import attach_similar_companies_to_brief
from brain_os.brain.context_graph_expand import attach_graph_context_to_brief
from brain_os.config import get_settings
from brain_os.knowledge.outbound_proof_registry import match_artifacts
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.account_brief import (
    AccountBrief,
    AccountBriefLLMSynthesis,
    AccountBriefSynthesisInput,
    AtlasProductionSnapshot,
    CrmSnapshot,
    MailSnapshot,
    ShopfloorTruthSnapshot,
    SourceRef,
    TimelineEvent,
    brief_context_for_llm,
)
from brain_os.schemas.account_journey import AccountJourney
from brain_os.service_keys import ServiceKey as SK
from brain_os.services.account_mailbox_journey import build_account_mailbox_journey
from brain_os.services.llm_client import LLMClient
from brain_os.services.operator_context import (
    attach_precedents_to_brief,
    record_operator_context_from_brief,
)

logger = logging.getLogger(__name__)

_BRIEF_TOTAL_TIMEOUT = 90.0
_CRM_TIMEOUT = 15.0
_MAIL_TIMEOUT = 20.0
_MAIL_DEEP_TIMEOUT = 70.0
_KB_TIMEOUT = 45.0
_ARGUS_TIMEOUT = 60.0

_SYNTHESIS_PROMPT = load_prompt("account_brief_synthesis")


async def build_account_brief(
    company: str,
    *,
    contact_email: str | None = None,
    deep: bool = False,
    use_llm: bool = True,
    skip_kb: bool = False,
    skip_mail: bool = False,
    pantheon: Any | None = None,
    shared_services: dict[str, Any] | None = None,
    email_processor: Any | None = None,
    llm_client: LLMClient | None = None,
    pipeline_run_id: str | None = None,
    record_context: bool = True,
) -> AccountBrief:
    """Assemble a single-account brief for operator prep."""

    async def _inner() -> AccountBrief:
        name = company.strip()
        if not name:
            raise ValueError("company name is required")

        needs_pantheon = (not skip_kb) or deep
        ph = pantheon
        services = dict(shared_services or {})
        if ph is None and needs_pantheon:
            from brain_os.runtime.cli_runtime import _build_pantheon

            ph, services = _build_pantheon()

        async def _pipeline() -> AccountBrief:
            nonlocal email_processor
            proc = email_processor
            if not skip_mail and proc is None and ph is not None:
                from brain_os.runtime.cli_runtime import (
                    _build_digestive,
                    _build_email_processor,
                )

                digestive, _ing, _qd = _build_digestive()
                proc = _build_email_processor(ph, digestive, services)

            crm_data, mail_data, kb_data, proof_data, argus_data = await _gather_sources(
                name,
                contact_email=contact_email,
                deep=deep,
                skip_kb=skip_kb,
                skip_mail=skip_mail,
                pantheon=ph,
                shared_services=services,
                email_processor=proc,
            )

            brief = _assemble_brief(
                name,
                contact_email=contact_email,
                crm_data=crm_data,
                mail_data=mail_data,
                kb_data=kb_data,
                proof_data=proof_data,
                argus_data=argus_data,
            )

            if use_llm:
                brief = await _apply_llm_synthesis(
                    brief,
                    argus_excerpt=str(argus_data.get("excerpt") or ""),
                    llm_client=llm_client,
                )
            else:
                brief = brief.model_copy(
                    update={"executive_summary": _template_executive_summary(brief)}
                )

            brief = await attach_precedents_to_brief(brief)
            brief = await attach_graph_context_to_brief(brief)
            brief = await attach_similar_companies_to_brief(brief)
            if record_context:
                op_id = await record_operator_context_from_brief(
                    brief,
                    kind="account_brief",
                    pipeline_run_id=pipeline_run_id,
                )
                if op_id:
                    brief = brief.model_copy(update={"operator_context_run_id": op_id})
                    brief = await attach_precedents_to_brief(
                        brief,
                        current_run_id=op_id,
                    )
            from brain_os.services.math_mode import (
                attach_math_advisory_if_enabled,
                record_math_shadow_if_enabled,
            )

            record_math_shadow_if_enabled(brief, surface="account_brief")
            brief = attach_math_advisory_if_enabled(brief)
            from brain_os.services.evidence_freshness import attach_evidence_freshness

            brief = attach_evidence_freshness(brief)
            from brain_os.services.account_state import (
                account_state_attach_to_brief_enabled,
                build_account_state_from_brief,
            )

            if account_state_attach_to_brief_enabled():
                from brain_os.services.deal_dynamics_predictor import attach_deal_dynamics_if_enabled
                from brain_os.services.triangulation_gaps import triangulation_gaps

                gaps = triangulation_gaps(brief, card=None, hex=False)
                dynamics = await attach_deal_dynamics_if_enabled(brief)
                brief = brief.model_copy(
                    update={
                        "account_state": build_account_state_from_brief(
                            brief,
                            deal_dynamics=dynamics,
                            triangulation_gaps=gaps,
                        ),
                    },
                )
            return brief

        if ph is not None and (needs_pantheon or deep or (not skip_mail)):
            async with ph:
                return await _pipeline()
        return await _pipeline()

    return await asyncio.wait_for(_inner(), timeout=_BRIEF_TOTAL_TIMEOUT)


async def _gather_sources(
    company: str,
    *,
    contact_email: str | None,
    deep: bool,
    skip_kb: bool,
    skip_mail: bool,
    pantheon: Any | None,
    shared_services: dict[str, Any],
    email_processor: Any | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    tasks: list[tuple[str, Any]] = [
        (
            "crm",
            asyncio.wait_for(
                _fetch_crm(company, contact_email=contact_email),
                timeout=_CRM_TIMEOUT,
            ),
        ),
    ]
    if not skip_mail and email_processor is not None:
        tasks.append(
            (
                "mail",
                asyncio.wait_for(
                    _fetch_mail(email_processor, company, contact_email, deep=deep),
                    timeout=_MAIL_DEEP_TIMEOUT if deep else _MAIL_TIMEOUT,
                ),
            )
        )
    elif not skip_mail:
        tasks.append(("mail", _empty_mail()))

    if not skip_kb and pantheon is not None and shared_services.get(SK.RETRIEVER) is not None:
        tasks.append(
            (
                "kb",
                asyncio.wait_for(
                    _fetch_kb(shared_services[SK.RETRIEVER], company),
                    timeout=_KB_TIMEOUT,
                ),
            )
        )
    else:
        tasks.append(("kb", _empty_kb()))

    tasks.append(("proof", asyncio.to_thread(_fetch_proof, company)))

    if deep and pantheon is not None:
        tasks.append(
            (
                "argus",
                asyncio.wait_for(
                    _fetch_argus(pantheon, company, contact_email),
                    timeout=_ARGUS_TIMEOUT,
                ),
            )
        )
    else:
        tasks.append(("argus", _empty_argus()))

    keys = [t[0] for t in tasks]
    results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

    out: dict[str, dict[str, Any]] = {
        "crm": {},
        "mail": {},
        "kb": {},
        "proof": {},
        "argus": {},
    }
    for key, res in zip(keys, results, strict=True):
        if isinstance(res, Exception):
            logger.warning("account brief source %s failed: %s", key, res)
            if key == "crm":
                out[key] = {"ok": False, "error": str(res)}
            elif key == "argus":
                out[key] = {"excerpt": "", "error": str(res)}
            continue
        out[key] = res if isinstance(res, dict) else {}
    return out["crm"], out["mail"], out["kb"], out["proof"], out["argus"]


async def _fetch_crm(
    company: str,
    *,
    contact_email: str | None = None,
) -> dict[str, Any]:
    from brain_os.data.crm import CRMDatabase
    from brain_os.data.crm_tier1 import crm_source_label
    from brain_os.services.revenue_mode_research import fetch_crm_research_bundle

    bundle = await fetch_crm_research_bundle(company)
    contacts = list(bundle.get("contacts") or [])
    snapshot: CrmSnapshot | None = None
    timeline: list[TimelineEvent] = []
    sources: list[SourceRef] = []

    if bundle.get("ok") and contacts:
        sources.append(SourceRef(channel="crm", label=crm_source_label()))
        want = (contact_email or "").strip().lower()
        primary = contacts[0]
        if want:
            for row in contacts:
                if str(row.get("email") or "").strip().lower() == want:
                    primary = row
                    break
        contact_id = primary.get("id")
        contact_name = str(primary.get("name") or "")
        contact_mail = str(primary.get("email") or "")
        stage = None
        value = None
        currency = None
        machine = None
        last_activity = None

        if contact_id:
            try:
                crm = CRMDatabase()
                await crm.create_tables()
                deals = await crm.get_deals_for_contact(str(contact_id))
                if deals:
                    d0 = deals[0]
                    stage = str(d0.get("stage") or "") or None
                    value = d0.get("value")
                    if value is not None:
                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            value = None
                    currency = str(d0.get("currency") or "") or None
                    machine = str(d0.get("machine_model") or "") or None
                    updated = d0.get("updated_at")
                    if updated is not None:
                        last_activity = str(updated)[:19]
                    timeline.append(
                        TimelineEvent(
                            date=last_activity or "",
                            source="crm",
                            summary=f"Deal: {d0.get('title') or 'untitled'} — {stage or 'stage unknown'}",
                        )
                    )
            except Exception as exc:
                logger.warning("CRM deal lookup failed: %s", exc)

        snapshot = CrmSnapshot(
            company_name=company,
            stage=stage,
            value=value,
            currency=currency,
            machine_model=machine,
            last_activity=last_activity,
            contact_count=len(contacts),
        )
        return {
            "ok": True,
            "snapshot": snapshot,
            "timeline": timeline,
            "sources": sources,
            "contact_name": contact_name,
            "contact_email": contact_mail or None,
            "crm_text": _crm_bundle_text(snapshot, contacts),
        }

    return {
        "ok": bool(bundle.get("ok")),
        "snapshot": snapshot,
        "timeline": timeline,
        "sources": sources if bundle.get("ok") else [],
        "contact_name": None,
        "contact_email": None,
        "crm_text": "",
        "error": bundle.get("error"),
    }


def _crm_bundle_text(snapshot: CrmSnapshot | None, contacts: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    if snapshot:
        parts.append(
            f"Company: {snapshot.company_name}; contacts: {snapshot.contact_count}; "
            f"stage: {snapshot.stage or 'n/a'}; value: {snapshot.value}; "
            f"machine: {snapshot.machine_model or 'n/a'}; last: {snapshot.last_activity or 'n/a'}"
        )
    for c in contacts[:5]:
        parts.append(f"Contact: {c.get('name')} <{c.get('email')}> score={c.get('lead_score', 0)}")
    return "\n".join(parts)


async def _empty_mail() -> dict[str, Any]:
    return {"snapshot": None, "timeline": [], "sources": [], "mail_text": "", "journey": None}


async def _fetch_mail(
    email_processor: Any,
    company: str,
    contact_email: str | None,
    *,
    deep: bool = False,
) -> dict[str, Any]:
    query_parts: list[str] = []
    if contact_email and "@" in contact_email:
        query_parts.append(f"from:{contact_email} OR to:{contact_email}")
    else:
        token = company.split()[0] if company.split() else company
        if len(token) >= 3:
            query_parts.append(token)

    if not query_parts:
        return await _empty_mail()

    query = " ".join(query_parts)
    if deep:
        app_cfg = get_settings().app
        journey = await build_account_mailbox_journey(
            email_processor=email_processor,
            company=company,
            contact_email=contact_email,
            include_attachments=True,
            resume=False,
            max_messages=app_cfg.account_journey_max_messages,
            max_threads=app_cfg.account_journey_max_threads,
            max_attachment_chars=app_cfg.account_journey_max_attachment_chars,
            page_size=app_cfg.account_journey_page_size,
        )
        if not journey.events:
            return await _empty_mail()
        recent_events = list(journey.events)[-8:]
        last_sent = next((e for e in reversed(journey.events) if e.direction == "outbound"), None)
        last_recv = next((e for e in reversed(journey.events) if e.direction == "inbound"), None)
        latest = journey.events[-1]
        snapshot = MailSnapshot(
            last_sent=last_sent.date if last_sent else None,
            last_received=last_recv.date if last_recv else None,
            subject_hint=(latest.subject or "")[:120] or None,
            thread_id=latest.thread_id or None,
        )
        timeline = [
            TimelineEvent(
                date=e.date,
                source="gmail_journey",
                summary=f"{e.subject or '(no subject)'} — {e.from_address}",
            )
            for e in recent_events
        ]
        meeting_context = journey.meeting_context
        mail_text = (
            f"Journey scanned {journey.total_messages_scanned} messages across "
            f"{journey.total_threads_scanned} threads. Current need: {meeting_context.current_need} "
            f"Constraints: {'; '.join(meeting_context.constraints[:3]) or 'n/a'}."
        )
        return {
            "snapshot": snapshot,
            "timeline": timeline,
            "sources": [SourceRef(channel="gmail", label="account_mailbox_journey")],
            "mail_text": mail_text,
            "journey": journey,
        }

    emails = await email_processor.search_emails(query=query, max_results=15)
    if not emails:
        return await _empty_mail()

    sent: list[Any] = []
    received: list[Any] = []
    for em in emails:
        labels = [str(x).upper() for x in (em.labels or [])]
        if "SENT" in labels:
            sent.append(em)
        else:
            received.append(em)

    def _ts(em: Any) -> datetime:
        return em.received_at

    last_sent = max(sent, key=_ts) if sent else None
    last_recv = max(received, key=_ts) if received else None
    latest = max(emails, key=_ts)

    snapshot = MailSnapshot(
        last_sent=last_sent.received_at.isoformat()[:19] if last_sent else None,
        last_received=last_recv.received_at.isoformat()[:19] if last_recv else None,
        subject_hint=(latest.subject or "")[:120] or None,
        thread_id=latest.thread_id,
    )

    timeline: list[TimelineEvent] = []
    for em in sorted(emails, key=_ts, reverse=True)[:8]:
        timeline.append(
            TimelineEvent(
                date=em.received_at.isoformat()[:19],
                source="gmail",
                summary=f"{em.subject or '(no subject)'} — {em.from_address}",
            )
        )

    mail_text = "\n".join(
        f"{em.received_at.isoformat()[:10]} {em.subject or ''} from {em.from_address}"
        for em in sorted(emails, key=_ts, reverse=True)[:6]
    )
    return {
        "snapshot": snapshot,
        "timeline": timeline,
        "sources": [SourceRef(channel="gmail", label="search_emails")],
        "mail_text": mail_text,
        "journey": None,
    }


async def _empty_kb() -> dict[str, Any]:
    return {"highlights": [], "sources": [], "kb_text": ""}


async def _fetch_kb(retriever: Any, company: str) -> dict[str, Any]:
    query = f"{company.strip()} industrial forming forming manufacturing plastics industrial".strip()
    hits = await retriever.search(query, limit=5)
    highlights: list[str] = []
    for h in hits:
        excerpt = str(h.get("content") or "")[:280].strip()
        src = str(h.get("source") or h.get("doc_id") or "kb")
        if excerpt:
            highlights.append(f"[{src}] {excerpt}")
    return {
        "highlights": highlights,
        "sources": [SourceRef(channel="kb", label="unified_retriever")],
        "kb_text": "\n".join(highlights),
    }


def _fetch_proof(company: str) -> dict[str, Any]:
    arts = match_artifacts(tag_filter=company, limit=5)
    urls: list[str] = []
    for art in arts:
        url = str(art.get("url") or "").strip()
        if url:
            urls.append(url)
    sources = [SourceRef(channel="proof_registry", label="outbound_proof_artifacts.json")]
    return {"urls": urls, "sources": sources}


async def _empty_argus() -> dict[str, Any]:
    return {"excerpt": "", "sources": []}


def _fetch_website_profile(domain: str | None) -> dict[str, Any]:
    if not domain:
        return {"excerpt": "", "sources": []}
    try:
        from brain_os.systems.thermoformer_site_research import load_profile

        prof = load_profile(domain)
    except Exception:
        return {"excerpt": "", "sources": []}
    if prof is None:
        return {"excerpt": "", "sources": []}
    lines = [
        f"Max forming: {prof.max_thermoforming_dimensions() or 'not_disclosed'}",
        f"Industries: {', '.join(prof.industries[:8]) or 'not_disclosed'}",
    ]
    for m in prof.machinery[:6]:
        qty = f"{m.quantity}× " if m.quantity else ""
        lines.append(f"{qty}{m.machine_type}: {m.max_dimensions_mm or 'not_disclosed'}")
    return {
        "excerpt": "\n".join(lines),
        "sources": [
            SourceRef(
                channel="website_profile",
                label=prof.meta.profile_path or domain,
            )
        ],
    }


async def _fetch_argus(
    pantheon: Any,
    company: str,
    contact_email: str | None,
) -> dict[str, Any]:
    question = (
        f"Single-account dossier for {company}. "
        f"Contact: {contact_email or 'unknown'}. "
        "Return: what they do, industrial forming fit, prior Acme Corp touchpoints, "
        "key people, and recommended next step. Be concise."
    )
    async with pantheon:
        agent = pantheon.get_agent("argus")
        if agent is None:
            return {"excerpt": "", "sources": [], "error": "argus not available"}
        text = await agent.handle(question, context={"user_id": "operator"})
    excerpt = (text or "")[:1500]
    return {
        "excerpt": excerpt,
        "sources": [SourceRef(channel="argus", label="dossier")],
    }


def _assemble_brief(
    company: str,
    *,
    contact_email: str | None,
    crm_data: dict[str, Any],
    mail_data: dict[str, Any],
    kb_data: dict[str, Any],
    proof_data: dict[str, Any],
    argus_data: dict[str, Any],
) -> AccountBrief:
    resolved_email = contact_email or crm_data.get("contact_email")
    resolved_name = crm_data.get("contact_name")
    domain = None
    if resolved_email and "@" in resolved_email:
        domain = resolved_email.split("@", 1)[1].lower()

    website_data = _fetch_website_profile(domain)

    timeline = list(crm_data.get("timeline") or []) + list(mail_data.get("timeline") or [])
    sources: list[SourceRef] = []
    for block in (crm_data, mail_data, kb_data, proof_data, argus_data, website_data):
        sources.extend(list(block.get("sources") or []))

    if argus_data.get("error"):
        timeline.append(
            TimelineEvent(
                source="argus",
                summary=f"Argus skipped or failed: {argus_data.get('error')}",
            )
        )

    from brain_os.brain.atlas_production_portfolio import match_production_for_account
    from brain_os.services.shopfloor_before_customer import evaluate_shopfloor_for_brief

    atlas_raw = match_production_for_account(company, domain=domain)
    atlas_snap = AtlasProductionSnapshot.model_validate(atlas_raw)
    sources.append(
        SourceRef(
            channel="atlas",
            label="production_portfolio",
            freshness=atlas_snap.portfolio_updated_at,
        )
    )
    if atlas_snap.matched:
        timeline.append(
            TimelineEvent(
                source="atlas",
                summary=atlas_snap.summary_line[:400],
            )
        )

    risks: list[str] = []
    shopfloor: ShopfloorTruthSnapshot | None = None
    brief_for_sf = AccountBrief(
        company_name=company,
        domain=domain,
        contact_email=resolved_email,
        contact_name=resolved_name,
        crm=crm_data.get("snapshot"),
        atlas_production=atlas_snap,
        generated_at=datetime.now(UTC),
    )
    try:
        shopfloor = evaluate_shopfloor_for_brief(brief_for_sf)
        if shopfloor.status == "contradiction":
            risks.append(
                shopfloor.detail
                or (
                    f"Shopfloor contradiction: Atlas {shopfloor.atlas_relation} "
                    f"vs CRM {shopfloor.crm_stage or '(none)'} — treat as LIVE customer"
                )
            )
    except Exception:
        logger.debug("shopfloor_truth evaluation skipped", exc_info=True)
        shopfloor = None

    return AccountBrief(
        company_name=company,
        domain=domain,
        contact_email=resolved_email,
        contact_name=resolved_name,
        executive_summary="",
        timeline=timeline,
        crm=crm_data.get("snapshot"),
        mail=mail_data.get("snapshot"),
        kb_highlights=list(kb_data.get("highlights") or []),
        proof_links=list(proof_data.get("urls") or []),
        website_profile_excerpt=str(website_data.get("excerpt") or "") or None,
        account_journey=(
            mail_data.get("journey")
            if isinstance(mail_data.get("journey"), AccountJourney)
            else None
        ),
        risks_gaps=risks,
        suggested_opener=None,
        sources=sources,
        atlas_production=atlas_snap,
        shopfloor_truth=shopfloor,
        generated_at=datetime.now(UTC),
    )


async def _apply_llm_synthesis(
    brief: AccountBrief,
    *,
    argus_excerpt: str = "",
    llm_client: LLMClient | None,
) -> AccountBrief:
    client = llm_client or LLMClient()
    payload = AccountBriefSynthesisInput(
        company_name=brief.company_name,
        contact_email=brief.contact_email,
        contact_name=brief.contact_name,
        crm_text=_crm_snapshot_text(brief.crm),
        mail_text=_mail_snapshot_text(brief.mail),
        kb_bullets=brief.kb_highlights,
        proof_urls=brief.proof_links,
        argus_excerpt=argus_excerpt[:1500],
        timeline_lines=[f"{e.date} [{e.source}] {e.summary}" for e in brief.timeline[:10]],
        journey_text=_journey_text(brief.account_journey),
        graph_context_lines=brief.graph_context_lines,
    )
    try:
        synthesis = await client.generate_structured(
            _SYNTHESIS_PROMPT,
            brief_context_for_llm(payload.model_dump()),
            AccountBriefLLMSynthesis,
            name="account_brief.synthesis",
            temperature=0.2,
            max_tokens=1024,
        )
        merged_gaps = list(brief.risks_gaps)
        for g in synthesis.risks_gaps or []:
            if g and g not in merged_gaps:
                merged_gaps.append(g)
        return brief.model_copy(
            update={
                "executive_summary": synthesis.executive_summary,
                "risks_gaps": merged_gaps,
                "suggested_opener": synthesis.suggested_opener,
            }
        )
    except Exception as exc:
        logger.warning("LLM synthesis failed, using template: %s", exc)
        gaps = list(brief.risks_gaps)
        gaps.append(f"LLM synthesis unavailable: {exc}")
        return brief.model_copy(
            update={
                "executive_summary": _template_executive_summary(brief),
                "risks_gaps": gaps,
            }
        )


def _crm_snapshot_text(crm: CrmSnapshot | None) -> str:
    if crm is None:
        return ""
    return (
        f"stage={crm.stage}; value={crm.value} {crm.currency or ''}; "
        f"machine={crm.machine_model}; last={crm.last_activity}; contacts={crm.contact_count}"
    )


def _mail_snapshot_text(mail: MailSnapshot | None) -> str:
    if mail is None:
        return ""
    return (
        f"last_sent={mail.last_sent}; last_received={mail.last_received}; "
        f"subject={mail.subject_hint}"
    )


def _journey_text(journey: AccountJourney | None) -> str:
    if journey is None:
        return ""
    bits: list[str] = [
        f"messages={journey.total_messages_scanned}",
        f"threads={journey.total_threads_scanned}",
    ]
    if journey.meeting_context.current_need:
        bits.append(f"need={journey.meeting_context.current_need}")
    if journey.meeting_context.constraints:
        bits.append("constraints=" + "; ".join(journey.meeting_context.constraints[:3]))
    return " | ".join(bits)


def _template_executive_summary(brief: AccountBrief) -> str:
    parts: list[str] = [f"Account: {brief.company_name}."]
    if brief.crm and brief.crm.stage:
        parts.append(f"CRM stage: {brief.crm.stage}.")
    if brief.mail and brief.mail.last_received:
        parts.append(f"Last inbound mail: {brief.mail.last_received}.")
    elif brief.mail and brief.mail.last_sent:
        parts.append(f"Last outbound mail: {brief.mail.last_sent}.")
    if not brief.crm and not brief.mail:
        parts.append("Limited CRM and mail context available.")
    if brief.kb_highlights:
        parts.append("KB snippets attached below.")
    return " ".join(parts)
