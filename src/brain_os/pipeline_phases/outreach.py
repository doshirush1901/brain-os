"""Phase: outreach helper utilities extracted from pipeline.py.

Phase modules MUST NOT import from brain_os.pipeline (circular). Shared helpers
live in ira.pipeline_runtime; import from there when this slice needs them.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from brain_os.schemas.llm_outputs import OutreachRankingOutput
from brain_os.service_keys import ServiceKey as SK
from brain_os.services.hot_leads_board_sync import outreach_board_boost, outreach_should_exclude

_OWN_MAIL_DOMAINS = frozenset(
    {"example-company.org", "acme-corp.com", "example-company.in", "gmail.com"}
)

_OUTREACH_CRM_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    ImportError,
    TimeoutError,
)


def has_thread_evidence(tool_audit: Any) -> bool:
    if not isinstance(tool_audit, list):
        return False
    for item in tool_audit:
        if not isinstance(item, dict):
            continue
        if item.get("tool") == "read_email_thread" and bool(item.get("success")):
            return True
    return False


def coerce_contact_type(contact: Any) -> str:
    raw = getattr(contact, "contact_type", None)
    val = getattr(raw, "value", raw)
    return str(val or "").upper()


def get_email_processor(pantheon: Any) -> Any | None:
    for agent_name in ("athena", "prometheus", "clio"):
        agent = pantheon.get_agent(agent_name)
        services = getattr(agent, "_services", {}) if agent is not None else {}
        if isinstance(services, dict) and services.get(SK.EMAIL_PROCESSOR) is not None:
            return services[SK.EMAIL_PROCESSOR]
    return None


def extract_thread_reference(
    contact: Any, *, thread_id_pattern: re.Pattern[str]
) -> tuple[str, str]:
    tags = getattr(contact, "tags", None)
    thread_id = ""
    mailbox = ""
    if isinstance(tags, dict):
        tid = tags.get("gmail_thread_id") or tags.get("thread_id")
        if isinstance(tid, str):
            thread_id = tid.strip()
        mb = tags.get("source_mailbox") or tags.get("mailbox")
        if isinstance(mb, str):
            mailbox = mb.strip()
    if not thread_id:
        summary = str(getattr(contact, "account_summary", "") or "")
        match = thread_id_pattern.search(summary)
        if match:
            thread_id = match.group(1)
    return thread_id, mailbox


async def build_outreach_shortlist(
    *,
    crm: Any | None,
    query: str,
    thread_id_pattern: re.Pattern[str],
    logger: Any,
    max_contacts: int = 8,
) -> list[dict[str, Any]]:
    """Build a deterministic candidate shortlist for outreach queries."""
    if crm is None:
        return []
    try:
        contacts = await crm.list_contacts()
    except Exception:
        logger.warning("Outreach shortlist: CRM contacts lookup failed", exc_info=True)
        return []
    if not contacts:
        return []

    allowed_types = {"LEAD_WITH_INTERACTIONS", "LIVE_CUSTOMER", "PAST_CUSTOMER"}
    type_weight = {"LIVE_CUSTOMER": 3.0, "LEAD_WITH_INTERACTIONS": 2.0, "PAST_CUSTOMER": 1.0}
    warmth_weight = {"HOT": 3.0, "WARM": 2.0, "COOL": 1.0, "COLD": -0.5}
    tokens = [t for t in re.findall(r"[a-zA-Z0-9@._-]+", (query or "").lower()) if len(t) >= 4]
    rows: list[tuple[float, str, dict[str, Any]]] = []

    for c in contacts:
        ctype = coerce_contact_type(c)
        if ctype not in allowed_types:
            continue
        thread_id, mailbox = extract_thread_reference(c, thread_id_pattern=thread_id_pattern)
        blob = " ".join(
            str(getattr(c, k, "") or "") for k in ("name", "email", "account_summary")
        ).lower()
        score = float(sum(1 for tok in tokens if tok in blob)) + type_weight.get(ctype, 0.0)

        warmth_raw = getattr(c, "warmth_level", None)
        warmth = str(getattr(warmth_raw, "value", warmth_raw) or "").upper()
        score += warmth_weight.get(warmth, 0.0)

        try:
            lead_score = float(getattr(c, "lead_score", 0.0) or 0.0)
        except Exception:
            lead_score = 0.0
        score += min(3.0, lead_score / 40.0)
        if thread_id:
            score += 1.5

        company_name = ""
        company = getattr(c, "company", None)
        if company is not None:
            company_name = str(getattr(company, "name", "") or "")
        stamp = getattr(c, "updated_at", None) or getattr(c, "created_at", None)
        sort_stamp = stamp.isoformat() if hasattr(stamp, "isoformat") else str(stamp or "")

        if company_name and outreach_should_exclude(company_name, query):
            continue

        score += outreach_board_boost(company_name)

        rows.append(
            (
                score,
                sort_stamp,
                {
                    "name": str(getattr(c, "name", "") or ""),
                    "email": str(getattr(c, "email", "") or ""),
                    "company": company_name,
                    "contact_type": ctype,
                    "warmth_level": warmth,
                    "lead_score": lead_score,
                    "thread_id": thread_id,
                    "mailbox": mailbox,
                    "updated_at": sort_stamp,
                },
            )
        )

    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in rows[:max_contacts]]


async def prefetch_outreach_thread_evidence(
    *,
    crm: Any | None,
    email_processor: Any | None,
    query: str,
    thread_id_pattern: re.Pattern[str],
    logger: Any,
    max_contacts: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    """Prefetch recent live thread snippets for outreach ranking queries."""
    if crm is None or email_processor is None:
        return "", []

    try:
        contacts = await crm.list_contacts()
    except Exception:
        logger.warning("Outreach prefetch: CRM contacts lookup failed", exc_info=True)
        return "", []
    if not contacts:
        return "", []

    allowed_types = {"LEAD_WITH_INTERACTIONS", "LIVE_CUSTOMER", "PAST_CUSTOMER"}
    q = (query or "").lower()
    tokens = [t for t in re.findall(r"[a-zA-Z0-9@._-]+", q) if len(t) >= 4]

    candidates: list[tuple[int, str, Any, str, str]] = []
    for c in contacts:
        if coerce_contact_type(c) not in allowed_types:
            continue
        thread_id, mailbox = extract_thread_reference(c, thread_id_pattern=thread_id_pattern)
        if not thread_id:
            continue

        blob = " ".join(
            str(getattr(c, k, "") or "") for k in ("name", "email", "account_summary")
        ).lower()
        score = sum(1 for tok in tokens if tok in blob)

        stamp = getattr(c, "updated_at", None) or getattr(c, "created_at", None)
        sort_stamp = stamp.isoformat() if hasattr(stamp, "isoformat") else str(stamp or "")

        candidates.append((score, sort_stamp, c, thread_id, mailbox))

    if not candidates:
        return "", []

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    snippets: list[str] = []
    audit: list[dict[str, Any]] = []

    for _, _, contact, thread_id, mailbox in candidates[:max_contacts]:
        try:
            emails = await asyncio.wait_for(
                email_processor.get_thread(thread_id, mailbox=mailbox or None),
                timeout=10.0,
            )
        except Exception as exc:
            audit.append(
                {
                    "agent": "pipeline",
                    "tool": "read_email_thread",
                    "success": False,
                    "thread_id": thread_id,
                    "error": str(exc)[:120],
                }
            )
            continue

        if not emails:
            audit.append(
                {
                    "agent": "pipeline",
                    "tool": "read_email_thread",
                    "success": False,
                    "thread_id": thread_id,
                    "error": "empty_thread",
                }
            )
            continue

        latest = max(
            emails,
            key=lambda e: str(getattr(e, "received_at", "") or ""),
        )
        body = re.sub(r"\s+", " ", str(getattr(latest, "body", "") or "")).strip()
        preview = body[:240] if body else "(no body preview)"
        received_at = getattr(latest, "received_at", "")
        received_txt = (
            received_at.isoformat() if hasattr(received_at, "isoformat") else str(received_at or "")
        )

        name = str(getattr(contact, "name", "") or "").strip() or "Unknown"
        email = str(getattr(contact, "email", "") or "").strip()
        sender = str(getattr(latest, "from_address", "") or "").strip()

        snippets.append(
            f"- {name} <{email}> | thread={thread_id} | latest_from={sender} | "
            f"at={received_txt} | {preview}"
        )
        audit.append(
            {
                "agent": "pipeline",
                "tool": "read_email_thread",
                "success": True,
                "thread_id": thread_id,
            }
        )

    if not snippets:
        return "", audit

    guidance = (
        "Recent live thread evidence (auto-prefetched for outreach prioritization):\n"
        + "\n".join(snippets)
        + "\nUse this evidence before naming the next contact."
    )
    return guidance, audit


async def _crm_snapshot_line_for_thread(
    crm: Any | None,
    messages: list[Any],
) -> str:
    """Lightweight CRM one-liner from thread participant emails (no full brief)."""
    if crm is None or not messages:
        return ""
    seen_emails: list[str] = []
    for msg in messages:
        for attr in ("from_address", "to_address"):
            raw = (getattr(msg, attr, None) or "").strip().lower()
            if not raw or "@" not in raw:
                continue
            dom = raw.split("@", 1)[-1]
            if dom in _OWN_MAIL_DOMAINS:
                continue
            if raw not in seen_emails:
                seen_emails.append(raw)
    for em in seen_emails[:4]:
        try:
            contact = await crm.get_contact_by_email(em)
        except _OUTREACH_CRM_ERRORS:
            continue
        if contact is None:
            continue
        company = ""
        comp = getattr(contact, "company", None)
        if comp is not None:
            company = (getattr(comp, "name", None) or "").strip()
        stage = ""
        try:
            deals = await crm.get_deals_for_contact(str(getattr(contact, "id", "") or ""))
        except _OUTREACH_CRM_ERRORS:
            deals = []
        if deals:
            d0 = deals[0]
            stage = str(d0.get("stage") or d0.get("status") or "").strip()
            title = str(d0.get("title") or "").strip()
            parts = [p for p in (company or "CRM contact", stage, title) if p]
            return "CRM snapshot: " + " | ".join(parts[:3])
        if company:
            return f"CRM snapshot: {company} (contact {em})"
    return ""


async def prefetch_gmail_thread_by_id(
    thread_ids: list[str],
    *,
    email_processor: Any | None = None,
    crm: Any | None = None,
    max_threads: int = 1,
    body_chars: int = 4000,
) -> tuple[str, list[dict[str, Any]]]:
    """Prefetch one or more Gmail threads by id (no search_emails sweep)."""
    audit: list[dict[str, Any]] = []
    if not thread_ids:
        return "", audit
    ep = email_processor or get_email_processor()
    if ep is None:
        audit.append(
            {
                "tool": "read_email_thread",
                "success": False,
                "error": "email_processor_unavailable",
            }
        )
        return "", audit

    snippets: list[str] = []
    last_messages: list[Any] | None = None
    for thread_id in thread_ids[:max_threads]:
        try:
            messages = await asyncio.wait_for(
                ep.get_thread(thread_id),
                timeout=10.0,
            )
        except TimeoutError:
            audit.append(
                {
                    "tool": "read_email_thread",
                    "thread_id": thread_id,
                    "success": False,
                    "error": "timeout",
                }
            )
            continue
        except _OUTREACH_CRM_ERRORS as exc:
            audit.append(
                {
                    "tool": "read_email_thread",
                    "thread_id": thread_id,
                    "success": False,
                    "error": str(exc)[:200],
                }
            )
            continue

        if not messages:
            audit.append(
                {
                    "tool": "read_email_thread",
                    "thread_id": thread_id,
                    "success": False,
                    "error": "empty_thread",
                }
            )
            continue

        last_messages = messages
        latest = messages[-1]
        body = (getattr(latest, "body", None) or "")[:body_chars]
        snippets.append(
            f"- Thread `{thread_id}` | latest {getattr(latest, 'received_at', '')} | "
            f"From: {getattr(latest, 'from_address', '')} | "
            f"To: {getattr(latest, 'to_address', '')} | "
            f"Subject: {getattr(latest, 'subject', '')}\n"
            f"  Body excerpt:\n{body}"
        )
        audit.append(
            {
                "tool": "read_email_thread",
                "thread_id": thread_id,
                "success": True,
                "message_count": len(messages),
            }
        )

    if not snippets:
        return "", audit

    crm_line = ""
    if crm is not None and last_messages:
        try:
            crm_line = await _crm_snapshot_line_for_thread(crm, last_messages)
        except _OUTREACH_CRM_ERRORS:
            crm_line = ""

    guidance = (
        "Gmail thread evidence (prefetched by thread id — use read_email_thread only; "
        "do not run broad search_emails for this request):\n" + "\n".join(snippets)
    )
    if crm_line:
        guidance = f"{crm_line}\n\n{guidance}"
    return guidance, audit


def compose_outreach_workflow_query(
    query: str,
    shortlist: list[dict[str, Any]],
) -> str:
    """Compose a stable prompt for outreach specialist routing."""
    shortlist_blob = json.dumps(shortlist[:8], default=str, ensure_ascii=True)
    return (
        "OUTREACH PRIORITIZATION WORKFLOW\n"
        "Task: Decide who Acme Corp should contact next.\n"
        "Rules:\n"
        "1) Use latest live thread evidence and shortlist signals.\n"
        "2) If evidence is weak or contradictory, say so explicitly.\n"
        "3) Return ranked candidates (top 3) with one evidence line each.\n"
        "4) Every ranked candidate MUST include a thread_id from evidence.\n"
        "5) Output ONLY valid JSON (no prose) matching this schema:\n"
        "{\n"
        '  "recommendation": {\n'
        '    "rank": 1,\n'
        '    "contact_name": "...",\n'
        '    "contact_email": "...",\n'
        '    "company": "...",\n'
        '    "thread_id": "...",\n'
        '    "evidence_line": "...",\n'
        '    "reason": "..."\n'
        "  },\n"
        '  "ranked": [{... up to 3 entries ...}],\n'
        '  "confidence": "high|medium|low",\n'
        '  "freshness": "current|possibly_stale|historical",\n'
        '  "notes": "..."\n'
        "}\n\n"
        f"User request:\n{query}\n\n"
        f"Shortlist candidates (JSON):\n{shortlist_blob}\n"
    )


def extract_json_payload(text: str) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fenced:
        block = fenced.group(1).strip()
        if "{" in block and "}" in block:
            return block[block.find("{") : block.rfind("}") + 1]
    if "{" in raw and "}" in raw:
        return raw[raw.find("{") : raw.rfind("}") + 1]
    return None


def render_outreach_ranking(parsed: OutreachRankingOutput) -> str:
    lines = [
        "**Outreach recommendation**",
        (
            f"Top pick: **{parsed.recommendation.contact_name}**"
            f" ({parsed.recommendation.contact_email})"
            + (f" · {parsed.recommendation.company}" if parsed.recommendation.company else "")
            + f" · thread `{parsed.recommendation.thread_id}`"
        ),
        f"Reason: {parsed.recommendation.reason or parsed.recommendation.evidence_line}",
        "",
        "**Ranked candidates (top 3)**",
    ]
    for c in parsed.ranked[:3]:
        lines.append(
            f"{c.rank}. {c.contact_name} ({c.contact_email})"
            + (f" · {c.company}" if c.company else "")
            + f" · thread `{c.thread_id}`"
        )
        lines.append(f"   Evidence: {c.evidence_line}")
    lines.extend(
        [
            "",
            f"Confidence: {parsed.confidence}",
            f"Freshness: {parsed.freshness}",
        ]
    )
    if parsed.notes:
        lines.append(f"Notes: {parsed.notes}")
    return "\n".join(lines)


__all__ = [
    "build_outreach_shortlist",
    "coerce_contact_type",
    "compose_outreach_workflow_query",
    "extract_json_payload",
    "extract_thread_reference",
    "get_email_processor",
    "has_thread_evidence",
    "prefetch_gmail_thread_by_id",
    "prefetch_outreach_thread_evidence",
    "render_outreach_ranking",
]
