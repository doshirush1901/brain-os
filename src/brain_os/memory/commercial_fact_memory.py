"""Commercial-fact memories from the quote registry + operator gap-fill seeds.

Fills the three proven Mem0 recall gaps from the 2026-08-02 audit:
(a) quote commercial facts, (b) morning-brief operator preference,
(c) AcmeVIP/Misha VIP relationship facts from operator_account_pins.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

MORNING_BRIEF_PREFERENCE = (
    "Operator preference (judgment-sprint ground truth 2026-08-02): morning brief "
    "must use three lanes — HEAT NOW (live chase accounts above heat threshold), "
    "CURRENT ORDERS (Atlas/order-book production truth, never mixed with leads), "
    "and HOUSEKEEPING (interested/question replies capped, noise blocklist excluded). "
    "Always show counts per lane. VIP P0 pins (Demo/AcmeVIP, Liebherr, KTX) stay "
    "visible even when CRM stage is thin. Sticky replied noise (Alphaform, Erie, etc.) "
    "is muted, not a chase."
)

METAKAM_VIP_FACT = (
    "Demo/AcmeVIP (aliases ACMEVIP, ACMEVIP, AcmeVIP LLC; contact vip.contact@example.com / "
    "Demo VIP Contact) is VIP P0 — Russia channel with a live quote. "
    "Mailbox triage: treat as highest priority (operator_account_pins + "
    "mailbox_operator_priority). Do not bury Cyrillic/mail.ru threads under "
    "newsletters or sibling-owned CC."
)


def format_commercial_fact(
    *,
    company: str,
    machine: str,
    value: Any,
    currency: str,
    when: str,
    status: str,
    quote_number: str = "",
) -> str:
    """Canonical commercial-fact line for Mem0."""
    co = (company or "Unknown company").strip() or "Unknown company"
    machine_s = (machine or "machine").strip() or "machine"
    cur = (currency or "INR").strip() or "INR"
    try:
        val_s = f"{float(value):,.0f}" if value is not None and str(value) != "" else "unknown"
    except (TypeError, ValueError):
        val_s = str(value or "unknown")
    status_s = (status or "DRAFT").strip() or "DRAFT"
    date_s = (when or "").strip() or "unknown date"
    qn = f" ({quote_number})" if quote_number else ""
    return f"{co}: quoted {machine_s} at {val_s} {cur} on {date_s}, status {status_s}{qn}"


def commercial_fact_from_quote(quote: Any) -> str | None:
    """Build a commercial-fact string from a QuoteModel-like object."""
    company = getattr(quote, "company_name", None) or ""
    machine = getattr(quote, "machine_model", None) or ""
    value = getattr(quote, "estimated_value", None)
    currency = getattr(quote, "currency", None) or "INR"
    status = getattr(quote.status, "value", None) or getattr(quote, "status", None) or "DRAFT"
    qn = getattr(quote, "quote_number", None) or ""
    created = getattr(quote, "created_at", None) or getattr(quote, "sent_at", None)
    if hasattr(created, "date"):
        when = created.date().isoformat()
    elif isinstance(created, date):
        when = created.isoformat()
    else:
        when = str(created or "")[:10]
    if not company and not machine and not value:
        return None
    return format_commercial_fact(
        company=str(company),
        machine=str(machine),
        value=value,
        currency=str(currency),
        when=when,
        status=str(status),
        quote_number=str(qn or ""),
    )


async def store_commercial_fact_for_quote(
    long_term: Any,
    quote: Any,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Store one commercial-fact memory for a registry quote (idempotent intent)."""
    body = commercial_fact_from_quote(quote)
    if not body:
        return {"stored": False, "reason": "empty_quote"}
    qn = str(getattr(quote, "quote_number", "") or "")
    meta = {
        "type": "commercial_fact",
        "memory_category": "commercial_fact",
        "quote_number": qn,
        "company_name": getattr(quote, "company_name", "") or "",
        "machine_model": getattr(quote, "machine_model", "") or "",
        "confidence": 0.9,
        "verified": True,
    }
    if dry_run:
        return {"stored": False, "dry_run": True, "content": body, "metadata": meta}
    if long_term is None:
        return {"stored": False, "reason": "no_long_term"}
    gated = await long_term.store_gated(
        body,
        user_id="global",
        metadata=meta,
        source=f"quote:{qn or 'unknown'}",
        category="commercial_fact",
        bypass_salience=True,
    )
    if gated.get("skipped"):
        return {"stored": False, "reason": gated.get("reason"), "content": body}
    return {"stored": True, "content": body, "entries": gated.get("entries")}


async def backfill_commercial_facts_from_registry(
    *,
    long_term: Any | None = None,
    dry_run: bool = True,
    limit: int = 0,
    batch_pause_every: int = 50,
) -> dict[str, Any]:
    """Backfill commercial-fact memories from Postgres quote registry (~879 rows)."""
    from brain_os.data.crm import CRMDatabase
    from brain_os.data.quotes import QuoteManager
    from brain_os.memory.long_term import LongTermMemory

    ltm = long_term or LongTermMemory()
    crm = CRMDatabase()
    qm = QuoteManager(crm.session_factory)
    quotes = await qm.list_quotes()
    if limit and limit > 0:
        quotes = quotes[:limit]

    attempted = 0
    written = 0
    failed = 0
    skipped = 0
    samples: list[str] = []

    for q in quotes:
        attempted += 1
        try:
            result = await store_commercial_fact_for_quote(ltm, q, dry_run=dry_run)
            if result.get("dry_run"):
                written += 1  # would-write count
                if len(samples) < 10 and result.get("content"):
                    samples.append(str(result["content"]))
            elif result.get("stored"):
                written += 1
                if len(samples) < 10 and result.get("content"):
                    samples.append(str(result["content"]))
            else:
                skipped += 1
        except Exception:
            logger.exception("commercial fact backfill failed for quote")
            failed += 1
        if not dry_run and batch_pause_every and attempted % batch_pause_every == 0:
            import asyncio

            await asyncio.sleep(0.25)

    status = "ok" if failed == 0 else "partial_error"
    return {
        "status": status,
        "dry_run": dry_run,
        "attempted": attempted,
        "written": written,
        "failed": failed,
        "skipped": skipped,
        "samples": samples,
    }


async def seed_gap_fill_memories(
    *,
    long_term: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Store morning-brief preference + AcmeVIP VIP relationship facts."""
    from brain_os.memory.long_term import LongTermMemory

    ltm = long_term or LongTermMemory()
    results: list[dict[str, Any]] = []

    seeds = [
        (
            MORNING_BRIEF_PREFERENCE,
            "preference",
            "operator:morning_brief_spec",
            {
                "type": "preference",
                "preference_type": "morning_brief_lanes",
                "memory_category": "preference",
                "provenance": "judgment_sprint_2026-08-02",
            },
        ),
        (
            METAKAM_VIP_FACT,
            "relationship",
            "operator_account_pins:misha_metakam",
            {
                "type": "relationship",
                "memory_category": "relationship",
                "company": "Demo/AcmeVIP",
                "pin": "hot",
                "provenance": "operator_account_pins",
            },
        ),
    ]

    for content, category, source, meta in seeds:
        if dry_run:
            results.append(
                {"stored": False, "dry_run": True, "content": content, "category": category}
            )
            continue
        gated = await ltm.store_gated(
            content,
            user_id="global",
            metadata=meta,
            source=source,
            category=category,
            bypass_salience=True,
        )
        results.append(
            {
                "stored": not gated.get("skipped"),
                "reason": gated.get("reason"),
                "category": category,
                "content": content[:160],
            }
        )

    written = sum(1 for r in results if r.get("stored") or r.get("dry_run"))
    return {
        "status": "ok",
        "dry_run": dry_run,
        "attempted": len(seeds),
        "written": written,
        "failed": 0,
        "results": results,
    }
