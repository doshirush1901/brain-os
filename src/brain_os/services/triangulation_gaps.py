"""Shared triangulation gap detection (triangle + optional hex) for AccountBrief."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from brain_os.schemas.account_brief import AccountBrief

Layer = Literal["triangle", "hex", "tinder"]


@dataclass(frozen=True)
class LegSpec:
    key: str
    label: str
    next_step: str
    layer: Layer


_PRODUCTION_STAGES = frozenset(
    {
        "CUSTOMER",
        "PRODUCTION",
        "DELIVERED",
        "WON",
        "CLOSED_WON",
        "ACTIVE",
        "ENGAGED",
    }
)

_TRIANGLE_LEGS: tuple[LegSpec, ...] = (
    LegSpec("intent_kb", "Intent (KB)", "search_knowledge / quote PDFs in KB", "triangle"),
    LegSpec(
        "relationship_mail",
        "Relationship (mail)",
        "search_emails / read_email_thread",
        "triangle",
    ),
    LegSpec(
        "identity_contact",
        "Identity (domain)",
        "find_company_contacts; Argus if thin",
        "triangle",
    ),
)

_HEX_LEGS: tuple[LegSpec, ...] = (
    LegSpec(
        "production_atlas",
        "Production (Atlas)",
        "ask_agent atlas / check active orders",
        "hex",
    ),
    LegSpec(
        "graph_quotes",
        "Graph (quotes)",
        "find_company_quotes / find_related_entities",
        "hex",
    ),
    LegSpec(
        "proof_registry",
        "Proof (registry)",
        "list_outbound_proof_artifacts",
        "hex",
    ),
)

_TINDER_EXTRA_LEGS: tuple[LegSpec, ...] = (
    LegSpec(
        "proof_registry",
        "Proof (registry)",
        "list_outbound_proof_artifacts",
        "tinder",
    ),
    LegSpec("crm_snapshot", "CRM snapshot", "search_crm / get_deal", "tinder"),
    LegSpec(
        "outbound_cadence",
        "Outbound cadence",
        "ira email touch-audit",
        "tinder",
    ),
    LegSpec(
        "icp_buyer_fit",
        "ICP (industrial former buyer)",
        "ira tinder card — site + classifier",
        "tinder",
    ),
)


def _has_gmail_timeline(brief: AccountBrief) -> bool:
    return any(e.source == "gmail" for e in brief.timeline)


def _has_graph_signal(brief: AccountBrief) -> bool:
    for src in brief.sources:
        label = (src.label or "").lower()
        if "neo4j" in label or "graph" in label:
            return True
    for line in brief.kb_highlights:
        low = line.lower()
        if "quote" in low or "mct-" in low or "pf1-" in low:
            return True
    return False


def _has_production_signal(brief: AccountBrief) -> bool:
    ap = brief.atlas_production
    if ap is not None and ap.relation in ("in_flight", "installation", "quote_only"):
        return True
    if brief.crm is None:
        return False
    stage = (brief.crm.stage or "").upper().replace(" ", "_")
    if stage in _PRODUCTION_STAGES:
        return True
    if brief.crm.machine_model:
        return True
    return False


def _leg_missing(key: str, brief: AccountBrief, *, card: dict[str, Any] | None) -> bool:
    if key == "intent_kb":
        return not brief.kb_highlights
    if key == "relationship_mail":
        return brief.mail is None and not _has_gmail_timeline(brief)
    if key == "identity_contact":
        return not brief.contact_email and not brief.domain
    if key == "production_atlas":
        return not _has_production_signal(brief)
    if key == "graph_quotes":
        return not _has_graph_signal(brief)
    if key == "proof_registry":
        return not brief.proof_links
    if key == "crm_snapshot":
        return brief.crm is None
    if key == "outbound_cadence":
        return bool(card and card.get("touch_audit_flag"))
    if key == "icp_buyer_fit":
        if card is None:
            return True
        intel = card.get("company_intel") or {}
        if intel.get("icp_approved"):
            return False
        if intel.get("icp_verdict"):
            return not bool(intel.get("icp_approved"))
        return True
    return False


def triangulation_gaps(
    brief: AccountBrief,
    *,
    hex: bool = False,
    card: dict[str, Any] | None = None,
) -> list[str]:
    """Return missing leg keys (backward compatible with Tinder + pipeline)."""
    keys: list[str] = []
    for leg in _TRIANGLE_LEGS:
        if _leg_missing(leg.key, brief, card=card):
            keys.append(leg.key)
    if hex:
        for leg in _HEX_LEGS:
            if _leg_missing(leg.key, brief, card=card) and leg.key not in keys:
                keys.append(leg.key)
    if card is not None:
        for leg in _TINDER_EXTRA_LEGS:
            if _leg_missing(leg.key, brief, card=card) and leg.key not in keys:
                keys.append(leg.key)
    return keys


def leg_status_rows(
    brief: AccountBrief,
    *,
    hex: bool = False,
) -> list[tuple[str, str, str]]:
    """(label, status, next_step) for gaps-only operator tables."""
    rows: list[tuple[str, str, str]] = []
    specs: list[LegSpec] = list(_TRIANGLE_LEGS)
    if hex:
        specs.extend(_HEX_LEGS)
    for leg in specs:
        missing = _leg_missing(leg.key, brief, card=None)
        status = "MISSING" if missing else "OK"
        rows.append((leg.label, status, leg.next_step if missing else "—"))
    return rows


def format_gaps_only_block(
    *,
    company: str,
    contact_email: str | None,
    mode: Literal["triangle", "hex"],
    brief: AccountBrief | None,
    gap_keys: list[str] | None = None,
    unresolved_company: bool = False,
) -> str | None:
    """Plain-text gaps appendix; None when nothing to report."""
    if unresolved_company:
        return (
            "\n\n---\n"
            "Triangulation (SOUL / Anekantavada) — gaps only\n"
            "Could not resolve company from query.\n"
            'Run: poetry run brain brief "{company}" --contact email@domain.com --json'
        )

    if brief is None:
        return (
            "\n\n---\n"
            "Triangulation (SOUL / Anekantavada) — gaps only\n"
            f"Company: {company} | Mode: {mode}\n"
            "| Leg | Status | Next step |\n"
            "| UNVERIFIED | brief timeout or error | retry ira brief |"
        )

    rows = leg_status_rows(brief, hex=mode == "hex")
    missing_rows = [r for r in rows if r[1] == "MISSING"]
    if not missing_rows and not gap_keys:
        return None

    contact = contact_email or brief.contact_email or "—"
    lines = [
        "",
        "---",
        "Triangulation (SOUL / Anekantavada) — gaps only",
        f"Company: {company} | Contact: {contact} | Mode: {mode}",
        "",
        "| Leg | Status | Next step |",
        "|-----|--------|-----------|",
    ]
    for label, status, nxt in rows:
        if status == "MISSING":
            lines.append(f"| {label} | {status} | {nxt} |")
    if not missing_rows and gap_keys:
        lines.append(f"| (keys) | MISSING | {', '.join(gap_keys)} |")
    cmd_contact = f" --contact {contact}" if contact and contact != "—" else ""
    lines.append("")
    lines.append(f'Run: poetry run brain brief "{company}"{cmd_contact} --json')
    return "\n".join(lines)
