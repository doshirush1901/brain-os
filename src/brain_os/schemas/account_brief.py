"""Structured account brief for operator prep (calls, email, travel)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from brain_os.schemas.account_journey import AccountJourney
from brain_os.schemas.account_state import AccountState
from brain_os.schemas.math_mode import MathModeAdvisory
from brain_os.schemas.operator_context import ContextPrecedent
from brain_os.schemas.similar_company import SimilarCompany


class SourceRef(BaseModel):
    channel: str = Field(..., description="e.g. crm, gmail, kb, argus, proof_registry")
    ref_id: str = ""
    label: str = ""
    freshness: str | None = None


class EvidenceLeg(BaseModel):
    leg: str = Field(..., description="crm | mail | atlas | website | proof")
    label: str = ""
    mode: str = Field(default="missing", description="live | static | missing | skipped")
    age_days: int | None = None
    stale: bool = False
    detail: str = ""


class TimelineEvent(BaseModel):
    date: str = ""
    source: str = ""
    summary: str = ""


class CrmSnapshot(BaseModel):
    company_name: str = ""
    stage: str | None = None
    value: float | None = None
    currency: str | None = None
    machine_model: str | None = None
    last_activity: str | None = None
    contact_count: int = 0


class MailSnapshot(BaseModel):
    last_sent: str | None = None
    last_received: str | None = None
    subject_hint: str | None = None
    thread_id: str | None = None


class AtlasProductionSnapshot(BaseModel):
    """Atlas canonical order-book match (operator-maintained portfolio)."""

    matched: bool = False
    relation: str = Field(
        default="none",
        description="in_flight | installation | quote_only | none",
    )
    in_flight: bool = False
    quote_only_hint: bool = True
    matched_projects: list[dict[str, Any]] = Field(default_factory=list)
    summary_line: str = ""
    portfolio_updated_at: str | None = None


class ShopfloorTruthSnapshot(BaseModel):
    """CRM vs Atlas shopfloor contradiction (brain growth #6)."""

    ok: bool = True
    blocked: bool = False
    status: str = Field(
        default="ok",
        description="ok | contradiction | not_applicable | skipped",
    )
    display_label: str = ""
    atlas_relation: str = ""
    crm_stage: str = ""
    failures: list[str] = Field(default_factory=list)
    detail: str = ""


class AccountBriefSynthesisInput(BaseModel):
    """Compact facts passed to the LLM synthesizer."""

    company_name: str
    contact_email: str | None = None
    contact_name: str | None = None
    crm_text: str = ""
    mail_text: str = ""
    kb_bullets: list[str] = Field(default_factory=list)
    proof_urls: list[str] = Field(default_factory=list)
    argus_excerpt: str = ""
    timeline_lines: list[str] = Field(default_factory=list)
    journey_text: str = ""
    graph_context_lines: list[str] = Field(default_factory=list)


class AccountBriefLLMSynthesis(BaseModel):
    """LLM-generated prose fields only (merged with grounded snapshots)."""

    executive_summary: str = ""
    risks_gaps: list[str] = Field(default_factory=list)
    suggested_opener: str | None = None


class AccountBrief(BaseModel):
    company_name: str
    domain: str | None = None
    contact_email: str | None = None
    contact_name: str | None = None
    executive_summary: str = ""
    timeline: list[TimelineEvent] = Field(default_factory=list)
    crm: CrmSnapshot | None = None
    mail: MailSnapshot | None = None
    kb_highlights: list[str] = Field(default_factory=list)
    proof_links: list[str] = Field(default_factory=list)
    website_profile_excerpt: str | None = None
    account_journey: AccountJourney | None = None
    risks_gaps: list[str] = Field(default_factory=list)
    suggested_opener: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    precedents: list[ContextPrecedent] = Field(
        default_factory=list,
        description="Prior operator runs (brief, Tinder draft, quote prep) for this account.",
    )
    graph_context_lines: list[str] = Field(
        default_factory=list,
        description="1-hop Neo4j context (contacts, quotes, operator/pipeline runs).",
    )
    similar_companies: list[SimilarCompany] = Field(
        default_factory=list,
        description="Peer accounts by Company embedding cosine similarity (P3).",
    )
    operator_context_run_id: str | None = Field(
        default=None,
        description="Run id of the operator context record created for this brief.",
    )
    math_advisory: MathModeAdvisory | None = Field(
        default=None,
        description="Math Mode scores when APP__MATH_MODE_ADVISORY_ENABLED=true (phase 3).",
    )
    atlas_production: AtlasProductionSnapshot | None = Field(
        default=None,
        description="Match against data/knowledge/atlas_production_portfolio.json.",
    )
    shopfloor_truth: ShopfloorTruthSnapshot | None = Field(
        default=None,
        description="CRM lead vs Atlas in-flight contradiction (brain growth #6).",
    )
    evidence_legs: list[EvidenceLeg] = Field(
        default_factory=list,
        description="Triangle/hex leg freshness for operator cadence.",
    )
    account_state: AccountState | None = Field(
        default=None,
        description="Canonical structured latent (encoder→predictor→generator).",
    )
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_plain_text(self) -> str:
        """Perform the to plain text operation.

        Returns:
            str: Result produced by this operation.
        """
        return format_account_brief_plain_text(self)


def format_account_brief_plain_text(brief: AccountBrief) -> str:
    """Gmail-safe plain text (no markdown tables)."""
    lines: list[str] = [
        f"ACCOUNT BRIEF — {brief.company_name}",
        f"Generated: {brief.generated_at.isoformat()}",
        "",
    ]
    if brief.contact_name or brief.contact_email:
        who = brief.contact_name or ""
        if brief.contact_email:
            who = f"{who} <{brief.contact_email}>".strip()
        lines.extend(["Contact", who, ""])

    if brief.executive_summary.strip():
        lines.extend(["Summary", brief.executive_summary.strip(), ""])

    if brief.atlas_production is not None:
        ap = brief.atlas_production
        lines.append("Production (Atlas order book)")
        lines.append(f"- {ap.summary_line}")
        if ap.portfolio_updated_at:
            lines.append(f"- Portfolio updated: {ap.portfolio_updated_at}")
        lines.append("")

    if brief.shopfloor_truth is not None and (
        brief.shopfloor_truth.display_label or brief.shopfloor_truth.status == "contradiction"
    ):
        sf = brief.shopfloor_truth
        lines.append("Shopfloor truth (relationship label)")
        if sf.display_label:
            lines.append(f"- Treat as: {sf.display_label}")
        if sf.status == "contradiction":
            lines.append(
                f"- CONTRADICTION: CRM stage {sf.crm_stage or '(none)'} vs Atlas "
                f"{sf.atlas_relation} — do not call this a lead/prospect"
            )
        if sf.detail and sf.status == "contradiction":
            lines.append(f"- {sf.detail}")
        lines.append("")

    if brief.crm is not None:
        c = brief.crm
        lines.append("CRM")
        if c.stage:
            lines.append(f"- Stage: {c.stage}")
        if c.value is not None:
            cur = c.currency or "USD"
            lines.append(f"- Value: {cur} {c.value:,.0f}")
        if c.machine_model:
            lines.append(f"- Machine: {c.machine_model}")
        if c.last_activity:
            lines.append(f"- Last activity: {c.last_activity}")
        if c.contact_count:
            lines.append(f"- Contacts in CRM: {c.contact_count}")
        lines.append("")

    if brief.mail is not None:
        m = brief.mail
        lines.append("Mail")
        if m.last_sent:
            lines.append(f"- Last sent: {m.last_sent}")
        if m.last_received:
            lines.append(f"- Last received: {m.last_received}")
        if m.subject_hint:
            lines.append(f"- Subject: {m.subject_hint}")
        lines.append("")

    if brief.kb_highlights:
        lines.append("Knowledge base")
        for h in brief.kb_highlights[:5]:
            lines.append(f"- {h}")
        lines.append("")

    if brief.website_profile_excerpt and brief.website_profile_excerpt.strip():
        lines.append("Website profile (cached site research)")
        for ln in brief.website_profile_excerpt.strip().splitlines():
            lines.append(ln if ln.startswith("-") else f"- {ln}")
        lines.append("")

    if brief.proof_links:
        lines.append("Approved proof links")
        for url in brief.proof_links[:5]:
            lines.append(f"- {url}")
        lines.append("")

    if brief.account_journey is not None:
        j = brief.account_journey
        lines.append("Account journey")
        lines.append(f"- Messages scanned: {j.total_messages_scanned}")
        lines.append(f"- Threads scanned: {j.total_threads_scanned}")
        if j.yearly_narrative:
            lines.append("- Yearly highlights:")
            for block in j.yearly_narrative[:6]:
                if block.highlights:
                    lines.append(f"  {block.year}: {block.highlights[0]}")
        if j.meeting_context.current_need:
            lines.append(f"- Current need: {j.meeting_context.current_need}")
        lines.append("")

    if brief.timeline:
        lines.append("Timeline")
        for ev in brief.timeline[:12]:
            prefix = f"{ev.date}: " if ev.date else ""
            src = f" [{ev.source}]" if ev.source else ""
            lines.append(f"- {prefix}{ev.summary}{src}")
        lines.append("")

    if brief.risks_gaps:
        lines.append("Risks / gaps")
        for r in brief.risks_gaps[:8]:
            lines.append(f"- {r}")
        lines.append("")

    if brief.evidence_legs:
        from brain_os.services.evidence_freshness import format_evidence_freshness_block

        block = format_evidence_freshness_block(brief.evidence_legs)
        if block:
            lines.extend([block, ""])

    if brief.math_advisory is not None:
        m = brief.math_advisory
        lines.extend(
            [
                "Math advisory (advisory-only — not a send trigger)",
                f"- {m.math_summary}",
                f"- Readiness: {m.readiness:.0f} ({m.readiness_confidence})",
                f"- Cadence: {m.cadence:.0f} ({m.cadence_confidence})",
                f"- Priority: {m.action_priority:.0f} ({m.priority_confidence})",
                f"- Next action: {m.next_action}",
                "",
            ]
        )
        if m.gaps:
            lines.append("Math gaps")
            for g in m.gaps[:6]:
                lines.append(f"- {g}")
            lines.append("")

    if brief.suggested_opener and brief.suggested_opener.strip():
        lines.extend(["Suggested opener", brief.suggested_opener.strip(), ""])

    if brief.graph_context_lines:
        lines.append("Graph context (Neo4j 1-hop)")
        for ln in brief.graph_context_lines[:10]:
            lines.append(f"- {ln}")
        lines.append("")

    if brief.similar_companies:
        lines.append("Similar accounts (embedding)")
        for s in brief.similar_companies[:5]:
            hint = f" — {s.snippet}" if s.snippet else ""
            lines.append(f"- {s.company_name} (score {s.score:.2f}){hint}")
        lines.append("")

    if brief.precedents:
        lines.append("Precedents (prior operator runs)")
        for p in brief.precedents[:5]:
            ok = ""
            if p.success is True:
                ok = " [sent/ok]"
            elif p.success is False:
                ok = " [failed]"
            kind = p.kind.replace("_", " ")
            lines.append(f"- {kind}{ok}: {p.summary or 'no summary'} ({p.match_reason or 'prior'})")
        lines.append("")

    if brief.sources:
        lines.append("Sources")
        for s in brief.sources[:15]:
            label = s.label or s.channel
            fresh = f" ({s.freshness})" if s.freshness else ""
            lines.append(f"- {label}{fresh}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def brief_context_for_llm(payload: dict[str, Any]) -> str:
    """Serialize synthesis input for the LLM user message."""
    return AccountBriefSynthesisInput.model_validate(payload).model_dump_json(indent=2)
