"""Pydantic models for the unified operator approval inbox."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

OperatorInboxKind = Literal["outbound_email", "lead_review", "quote_draft"]
OperatorInboxRisk = Literal["internal", "external_visible"]
OperatorInboxDecision = Literal["approve", "reject", "snooze"]


class OperatorInboxItem(BaseModel):
    """One scannable approval card in the operator queue."""

    id: str
    kind: OperatorInboxKind
    title: str
    subtitle: str = ""
    company_name: str = ""
    risk: OperatorInboxRisk = "internal"
    preview: dict[str, Any] = Field(default_factory=dict)
    source: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class OperatorInboxSummary(BaseModel):
    """Aggregate counts for the inbox header."""

    outbound_email: int = 0
    lead_review: int = 0
    quote_draft: int = 0
    total: int = 0
    external_pending: int = 0


class OperatorInboxPayload(BaseModel):
    """Full inbox response."""

    summary: OperatorInboxSummary
    items: list[OperatorInboxItem] = Field(default_factory=list)
    released: bool = False
    released_until: str | None = None
    math_advisory_enabled: bool = False
    math_priority_hints: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top accounts by latest shadow/advisory priority (phase 3).",
    )
    math_promote_checklist: dict[str, Any] = Field(
        default_factory=dict,
        description="Phase 4: candidate/active weights + backtest gate checklist.",
    )


class OperatorReleaseSession(BaseModel):
    """Time-boxed operator release (Turn Ira loose)."""

    released_until: str | None = None
    released_by: str | None = None
    pending_at_release: int = 0
    released: bool = False
