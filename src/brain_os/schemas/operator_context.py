"""Operator context runs — precedents for account brief / outbound prep."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

OperatorContextKind = Literal[
    "account_brief",
    "tinder_draft",
    "quote_prep",
    "persuasion_sprint",
]
OperatorContextOutcome = Literal["ok", "draft_created", "sent", "skipped", "error"]


class ContextPrecedent(BaseModel):
    """Compact prior operator run for the same account or similar context."""

    run_id: str
    kind: OperatorContextKind
    ts: float
    company_name: str = ""
    outcome: OperatorContextOutcome = "ok"
    success: bool | None = None
    machine_model: str | None = None
    crm_stage: str | None = None
    summary: str = ""
    match_reason: str = Field(
        default="",
        description="e.g. same_company, same_domain, same_machine_model",
    )


class OperatorContextRun(BaseModel):
    """Persisted high-signal operator action (brief, Tinder draft, quote prep)."""

    run_id: str
    ts: float
    kind: OperatorContextKind
    company_key: str
    company_name: str = ""
    domain: str | None = None
    contact_email: str | None = None
    machine_model: str | None = None
    crm_stage: str | None = None
    outcome: OperatorContextOutcome = "ok"
    success: bool | None = None
    pipeline_run_id: str | None = None
    summary: str = ""
    brief_snapshot: dict[str, Any] | None = Field(
        default=None,
        description="Serialized AccountBrief for triangulation reuse (account_brief kind).",
    )
