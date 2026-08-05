"""Structured run records for pipeline observability (Phase 0+1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RunOutcome = Literal["ok", "early_exit", "error", "timeout", "budget_gate"]

_SUMMARY_INPUT_MAX = 500
_SUMMARY_RESPONSE_MAX = 500


class DelegationHop(BaseModel):
    """One ``ask_agent`` delegation hop."""

    from_agent: str = Field(..., alias="from")
    to_agent: str = Field(..., alias="to")
    duration_ms: int = 0
    ok: bool = True

    model_config = {"populate_by_name": True}


class ToolInvocationRef(BaseModel):
    """Normalized tool call from pipeline ``_tool_audit``."""

    agent: str = ""
    tool: str
    success: bool = True
    duration_ms: int | None = None
    error_code: str | None = None


class EvidenceRef(BaseModel):
    """Lightweight evidence pointer (no full chunk bodies)."""

    kind: Literal["kb", "email_thread"]
    ref: str
    score: float | None = None


class SafetyFlags(BaseModel):
    """Safety / compliance signals for the run."""

    had_provenance_warning: bool = False
    had_dlp_flag: bool = False
    faithfulness_tier: str | None = None
    caveat_appended: bool = False


class RouteInfo(BaseModel):
    """Routing method and telemetry snapshot."""

    method: str = ""
    telemetry: dict[str, Any] = Field(default_factory=dict)


class BranchTelemetry(BaseModel):
    """Early-branch audit fields (Letta peer spike; derived from trace when not explicit).

    Maps to ``docs/LETTA_PEER_BRANCHING_SPIKE.md``. Populated at run-record assembly
    from ``trace`` / ``meta`` so operators can see why a request took a cheap path.
    """

    perceive_class: str = ""
    fast_memory_path: bool | None = None
    memory_confidence: float | None = None
    domain_match: float | None = None
    route_skip: bool | None = None
    bypass_cheap_exits: bool | None = None
    post_remember_node: str | None = None
    branch_source: Literal["derived", "explicit"] = "derived"


class QualityInfo(BaseModel):
    """Quality and timing metadata."""

    confidence: float | None = None
    metis_score: int | None = None
    stage_timings: dict[str, float] = Field(default_factory=dict)


class CostEstimate(BaseModel):
    """Heuristic cost signals (not billing-grade)."""

    tokens_heuristic: int = 0


class RunRecordArtifacts(BaseModel):
    """Cross-run links (dedup, tasks, feedback, outbound send, etc.)."""

    dedup: dict[str, Any] | None = None
    links: list[dict[str, Any]] = Field(default_factory=list)


class RunRecordSummary(BaseModel):
    """Compact list view for Phase 2 APIs."""

    run_id: str
    ts_end: float
    channel: str
    outcome: RunOutcome
    route_method: str = ""
    agents_count: int = 0
    early_exit: str | None = None
    transition_phases: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    """Full persisted evidence bundle for one pipeline run."""

    run_id: str
    ts_start: float
    ts_end: float
    duration_ms: int
    channel: str
    sender_id: str
    outcome: RunOutcome
    early_exit: str | None = None
    input_summary: str = ""
    response_summary: str = ""
    route: RouteInfo = Field(default_factory=RouteInfo)
    email_scope: str | None = None
    agents: list[str] = Field(default_factory=list)
    delegations: list[DelegationHop] = Field(default_factory=list)
    tools: list[ToolInvocationRef] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    safety: SafetyFlags = Field(default_factory=SafetyFlags)
    quality: QualityInfo = Field(default_factory=QualityInfo)
    cost_estimate: CostEstimate = Field(default_factory=CostEstimate)
    artifacts: RunRecordArtifacts = Field(default_factory=RunRecordArtifacts)
    transitions: dict[str, Any] = Field(default_factory=dict)
    branch_telemetry: BranchTelemetry | None = None

    def summary(self) -> RunRecordSummary:
        """Perform the summary operation.

        Returns:
            RunRecordSummary: Result produced by this operation.
        """
        phases = [str(k) for k in self.transitions if k]
        return RunRecordSummary(
            run_id=self.run_id,
            ts_end=self.ts_end,
            channel=self.channel,
            outcome=self.outcome,
            route_method=self.route.method,
            agents_count=len(self.agents),
            early_exit=self.early_exit,
            transition_phases=phases,
        )


def truncate_summary(text: str, *, max_len: int) -> str:
    """Perform the truncate summary operation.

    Args:
        text (str): Parameter used by this operation.
        max_len (int): Parameter used by this operation.

    Returns:
        str: Result produced by this operation.
    """
    s = (text or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
