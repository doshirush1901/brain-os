"""Shared FastAPI request/response Pydantic models.

Models used by multiple route modules (after W2 STEP 3) live here.
Single-domain-only models may move beside their router in a later pass.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from brain_os.schemas.run_record import RunRecord, RunRecordSummary


class QueryRequest(BaseModel):
    query: str
    user_id: str | None = None
    context: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    response: str
    agents_consulted: list[str] | None = None
    delegation_hops: list[dict[str, Any]] | None = None
    #: Correlates logs, Graphe, and write-contract receipts for this turn (UUID hex).
    run_id: str = ""
    run_summary: RunRecordSummary | None = None
    #: Selected pipeline metadata pass-through (phase-2 hardening).
    #: When the response came from a dedup cache hit, ``metadata["dedup"]`` is
    #: ``{"hit": "redis"|"inproc", "original_run_id": ..., "original_agents_used": [...]}``.
    metadata: dict[str, Any] | None = None


class QueryAgentRequest(BaseModel):
    """Ask a specific agent directly (bypasses pipeline routing). Use for Cursor/Ira flows where Ira must contribute a minimum share."""

    query: str
    agent_name: str  # e.g. "themis", "calliope", "clio"
    user_id: str | None = None
    context: dict[str, Any] | None = None


class BoardMeetingRequest(BaseModel):
    topic: str
    participants: list[str] | None = None


class FeedbackRequest(BaseModel):
    correction: str
    previous_query: str
    previous_response: str
    user_id: str | None = None
    severity: str = "HIGH"
    run_id: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    polarity: str
    correction_id: int | None = None
    micro_learning_triggered: bool = False
    procedure_reinforced: bool = False
    praise_recorded: bool = False
    graphe_meta_patched: bool = False


class EmailSearchRequest(BaseModel):
    from_address: str = ""
    to_address: str = ""
    subject: str = ""
    label: str = ""
    query: str = ""
    after: str = ""
    before: str = ""
    max_results: int = 10


class EmailDraftRequest(BaseModel):
    to: str
    subject: str
    context: str
    tone: str = "professional"
    thread_id: str = ""
    intent: str = "meeting_request"
    date_options: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    #: Operator override — skip triangulation gate (SOUL outbound enforcement).
    skip_triangulation: bool = False
    #: Force rebuild account brief (ignore recent operator_context snapshot).
    force_triangulation: bool = False
    #: When triangulation blocks, return structured JSON in ``detail`` (422).
    triangulation_json: bool = False
    #: Attach claim_map + Aletheia outbound check to response (never blocks unless enforce).
    verify_outbound_claims: bool = False


class EmailSendRequest(BaseModel):
    """Payload for sending an email (only when user explicitly said 'send')."""

    to: str
    subject: str
    body: str
    cc: str | None = None
    thread_id: str | None = None
    #: Gmail account to send from (e.g. sales@example.com). Omit to use primary token.
    from_mailbox: str | None = None
    #: If True, remove INBOX after send (when ``thread_id`` set). If None, use ``APP__EMAIL_ARCHIVE_THREAD_AFTER_SEND``.
    archive_from_inbox: bool | None = None
    #: Paths relative to repo root or absolute; must exist on the server host.
    attachment_paths: list[str] | None = None
    #: Stable campaign slug for idempotency (``cid:{campaign_id}:{recipient}``). Omit for one-off UI sends.
    campaign_id: str | None = None
    #: Explicit dedupe key (overrides ``campaign_id`` when both set). Max 512 chars.
    idempotency_key: str | None = None
    #: Optional interconnection brief id (required for campaign sends when configured).
    brief_id: str | None = None
    #: When set, a successful Gmail send updates ``data/revenue_mode/`` pipeline to ``sent`` and outcome ``sent``.
    pipeline_company_name: str | None = None
    run_id: str | None = None


class EmailInterconnectionsRequest(BaseModel):
    thread_id: str = ""
    recipient_email: str
    intent: str = "meeting_request"
    target_outcome: str = "book_meeting"
    context_text: str = ""
    date_options: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)


class EmailDraftFromBriefRequest(BaseModel):
    brief_id: str
    to: str
    subject_hint: str = ""
    tone: str = "professional"


class SchedulingWindowRequest(BaseModel):
    date: str
    start_time: str
    end_time: str
    timezone: str | None = None


class SchedulingProposeRequest(BaseModel):
    contact_email: str
    contact_name: str | None = None
    purpose: str = "a quick meeting"
    windows: list[SchedulingWindowRequest]
    duration_minutes: int = 30
    max_slots: int = 5
    buffer_minutes: int | None = None
    slot_step_minutes: int | None = None
    thread_id: str | None = None
    owner_mailbox: str | None = None
    public_base_url: str | None = None


class SchedulingBookRequest(BaseModel):
    token: str
    booked_email: str | None = None
    attendee_name: str | None = None


class OutboundMessageRequest(BaseModel):
    to: str
    subject: str
    body: str


class OutboundDraftBatchRequest(BaseModel):
    campaign_name: str
    created_by: str
    messages: list[OutboundMessageRequest]


class OutboundApproveRequest(BaseModel):
    batch_id: str
    approved_by: str


class OutboundRejectRequest(BaseModel):
    batch_id: str
    rejected_by: str
    reason: str = ""


class OperatorInboxDecideRequest(BaseModel):
    item_id: str
    decision: str  # approve | reject | snooze
    actor: str = "web_operator"
    snooze_days: int = Field(default=7, ge=1, le=90)
    to_address: str | None = None


class OperatorReleaseRequest(BaseModel):
    actor: str = "web_operator"
    ttl_hours: float | None = None
    force: bool = False


class TaskRequest(BaseModel):
    goal: str
    user_id: str | None = None
    output_format: str = "markdown"


class TaskClarificationRequest(BaseModel):
    task_id: str
    answer: str
    user_id: str | None = None


class TaskAbortRequest(BaseModel):
    task_id: str
    reason: str = ""


class TaskRetryRequest(BaseModel):
    task_id: str
    from_phase: int | None = None


class AnuScoreRequest(BaseModel):
    candidate_profile: dict[str, Any]
    job_description: str = ""


class AnuChatRequest(BaseModel):
    candidate_profile: dict[str, Any]
    message: str
    conversation_history: list[dict[str, str]] | None = None


class AnuParseTextRequest(BaseModel):
    resume_text: str


class AnuCvParsedUpdateRequest(BaseModel):
    """Body for updating a candidate's CV-parsed profile (from Anu parse-resume)."""

    candidate_profile: dict[str, Any]


class AnuDraftRecruitmentStage2Request(BaseModel):
    """Inputs for drafting Stage 2 recruitment email (case study + DICE + skills)."""

    candidate_name: str = ""
    role: str = ""
    case_study_text: str = ""
    dice_questions: str = ""
    skills_questions: str = ""
    company_intro_short: str = ""
    job_description_or_context: str = ""


class AnuExportRequest(BaseModel):
    candidate_profile: dict[str, Any]
    scoring: dict[str, Any] | None = None
    format: str = "text"  # "text" or "pdf"


class RecruitmentUpsertRequest(BaseModel):
    """Upsert a candidate in the recruitment database."""

    email: str
    name: str | None = None
    phone: str | None = None
    role_applied: str | None = None
    profile: dict[str, Any] | None = None
    cv_parsed: dict[str, Any] | None = None
    score: dict[str, Any] | None = None
    ctc_current: str | None = None
    source_type: str | None = None
    source_id: str | None = None
    notes: str | None = None


class RecruitmentStageEventRequest(BaseModel):
    """Record a stage event for a candidate (e.g. stage2_sent, call_invited)."""

    stage: str
    event_at: str | None = None  # ISO datetime; default now
    metadata: dict[str, Any] | None = None


class RecruitmentCandidateUpdateRequest(BaseModel):
    """Update candidate fields (ctc_current, notes, etc.)."""

    ctc_current: str | None = None
    notes: str | None = None
    name: str | None = None
    phone: str | None = None
    role_applied: str | None = None


class RecruitmentScoreRequest(BaseModel):
    """Request to compute and store applicant score (dimension-based)."""

    role_applied: str = "Procurement"
    stage2_response_text: str = (
        ""  # Optional: paste Stage 2 reply for case_study/behavioural scoring
    )


class SyncApolloRequest(BaseModel):
    """Optional body for POST /api/crm/sync-apollo."""

    dry_run: bool = False
    limit: int | None = None
    contact_type: str | None = None
    contacts_only: bool = False


class SeedActive21Request(BaseModel):
    """Optional body for POST /api/crm/seed-active-21."""

    dry_run: bool = False
    enrich: bool = False
    max_messages: int = 120


class EnrichProgrammeRequest(BaseModel):
    """Body for POST /api/crm/enrich-programme."""

    programme_id: str = "active_21"
    dry_run: bool = False
    max_messages: int = 120


class ReingestRequest(BaseModel):
    min_file_size_mb: int = 5
    base_path: str = "data/imports"
    min_chars_per_page: int = 25


class MemoryStoreRequest(BaseModel):
    content: str
    user_id: str = "global"
    metadata: dict[str, Any] | None = None


class RunRecordListResponse(BaseModel):
    enabled: bool = True
    count: int = 0
    runs: list[RunRecordSummary] = Field(default_factory=list)


class RunRecordResponse(BaseModel):
    enabled: bool = True
    record: RunRecord


class EmailTrashRequest(BaseModel):
    message_id: str


class EmailRescanRequest(BaseModel):
    after: str = "2023/03/08"
    before: str = "2026/03/08"
    dry_run: bool = False
    resume: bool = False
    throttle: float = 0.1
    skip_crm_populate: bool = False
    user_id: str | None = None


class EmailAccountJourneyRequest(BaseModel):
    company: str
    contact_email: str | None = None
    domain: str | None = None
    after: str = ""
    before: str = ""
    mailbox: str | None = None
    include_attachments: bool = True
    resume: bool = False


__all__ = [
    "AnuChatRequest",
    "AnuCvParsedUpdateRequest",
    "AnuDraftRecruitmentStage2Request",
    "AnuExportRequest",
    "AnuParseTextRequest",
    "AnuScoreRequest",
    "BoardMeetingRequest",
    "EmailAccountJourneyRequest",
    "EmailDraftFromBriefRequest",
    "EmailDraftRequest",
    "EmailInterconnectionsRequest",
    "EmailRescanRequest",
    "EmailSearchRequest",
    "EmailSendRequest",
    "EmailTrashRequest",
    "EnrichProgrammeRequest",
    "FeedbackRequest",
    "FeedbackResponse",
    "MemoryStoreRequest",
    "OutboundApproveRequest",
    "OutboundDraftBatchRequest",
    "OutboundMessageRequest",
    "OutboundRejectRequest",
    "QueryAgentRequest",
    "QueryRequest",
    "QueryResponse",
    "RecruitmentCandidateUpdateRequest",
    "RecruitmentScoreRequest",
    "RecruitmentStageEventRequest",
    "RecruitmentUpsertRequest",
    "ReingestRequest",
    "RunRecordListResponse",
    "RunRecordResponse",
    "SchedulingBookRequest",
    "SchedulingProposeRequest",
    "SchedulingWindowRequest",
    "SeedActive21Request",
    "SyncApolloRequest",
    "TaskAbortRequest",
    "TaskClarificationRequest",
    "TaskRequest",
    "TaskRetryRequest",
]
