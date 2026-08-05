"""Pydantic models for structured LLM outputs.
Every JSON schema that was previously parsed via ``json.loads()`` from raw
LLM text is defined here as a Pydantic model.  These models are used with
``LLMClient.generate_structured()`` for type-safe, validated responses.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from brain_os.schemas.outbound_voice import OutboundVoiceRubric as OutboundVoiceRubric

# ── Email intent gold-set calibration (filename labels + classifier eval) ─


class EmailIntentGoldBucket(StrEnum):
    """Coarse buckets aligned with `examples/acme/docs/28_Emails Gold` filename prefixes."""

    SALES_LEAD = "SALES_LEAD"
    CLIENT_ACCOUNT = "CLIENT_ACCOUNT"
    HR_RECRUITMENT = "HR_RECRUITMENT"
    INTERNAL_MC = "INTERNAL_MC"


class EmailIntentGoldClassifierOutput(BaseModel):
    """Structured output for calibrating / evaluating inbound email triage."""

    bucket: EmailIntentGoldBucket
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=4000)


# ── ReAct loop ────────────────────────────────────────────────────────────


class ToolCall(BaseModel):
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ReActDecision(BaseModel):
    thought: str = ""
    tool_to_use: ToolCall | None = None
    final_answer: str | None = None


# ── Feedback ──────────────────────────────────────────────────────────────


class FeedbackClassification(BaseModel):
    polarity: str = "neutral"
    confidence: float = 0.5
    extracted_correction: str | None = None


# ── Retriever ─────────────────────────────────────────────────────────────


class EntityNames(BaseModel):
    entities: list[str] = Field(default_factory=list)


class SubQueries(BaseModel):
    queries: list[str] = Field(default_factory=list)


class CitationSpan(BaseModel):
    """One response fragment linked to an evidence chunk index."""

    claim_snippet: str = Field(default="", max_length=500)
    evidence_index: int = Field(
        default=0, ge=0, description="0-based index into enumerated evidence"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CitationAlignmentResult(BaseModel):
    """Structured citation alignment for optional provenance overlays."""

    spans: list[CitationSpan] = Field(default_factory=list, max_length=24)


# ── Imports metadata ──────────────────────────────────────────────────────


class DocumentMetadata(BaseModel):
    summary: str = ""
    doc_type: str = ""
    machines: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    intent_tags: list[str] = Field(default_factory=list)
    counterparty_type: str = "unknown"
    document_role: str = "other"
    intent_confidence: dict[str, float] = Field(default_factory=dict)


# ── Knowledge graph ───────────────────────────────────────────────────────


class CompanyEntity(BaseModel):
    name: str = ""
    region: str = ""
    industry: str = ""
    website: str = ""


class PersonEntity(BaseModel):
    name: str = ""
    email: str = ""
    company: str = ""
    role: str = ""


class MachineEntity(BaseModel):
    model: str = ""
    category: str = ""
    description: str = ""


class ProjectEntity(BaseModel):
    project_id: str = ""
    customer: str = ""
    machine_model: str = ""
    status: str = ""


class ApplicationEntity(BaseModel):
    name: str = ""
    description: str = ""


class MaterialEntity(BaseModel):
    name: str = ""
    category: str = ""


class ExhibitionEntity(BaseModel):
    name: str = ""
    location: str = ""
    year: str = ""


class GraphRelationship(BaseModel):
    from_type: str = ""
    from_key: str = ""
    rel: str = ""
    to_type: str = ""
    to_key: str = ""
    properties: dict[str, str] = Field(default_factory=dict)


class GraphEntities(BaseModel):
    companies: list[CompanyEntity] = Field(default_factory=list)
    people: list[PersonEntity] = Field(default_factory=list)
    machines: list[MachineEntity] = Field(default_factory=list)
    projects: list[ProjectEntity] = Field(default_factory=list)
    applications: list[ApplicationEntity] = Field(default_factory=list)
    materials: list[MaterialEntity] = Field(default_factory=list)
    exhibitions: list[ExhibitionEntity] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)


# ── Digestive system ─────────────────────────────────────────────────────


class NutrientClassification(BaseModel):
    protein: list[str] = Field(default_factory=list)
    carbs: list[str] = Field(default_factory=list)
    waste: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def truncate_lists(self) -> NutrientClassification:
        """Cap list lengths so LLM output never exceeds token limit when parsed."""
        max_protein_carbs = 30
        max_waste = 15
        if (
            len(self.protein) <= max_protein_carbs
            and len(self.carbs) <= max_protein_carbs
            and len(self.waste) <= max_waste
        ):
            return self
        return self.model_copy(
            update={
                "protein": self.protein[:max_protein_carbs],
                "carbs": self.carbs[:max_protein_carbs],
                "waste": self.waste[:max_waste],
            }
        )


class DigestiveSummary(BaseModel):
    statements: list[str] = Field(default_factory=list)


class EmailSenderInfo(BaseModel):
    name: str = ""
    email: str = ""
    company: str = ""
    role: str = ""


class EmailMetadata(BaseModel):
    sender_info: EmailSenderInfo = Field(default_factory=EmailSenderInfo)
    company_mentions: list[str] = Field(default_factory=list)
    machine_mentions: list[str] = Field(default_factory=list)
    pricing_mentions: list[str] = Field(default_factory=list)
    dates_deadlines: list[str] = Field(default_factory=list)


class ExtractedContact(BaseModel):
    """Single contact extracted from unstructured text (e.g. PDF, email body)."""

    name: str = ""
    email: str = ""
    company: str = ""
    machine_model: str = ""


class ExtractedContacts(BaseModel):
    """List of contacts extracted from text for CRM populator."""

    contacts: list[ExtractedContact] = Field(default_factory=list)


# ── Emotional intelligence ───────────────────────────────────────────────


class EmotionDetection(BaseModel):
    state: str = "NEUTRAL"
    intensity: str = "MILD"
    indicators: list[str] = Field(default_factory=list)


# ── Goal manager ─────────────────────────────────────────────────────────


class GoalDetection(BaseModel):
    should_initiate: bool = False
    goal_type: str = ""
    reason: str = ""


# ── Agent loop standing objectives ─────────────────────────────────────────


class StandingGoalJudgeResult(BaseModel):
    """Structured verdict for whether a standing objective is satisfied."""

    verdict: str = Field(
        default="continue",
        description="done or continue",
    )
    reason: str = Field(default="", max_length=2000)


class GoalSlots(BaseModel):
    slots: dict[str, str | None] = Field(default_factory=dict)


# ── Inner voice ──────────────────────────────────────────────────────────


class InnerReflection(BaseModel):
    reflection_type: str = "OBSERVATION"
    content: str = ""
    should_surface: bool = False


# ── Metacognition ────────────────────────────────────────────────────────


class KnowledgeAssessment(BaseModel):
    state: str = "PARTIAL"
    confidence: float = 0.5
    conflicts: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


# ── Conversation memory ──────────────────────────────────────────────────


class ConversationEntities(BaseModel):
    companies: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)
    machines: list[str] = Field(default_factory=list)
    quote_ids: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)


# ── Episodic memory ──────────────────────────────────────────────────────


class EpisodeConsolidation(BaseModel):
    narrative: str = ""
    key_topics: list[str] = Field(default_factory=list)
    decisions_made: list[str] = Field(default_factory=list)
    commitments: list[str] = Field(default_factory=list)
    memorable_moments: list[str] = Field(default_factory=list)
    emotional_tone: str = ""
    relationship_impact: str = "maintained"


# ── Relationship memory ──────────────────────────────────────────────────


class MemorableMoment(BaseModel):
    type: str = ""
    content: str = ""
    importance: str = "medium"


class MemorableMoments(BaseModel):
    moments: list[MemorableMoment | str] = Field(default_factory=list)


# ── Procedural memory ────────────────────────────────────────────────────


class PatternExtraction(BaseModel):
    trigger: str = ""
    steps: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    category: str = ""


# ── Dream mode ────────────────────────────────────────────────────────────


class DreamPattern(BaseModel):
    description: str = ""
    frequency: int = 0
    examples: list[str] = Field(default_factory=list)


class DreamContradiction(BaseModel):
    description: str = ""
    sources: list[str] = Field(default_factory=list)


class MemoryContradictionCandidate(BaseModel):
    """Cross-store contradiction found during dream stage 3i."""

    entity: str = ""
    claim_a: str = ""
    claim_b: str = ""
    source_a: str = ""
    source_b: str = ""
    correct_value: str = ""
    wrong_value: str = ""
    confidence: str = "LOW"
    category: str = "GENERAL"
    rationale: str = ""


class MemoryReconciliationResult(BaseModel):
    """Structured contradiction candidates returned by the reconciliation model."""

    contradictions: list[MemoryContradictionCandidate] = Field(default_factory=list)


class DreamInsightItem(BaseModel):
    insight: str = ""
    confidence: str = ""
    evidence: list[str] = Field(default_factory=list)


class DreamRecommendation(BaseModel):
    action: str = ""
    priority: str = ""
    rationale: str = ""


class DreamInsight(BaseModel):
    patterns: list[DreamPattern] = Field(default_factory=list)
    contradictions: list[DreamContradiction] = Field(default_factory=list)
    insights: list[DreamInsightItem] = Field(default_factory=list)
    recommendations: list[DreamRecommendation] = Field(default_factory=list)


class DreamGap(BaseModel):
    topic: str = ""
    description: str = ""
    priority: str = ""
    related_queries: list[str] = Field(default_factory=list)


class DreamGaps(BaseModel):
    gaps: list[DreamGap] = Field(default_factory=list)


class DreamConnection(BaseModel):
    insight: str = ""
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence: str = ""


class DreamCreative(BaseModel):
    connections: list[DreamConnection] = Field(default_factory=list)


class DreamCampaignInsights(BaseModel):
    insights: list[str] = Field(default_factory=list)


class DreamProcedure(BaseModel):
    trigger: str = ""
    steps: list[str] = Field(default_factory=list)
    expected_outcome: str = ""
    confidence: str = ""


class DreamProcedures(BaseModel):
    procedures: list[DreamProcedure] = Field(default_factory=list)


class PredictionProcedureCandidate(BaseModel):
    trigger: str = ""
    steps: list[str] = Field(default_factory=list)
    confidence: str = ""


class PredictionReflection(BaseModel):
    lessons: list[str] = Field(default_factory=list)
    false_positive_patterns: list[str] = Field(default_factory=list)
    procedure_candidates: list[PredictionProcedureCandidate] = Field(default_factory=list)


class DreamPruneSummary(BaseModel):
    ids: list[int] = Field(default_factory=list)
    summary: str = ""


class DreamPrune(BaseModel):
    keep: list[int] = Field(default_factory=list)
    summarise: list[DreamPruneSummary] = Field(default_factory=list)
    archive: list[int] = Field(default_factory=list)


# ── Context compaction (Hermes-inspired rolling summary) ─────────────────


class CompactionSummary(BaseModel):
    """Structured rolling compaction for pipeline enrichment or ReAct scratchpad."""

    goal: str = ""
    done: list[str] = Field(default_factory=list)
    in_progress: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    key_files: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


# ── Sleep trainer ─────────────────────────────────────────────────────────


class TruthHint(BaseModel):
    pattern: str = ""
    answer: str = ""
    entity: str = ""
    category: str = ""


class TruthHints(BaseModel):
    hints: list[TruthHint] = Field(default_factory=list)


# ── Realtime observer ────────────────────────────────────────────────────


class ObservedTurn(BaseModel):
    facts: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)


# ── Knowledge discovery ──────────────────────────────────────────────────


class KnowledgeGap(BaseModel):
    gap_type: str = ""
    description: str = ""
    suggested_search: str = ""


class DeepFact(BaseModel):
    content: str = ""
    entity: str = ""
    category: str = ""


class DeepFacts(BaseModel):
    facts: list[DeepFact] = Field(default_factory=list)


# ── Learning hub ─────────────────────────────────────────────────────────


class CorrectionAnalysis(BaseModel):
    error_category: str = ""
    what_was_wrong: str = ""
    correct_behaviour: str = ""


class GapAnalysis(BaseModel):
    gap_type: str = ""
    description: str = ""
    suggested_skill_name: str = ""
    suggested_skill_description: str = ""
    suggested_knowledge_source: str = ""


class ProcedureSteps(BaseModel):
    steps: list[str] = Field(default_factory=list)


class SuccessPlaybookExtraction(BaseModel):
    """Structured playbook extracted from a praised turn + method trace."""

    trigger_pattern: str = ""
    steps: list[str] = Field(default_factory=list)
    skill_hint: str = ""


# ── Quotes ────────────────────────────────────────────────────────────────


class MachineInfo(BaseModel):
    machine_model: str = ""
    configuration: dict[str, Any] = Field(default_factory=dict)


# ── Guardrails ────────────────────────────────────────────────────────────


class UnsupportedClaim(BaseModel):
    claim: str = ""
    reason: str = ""


class FaithfulnessResult(BaseModel):
    faithful: bool = True
    score: float = 1.0
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)


class ConfidentialityResult(BaseModel):
    safe: bool = True
    leaked_categories: list[str] = Field(default_factory=list)
    flagged_snippets: list[str] = Field(default_factory=list)


# ── Mailbox → CRM digest (outbound sales threads) ─────────────────────────


class MailboxOutboundDigestRow(BaseModel):
    """One external contact’s consolidated view from Gmail threads (LLM-filled)."""

    contact_email: str = Field(..., max_length=320)
    display_name: str = Field(default="", max_length=500)
    company_inferred: str = Field(default="", max_length=500)
    conversation_summary: str = Field(
        default="",
        max_length=8000,
        description="Neutral narrative of what was discussed across the thread(s).",
    )
    machines_discussed: list[str] = Field(default_factory=list)
    tech_specs_notes: str = Field(default="", max_length=4000)
    price_or_commercial_notes: str = Field(
        default="",
        max_length=2000,
        description="Only facts present in the email text; otherwise say not stated.",
    )
    suggested_next_steps: list[str] = Field(default_factory=list)
    confidence_notes: str = Field(default="", max_length=800)


# ── Outreach prioritization workflow ───────────────────────────────────────


class OutreachRankedCandidate(BaseModel):
    rank: int = Field(default=1, ge=1, le=3)
    contact_name: str = Field(default="", max_length=255)
    contact_email: str = Field(default="", max_length=320)
    company: str = Field(default="", max_length=255)
    thread_id: str = Field(default="", max_length=128)
    evidence_line: str = Field(
        default="",
        max_length=2000,
        description="One concrete evidence line grounded in a recent thread.",
    )
    reason: str = Field(default="", max_length=1200)


class OutreachRankingOutput(BaseModel):
    recommendation: OutreachRankedCandidate
    ranked: list[OutreachRankedCandidate] = Field(default_factory=list, min_length=1, max_length=3)
    confidence: str = Field(default="medium", max_length=16)
    freshness: str = Field(default="current", max_length=32)
    notes: str = Field(default="", max_length=1200)

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> OutreachRankingOutput:
        ranked_sorted = sorted(
            self.ranked,
            key=lambda c: (c.rank, c.contact_email.lower(), c.contact_name.lower()),
        )[:3]
        seen: set[str] = set()
        deduped: list[OutreachRankedCandidate] = []
        for c in ranked_sorted:
            key = (c.contact_email or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        if not deduped:
            raise ValueError("ranked must include at least one candidate with contact_email")

        rec_email = (self.recommendation.contact_email or "").strip().lower()
        if not rec_email:
            raise ValueError("recommendation.contact_email is required")
        if rec_email not in {(c.contact_email or "").strip().lower() for c in deduped}:
            deduped.insert(0, self.recommendation)
        self.ranked = deduped[:3]
        return self


# ── Company screening / fit ranking ───────────────────────────────────────


class CompanyEvidence(BaseModel):
    source_url: str = ""
    quote: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CompanyScreeningProfile(BaseModel):
    company: str = ""
    website: str = ""
    country: str = ""
    address: str = ""
    contact_email: str = ""
    company_established_year: int | None = None
    approx_turnover_eur_million: float | None = None
    employee_count: int | None = None
    last_investment_period: str = ""
    known_machine_models: list[str] = Field(default_factory=list)
    no_of_thermoforming_machines: int | None = None
    no_of_cnc_machines: int | None = None
    services: list[str] = Field(default_factory=list)
    applications: list[str] = Field(default_factory=list)
    automotive_applications: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    max_part_capability_mm: str = ""
    min_part_capability_mm: str = ""
    twin_sheet_capability: bool | None = None
    high_pressure_forming_capability: bool | None = None
    competitor_risk: bool = False
    notes: str = ""
    evidence: list[CompanyEvidence] = Field(default_factory=list)


# ── NA industrial forming Firecrawl discovery (LLM validation) ─────────────────


class ThermoformingSiteFirecrawlLLMOutput(BaseModel):
    """Structured site classification for custom heavy-gauge industrial forming fit."""

    is_custom_heavy_gauge_thermoformer: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_phrases: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=2000)


class GaugeTier(StrEnum):
    """Sheet/process gauge for industrial forming buyers (DEMO vs packaging lines)."""

    HEAVY_GAUGE = "heavy_gauge"
    THIN_GAUGE = "thin_gauge"
    MIXED = "mixed"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


# ── India / generic industrial former-buyer classifier (Ollama-friendly) ───────


class CompanyCategory(StrEnum):
    """Coarse buckets for Acme Corp ICP triage from a homepage."""

    THERMOFORMER = "industrial former"  # runs industrial forming / vacuum forming lines (= buyer)
    MACHINE_BUILDER = "machine_builder"  # builds industrial forming / packaging machines (= peer)
    MOULD_MAKER = "mould_maker"  # tool / mould / die maker
    INJECTION_OR_BLOW_MOULDER = "injection_or_blow_moulder"
    EXTRUDER_OR_SHEET = "extruder_or_sheet"  # extrusion / sheet / film maker, no industrial forming
    FOAM_FABRICATOR = "foam_fabricator"  # foam cutting, dunnage, CNC — not industrial forming OEM buyer
    FLEX_PACK_OR_PRINTER = "flex_pack_or_printer"  # flexible packaging / printer / converter
    TRADER_OR_DISTRIBUTOR = "trader_or_distributor"
    INTERNAL_GROUP = "internal_group"  # Acme Corp sister / group / family entity (NOT a lead)
    UNRELATED = "unrelated"
    UNKNOWN = "unknown"


class ThermoformerLeadClassification(BaseModel):
    """Classify one company homepage as a Acme Corp industrial forming-machine BUYER (or not)."""

    category: CompanyCategory = CompanyCategory.UNKNOWN
    is_thermoformer: bool = False
    gauge_tier: GaugeTier = GaugeTier.UNKNOWN
    gauge_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_phrases: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="", max_length=1500)

    @model_validator(mode="after")
    def _gauge_tier_matches_category(self) -> ThermoformerLeadClassification:
        if self.category != CompanyCategory.THERMOFORMER:
            self.gauge_tier = GaugeTier.NOT_APPLICABLE
        elif self.gauge_tier == GaugeTier.NOT_APPLICABLE:
            self.gauge_tier = GaugeTier.UNKNOWN
        return self


# ── Task orchestration ────────────────────────────────────────────────────


class ClarityAssessment(BaseModel):
    clear: bool = True
    ambiguity_reason: str = ""
    clarifying_questions: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    can_answer_partially: bool = False
    suggested_default_scope: str = ""


class SocraticQuestionType(StrEnum):
    """Typed counter-question categories for Sphinx v2 Socratic gate."""

    SCOPE = "scope"
    TRADEOFF = "tradeoff"
    GROUND_TRUTH = "ground_truth"
    SUCCESS_CRITERIA = "success_criteria"
    ASSUMPTION = "assumption"


class SocraticQuestion(BaseModel):
    """One Socratic counter-question with a proposed default."""

    type: SocraticQuestionType = SocraticQuestionType.ASSUMPTION
    question: str = ""
    proposed_default: str = ""
    why_it_matters: str = ""
    candidates: list[str] = Field(default_factory=list)
    context_key: str = ""


class SocraticQuestionSet(BaseModel):
    """Strict LLM schema for Sphinx Socratic question generation (max 4)."""

    questions: list[SocraticQuestion] = Field(default_factory=list, max_length=4)
    ambiguity_summary: str = ""


class SocraticMappedAnswer(BaseModel):
    """One mapped answer keyed to a pending Socratic context_key."""

    context_key: str = ""
    answer: str = ""


class SocraticResumeMatch(BaseModel):
    """Strict schema: does this message answer the pending Socratic gate?"""

    is_answer: bool = False
    mapped_answers: list[SocraticMappedAnswer] = Field(default_factory=list)
    reason: str = ""


class ClarificationPayload(BaseModel):
    """Normalized clarify/can-proceed contract shared across runtime paths."""

    needs_clarification: bool = False
    questions: list[str] = Field(default_factory=list)
    reason: str = ""
    missing_slots: list[str] = Field(default_factory=list)
    can_answer_partially: bool = False


def parse_clarification_text(raw_text: str) -> ClarificationPayload:
    """Parse legacy tag-formatted clarify text into a normalized payload.

    Backward compatible with:
    - "[CLARIFY] <question text>"
    - "[CLEAR] <optional message>"
    """
    text = (raw_text or "").strip()
    upper = text.upper()
    if upper.startswith("[CLARIFY]"):
        body = text[len("[CLARIFY]") :].strip()
        questions = [line.strip("- ").strip() for line in body.splitlines() if line.strip()]
        if not questions and body:
            questions = [body]
        return ClarificationPayload(
            needs_clarification=True,
            questions=questions,
            reason="Agent requested clarification",
        )
    if upper.startswith("[CLEAR]"):
        return ClarificationPayload(needs_clarification=False)
    return ClarificationPayload(needs_clarification=False)


class TaskPlanPhase(BaseModel):
    title: str = ""
    agent: str = ""
    description: str = ""
    depends_on: list[int] = Field(default_factory=list)


class TaskPlanValidationContract(BaseModel):
    """Per-task assertion contract (CLI ``brain task`` / TaskOrchestrator)."""

    assertions: list[str] = Field(default_factory=list)
    phase_assertion_map: dict[int, list[int]] = Field(default_factory=dict)
    completion_gate: str = ""
    validator_mode: str = "strict"


class TaskPlan(BaseModel):
    goal: str = ""
    phases: list[TaskPlanPhase] = Field(default_factory=list)
    reasoning: str = ""
    validation_contract: TaskPlanValidationContract | None = None


# ── GEPA strategy overlay compile (Dream batch) ──────────────────────────


class GepaStrategyOverlayItem(BaseModel):
    agent: str = Field(default="", max_length=64)
    pattern: str = Field(default="", max_length=500)
    append: str = Field(default="", max_length=2000)


class GepaStrategyOverlayCompile(BaseModel):
    """LLM output: short append-only strategy hints keyed by agent (optional query regex)."""

    overlays: list[GepaStrategyOverlayItem] = Field(default_factory=list, max_length=8)


class GepaOverlayGateDecision(BaseModel):
    approve: bool = False
    reason: str = Field(default="", max_length=500)


class InboundReplyClassification(BaseModel):
    """LLM output: classify an inbound reply to an outbound campaign send.

    Canonical classes and routing: ``docs/EMAIL_TAXONOMY.md``.
    ``confidence`` is required for routing (high/medium → consumers; low →
    morning-brief unclassified).

    ``category`` is one of: rfq | interested | question | objection | not_now |
    not_interested | unsubscribe | bounce | auto_reply |
    production_update | drawing_approval | payment_confirmation |
    dispatch_notice | installation_report |
    vendor_quote | vendor_order_confirmation |
    complaint | support | general_inquiry.
    """

    category: Literal[
        "rfq",
        "interested",
        "question",
        "objection",
        "not_now",
        "not_interested",
        "unsubscribe",
        "bounce",
        "auto_reply",
        "production_update",
        "drawing_approval",
        "payment_confirmation",
        "dispatch_notice",
        "installation_report",
        "vendor_quote",
        "vendor_order_confirmation",
        "complaint",
        "support",
        "general_inquiry",
    ] = "question"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    summary: str = Field(default="", max_length=500)
    requested_next_step: str = Field(default="", max_length=200)
    language: str = Field(default="en", max_length=16)


class RfqParsedRequirements(BaseModel):
    """Strict schema for one LLM RFQ extraction call (CPQ Wave 3)."""

    machine_type: str = Field(default="", max_length=128)
    machine_series: str = Field(default="", max_length=64)
    machine_hint: str = Field(default="", max_length=255)
    forming_area_mm: str = Field(
        default="",
        max_length=64,
        description="Forming / platen area as W×L mm, e.g. 1400×1000",
    )
    draw_depth_mm: str = Field(default="", max_length=32)
    clamp_force: str = Field(default="", max_length=64, description="e.g. 10 t max")
    material: str = Field(default="", max_length=128)
    gauge: str = Field(default="", max_length=64)
    quantity: str = Field(default="", max_length=32)
    voltage: str = Field(default="", max_length=64, description="e.g. 200 V, 3-phase, 50 Hz")
    region_signal: str = Field(default="", max_length=64)
    controls: str = Field(default="", max_length=255, description="PLC/HMI brand + series")
    tooling: str = Field(default="", max_length=255)
    timeline: str = Field(default="", max_length=128)
    delivery_terms: str = Field(default="", max_length=128)
    process_notes: str = Field(default="", max_length=500)
    attachment_filenames: list[str] = Field(default_factory=list, max_length=40)
    field_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Per-field confidence 0-1 for extracted values",
    )
    missing_fields: list[str] = Field(default_factory=list, max_length=30)
    summary: str = Field(default="", max_length=500)


class VendorQuoteLineItem(BaseModel):
    """One line on a supplier quotation."""

    part: str = Field(default="", max_length=255)
    qty: float = Field(default=0.0, ge=0.0)
    unit_price: float = Field(default=0.0, ge=0.0)
    currency: str = Field(default="EUR", max_length=10)


class VendorQuoteParsed(BaseModel):
    """Strict schema for one LLM vendor-quote extraction call (ERP Stage 6 W1)."""

    vendor: str = Field(default="", max_length=255)
    line_items: list[VendorQuoteLineItem] = Field(default_factory=list, max_length=40)
    total: float | None = Field(default=None, ge=0.0)
    currency: str = Field(default="", max_length=10)
    production_days: int | None = Field(default=None, ge=0)
    freight_days: int | None = Field(default=None, ge=0)
    freight_mode: str = Field(default="", max_length=64)
    payment_terms: str = Field(default="", max_length=128)
    validity: str = Field(default="", max_length=128)
    wo_hint: str = Field(
        default="",
        max_length=64,
        description="Inferred WO number e.g. 23011 when mentioned",
    )
    field_confidence: dict[str, float] = Field(default_factory=dict)
    summary: str = Field(default="", max_length=500)


class SerendipityCollisionVerdict(BaseModel):
    """LLM output: forced collision pair evaluation for the Serendipity Engine."""

    connection: str = Field(
        default="NONE",
        max_length=800,
        description="Concrete business link for Acme Corp, or NONE.",
    )
    why_now: str = Field(default="", max_length=1000)
    suggested_play: Literal[
        "draft_outreach",
        "add_to_campaign",
        "create_campaign_design",
        "revive_quote",
        "research_deeper",
        "tell_operator",
    ] = "tell_operator"
    confidence: int = Field(default=0, ge=0, le=100)
    required_evidence_check: str = Field(default="", max_length=800)


class LocalizedEmailDraft(BaseModel):
    """LLM output: a localized outbound email draft plus a back-translation.

    ``localized_subject`` / ``localized_body`` are in the target language with
    every protected glossary term kept verbatim; ``back_translation`` is an
    English re-translation of the localized body for operator review.
    """

    localized_subject: str = Field(default="", max_length=300)
    localized_body: str = Field(default="", max_length=8000)
    back_translation: str = Field(default="", max_length=8000)
    target_language: str = Field(default="", max_length=16)
    notes: str = Field(default="", max_length=500)


class SessionMineExtraction(BaseModel):
    """Per-session learning extract from a Graphe / Cursor transcript (cheap model)."""

    lessons: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(
        default_factory=list,
        description="Operator corrections — candidate Mnemon ledger entries",
    )
    preferences: list[str] = Field(default_factory=list)
    procedures: list[str] = Field(
        default_factory=list,
        description="Procedures that worked — candidate promotions",
    )
    open_loops: list[str] = Field(
        default_factory=list,
        description="Open TODOs / unfinished threads",
    )
    summary: str = Field(default="", max_length=600)


# ── Weekly agent peer review (Vera / Metis / Sophia panel) ────────────────


class AgentPeerPanelReview(BaseModel):
    """One reviewer's strict-schema verdict for a weekly agent critic pass."""

    reviewer: Literal["vera", "metis", "sophia"]
    grounding_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    recurring_issue: str = Field(default="", max_length=500)
    notes: str = Field(default="", max_length=2000)


class AgentPeerScorecard(BaseModel):
    """Composite weekly peer-review scorecard for one high-traffic agent."""

    agent: str = Field(default="", max_length=64)
    week: str = Field(default="", max_length=16)
    samples: int = Field(default=0, ge=0)
    grounding_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    top_recurring_issue: str = Field(default="", max_length=500)
    one_suggested_prompt_improvement: str = Field(default="", max_length=2000)
    panel: list[AgentPeerPanelReview] = Field(default_factory=list, max_length=3)


# ── Wonder curiosity digest ───────────────────────────────────────────────
class CuriosityConnection(BaseModel):
    """One evidence-cited cross-agent connection from the weekly digest."""

    summary: str = Field(default="", max_length=400)
    agents: list[str] = Field(default_factory=list, max_length=4)
    evidence_refs: list[str] = Field(default_factory=list, max_length=4)
    shared_token: str = Field(default="", max_length=64)


class CuriosityDigest(BaseModel):
    """Strict-schema weekly connections between agents' domains (max 3)."""

    connections: list[CuriosityConnection] = Field(default_factory=list, max_length=3)
