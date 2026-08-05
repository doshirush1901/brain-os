"""Pipeline timeouts, ReAct, routing, compression knobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator


class AppPipelineMixin:
    """Mixin slice of :class:`~brain_os.config.app.AppConfig` (MOVE only)."""

    react_max_iterations: int = 8

    react_tool_transient_retries: int = 1

    react_tool_timeout_seconds: float = Field(default=45.0, ge=1.0, le=600.0)

    mcp_tool_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)

    pipeline_timeout: int = 600

    pipeline_stage_budgets: dict[str, float] = Field(
        default_factory=lambda: {
            "perceive": 5.0,
            "remember": 8.0,
            "route": 4.0,
            "enrich": 20.0,
            "plan": 15.0,
            "execute": 60.0,
            "validate": 5.0,
            "compile": 8.0,
            "learn": 0.0,
        }
    )

    agent_timeout: int = 90

    max_parallel_agents: int = 5

    brain_route_provider: str = ""

    brain_synthesis_provider: str = ""

    brain_embedding_provider: str = ""

    athena_synthesis_timeout: int = 90

    max_delegation_depth: int = 5

    progressive_tool_discovery: bool = False

    standing_goal_enabled: bool = True

    standing_goal_max_turns: int = Field(default=20, ge=1, le=100)

    standing_goal_max_phases: int = Field(default=20, ge=1, le=100)

    task_state_ttl_seconds: int = Field(
        default=86400,
        ge=-1,
        le=30 * 86400,
        validation_alias=AliasChoices(
            "APP__TASK_STATE_TTL_SECONDS",
            "TASK_STATE_TTL_SECONDS",
        ),
    )

    task_workspace_persist_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "APP__TASK_WORKSPACE_PERSIST_ENABLED",
            "TASK_WORKSPACE_PERSIST_ENABLED",
        ),
    )

    task_wave_max_phases: int = Field(
        default=20,
        ge=1,
        le=100,
        validation_alias=AliasChoices(
            "APP__TASK_WAVE_MAX_PHASES",
            "TASK_WAVE_MAX_PHASES",
        ),
    )

    phase_validator_fail_open: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "APP__PHASE_VALIDATOR_FAIL_OPEN",
            "PHASE_VALIDATOR_FAIL_OPEN",
        ),
    )

    task_phase_validator_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "APP__TASK_PHASE_VALIDATOR_ENABLED",
            "TASK_PHASE_VALIDATOR_ENABLED",
        ),
    )

    task_phase_validator_retry: int = Field(
        default=1,
        ge=0,
        le=3,
        validation_alias=AliasChoices(
            "APP__TASK_PHASE_VALIDATOR_RETRY",
            "TASK_PHASE_VALIDATOR_RETRY",
        ),
    )

    progressive_tool_discovery_top_k: int = Field(default=8, ge=1, le=50)

    progressive_tool_discovery_expand_on_low_confidence: bool = True

    tool_selection_telemetry: bool = False

    mcp_fastlane_enabled: bool = True

    mcp_fastlane_min_score: float = Field(default=0.45, ge=0.0, le=1.0)

    mcp_fastlane_max_snippets: int = Field(default=3, ge=1, le=10)

    mcp_email_default_scope: Literal["primary", "secondary", "both"] = "primary"

    mcp_email_search_timeout_seconds: float = Field(default=35.0, ge=1.0, le=120.0)

    mcp_email_search_max_results: int = Field(default=10, ge=1, le=25)

    mcp_route_prefer_cloud_fast: bool = True

    mcp_ollama_retry_budget: int = Field(default=0, ge=0, le=10)

    mcp_fast_fail_to_openai: bool = True

    request_prompt_snapshot: bool = False

    pipeline_query_max_chars: int = Field(default=100_000, ge=4096, le=500_000)

    preflight_compression_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "APP__PREFLIGHT_COMPRESSION_ENABLED",
            "PREFLIGHT_COMPRESSION_ENABLED",
        ),
    )

    preflight_compression_threshold: int = Field(
        default=12_000,
        ge=1024,
        le=200_000,
        validation_alias=AliasChoices(
            "APP__PREFLIGHT_COMPRESSION_THRESHOLD",
            "PREFLIGHT_COMPRESSION_THRESHOLD",
        ),
    )

    react_scratchpad_compression_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "APP__REACT_SCRATCHPAD_COMPRESSION_ENABLED",
            "REACT_SCRATCHPAD_COMPRESSION_ENABLED",
        ),
    )

    react_scratchpad_compression_threshold: int = Field(
        default=6_000,
        ge=512,
        le=100_000,
        validation_alias=AliasChoices(
            "APP__REACT_SCRATCHPAD_COMPRESSION_THRESHOLD",
            "REACT_SCRATCHPAD_COMPRESSION_THRESHOLD",
        ),
    )

    react_scratchpad_compression_keep_recent: int = Field(
        default=2,
        ge=1,
        le=8,
        validation_alias=AliasChoices(
            "APP__REACT_SCRATCHPAD_COMPRESSION_KEEP_RECENT",
            "REACT_SCRATCHPAD_COMPRESSION_KEEP_RECENT",
        ),
    )

    structured_compaction_summary_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "APP__STRUCTURED_COMPACTION_SUMMARY_ENABLED",
            "STRUCTURED_COMPACTION_SUMMARY_ENABLED",
        ),
    )

    tool_output_pruning_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "APP__TOOL_OUTPUT_PRUNING_ENABLED",
            "TOOL_OUTPUT_PRUNING_ENABLED",
        ),
    )

    tool_output_max_chars: int = Field(
        default=800,
        ge=200,
        le=16_000,
        validation_alias=AliasChoices(
            "APP__TOOL_OUTPUT_MAX_CHARS",
            "TOOL_OUTPUT_MAX_CHARS",
        ),
    )

    tool_output_pruning_keep_recent: int = Field(
        default=2,
        ge=1,
        le=8,
        validation_alias=AliasChoices(
            "APP__TOOL_OUTPUT_PRUNING_KEEP_RECENT",
            "TOOL_OUTPUT_PRUNING_KEEP_RECENT",
        ),
    )

    enrichment_part_max_chars: int = Field(
        default=4000,
        ge=512,
        le=32_000,
        validation_alias=AliasChoices(
            "APP__ENRICHMENT_PART_MAX_CHARS",
            "ENRICHMENT_PART_MAX_CHARS",
        ),
    )

    cheap_exit_bypass_enabled: bool = True

    cheap_exit_bypass_keywords: str = (
        "send,dispatch,draft,reply,follow-up,follow up,outreach,campaign,"
        "latest,recent,today,yesterday,just sent,unread,inbox,"
        "update,create,delete,close deal,mark as,"
        "confidential,internal,private,margin,pricing,quote,what did"
    )

    cheap_exit_allowlist_keywords: str = ""

    optional_agent_execution_enabled: bool = True

    optional_agent_budget: int = Field(default=2, ge=0, le=10)

    required_tool_enforcement: bool = False

    router_deterministic_margin_min: float = Field(default=0.0, ge=0.0, le=20.0)

    router_embedding_tiebreak_enabled: bool = False

    router_embedding_tiebreak_max_margin: float = Field(default=1.25, ge=0.0, le=10.0)

    router_optional_keyword_ranking: bool = True

    pipeline_attach_observability_trace: bool = True

    response_include_execution_summary: bool = False

    llm_route_prefer_ollama: bool = False

    llm_route_high_stakes_cloud_escalation: bool = True

    retriever_over_retrieve_factor: int = Field(default=3, ge=1, le=12)

    retriever_diversity_overlap_threshold: float = Field(default=0.55, ge=0.1, le=0.95)

    retriever_rerank_candidates_factor: int = Field(default=2, ge=1, le=8)

    retriever_keyword_boost_weight: float = Field(default=0.35, ge=0.0, le=1.0)

    retriever_access_boost_enabled: bool = True

    retriever_access_boost_weight: float = Field(default=1.0, ge=0.0, le=2.0)

    retriever_access_boost_max_influence: float = Field(default=0.12, ge=0.0, le=0.25)

    retriever_access_boost_half_life_days: float = Field(default=30.0, ge=1.0, le=365.0)

    retriever_second_pass_enabled: bool = False

    retriever_second_pass_keyword_threshold: float = Field(default=0.22, ge=0.0, le=0.95)

    retriever_trace_enabled: bool = False

    retriever_dedup_enabled: bool = False

    retriever_dedup_max_jaccard: float = Field(default=0.92, ge=0.5, le=1.0)

    retriever_dedup_content_chars: int = Field(default=800, ge=200, le=12000)

    retriever_default_profile: str = "default"

    retriever_evidence_max_chars: int = Field(default=0, ge=0, le=500_000)

    citation_aligner_enabled: bool = False

    crm_pipeline_cache_ttl_seconds: float = Field(default=15.0, ge=0.0, le=300.0)

    pipeline_triangulation_gaps_enabled: bool = True

    pipeline_triangulation_gaps_hex: bool = False

    pipeline_triangulation_gaps_timeout_s: float = Field(default=25.0, ge=5.0, le=120.0)

    pipeline_shape_require_triangulation: bool = True

    pipeline_shape_triangulation_hex: bool = False

    pipeline_shape_triangulation_block_on_gaps: bool | None = None

    pipeline_triangulation_gaps_skip_if_agent_ran_brief: bool = True

    pipeline_triangulation_gaps_channel_allowlist: str = "cli,api,cursor"

    # Sphinx v2 — Socratic counter-questions on high-stakes × ambiguous asks
    socratic_gate_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "APP__SOCRATIC_GATE_ENABLED",
            "SOCRATIC_GATE_ENABLED",
        ),
    )

    socratic_gate_threshold: float = Field(
        default=0.28,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "APP__SOCRATIC_GATE_THRESHOLD",
            "SOCRATIC_GATE_THRESHOLD",
        ),
    )

    socratic_gate_promotion_count: int = Field(
        default=3,
        ge=2,
        le=20,
        validation_alias=AliasChoices(
            "APP__SOCRATIC_GATE_PROMOTION_COUNT",
            "SOCRATIC_GATE_PROMOTION_COUNT",
        ),
    )

    socratic_crm_write_threshold: int = Field(
        default=3,
        ge=1,
        le=100,
        validation_alias=AliasChoices(
            "APP__SOCRATIC_CRM_WRITE_THRESHOLD",
            "SOCRATIC_CRM_WRITE_THRESHOLD",
        ),
    )

    socratic_resume_ttl_minutes: int = Field(
        default=30,
        ge=1,
        le=24 * 60,
        validation_alias=AliasChoices(
            "APP__SOCRATIC_RESUME_TTL_MINUTES",
            "SOCRATIC_RESUME_TTL_MINUTES",
        ),
    )
