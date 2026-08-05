"""Runtime identity and leftover AppConfig fields."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator


class AppCoreMixin:
    """Mixin slice of :class:`~brain_os.config.app.AppConfig` (MOVE only)."""

    log_level: str = "INFO"

    log_format: str = "text"

    environment: str = "development"

    api_secret_key: SecretStr = SecretStr("")

    cors_origins: str = "http://localhost:3000"

    crm_backend: Literal["postgres", "twenty", "hybrid"] = Field(
        default="postgres",
        validation_alias=AliasChoices("APP__CRM_BACKEND", "CRM_BACKEND"),
    )

    deployment_profile: Literal["full", "edge"] = "full"

    vault_export_default_dir: str = ""

    qdrant_hygiene_scan_cap: int = 5000

    qdrant_hygiene_batch_size: int = 200

    neo4j_max_pool_size: int = 50

    hf_cache_dir: str = ""

    flashrank_cache_dir: str = ""

    message_bus_log_maxlen: int = Field(default=1000, ge=100, le=500_000)

    unified_context_max_history: int = Field(default=50, ge=4, le=10_000)

    unified_context_max_users: int = Field(default=500, ge=1, le=100_000)

    embedding_sqlite_cache_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "APP__EMBEDDING_SQLITE_CACHE_ENABLED",
            "APP__EMBEDDING_CACHE_SQLITE_ENABLED",
            "EMBEDDING_SQLITE_CACHE_ENABLED",
            "EMBEDDING_CACHE_SQLITE_ENABLED",
        ),
    )

    embedding_sqlite_cache_retention_days: int = Field(
        default=30,
        ge=1,
        le=3650,
        validation_alias=AliasChoices(
            "APP__EMBEDDING_SQLITE_CACHE_RETENTION_DAYS",
            "EMBEDDING_SQLITE_CACHE_RETENTION_DAYS",
        ),
    )

    embedding_sqlite_cache_max_rows: int = Field(
        default=100_000,
        ge=1_000,
        le=50_000_000,
        validation_alias=AliasChoices(
            "APP__EMBEDDING_SQLITE_CACHE_MAX_ROWS",
            "EMBEDDING_SQLITE_CACHE_MAX_ROWS",
        ),
    )

    embedding_sqlite_cache_prune_every_n_writes: int = Field(
        default=1_000,
        ge=1,
        le=1_000_000,
        validation_alias=AliasChoices(
            "APP__EMBEDDING_SQLITE_CACHE_PRUNE_EVERY_N_WRITES",
            "EMBEDDING_SQLITE_CACHE_PRUNE_EVERY_N_WRITES",
        ),
    )

    embedding_sqlite_cache_max_bytes: int = Field(
        default=300 * 1024 * 1024,
        ge=1_048_576,
        le=50 * 1024 * 1024 * 1024,
        validation_alias=AliasChoices(
            "APP__EMBEDDING_SQLITE_CACHE_MAX_BYTES",
            "EMBEDDING_SQLITE_CACHE_MAX_BYTES",
        ),
    )

    data_dir_lock_narrow: bool = False

    lettie_enabled: bool = False

    letta_base_url: str = "http://localhost:8283"

    letta_api_key: SecretStr = SecretStr("")

    letta_agent_id: str = ""

    letta_timeout_seconds: float = 45.0

    lettie_conversation_log_enabled: bool = True

    lettie_loop_guard_enabled: bool = True

    lettie_reference_turn_limit: int = 3

    digestive_llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"

    digestive_openai_model: str = ""

    digestive_anthropic_model: str = ""

    helicone_proxy_anthropic: bool = True

    # Authoritative reader collection — see docs/RETRIEVAL.md (hybrid-first). Permanent.
    use_sparse_hybrid: bool = True

    #: Hybrid fusion: ``weighted`` (dense/sparse score blend) or legacy ``rrf``.
    hybrid_fusion_mode: Literal["weighted", "rrf"] = "weighted"

    #: Dense channel weight for score-weighted fusion (normalized with sparse).
    hybrid_dense_weight: float = Field(default=0.8, ge=0.0, le=1.0)

    #: Sparse channel weight for score-weighted fusion (normalized with dense).
    hybrid_sparse_weight: float = Field(default=0.2, ge=0.0, le=1.0)

    #: When the query is code-like (DEMO-X-0707 / MCT-2026), prefer sparse.
    hybrid_code_dense_weight: float = Field(default=0.35, ge=0.0, le=1.0)

    hybrid_code_sparse_weight: float = Field(default=0.65, ge=0.0, le=1.0)

    default_user_id: str = "brain_default_user"

    auto_fallback_api_on_data_dir_lock: bool = False

    email_archive_thread_after_send: bool = False

    rate_limit_enabled: bool = True

    rate_limit_storage_uri: str = ""

    rate_limit_query_per_minute: int = Field(default=120, ge=1, le=50_000)

    rate_limit_query_stream_per_minute: int = Field(default=60, ge=1, le=50_000)

    rate_limit_task_stream_per_minute: int = Field(default=40, ge=1, le=50_000)

    rate_limit_send_per_minute: int = Field(default=24, ge=1, le=5000)

    rate_limit_ingest_per_minute: int = Field(default=40, ge=1, le=5000)

    ingest_semantic_similarity_threshold: float = Field(default=0.7, ge=0.2, le=0.99)

    ingest_semantic_similarity_window: int = Field(default=3, ge=1, le=20)

    ingest_chunk_quote_tokens: int = Field(default=256, ge=64, le=2048)

    ingest_chunk_quote_overlap: int = Field(default=64, ge=0, le=512)

    ingest_chunk_technical_tokens: int = Field(default=384, ge=64, le=2048)

    ingest_chunk_technical_overlap: int = Field(default=96, ge=0, le=512)

    ingest_chunk_default_tokens: int = Field(default=512, ge=128, le=8192)

    ingest_chunk_default_overlap: int = Field(default=128, ge=0, le=512)

    llm_monthly_token_budget: int = Field(default=0, ge=0, le=2_000_000_000)

    llm_budget_scope: str = Field(default="user")

    #: Daily USD soft/hard governor (ledger-backed). 0 disables USD gate.
    daily_llm_budget_usd: float = Field(
        default=25.0,
        ge=0.0,
        le=100_000.0,
        validation_alias=AliasChoices("APP__DAILY_LLM_BUDGET", "APP__DAILY_LLM_BUDGET_USD"),
    )

    daily_llm_budget_soft_pct: float = Field(
        default=0.80,
        ge=0.5,
        le=0.99,
        validation_alias=AliasChoices("APP__DAILY_LLM_BUDGET_SOFT_PCT"),
    )

    llm_job_budget_drip_usd: float = Field(
        default=5.0,
        ge=0.0,
        le=10_000.0,
        validation_alias=AliasChoices("APP__LLM_JOB_BUDGET_DRIP_USD"),
    )

    llm_job_budget_dream_usd: float = Field(
        default=8.0,
        ge=0.0,
        le=10_000.0,
        validation_alias=AliasChoices("APP__LLM_JOB_BUDGET_DREAM_USD"),
    )

    llm_job_budget_universe_usd: float = Field(
        default=5.0,
        ge=0.0,
        le=10_000.0,
        validation_alias=AliasChoices("APP__LLM_JOB_BUDGET_UNIVERSE_USD"),
    )

    llm_job_budget_eval_usd: float = Field(
        default=3.0,
        ge=0.0,
        le=10_000.0,
        validation_alias=AliasChoices("APP__LLM_JOB_BUDGET_EVAL_USD"),
    )

    llm_spend_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("APP__LLM_SPEND_ENABLED"),
    )

    llm_spend_max_bytes: int = Field(
        default=20_000_000,
        ge=100_000,
        le=500_000_000,
        validation_alias=AliasChoices("APP__LLM_SPEND_MAX_BYTES"),
    )

    wonder_topics_per_run: int = Field(default=3, ge=1, le=10)

    wonder_max_searches_per_day: int = Field(default=9, ge=1, le=100)

    curious_ira_enabled: bool = False

    curious_ira_max_questions: int = Field(default=5, ge=1, le=10)

    serendipity_collectors_enabled: bool = True

    serendipity_collisions_enabled: bool = True

    serendipity_max_searches_per_day: int = Field(default=9, ge=1, le=100)

    aftermarket_max_triggers_per_day: int = Field(default=5, ge=1, le=50)

    stomach_quarantine_enabled: bool = True

    heartbeat_jobs_path: str = Field(default="data/heartbeat_jobs.json")

    heartbeat_server_enabled: bool = False

    heartbeat_server_interval_minutes: int = Field(default=15, ge=1, le=1440)

    heartbeat_catch_up_window_hours: float = Field(
        default=6.0,
        ge=0.0,
        le=48.0,
        description=(
            "After lid-close / daemon downtime, wall-clock (at:) jobs missed within "
            "this many hours still fire once on next wake/tick (catch_up=true)."
        ),
    )

    heartbeat_max_jobs_per_tick: int = Field(
        default=8,
        ge=0,
        le=200,
        description=(
            "Max due jobs executed per heartbeat tick (0 = unlimited). "
            "Wall-clock/catch-up jobs are ordered first after downtime."
        ),
    )

    slack_listen_enabled: bool = False

    event_task_reactor_enabled: bool = True

    event_task_rules_path: str = Field(default="data/event_task_rules.json")

    event_task_max_inflight: int = Field(default=2, ge=1, le=32)

    hephaestion_audit_path: str = "data/audit"

    hephaestion_langfuse_lookback_hours: int = Field(default=24, ge=1, le=168)

    hephaestion_golden_auto_write: bool = False

    hephaestion_skip_pytest_cov: bool = True

    hephaestion_brief_webhook: bool = False

    graphify_scheduled_refresh_enabled: bool = True

    graphify_refresh_timeout_s: float = Field(
        default=600.0,
        ge=60.0,
        le=3600.0,
        description="Per-step timeout (extract/cluster) for graphify_refresh heartbeat.",
    )

    graphify_prefetch_on_code_architecture_route: bool = True

    graphify_stale_commit_threshold: int = Field(
        default=5,
        ge=0,
        le=500,
        description="Refresh graph.json before nightly audit when HEAD is this many commits ahead of graph build (0 = only when missing).",
    )

    respiratory_inhale_hour: int = Field(default=6, ge=0, le=23)

    respiratory_inhale_minute: int = Field(default=0, ge=0, le=59)

    respiratory_exhale_hour: int = Field(default=22, ge=0, le=23)

    respiratory_exhale_minute: int = Field(default=0, ge=0, le=59)

    morning_brain_enabled: bool = True

    morning_brain_hour: int = Field(default=6, ge=0, le=23)

    morning_brain_minute: int = Field(default=30, ge=0, le=59)

    morning_brain_output_dir: str = Field(default="data/morning")

    morning_brain_inbox_path: str = Field(default="data/operations/morning_brain_inbox.json")

    #: Operator email for the Morning Brain brief (``APP__OPERATOR_BRIEF_EMAIL``).
    #: Empty → falls back to ``GOOGLE_IRA_EMAIL`` (the operator's own mailbox).
    operator_brief_email: str = ""

    morning_brain_include_standing_brief: bool = False

    morning_brain_evidence_sweep_limit: int = Field(default=25, ge=1, le=100)

    #: Long-horizon campaign goals — max concurrent active/plan_pending/blocked.
    max_active_goals: int = Field(default=3, ge=1, le=10)

    #: Wave 2 receivables — FX drift alerts on open foreign milestones (nice-to-have).
    receivables_fx_drift_alerts_enabled: bool = False

    receivables_fx_drift_pct: float = Field(default=5.0, ge=0.5, le=50.0)

    pantheon_conference_enabled: bool = True

    pantheon_conference_output_dir: str = Field(default="data/reports")

    pantheon_conference_batch_size: int = Field(default=4, ge=1, le=12)

    pantheon_conference_max_agents: int = Field(
        default=0,
        ge=0,
        description="0 = all pantheon agents except Athena",
    )

    pantheon_conference_weekday: int = Field(
        default=6,
        ge=0,
        le=6,
        description="Python weekday for scheduled run (0=Mon, 6=Sun)",
    )

    pantheon_conference_timezone: str = Field(default="Asia/Kolkata")

    ram_sabbath_enabled: bool = False

    ram_sabbath_output_dir: str = Field(default="data/reports/ram_sabbath")

    ram_sabbath_max_proposals: int = Field(default=3, ge=1, le=8)

    ram_sabbath_weekday: int = Field(
        default=6,
        ge=0,
        le=6,
        description="Python weekday: 0=Mon … 6=Sun (IST default Sabbath).",
    )

    ram_sabbath_timezone: str = Field(default="Asia/Kolkata")

    scheduling_public_base_url: str = ""

    scheduling_token_secret: SecretStr = SecretStr("")

    scheduling_default_timezone: str = "Asia/Kolkata"

    brain_birth_date: date = Field(default=date(2026, 3, 6))

    operator_timezone: str = "Asia/Kolkata"

    board_guest_mode_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("APP__BOARD_GUEST_MODE_ENABLED", "BOARD_GUEST_MODE_ENABLED"),
    )

    board_default_email_scope: str = Field(
        default="no_email",
        validation_alias=AliasChoices(
            "APP__BOARD_DEFAULT_EMAIL_SCOPE", "BOARD_DEFAULT_EMAIL_SCOPE"
        ),
    )

    board_pipeline_timeout: int = Field(
        default=300,
        ge=30,
        le=3600,
        validation_alias=AliasChoices("APP__BOARD_PIPELINE_TIMEOUT", "BOARD_PIPELINE_TIMEOUT"),
    )

    brain_temporal_context_enabled: bool = True

    brain_time_mode: Literal["awake", "dream", "briefing", "quiet"] = "awake"

    brain_internet_years_ratio: float = Field(default=7.0, ge=1.0, le=52.0)

    brain_experience_activity_weighting: bool = True

    brain_experience_idle_floor: float = Field(default=0.15, ge=0.0, le=1.0)

    brain_experience_lookback_days: int = Field(default=90, ge=7, le=3650)

    scheduling_default_expiry_days: int = Field(default=7, ge=1, le=90)

    scheduling_default_buffer_minutes: int = Field(default=15, ge=0, le=180)

    scheduling_default_slot_step_minutes: int = Field(default=30, ge=5, le=240)

    scheduling_busy_ics_url: str = ""

    scheduling_ical_api_key: SecretStr = SecretStr("")

    scheduling_ical_api_base_url: str = ""

    optout_token_secret: SecretStr = SecretStr("")

    optout_public_base_url: str = ""

    inbound_widget_api_key: SecretStr = SecretStr("")

    inbound_inquiry_rate_limit_per_hour: int = Field(default=5, ge=1, le=1000)

    startup_qdrant_ensure_timeout_s: float = Field(default=120.0, ge=10.0, le=600.0)

    startup_qdrant_ensure_required: bool = True

    startup_sql_schema_timeout_s: float = Field(default=120.0, ge=10.0, le=600.0)

    startup_sql_schema_required: bool = True

    startup_defer_to_background: bool = False

    claude_code_delegate_enabled: bool = False

    claude_code_delegate_command: str = "claude"

    claude_code_delegate_timeout_seconds: int = Field(default=600, ge=30, le=7200)

    claude_code_delegate_allowed_roots: str = ""

    claude_code_delegate_max_prompt_chars: int = Field(default=48_000, ge=500, le=200_000)

    claude_code_delegate_max_output_chars: int = Field(default=400_000, ge=10_000, le=2_000_000)

    git_ship_enabled: bool = False

    git_ship_allow_push: bool = False

    git_ship_allowed_roots: str = ""

    git_ship_timeout_seconds: int = Field(default=120, ge=10, le=900)

    git_ship_max_commit_message_chars: int = Field(default=8000, ge=20, le=50_000)

    operator_webhook_url: SecretStr = SecretStr("")

    operator_webhook_timeout_s: float = Field(default=8.0, ge=1.0, le=60.0)

    operator_webhook_events: str = ""

    employment_verify_ttl_days: float = Field(default=30.0, ge=0.0, le=365.0)

    firecrawl_scrape_max_retries: int = Field(default=2, ge=0, le=5)

    pantheon_conference_include_prior_improvements: bool = True

    pantheon_conference_slack_webhook_url: str = ""

    run_record_enabled: bool = False

    run_record_db_path: str = "brain/run_records.sqlite"

    run_record_retention_days: int = Field(default=30, ge=1, le=3650)

    run_record_max_rows: int = Field(default=100_000, ge=1_000, le=50_000_000)

    operator_context_enabled: bool = True

    operator_context_db_path: str = "brain/operator_context.sqlite"

    operator_context_retention_days: int = Field(default=180, ge=1, le=3650)

    operator_context_max_rows: int = Field(default=50_000, ge=100, le=50_000_000)

    operator_context_precedent_limit: int = Field(default=5, ge=1, le=20)

    operator_context_neo4j_sync: bool = False

    operator_context_graph_expand_enabled: bool = True

    company_similarity_enabled: bool = True

    company_similarity_limit: int = Field(default=5, ge=1, le=20)

    company_similarity_min_score: float = Field(default=0.72, ge=0.0, le=1.0)

    company_similarity_candidate_pool: int = Field(default=500, ge=10, le=5000)

    company_similarity_auto_embed_on_brief: bool = True
