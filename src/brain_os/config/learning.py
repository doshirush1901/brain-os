"""Dream / GEPA / Mem0 / procedural / prediction learning flags."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator


class AppLearningMixin:
    """Mixin slice of :class:`~brain_os.config.app.AppConfig` (MOVE only)."""

    mem0_timeout: float = 15.0

    mem0_search_cache_ttl_seconds: int = 300

    retriever_mem0_scoped_fanout: bool = True

    dream_mem0_forgetting_enabled: bool = True

    mem0_forget_min_age_days: int = 120

    mem0_forget_max_confidence: float = 0.7

    mem0_forget_run_cap: int = 200

    mem0_forget_hard_delete_after_days: int = 30

    mem0_forget_salience_protect: float = 0.35

    mem0_forgetting_heartbeat_enabled: bool = True

    mem0_salience_min_score: float = 0.35

    mem0_reconsolidation_enabled: bool = True

    mem0_reconsolidation_window_hours: int = 48

    mem0_evidence_score_boost: float = 1.15

    mem0_lore_score_penalty: float = 0.92

    dream_mem0_replay_enabled: bool = True

    dream_replay_top_k: int = 25

    graphe_learning_meta_coverage_floor: float = 0.6

    graphe_learning_meta_backfill_enabled: bool = True

    learning_route_eligible_floor: int = Field(default=5, ge=0, le=500)

    dream_qdrant_hygiene_enabled: bool = False

    plan_episodes_scoped: bool = True

    dream_distributed_mutex_enabled: bool = False

    dream_distributed_mutex_ttl_seconds: int = 7200

    dream_min_history_messages: int = Field(default=2, ge=1, le=20)

    dream_lookback_hours: int = Field(default=24, ge=1, le=168)

    dream_creative_recent_episodes: int = Field(default=5, ge=1, le=50)

    dream_revenue_reflection_enabled: bool = True

    dream_aftermarket_triggers_enabled: bool = True

    dream_aftermarket_reflection_enabled: bool = True

    dream_revenue_dir: str = Field(default="data/revenue_mode")

    pending_memory_queue_path: str = Field(default="data/brain/pending_memory_queue.sqlite")

    session_mine_queue_path: str = Field(default="data/brain/session_mine_queue.jsonl")

    session_mine_batch_limit: int = Field(default=25, ge=1, le=200)

    deep_consolidation_interval_hours: float = Field(default=6.0, ge=0.0, le=168.0)

    dream_stale_hours: float = Field(default=24.0, ge=1.0, le=168.0)

    dream_stale_alert_enabled: bool = True

    dream_stale_slack_alert: bool = True

    learned_procedure_promotion_enabled: bool = True

    learned_routing_enabled: bool = False

    learning_compiler_min_evidence: int = Field(default=2, ge=1, le=10_000)

    learning_trial_decay_days: int = Field(default=14, ge=1, le=365)

    dream_stage4_candidates_only: bool = True

    dream_procedural_prune_enabled: bool = True

    instinct_ttl_days: int = Field(default=30, ge=1, le=3650)

    instinct_ttl_prune_enabled: bool = True

    instinct_ttl_soft_expire_enabled: bool = True

    #: Procedural match: min cosine similarity (Voyage embeddings) to route.
    #: Tuned on paraphrase fixture in ``tests/test_procedural_embedding_match.py``
    #: (live Voyage: highest thr with TP=TN=10 on 10+10 pairs → 0.63).
    procedural_match_cosine_min: float = Field(default=0.63, ge=0.0, le=1.0)

    #: Among candidates within this cosine of the top score, pick highest Jaccard.
    procedural_match_cosine_tiebreak: float = Field(default=0.02, ge=0.0, le=0.5)

    #: When True and an embedder is injected, prefer cosine kNN over Jaccard.
    procedural_embed_match_enabled: bool = True

    instinct_evolve_enabled: bool = False

    instinct_evolve_min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    instinct_evolve_min_cluster: int = Field(default=3, ge=2, le=1000)

    dream_moat_compounding_enabled: bool = False

    learning_promote_min_dream_cycles: int = Field(default=2, ge=0, le=100)

    procedure_deprecated_warn_enabled: bool = True

    procedure_auto_retire_deprecated_uses: int = Field(default=5, ge=0, le=10_000)

    memory_health_corrections_24h_warn: int = Field(default=3, ge=0, le=10_000)

    memory_health_low_metis_sessions_24h_warn: int = Field(default=5, ge=0, le=10_000)

    memory_health_low_metis_floor: float = Field(default=40.0, ge=0.0, le=100.0)

    memory_health_stale_procedure_ratio_warn: float = Field(default=0.25, ge=0.0, le=1.0)

    memory_health_candidate_backlog_warn: int = Field(default=20, ge=0, le=10_000)

    memory_health_deprecated_use_24h_warn: int = Field(default=10, ge=0, le=10_000)

    memory_health_metis_rolling_warn: float = Field(default=60.0, ge=0.0, le=100.0)

    dream_cadence_gate_enabled: bool = True

    dream_cadence_min_health_score: int = Field(default=70, ge=0, le=100)

    dream_cadence_defer_hours: float = Field(default=6.0, ge=0.5, le=168.0)

    praise_learning_enabled: bool = True

    praise_compiler_min_evidence: int = Field(default=1, ge=1, le=100)

    skill_candidate_compile_enabled: bool = False

    skill_candidate_min_evidence: int = Field(default=2, ge=1, le=100)

    learning_max_promotions_per_week: int = Field(default=10, ge=0, le=500)

    learning_routing_max_weight_delta: float = Field(default=2.0, ge=0.0, le=20.0)

    learning_promotions_audit_filename: str = "brain/learning/promotions.jsonl"

    prediction_reconciliation_enabled: bool = True

    prediction_reconciliation_max_accounts: int = Field(default=50, ge=1, le=500)

    dream_memory_reconciliation_enabled: bool = True

    dream_memory_reconciliation_max_claims: int = Field(default=40, ge=5, le=120)

    dream_memory_reconciliation_max_candidates: int = Field(default=20, ge=1, le=100)

    dream_memory_reconciliation_enqueue: bool = False

    dream_memory_reconciliation_max_enqueue: int = Field(default=10, ge=1, le=50)

    prediction_record_must_act: bool = True

    prediction_must_act_accuracy_floor: float = Field(default=0.25, ge=0.0, le=1.0)

    prediction_must_act_accuracy_target: float = Field(default=0.85, ge=0.0, le=1.0)

    prediction_must_act_min_reconciled: int = Field(default=5, ge=0, le=500)

    prediction_must_act_blocking_only: bool = True

    prediction_must_act_narrow_when_below_target: bool = True

    prediction_must_act_max_per_cycle: int = Field(default=12, ge=1, le=100)

    prediction_must_act_procedural_learning: bool = False

    correction_ledger_hygiene_enabled: bool = True

    correction_ledger_hygiene_fix_missing_dates: bool = True

    correction_ledger_hygiene_crm_hints: bool = True

    prediction_record_tyche_deals: bool = True

    prediction_record_intraday_on_board_write: bool = False

    prediction_reconcile_heartbeat_skip_llm: bool = True

    gepa_persist_tool_invocations: bool = False

    gepa_tool_invocations_db_path: str = "brain/tool_invocations.sqlite"

    gepa_tool_invocations_retention_days: int = Field(default=30, ge=1, le=3650)

    gepa_tool_invocations_max_rows: int = Field(default=500_000, ge=1_000, le=50_000_000)

    gepa_strategy_overlay_runtime: bool = True

    gepa_dream_compile_enabled: bool = True

    gepa_promote_strategy_overlays: bool = True

    gepa_nemesis_gate_enabled: bool = False

    gepa_compile_min_invocations_per_pair: int = Field(default=15, ge=1, le=100_000)

    gepa_compile_max_success_rate: float = Field(default=0.85, ge=0.0, le=1.0)

    gepa_compile_window_days: int = Field(default=90, ge=1, le=365)

    gepa_max_strategy_overlay_promotions_per_week: int = Field(default=5, ge=0, le=500)
