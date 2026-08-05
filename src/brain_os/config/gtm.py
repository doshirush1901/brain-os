"""GTM / agency / tinder / drip / outbound / math-mode flags."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator


class AppGtmMixin:
    """Mixin slice of :class:`~brain_os.config.app.AppConfig` (MOVE only)."""

    drip_llm_insights: bool = False

    drip_llm_adjust: bool = False

    drip_tier1_auto_send_enabled: bool = False

    drip_tier1_daily_cap: int = Field(default=10, ge=0, le=100)

    drip_tier1_min_quality: float = Field(default=0.85, ge=0.0, le=1.0)

    drip_tier1_cold_list_only: bool = True

    drip_tier1_never_replied_only: bool = True

    drip_tier1_require_warmed_sender: bool = True

    interconnections_enabled: bool = False

    interconnections_top_k: int = Field(default=6, ge=3, le=20)

    interconnections_brief_ttl_hours: int = Field(default=72, ge=1, le=720)

    interconnections_require_recent_brief_on_campaign_send: bool = False

    agency_enabled: bool = True

    agency_max_daily_cards: int = Field(default=8, ge=1, le=50)

    agency_goals_path: str = ""

    agency_prefs_path: str = ""

    agency_accept_use_persuasion: bool = False

    tinder_triangulate_before_draft: bool = True

    tinder_triangulate_deep: bool = False

    tinder_require_company_card_before_draft: bool = True

    media_vision_provider: str = Field(
        default="anthropic",
        validation_alias=AliasChoices("APP__MEDIA_VISION_PROVIDER", "MEDIA_VISION_PROVIDER"),
    )

    media_vision_model: str = Field(
        default="",
        validation_alias=AliasChoices("APP__MEDIA_VISION_MODEL", "MEDIA_VISION_MODEL"),
    )

    media_vision_max_long_edge: int = Field(
        default=1280,
        ge=512,
        le=2048,
        validation_alias=AliasChoices("APP__MEDIA_VISION_MAX_LONG_EDGE"),
    )

    founder_voice_enabled: bool = True

    tinder_auto_fetch_company_card_on_status: bool = True

    tinder_icp_gate_enabled: bool = True

    tinder_icp_auto_skip_min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)

    tinder_icp_scrape_on_card: bool = True

    tinder_icp_scrape_timeout_s: float = Field(default=25.0, ge=5.0, le=120.0)

    tinder_icp_max_auto_skips_per_status: int = Field(default=25, ge=1, le=100)

    tinder_exclude_batch_spray_subjects: bool = True

    #: Promote weekly universe-refresh shortlist rows into CRM. Off (report-only) by
    #: default — never auto-sends.
    universe_write_crm: bool = False

    universe_write_crm_limit: int = Field(default=50, ge=1, le=500)

    #: When promoting, optionally call Apollo org enrich (uses credits). Default off.
    universe_write_crm_apollo_org: bool = False

    #: Weekly buyer-persona discovery (Apollo people → CRM named contact). Off by default.
    buyer_persona_refresh_enabled: bool = False

    buyer_persona_max_domains: int = Field(default=10, ge=1, le=100)

    buyer_persona_max_email_reveals: int = Field(default=2, ge=0, le=5)

    #: When True, call NeverBounce before CRM upsert (requires NEVERBOUNCE_API_KEY).
    buyer_persona_verify_email: bool = False

    top100_campaign_enabled: bool = False

    top100_weekly_dow: int = Field(default=6, ge=0, le=6)

    #: Cold/top100 is the EXPERIMENT lane — capped; warm_lane owns the daily tap.
    top100_daily_draft_limit: int = Field(default=10, ge=1, le=100)

    #: Warm-lane daily draft batch size (heartbeat ``warm_lane_daily``).
    warm_lane_daily_draft_limit: int = Field(default=5, ge=1, le=25)

    #: Inbound RFQ/interested/question → response SLA (hours).
    warm_lane_sla_hours: int = Field(default=24, ge=1, le=168)

    outreach_angles: list[str] = Field(
        default_factory=lambda: [
            "replacement",
            "roadshow",
            "technical",
            "insight",
            "intro",
        ]
    )

    subject_styles: list[str] = Field(default_factory=lambda: ["question", "proof_led", "direct"])

    tinder_apollo_contact_resolution: bool = True

    tinder_apollo_max_reveals: int = Field(default=3, ge=0, le=10)

    tinder_domain_mail_max_results: int = Field(default=40, ge=5, le=100)

    tinder_calliope_latent_json_only: bool = True

    account_state_attach_to_brief: bool = True

    deal_dynamics_before_outbound: bool = True

    demo_quote_generator_url: str = Field(
        default="https://quote-demo.example.com/",
        validation_alias=AliasChoices(
            "APP__DEMO_QUOTE_MANUS_URL",
            "DEMO_QUOTE_MANUS_URL",
        ),
    )

    demo_quote_access_code: str = Field(
        default="",
        validation_alias=AliasChoices(
            "APP__DEMO_QUOTE_MANUS_ACCESS_CODE",
            "DEMO_QUOTE_MANUS_ACCESS_CODE",
        ),
    )

    demo_quote_render_url: str = Field(
        default="",
        validation_alias=AliasChoices(
            "APP__DEMO_QUOTE_RENDER_URL",
            "DEMO_QUOTE_RENDER_URL",
        ),
    )

    demo_quote_render_timeout_seconds: int = Field(
        default=120,
        ge=30,
        le=600,
        validation_alias=AliasChoices(
            "APP__DEMO_QUOTE_RENDER_TIMEOUT_SECONDS",
            "DEMO_QUOTE_RENDER_TIMEOUT_SECONDS",
        ),
    )

    demo_quote_render_from_spec_url: str = Field(
        default="",
        validation_alias=AliasChoices(
            "APP__DEMO_QUOTE_RENDER_FROM_SPEC_URL",
            "DEMO_QUOTE_RENDER_FROM_SPEC_URL",
        ),
    )

    demo_quote_pdf_max_bytes: int = Field(
        default=5 * 1024 * 1024,
        ge=512_000,
        le=50 * 1024 * 1024,
        validation_alias=AliasChoices(
            "APP__DEMO_QUOTE_PDF_MAX_BYTES",
            "DEMO_QUOTE_PDF_MAX_BYTES",
        ),
    )

    demo_quote_pdf_verify_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "APP__DEMO_QUOTE_PDF_VERIFY_ENABLED",
            "DEMO_QUOTE_PDF_VERIFY_ENABLED",
        ),
    )

    #: Quote date → deal.expected_close_date offset (CPQ Wave 1).
    quote_deal_cycle_days: int = Field(
        default=90,
        ge=1,
        le=730,
        validation_alias=AliasChoices(
            "APP__QUOTE_DEAL_CYCLE_DAYS",
            "QUOTE_DEAL_CYCLE_DAYS",
        ),
    )

    #: Commercial invoice number series key (see data/knowledge/invoice_number_series.json).
    invoice_number_series: str = Field(
        default="export_e_fy",
        validation_alias=AliasChoices(
            "APP__INVOICE_NUMBER_SERIES",
            "INVOICE_NUMBER_SERIES",
        ),
    )

    thermoformer_research_max_pages: int = Field(default=12, ge=3, le=40)

    thermoformer_research_delay_s: float = Field(default=2.0, ge=0.0, le=60.0)

    thermoformer_research_auto_ingest: bool = True

    thermoformer_research_llm_provider: Literal["openai", "anthropic", "ollama"] = "anthropic"

    neo4j_icp_gauge_sync_enabled: bool = True

    agency_digest_webhook_top_n: int = Field(default=3, ge=1, le=10)

    agency_digest_followup_idle_days: int = Field(default=15, ge=1, le=60)

    agency_digest_webhook_enabled: bool = True

    outbound_touch_audit_enabled: bool = True

    outbound_min_days_without_reply: int = Field(default=15, ge=1, le=120)

    outbound_touch_audit_sent_max: int = Field(default=25, ge=5, le=100)

    na_call_max_attempts: int = Field(default=2, ge=1, le=10)

    na_call_retry_days: int = Field(default=3, ge=1, le=30)

    na_call_cooldown_days: int = Field(default=14, ge=1, le=90)

    account_journey_max_messages: int = Field(default=500, ge=20, le=5000)

    account_journey_max_threads: int = Field(default=120, ge=5, le=1000)

    account_journey_max_attachment_chars: int = Field(default=50_000, ge=1000, le=500_000)

    account_journey_page_size: int = Field(default=250, ge=20, le=500)

    outbound_proactive_hard_block: bool = True

    outbound_dreamer_cooldown_days: int = Field(default=120, ge=30, le=365)

    outbound_dreamer_min_outbound: int = Field(default=5, ge=3, le=20)

    outbound_max_touches_per_domain_90d: int = Field(default=3, ge=1, le=20)

    twenty_sync_segment_in_job_title: bool = False

    maps_daily_budget: int = Field(default=0, ge=0, le=1_000_000)

    maps_agent_text_search_max: int = Field(default=5, ge=0, le=100)

    operator_release_required: bool = False

    operator_release_ttl_hours: float = Field(default=4.0, ge=0.25, le=168.0)

    triangulation_enforce_before_outbound_draft: bool = True

    triangulation_block_on_gaps: bool = True

    triangulation_hex_for_outbound_draft: bool = True

    triangulation_reuse_recent_brief_hours: float = Field(default=24.0, ge=0.0, le=168.0)

    triangulation_outbound_brief_timeout_s: float = Field(default=30.0, ge=5.0, le=120.0)

    outbound_claim_map_enforce_on_draft: bool = False

    firecrawl_scrape_cache_enabled: bool = True

    firecrawl_scrape_cache_ttl_hours: float = Field(default=168.0, ge=1.0, le=720.0)

    math_mode_shadow_enabled: bool = False

    math_mode_advisory_enabled: bool = False

    speculation_mode_enabled: bool = False

    speculation_mode_shadow_enabled: bool = True

    # ── Account heat (morning brief lane 1) ────────────────────────────────
    #: Hot operator pin score floor (muted = hard exclude).
    account_heat_hot_pin_floor: float = Field(default=50.0, ge=0.0, le=100.0)

    account_heat_watch_pin_bonus: float = Field(default=15.0, ge=0.0, le=50.0)

    #: Inbound recency half-life (days) for exponential decay.
    account_heat_inbound_half_life_days: float = Field(default=7.0, ge=0.5, le=60.0)

    account_heat_inbound_max: float = Field(default=30.0, ge=0.0, le=100.0)

    account_heat_intent_max: float = Field(default=20.0, ge=0.0, le=100.0)

    #: Extra points when category is rfq / quote_requested (live commercial heat).
    account_heat_rfq_boost: float = Field(default=12.0, ge=0.0, le=40.0)

    account_heat_deal_max: float = Field(default=15.0, ge=0.0, le=100.0)

    account_heat_vip_bonus: float = Field(default=10.0, ge=0.0, le=50.0)

    #: Minimum heat to appear in morning Lane 1 HEAT NOW.
    account_heat_lane1_threshold: float = Field(default=50.0, ge=0.0, le=100.0)

    account_heat_housekeeping_max_age_days: int = Field(default=14, ge=1, le=60)

    account_heat_housekeeping_cap: int = Field(default=5, ge=1, le=20)
