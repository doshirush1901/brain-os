"""Third-party integrations (Apollo, Vapi, Slack, search, docs, …)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from brain_os.config.common import _COMMON


class ExternalAPIsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEWSDATA_", **_COMMON)

    api_key: SecretStr = SecretStr("")


class ApolloConfig(BaseSettings):
    """Apollo.io API for contact/company enrichment. Uses credits per enrichment."""

    model_config = SettingsConfigDict(env_prefix="APOLLO_", **_COMMON)

    api_key: SecretStr = SecretStr("")


class PeopleDataLabsConfig(BaseSettings):
    """People Data Labs person/company enrich — employment truth vs Apollo lag."""

    model_config = SettingsConfigDict(env_prefix="PEOPLE_DATA_LABS_", **_COMMON)

    api_key: SecretStr = SecretStr("")
    #: Soft kill switch without clearing the key.
    enabled: bool = True


class DemoCampaignCampaignConfig(BaseSettings):
    """NA DEMO Maps → Vapi → email overnight campaign (state-by-state Places discovery)."""

    model_config = SettingsConfigDict(env_prefix="DEMO_CAMPAIGN_CAMPAIGN_", **_COMMON)

    campaign_id: str = "demo_campaign_maps_vapi"
    auto_send: bool = False
    business_hours_only: bool = True
    business_hour_start: int = Field(default=9, ge=0, le=23)
    business_hour_end: int = Field(default=17, ge=1, le=24)
    send_from: str = "operator@example.com"
    max_per_run: int = Field(default=1, ge=1, le=50)
    vapi_delay_s: float = Field(default=300.0, ge=0.0, le=3600.0)
    roadshow_window: str = "the week of June 22 through June 26"
    web_call_cta: str = "a short web call during the first week of June"
    places_manifest: str = "examples/acme/leads/places_demo.json"
    states_order_file: str = "examples/acme/leads/demo_states_order.txt"
    queries_file: str = "examples/acme/leads/queries_demo.txt"
    fetch_wait_seconds: float = Field(default=120.0, ge=0.0, le=600.0)


class VapiConfig(BaseSettings):
    """Vapi.ai outbound voice (assistant dial + poll). Never auto-dial without operator confirm."""

    model_config = SettingsConfigDict(env_prefix="VAPI_", **_COMMON)

    api_key: SecretStr = SecretStr("")
    assistant_id: str = ""
    phone_number_id: str = ""
    poll_interval_sec: float = Field(default=5.0, ge=1.0, le=60.0)
    poll_max_wait_sec: float = Field(default=180.0, ge=10.0, le=600.0)
    request_timeout_sec: float = Field(default=30.0, ge=5.0, le=120.0)
    #: Retries after SIP/provider faults (503, providerfault) on completed dials.
    dial_retry_max: int = Field(default=2, ge=0, le=5)
    dial_retry_delay_sec: float = Field(default=45.0, ge=5.0, le=300.0)
    #: Scrape website + ICP classify before dial (heavy-gauge DEMO buyer fit).
    icp_gate_enabled: bool = True
    icp_force_scrape: bool = True
    icp_scrape_timeout_s: float = Field(default=25.0, ge=5.0, le=120.0)
    icp_block_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    #: Use Brain OS industrial former site research (map/scrape/LLM profile) before dial; else fast Firecrawl snippet + Tinder ICP only.
    icp_use_ira_research: bool = True


class WhatsAppConfig(BaseSettings):
    """Meta WhatsApp Cloud API (outbound follow-ups + inbound webhook).

    Draft-first like email: agents only queue drafts; actual sends go through
    ``brain_os.services.whatsapp.client`` which enforces opt-in, blacklist, and the
    shared outbound dedupe registry. Webhook is flag-gated and signature-verified.
    """

    model_config = SettingsConfigDict(env_prefix="WHATSAPP_", **_COMMON)

    enabled: bool = False
    access_token: SecretStr = SecretStr("")
    phone_number_id: str = ""
    #: Token echoed back on webhook GET verification (``hub.verify_token``).
    verify_token: SecretStr = SecretStr("")
    #: Meta app secret for ``X-Hub-Signature-256`` webhook payload verification.
    app_secret: SecretStr = SecretStr("")
    api_version: str = "v20.0"
    request_timeout_sec: float = Field(default=30.0, ge=5.0, le=120.0)


class TwentyConfig(BaseSettings):
    """Twenty CRM GraphQL API (optional Tier-1 backend; see ``docs/TWENTY_CRM_EVAL.md``)."""

    model_config = SettingsConfigDict(env_prefix="TWENTY_", **_COMMON)

    api_url: str = ""
    api_key: SecretStr = SecretStr("")
    request_timeout_seconds: float = 30.0


class WolframConfig(BaseSettings):
    """WolframAlpha teacher-as-callable (Pattern B). See docs/TEACHERS_IMPLEMENTATION.md."""

    model_config = SettingsConfigDict(env_prefix="WOLFRAM_", **_COMMON)

    app_id: SecretStr = SecretStr("")
    #: Set ``WOLFRAM_ENABLED=false`` to disable when ``WOLFRAM_APP_ID`` is present.
    enabled: bool = True
    timeout_seconds: float = Field(default=30.0, ge=5.0, le=120.0)
    max_query_length: int = Field(default=500, ge=10, le=2000)


class SearchConfig(BaseSettings):
    model_config = SettingsConfigDict(**_COMMON)

    tavily_api_key: SecretStr = SecretStr("")
    searchapi_api_key: SecretStr = SecretStr("")
    #: SearchAPI.io ``engine`` query param when Brain OS uses SearchAPI (fallback chain or ``searchapi_search``).
    searchapi_engine: str = "google"
    serper_api_key: SecretStr = SecretStr("")


class JinaConfig(BaseSettings):
    """Jina Reader (r.jina.ai) and optional Jina Search (s.jina.ai). When set, improves rate limits."""

    model_config = SettingsConfigDict(env_prefix="JINA_", **_COMMON)

    api_key: SecretStr = SecretStr("")


class PdfCoConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PDFCO_", **_COMMON)

    api_key: SecretStr = SecretStr("")


class DocumentAIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCUMENT_AI_", **_COMMON)

    project_id: str = Field(default="", validation_alias="GOOGLE_CLOUD_PROJECT_ID")
    location: str = "us"
    processor_id: str = ""
    invoice_processor_id: str = ""
    form_processor_id: str = ""
    #: httpx read timeout for ``processors:process`` (large PDFs need more than 60s).
    request_timeout_seconds: float = Field(default=300.0, ge=30.0, le=900.0)


class LangfuseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LANGFUSE_", **_COMMON)

    public_key: str = ""
    secret_key: SecretStr = SecretStr("")
    base_url: str = "https://cloud.langfuse.com"


class FirecrawlConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FIRECRAWL_", **_COMMON)

    api_key: SecretStr = SecretStr("")


class UnstructuredConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UNSTRUCTURED_", **_COMMON)

    api_key: SecretStr = SecretStr("")
    api_url: str = "https://api.unstructuredapp.io/general/v0/general"


class SentryConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTRY_", **_COMMON)

    dsn: str = ""
    traces_sample_rate: float = 0.1


class SlackConfig(BaseSettings):
    """Optional RAAM Slack bot (``chat.postMessage`` from MCP / scripts)."""

    model_config = SettingsConfigDict(env_prefix="SLACK_", **_COMMON)

    bot_token: SecretStr = SecretStr("")
    #: App-level token (``xapp-``) with ``connections:write`` for Socket Mode DM listener.
    app_token: SecretStr = SecretStr("")
    default_channel_id: str = ""
    #: Optional ``#demo-sales-leads`` channel ID for operator lead digests.
    sales_leads_channel_id: str = ""
    #: Comma-separated channel IDs where ``@Brain OS`` mentions are answered (``app_mention``).
    allowed_channel_ids: str = ""
    http_timeout_s: float = Field(default=15.0, ge=3.0, le=60.0)
    pipeline_timeout_s: float = Field(default=90.0, ge=30.0, le=600.0)
    #: Enforce full agentic workflow on Slack (no listener/pipeline shortcut exits).
    strict_full_workflow_default: bool = True
    operational_fastpath_enabled: bool = True
    operational_fastpath_timeout_s: float = Field(default=75.0, ge=15.0, le=180.0)
    ack_text: str = "Working on that…"
    #: Show generic progress updates on the ack message while processing.
    progress_updates_enabled: bool = True
    progress_update_interval_s: float = Field(default=2.5, ge=1.0, le=10.0)
    #: If true, progress updates are sent on phase transitions (plus periodic heartbeat).
    progress_milestones_only: bool = True
    #: Post a second short disclosure message ("how I got this") in-thread.
    disclosure_enabled: bool = True
    #: Restrict disclosure posts to high-stakes asks (orders/pipeline/quote/send/outbound).
    disclosure_high_stakes_only: bool = True
    #: Enable Slack thread task mode (start/status/resume) for long-running asks.
    task_mode_enabled: bool = True
    max_concurrent_turns: int = Field(default=2, ge=1, le=10)
    dedupe_ttl_s: float = Field(default=600.0, ge=60.0, le=86400.0)
    #: Slack DM/channel: answer "top hot leads" from operator board + scored ledger (skip full pipeline).
    hot_leads_fastpath_enabled: bool = True
    #: Comma-separated account names never ranked as hot leads in Slack fast path (e.g. marketing noise).
    hot_leads_exclude_accounts: str = "Dyson India,Dyson,KTX"
    #: Optional Slack channel for weekly board brief posts (``SLACK_BOARD_CHANNEL``).
    board_channel_id: str = Field(
        default="",
        validation_alias=AliasChoices("SLACK_BOARD_CHANNEL", "SLACK_BOARD_CHANNEL_ID"),
    )
    #: Enable Slack-only deterministic KB retrieval path (UnifiedRetriever: Qdrant + Neo4j).
    kb_fastpath_enabled: bool = True
    #: Timeout for Slack KB fast path retrieval before falling back to guidance.
    kb_fastpath_timeout_s: float = Field(default=15.0, ge=2.0, le=60.0)
    #: Optional JSON object mapping Slack user IDs to canonical emails.
    #: Example: {"U01ABCDEF":"name@example-company.org"}
    user_map_json: str = ""


class BoardConfig(BaseSettings):
    """Family board portal — weekly brief output and push delivery."""

    model_config = SettingsConfigDict(**_COMMON)

    push_recipients: str = Field(
        default="",
        validation_alias=AliasChoices("BOARD_PUSH_RECIPIENTS"),
    )
    push_cc_interactive: str = Field(
        default="",
        validation_alias=AliasChoices("BOARD_PUSH_CC_INTERACTIVE"),
    )
    brief_output_dir: str = Field(
        default="data/board/briefs",
        validation_alias=AliasChoices("BOARD_BRIEF_OUTPUT_DIR"),
    )
    brief_cron_ist: str = Field(
        default="0 8 * * 1",
        validation_alias=AliasChoices("BOARD_BRIEF_CRON_IST"),
    )
    starter_prompts: list[str] = Field(
        default_factory=lambda: [
            "What changed in the pipeline this week?",
            "Any must-act items for the board?",
            "What are the top risks we should discuss?",
        ],
    )

    @field_validator("starter_prompts", mode="before")
    @classmethod
    def _parse_starter_prompts(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            text = v.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [p.strip() for p in text.split("|") if p.strip()]
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        return []

    @property
    def brief_output_dir_path(self) -> Path:
        return Path(self.brief_output_dir)


class HonchoConfig(BaseSettings):
    """Optional Honcho v3 bridge (dialectic user model + transcript sync). See ``brain_os.brain.honcho_user_model``.

    Env vars use the parent nested form, e.g. ``HONCHO__ENABLED``, ``HONCHO__API_KEY``.
    """

    model_config = SettingsConfigDict(**_COMMON)

    enabled: bool = False
    api_base: str = "https://api.honcho.dev"
    api_key: SecretStr = SecretStr("")
    workspace_id: str = ""
    assistant_peer_id: str = "brain"
    dialectic_query: str = (
        "In 6 bullet points or fewer, summarize this peer's communication style, "
        "stated goals, constraints, and how Brain OS should adapt. Be concrete; omit speculation."
    )
    dialectic_ttl_seconds: float = Field(default=3600.0, ge=60.0, le=604800.0)
    dialectic_reasoning_level: Literal["minimal", "low", "medium", "high", "max"] = "low"
    http_timeout: float = Field(default=30.0, ge=5.0, le=120.0)
    max_message_chars: int = Field(default=12000, ge=500, le=25000)
    #: When ``enabled`` is True, push user/assistant lines to Honcho after each turn (non-blocking).
    sync_transcripts: bool = True
    max_cached_peers: int = Field(default=100, ge=10, le=5000)
