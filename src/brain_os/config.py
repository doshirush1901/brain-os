from __future__ import annotations

"""Centralised Pydantic settings for Brain OS.

Prefer :func:`get_settings` (and nested config objects) for application code so
defaults, env aliases, and validation stay in one place. Use ``os.environ``
only for early bootstrap or tests that must run before settings load.
"""

import json
from datetime import date
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmailMode(str, Enum):
    """Operating mode for the EmailProcessor."""

    TRAINING = "TRAINING"
    OPERATIONAL = "OPERATIONAL"


_COMMON = dict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", **_COMMON)

    openai_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    # Default model for unspecified completions (ReAct, tools, misc.). Use profiles for frontier.
    openai_model: str = "gpt-4.1"
    anthropic_model: str = "claude-sonnet-4-20250514"
    # Task-specific model profiles. Source-of-truth data comes from CRM/RAG/email/tools.
    brain_model_fast: str = "gpt-4.1-mini"
    brain_model_reasoning: str = "gpt-5.5"
    brain_model_writing: str = "gpt-5.5"
    brain_model_verifier: str = "gpt-4.1-mini"
    #: Local OpenAI-compatible endpoint (Ollama). Example: http://localhost:11434/v1 or http://host:11434
    ollama_base_url: str = ""
    #: Optional; Ollama ignores this locally — defaults to a dummy in LLMClient when unset.
    ollama_api_key: SecretStr = SecretStr("")
    ollama_model: str = "llama3.2"
    #: httpx read timeout for Ollama chat calls (local models can be slow on first token).
    ollama_http_timeout_seconds: float = Field(
        default=180.0,
        ge=30.0,
        le=900.0,
        validation_alias=AliasChoices(
            "OLLAMA_HTTP_TIMEOUT_SECONDS", "OLLAMA_REQUEST_TIMEOUT_SECONDS"
        ),
    )
    #: httpx connect timeout for Ollama (wake-from-sleep / LAN can exceed a few seconds).
    ollama_connect_timeout_seconds: float = Field(
        default=45.0,
        ge=5.0,
        le=120.0,
        validation_alias=AliasChoices("OLLAMA_CONNECT_TIMEOUT_SECONDS"),
    )
    #: Global default for Pantheon agents when set to ``ollama`` and ``OLLAMA_BASE_URL`` is non-empty.
    default_llm_provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="openai",
        validation_alias=AliasChoices("BRAIN_DEFAULT_LLM_PROVIDER", "DEFAULT_LLM_PROVIDER"),
    )

    @field_validator("default_llm_provider", mode="before")
    @classmethod
    def _normalize_default_llm_provider(cls, v: Any) -> str:
        if v is None:
            return "openai"
        s = str(v).strip().lower()
        if s in ("openai", "anthropic", "ollama"):
            return s
        return "openai"

    def model_for_profile(self, profile: str | None) -> str:
        """Resolve a logical model profile to a concrete provider model name."""
        if not profile:
            return self.openai_model
        normalized = profile.strip().lower().replace("-", "_")
        mapping = {
            "default": self.openai_model,
            "fast": self.brain_model_fast,
            "extract": self.brain_model_fast,
            "extraction": self.brain_model_fast,
            "classification": self.brain_model_fast,
            "reasoning": self.brain_model_reasoning,
            "synthesis": self.brain_model_reasoning,
            "writing": self.brain_model_writing,
            "drafting": self.brain_model_writing,
            "verifier": self.brain_model_verifier,
            "verification": self.brain_model_verifier,
            "faithfulness": self.brain_model_verifier,
        }
        return mapping.get(normalized, self.openai_model)

    #: Optional per-profile **provider** override for tiered routing (empty = use baseline primary).
    #: When set to ``openai`` / ``anthropic`` / ``ollama``, :meth:`resolve_primary_provider_for_profile`
    #: uses this for matching ``model_profile`` on agent ``call_llm`` / ReAct ``_reason`` paths.
    brain_profile_fast_provider: str = ""
    brain_profile_reasoning_provider: str = ""
    brain_profile_writing_provider: str = ""
    brain_profile_verifier_provider: str = ""
    #: Optional Anthropic model per logical profile (empty = ``anthropic_model`` for that profile).
    brain_anthropic_model_fast: str = ""
    brain_anthropic_model_reasoning: str = ""
    brain_anthropic_model_writing: str = ""
    brain_anthropic_model_verifier: str = ""

    @field_validator(
        "brain_profile_fast_provider",
        "brain_profile_reasoning_provider",
        "brain_profile_writing_provider",
        "brain_profile_verifier_provider",
        mode="before",
    )
    @classmethod
    def _normalize_profile_provider(cls, v: Any) -> str:
        if v is None:
            return ""
        s = str(v).strip().lower()
        if s in ("openai", "anthropic", "ollama"):
            return s
        return ""

    def resolve_primary_provider_for_profile(
        self,
        *,
        baseline_primary: str,
        model_profile: str | None,
    ) -> str:
        """Pick API provider for a call when per-profile overrides are set.

        *baseline_primary* comes from :meth:`brain_os.agents.base_agent.BaseAgent._primary_llm_provider`
        or pipeline context. When no profile-specific override is set, returns *baseline_primary*.
        """
        base = (baseline_primary or "openai").strip().lower()
        if base not in ("openai", "anthropic", "ollama"):
            base = "openai"
        norm = (model_profile or "").strip().lower().replace("-", "_")
        fast_p = self.brain_profile_fast_provider
        reasoning_p = self.brain_profile_reasoning_provider
        writing_p = self.brain_profile_writing_provider
        verifier_p = self.brain_profile_verifier_provider
        by_profile: dict[str, str] = {
            "fast": fast_p,
            "extract": fast_p,
            "extraction": fast_p,
            "classification": fast_p,
            "reasoning": reasoning_p,
            "synthesis": reasoning_p,
            "writing": writing_p,
            "drafting": writing_p,
            "verifier": verifier_p,
            "verification": verifier_p,
            "faithfulness": verifier_p,
        }
        override = (by_profile.get(norm) or "").strip().lower()
        if override in ("openai", "anthropic", "ollama"):
            return override
        return base


class EmbeddingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOYAGE_", **_COMMON)

    api_key: SecretStr = SecretStr("")
    model: str = "voyage-3"
    rerank_model: str = "rerank-2.5"


class QdrantConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_", **_COMMON)

    url: str = "http://localhost:6333"
    api_key: SecretStr = SecretStr("")
    collection: str = "brain_knowledge_v3"
    # Request timeout in seconds (avoids indefinite hang when Qdrant is slow or unreachable)
    timeout: float = 30.0
    # Optional: second cluster used when the primary (url) is unreachable (connect/timeout).
    # Typical: QDRANT_URL=https://…cloud.qdrant.io, QDRANT_FALLBACK_URL=http://localhost:6333
    fallback_url: str = ""
    fallback_api_key: SecretStr = SecretStr("")
    # Optional: when set, every upsert and ensure_collection is mirrored to this
    # cluster so local and cloud stay in sync (dual-write).
    cloud_url: str = ""
    cloud_api_key: SecretStr = SecretStr("")

    # When APP__USE_SPARSE_HYBRID=true, this collection is used (dense + sparse vectors). Re-ingest to populate.
    collection_hybrid: str = "brain_knowledge_hybrid"


class Neo4jConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEO4J_", **_COMMON)

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("")
    auth: str = ""
    # Optional: mirror every graph write to this database (e.g. Neo4j Aura) while
    # keeping NEO4J_URI on local Docker. Set URI + cloud password or NEO4J_CLOUD_AUTH.
    cloud_uri: str = ""
    cloud_user: str = ""
    cloud_password: SecretStr = SecretStr("")
    cloud_auth: str = ""

    def resolved_auth(self) -> tuple[str, str]:
        """Resolve Neo4j credentials from explicit password or NEO4J_AUTH."""
        user = self.user.strip() or "neo4j"
        password = self.password.get_secret_value().strip()
        if password:
            return user, password

        auth = self.auth.strip()
        if "/" in auth:
            auth_user, auth_password = auth.split("/", 1)
            auth_user = auth_user.strip() or user
            auth_password = auth_password.strip()
            if auth_password:
                return auth_user, auth_password
        if "localhost" in self.uri or "127.0.0.1" in self.uri:
            # Local docker-compose default (safe dev fallback).
            return user, "brain_knowledge_graph"
        return user, ""

    def resolved_cloud_auth(self) -> tuple[str, str] | None:
        """Credentials for NEO4J_CLOUD_URI when dual-write is enabled.

        Returns ``None`` if ``cloud_uri`` is unset or no cloud password/auth is configured.
        """
        uri = self.cloud_uri.strip()
        if not uri:
            return None
        primary_user = self.user.strip() or "neo4j"
        user = self.cloud_user.strip() or primary_user
        pw = self.cloud_password.get_secret_value().strip()
        if pw:
            return user, pw
        ca = self.cloud_auth.strip()
        if "/" in ca:
            cu, cpw = ca.split("/", 1)
            cu = cu.strip() or user
            cpw = cpw.strip()
            if cpw:
                return cu, cpw
        return None


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATABASE_", **_COMMON)

    url: str = "postgresql+asyncpg://brain:brain@localhost:5432/brain_crm"


class MemoryConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEM0_", **_COMMON)

    api_key: SecretStr = SecretStr("")


class GoogleConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GOOGLE_", **_COMMON)

    credentials_path: Path = Path("credentials.json")
    token_path: Path = Path("token.json")
    oauth_client_id: str = ""
    oauth_client_secret: SecretStr = SecretStr("")
    brain_email: str = ""
    training_email: str = ""
    email_mode: EmailMode = Field(
        default=EmailMode.TRAINING,
        validation_alias="BRAIN_EMAIL_MODE",
    )
    email_poll_enabled: bool = Field(
        default=False,
        validation_alias="BRAIN_EMAIL_POLL",
    )
    # Optional second mailbox (read-only, e.g. procurement/vendor). When set, search and observe both.
    secondary_credentials_path: Path | None = Field(
        default=None, validation_alias="GOOGLE_SECONDARY_CREDENTIALS_PATH"
    )
    secondary_token_path: Path | None = Field(
        default=None, validation_alias="GOOGLE_SECONDARY_TOKEN_PATH"
    )
    secondary_oauth_client_id: str = ""
    secondary_oauth_client_secret: SecretStr = SecretStr("")
    #: When True with ``BRAIN_EMAIL_MODE=OPERATIONAL``, secondary mailbox uses the same Gmail
    #: scopes as primary (read, compose, send). Requires deleting the secondary token and
    #: re-authenticating after enabling. See ``docs/TROUBLESHOOTING.md`` (dual mailbox).
    secondary_full_access: bool = Field(
        default=False, validation_alias="GOOGLE_SECONDARY_FULL_ACCESS"
    )
    #: Gmail Pub/Sub push (``POST /api/gmail/push``); requires topic + ``users.watch`` registration.
    gmail_push_enabled: bool = False
    #: Maps Platform server key (Geocoding API, Directions API). Iris + MCP when set.
    maps_api_key: SecretStr = SecretStr("")


class ExternalAPIsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEWSDATA_", **_COMMON)

    api_key: SecretStr = SecretStr("")


class ApolloConfig(BaseSettings):
    """Apollo.io API for contact/company enrichment. Uses credits per enrichment."""

    model_config = SettingsConfigDict(env_prefix="APOLLO_", **_COMMON)

    api_key: SecretStr = SecretStr("")


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


class TwentyConfig(BaseSettings):
    """Twenty CRM GraphQL API (optional Tier-1 backend; see ``docs/TWENTY_CRM_EVAL.md``)."""

    model_config = SettingsConfigDict(env_prefix="TWENTY_", **_COMMON)

    api_url: str = ""
    api_key: SecretStr = SecretStr("")
    request_timeout_seconds: float = 30.0


class NeverBounceConfig(BaseSettings):
    """NeverBounce email verification API (query param ``key`` on ``/single/check``)."""

    model_config = SettingsConfigDict(env_prefix="NEVERBOUNCE_", **_COMMON)

    api_key: SecretStr = SecretStr("")
    #: REST API base (NeverBounce v4.2 default). Override only if docs specify another host.
    base_url: str = "https://api.neverbounce.com/v4.2"


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


class RedisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", **_COMMON)

    url: str = ""


class LLMEndpointsConfig(BaseSettings):
    """Phase-2 hardening: centralised LLM / embedding endpoint URLs.

    Older audits (`SYSTEM_AUDIT.md` 1.4e) flagged ~16 hardcoded URLs across
    brain modules. This config gives them a single home so an alt-endpoint,
    proxy, or local mock can be wired up by env var alone:

    ```
    LLM_ENDPOINTS__OPENAI_BASE_URL=...        # default https://api.openai.com/v1
    LLM_ENDPOINTS__OPENAI_MODELS_URL=...      # default https://api.openai.com/v1/models
    LLM_ENDPOINTS__ANTHROPIC_BASE_URL=...     # default https://api.anthropic.com
    LLM_ENDPOINTS__VOYAGE_EMBEDDINGS_URL=...  # default https://api.voyageai.com/v1/embeddings
    LLM_ENDPOINTS__VOYAGE_RERANK_URL=...      # default https://api.voyageai.com/v1/rerank
    ```

    Defaults preserve current behaviour (no env change needed). Migration
    of brain modules to read from these settings is incremental — see
    ``docs/PHASE2_HARDENING_AUDIT.md`` §4.5.
    """

    model_config = SettingsConfigDict(env_prefix="LLM_ENDPOINTS__", **_COMMON)

    openai_base_url: str = "https://api.openai.com/v1"
    openai_models_url: str = "https://api.openai.com/v1/models"
    anthropic_base_url: str = "https://api.anthropic.com"
    voyage_embeddings_url: str = "https://api.voyageai.com/v1/embeddings"
    voyage_rerank_url: str = "https://api.voyageai.com/v1/rerank"


class HeliconeConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HELICONE_", **_COMMON)

    api_key: SecretStr = SecretStr("")
    # Optional Helicone dashboard dimensions (Helicone-Property-* on every proxied LLM request).
    property_environment: str = ""
    property_app: str = ""


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
    #: Enable Slack-only deterministic KB retrieval path (UnifiedRetriever: Qdrant + Neo4j).
    kb_fastpath_enabled: bool = True
    #: Timeout for Slack KB fast path retrieval before falling back to guidance.
    kb_fastpath_timeout_s: float = Field(default=15.0, ge=2.0, le=60.0)
    #: Optional JSON object mapping Slack user IDs to canonical emails.
    #: Example: {"U01ABCDEF":"name@example-company.org"}
    user_map_json: str = ""


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


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(**_COMMON, populate_by_name=True)

    log_level: str = "INFO"
    log_format: str = "text"
    environment: str = "development"
    api_secret_key: SecretStr = SecretStr("")
    cors_origins: str = "http://localhost:3000"
    react_max_iterations: int = 8
    #: Extra attempts for ReAct / MCP tool calls on transient errors (429/5xx, timeouts).
    react_tool_transient_retries: int = 1
    react_tool_timeout_seconds: float = Field(default=45.0, ge=1.0, le=600.0)
    mcp_tool_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    crm_backend: Literal["postgres", "twenty", "hybrid"] = Field(
        default="postgres",
        validation_alias=AliasChoices("APP__CRM_BACKEND", "CRM_BACKEND"),
    )
    # Total request timeout (seconds). Typical presets: 30s, 2min=120, 5min=300, 10min=600, 20min=1200.
    # Future: Brain OS/Athena can choose by request type (e.g. learning model for time-to-complete).
    pipeline_timeout: int = 600
    #: Per-stage wall-clock budgets (seconds). ``0`` = no per-stage cap (learn is best-effort).
    #: Override via ``APP__PIPELINE_STAGE_BUDGETS`` JSON object. Global ``pipeline_timeout`` is
    #: raised to at least 2× the sum of positive budgets at runtime.
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
    # Per-sub-agent "slot": each parallel agent has this long to return best answer.
    agent_timeout: int = 90
    # Max sub-agents running in parallel (e.g. 5); Athena gets responses from up to this many at once.
    max_parallel_agents: int = 5
    deployment_profile: Literal["full", "edge"] = "full"
    faithfulness_heuristic_only: bool = False
    brain_route_provider: str = ""
    brain_synthesis_provider: str = ""
    brain_embedding_provider: str = ""
    vault_export_default_dir: str = ""
    # Athena's timeout to package the final answer (LLM synthesis) for Cursor/API.
    athena_synthesis_timeout: int = 90
    mem0_timeout: float = 15.0
    neo4j_max_pool_size: int = 50

    max_delegation_depth: int = 5

    faithfulness_threshold: float = 0.6
    faithfulness_hard_threshold: float = 0.3
    #: ``best_effort`` — current behavior with caveat/refusal tiers. ``strict`` tightens verifier errors
    #: and substantive bypass when the query looks pricing/quote/spec/delivery-related or Plutus/QB routed.
    faithfulness_mode: Literal["best_effort", "strict"] = "best_effort"
    confidence_floor: float = 0.3
    guardrails_fail_closed: bool = True
    mnemon_semantic_check: bool = False
    legacy_quarantine_strict: bool = False

    # Optional override for Hugging Face hub cache (e.g. when default cache disk is full)
    hf_cache_dir: str = ""

    # FlashRank local reranker (ONNX) cache. Empty = ``{BRAIN_DATA_DIR or ./data}/.flashrank_cache``
    # so CLI and server survive ``/tmp`` clears. Override if you want a shared cache path.
    flashrank_cache_dir: str = ""

    # In-process memory bounds (see Part 8 audit).
    message_bus_log_maxlen: int = Field(default=1000, ge=100, le=500_000)
    unified_context_max_history: int = Field(default=50, ge=4, le=10_000)
    unified_context_max_users: int = Field(default=500, ge=1, le=100_000)
    #: Toggle SQLite embedding cache (L2) in ``EmbeddingService``.
    #: Canonical env var: ``APP__EMBEDDING_SQLITE_CACHE_ENABLED``.
    #: Legacy alias kept for backward compatibility: ``APP__EMBEDDING_CACHE_SQLITE_ENABLED``.
    embedding_sqlite_cache_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "APP__EMBEDDING_SQLITE_CACHE_ENABLED",
            "APP__EMBEDDING_CACHE_SQLITE_ENABLED",
            "EMBEDDING_SQLITE_CACHE_ENABLED",
            "EMBEDDING_CACHE_SQLITE_ENABLED",
        ),
    )
    #: Retention window for SQLite embedding-cache rows.
    embedding_sqlite_cache_retention_days: int = Field(
        default=30,
        ge=1,
        le=3650,
        validation_alias=AliasChoices(
            "APP__EMBEDDING_SQLITE_CACHE_RETENTION_DAYS",
            "EMBEDDING_SQLITE_CACHE_RETENTION_DAYS",
        ),
    )
    #: Hard cap on SQLite embedding-cache rows; oldest rows are pruned first.
    embedding_sqlite_cache_max_rows: int = Field(
        default=100_000,
        ge=1_000,
        le=50_000_000,
        validation_alias=AliasChoices(
            "APP__EMBEDDING_SQLITE_CACHE_MAX_ROWS",
            "EMBEDDING_SQLITE_CACHE_MAX_ROWS",
        ),
    )
    #: Opportunistic prune cadence for SQLite embedding cache.
    embedding_sqlite_cache_prune_every_n_writes: int = Field(
        default=1_000,
        ge=1,
        le=1_000_000,
        validation_alias=AliasChoices(
            "APP__EMBEDDING_SQLITE_CACHE_PRUNE_EVERY_N_WRITES",
            "EMBEDDING_SQLITE_CACHE_PRUNE_EVERY_N_WRITES",
        ),
    )
    #: SQLite -> Postgres cutover flags. ``*_enabled`` turns on dual-write; ``*_read`` flips
    #: the primary read path after the parity window is complete.
    pg_store_ingest_enabled: bool = False
    pg_store_ingest_read: bool = False
    pg_store_corrections_enabled: bool = False
    pg_store_corrections_read: bool = False
    pg_store_agent_journal_enabled: bool = False
    pg_store_agent_journal_read: bool = False
    pg_store_atlas_logbook_enabled: bool = False
    pg_store_atlas_logbook_read: bool = False
    pg_store_asclepius_punch_enabled: bool = False
    pg_store_asclepius_punch_read: bool = False
    pg_store_learning_feedback_enabled: bool = False
    pg_store_learning_feedback_read: bool = False
    pg_store_conversation_enabled: bool = False
    pg_store_conversation_read: bool = False
    pg_store_relationship_enabled: bool = False
    pg_store_relationship_read: bool = False
    pg_store_pending_memory_enabled: bool = False
    pg_store_pending_memory_read: bool = False
    pg_store_tool_invocations_enabled: bool = False
    pg_store_tool_invocations_read: bool = False
    pg_store_procedural_enabled: bool = False
    pg_store_procedural_read: bool = False
    pg_store_cursor_sessions_enabled: bool = False
    pg_store_cursor_sessions_read: bool = False
    pg_store_episodes_enabled: bool = False
    pg_store_episodes_read: bool = False
    pg_store_goals_enabled: bool = False
    pg_store_goals_read: bool = False

    #: Lettie peer-system bridge to Letta (see docs/LETTA_BRIDGE_IMPLEMENTATION.md).
    lettie_enabled: bool = False
    letta_base_url: str = "http://localhost:8283"
    letta_api_key: SecretStr = SecretStr("")
    letta_agent_id: str = ""
    letta_timeout_seconds: float = 45.0

    # PII: when True, redact email/phone in ingested chunks before storing in Qdrant.
    redact_pii_at_ingest: bool = False

    #: Which API bills **structured** LLM steps in document ingestion (DigestiveSystem: classify,
    #: summarize, email metadata, contact extraction) and the KnowledgeGraph legacy entity fallback.
    #: Use ``anthropic`` to spend Claude credits during bulk ``brain ingest``, or ``ollama`` for local
    #: zero-cost inference when ``OLLAMA_BASE_URL`` is set; embeddings stay on Voyage.
    digestive_llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    #: When ``digestive_llm_provider`` is ``anthropic``, model id for ingest-only structured calls (alias ok).
    #: Empty = use global ``ANTHROPIC_MODEL``. For high-volume ingest, prefer Haiku, e.g. ``claude-haiku-4-5``.
    digestive_anthropic_model: str = ""
    #: When Helicone is enabled: proxy Anthropic via ``anthropic.helicone.ai`` (default True). Some newer Claude
    #: snapshots return 404 through that proxy — set False so Anthropic hits ``api.anthropic.com`` directly while
    #: OpenAI can remain on ``oai.helicone.ai``.
    helicone_proxy_anthropic: bool = True

    # When True, use dense + sparse hybrid in Qdrant (new collection, RRF). Requires re-ingest to populate sparse.
    use_sparse_hybrid: bool = False

    # Autonomous drip (`AutonomousDripEngine`): optional LLM layers using `prompts/drip_*.txt`.
    drip_llm_insights: bool = False
    drip_llm_adjust: bool = False

    # Progressive tool discovery for MCP/agent connectivity.
    progressive_tool_discovery: bool = False

    #: AgentLoop standing objectives (MCP plan_task): post-phase judge + auto-continuation.
    standing_goal_enabled: bool = True
    standing_goal_max_turns: int = Field(default=20, ge=1, le=100)
    standing_goal_max_phases: int = Field(default=20, ge=1, le=100)
    #: When False, phase validator LLM errors fail closed in strict mode (MCP + ``brain task``).
    phase_validator_fail_open: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "APP__PHASE_VALIDATOR_FAIL_OPEN",
            "PHASE_VALIDATOR_FAIL_OPEN",
        ),
    )
    #: Run per-phase contract validator after each ``brain task`` specialist phase.
    task_phase_validator_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "APP__TASK_PHASE_VALIDATOR_ENABLED",
            "TASK_PHASE_VALIDATOR_ENABLED",
        ),
    )
    #: Retry a failed task phase once when validator rejects output.
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
    # MCP fast-lane controls for short factual questions.
    mcp_fastlane_enabled: bool = True
    mcp_fastlane_min_score: float = Field(default=0.45, ge=0.0, le=1.0)
    mcp_fastlane_max_snippets: int = Field(default=3, ge=1, le=10)
    # MCP email scope default (tool-level callers may override per request).
    mcp_email_default_scope: Literal["primary", "secondary", "both"] = "primary"
    # MCP provider variance controls for routed LLM turns.
    mcp_route_prefer_cloud_fast: bool = True
    mcp_ollama_retry_budget: int = Field(default=0, ge=0, le=10)
    mcp_fast_fail_to_openai: bool = True

    #: When True, read ``SOUL.md`` once per pipeline request and attach
    #: ``_soul_preamble_snapshot`` to agent context (stable for that request;
    #: bypasses process-level ``load_soul_preamble`` cache for identity text).
    request_prompt_snapshot: bool = False

    # Bound single-turn user text after PERCEIVE/REMEMBER (NUL strip + length cap); see `untrusted_input.sanitize_pipeline_query`.
    pipeline_query_max_chars: int = Field(default=100_000, ge=4096, le=500_000)
    # Cheap exits (dedup / fast-path / quick-pipeline / truth-hints) safety policy.
    # When enabled, matching keywords force full pipeline execution for operational/time-sensitive asks.
    cheap_exit_bypass_enabled: bool = True
    # Comma-separated case-insensitive keywords/phrases (substring match).
    cheap_exit_bypass_keywords: str = (
        "send,dispatch,draft,reply,follow-up,follow up,outreach,campaign,"
        "latest,recent,today,yesterday,just sent,unread,inbox,"
        "update,create,delete,close deal,mark as,"
        "confidential,internal,private,margin,pricing,quote,what did"
    )
    # Optional comma-separated allowlist phrases that disable bypass for known-safe intents.
    cheap_exit_allowlist_keywords: str = ""

    # When HTTP/API (or other callers) omit user_id, use this stable key so
    # ConversationMemory + Mem0 grow under one identity. Prefer your work email
    # in .env, e.g. APP__DEFAULT_USER_ID=you@example.com
    default_user_id: str = "brain_default_user"

    #: When True, ``brain ask`` / ``brain chat`` / ``brain task`` behave like ``--auto-fallback-api`` was passed
    #: whenever the data-dir lock cannot be acquired: after ``GET /api/health`` preflight they call the
    #: running API (``POST /api/query`` or task stream). Use when you routinely keep ``brain server`` up and
    #: still want one-shot CLI from Cursor or scripts without passing the flag each time.
    auto_fallback_api_on_data_dir_lock: bool = False

    #: After ``send_message`` with a ``thread_id``, remove INBOX (archive) so Primary stays “to-do” only.
    #: Override per request via ``EmailSendRequest.archive_from_inbox`` (API) or ``send_message(..., archive_from_inbox=…)``.
    email_archive_thread_after_send: bool = False

    #: SlowAPI sliding-window limits per key (Bearer prefix or IP). Disable in tests via APP__RATE_LIMIT_ENABLED=false.
    rate_limit_enabled: bool = True
    #: Optional limits storage URI, e.g. ``redis://localhost:6379/1``. Empty = in-memory backend.
    rate_limit_storage_uri: str = ""
    rate_limit_query_per_minute: int = Field(default=120, ge=1, le=50_000)
    rate_limit_query_stream_per_minute: int = Field(default=60, ge=1, le=50_000)
    rate_limit_task_stream_per_minute: int = Field(default=40, ge=1, le=50_000)
    rate_limit_send_per_minute: int = Field(default=24, ge=1, le=5000)
    rate_limit_ingest_per_minute: int = Field(default=40, ge=1, le=5000)
    #: Execute optional deterministic-route specialists (bounded by optional_agent_budget).
    optional_agent_execution_enabled: bool = True
    optional_agent_budget: int = Field(default=2, ge=0, le=10)
    #: When True, deterministic routing fails closed if required tools are unavailable.
    required_tool_enforcement: bool = False
    #: When >0 and the regex winner's margin vs runner-up is below this, skip deterministic routing
    #: and delegate to Athena (reduces brittle misroutes). 0 disables.
    router_deterministic_margin_min: float = Field(default=0.0, ge=0.0, le=20.0)
    #: Embed query + cosine-compare with ``data/brain/router_intent_anchors.json`` when pattern
    #: margin is below ``router_embedding_tiebreak_max_margin`` (0 disables).
    router_embedding_tiebreak_enabled: bool = False
    router_embedding_tiebreak_max_margin: float = Field(default=1.25, ge=0.0, le=10.0)
    #: When True, deterministic ``optional_agents`` are ordered by lexical overlap with the query before budget cut.
    router_optional_keyword_ranking: bool = True
    #: Merge structured ``pipeline_trace`` into request metadata dict for callers (API/MCP/tests).
    pipeline_attach_observability_trace: bool = True
    #: Append a compact agentic execution summary (route/delegation/evidence) to final responses.
    response_include_execution_summary: bool = False
    #: Prefer local Ollama for LLM-routed turns (Athena orchestration path) when configured.
    #: Default False so cloud keys are used when present; set True for zero-cost local-first.
    llm_route_prefer_ollama: bool = False
    #: Escalate LLM-routed turns to cloud models for high-stakes prompts (pricing/contract/send etc.).
    llm_route_high_stakes_cloud_escalation: bool = True
    #: When True (or per-request ``context["uncensored_local_llm_mode"]``), prefer the configured
    #: local OpenAI-compatible endpoint for LLM calls and relax faithfulness / confidence-floor /
    #: guardrail *response replacements* so the model answer is not replaced by canned refusals.
    #: Does **not** disable Gmail send policy, outbound pre-checks, or Aegis post-Gapper BLOCK.
    uncensored_local_llm_mode: bool = False

    #: Semantic chunking (Chonkie) — only used when semantic chunking path succeeds.
    ingest_semantic_similarity_threshold: float = Field(default=0.7, ge=0.2, le=0.99)
    ingest_semantic_similarity_window: int = Field(default=3, ge=1, le=20)
    #: Per-doc-type tiktoken-ish chunk sizing (token targets + overlap).
    ingest_chunk_quote_tokens: int = Field(default=256, ge=64, le=2048)
    ingest_chunk_quote_overlap: int = Field(default=64, ge=0, le=512)
    ingest_chunk_technical_tokens: int = Field(default=384, ge=64, le=2048)
    ingest_chunk_technical_overlap: int = Field(default=96, ge=0, le=512)
    ingest_chunk_default_tokens: int = Field(default=512, ge=128, le=8192)
    ingest_chunk_default_overlap: int = Field(default=128, ge=0, le=512)

    #: Unified retriever knobs (were module constants in ``retriever.py``).
    retriever_over_retrieve_factor: int = Field(default=3, ge=1, le=12)
    retriever_diversity_overlap_threshold: float = Field(default=0.55, ge=0.1, le=0.95)
    retriever_rerank_candidates_factor: int = Field(default=2, ge=1, le=8)
    retriever_keyword_boost_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    #: When True and prerank keyword overlap is below the threshold, run one extra Qdrant
    #: search using the first LLM-decomposed sub-query (bounded; same Qdrant timeout).
    retriever_second_pass_enabled: bool = False
    #: Below this prerank ``keyword_overlap_score`` (query vs merged hit sample), trigger second pass.
    retriever_second_pass_keyword_threshold: float = Field(default=0.22, ge=0.0, le=0.95)
    #: DEBUG JSON lines for retrieval (``retrieval_trace`` logger); include ``run_id`` when set via context.
    retriever_trace_enabled: bool = False
    #: Greedy Jaccard de-dup on word sets (skip chunk if max similarity to kept set >= threshold).
    retriever_dedup_enabled: bool = False
    retriever_dedup_max_jaccard: float = Field(default=0.92, ge=0.5, le=1.0)
    retriever_dedup_content_chars: int = Field(default=800, ge=200, le=12000)
    #: Named profile for backend timeouts: ``default`` | ``quick`` | ``deep`` (see ``retrieval_slo``).
    retriever_default_profile: str = "default"
    #: Max total characters of KB evidence passed to faithfulness (0 = unlimited).
    retriever_evidence_max_chars: int = Field(default=0, ge=0, le=500_000)
    #: ``off`` | ``flag`` | ``redact`` for pipeline faithfulness evidence bundle only.
    retriever_bundle_pii_mode: str = "off"
    #: When True, run optional citation alignment after faithfulness (extra LLM call).
    citation_aligner_enabled: bool = False
    #: In-process TTL for ``get_pipeline_summary`` in quick-pipeline short-circuit (0 = no cache).
    crm_pipeline_cache_ttl_seconds: float = Field(default=15.0, ge=0.0, le=300.0)

    #: Monthly LLM usage gate (0 = disabled). Requires ``REDIS_URL``; usage is **estimated**
    #: per pipeline turn (see ``brain_os.systems.llm_budget``). Bucket resets by UTC month key.
    llm_monthly_token_budget: int = Field(default=0, ge=0, le=2_000_000_000)
    #: ``user`` — per ``sender_id`` / ``user_id``; ``global`` — one counter for the deployment.
    llm_budget_scope: str = Field(default="user")

    #: JSON list of heartbeat jobs (scheduled ``brain ask``-style runs). See ``scripts/heartbeat_jobs.example.json``.
    heartbeat_jobs_path: str = Field(default="data/heartbeat_jobs.json")
    #: In-process scheduler on the API server (``brain heartbeat run`` remains CLI/cron).
    heartbeat_server_enabled: bool = False
    heartbeat_server_interval_minutes: int = Field(default=15, ge=1, le=1440)
    #: Embed Slack Socket Mode listener in the API server (``brain server``); see ``docs/SLACK_CHAT.md``.
    slack_listen_enabled: bool = False
    #: ``EventTaskReactor`` — spawn bounded tasks from ``data/event_task_rules.json``.
    event_task_reactor_enabled: bool = False
    event_task_rules_path: str = Field(default="data/event_task_rules.json")
    event_task_max_inflight: int = Field(default=2, ge=1, le=32)
    hephaestion_audit_path: str = "data/audit"
    hephaestion_langfuse_lookback_hours: int = Field(default=24, ge=1, le=168)
    hephaestion_golden_auto_write: bool = False
    hephaestion_skip_pytest_cov: bool = True
    hephaestion_brief_webhook: bool = False
    #: SQLite queue drained into Mem0 during dream (``brain memory pending add`` / heartbeat enqueue action).
    pending_memory_queue_path: str = Field(default="data/brain/pending_memory_queue.sqlite")
    #: Background deep consolidation cadence (Dream cycle) while the API server is running.
    #: 0 disables interval scheduling; nightly exhale still runs.
    deep_consolidation_interval_hours: float = Field(default=6.0, ge=0.0, le=168.0)

    #: Weekly Pantheon AI Engineering Conference (``brain conference run`` / heartbeat).
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
    #: Ram (Ramanujan) weekly formula Sabbath — proposals only, never auto-promote.
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

    #: Public base URL for customer booking links. Empty keeps links relative (useful in tests/dev).
    scheduling_public_base_url: str = ""
    #: Secret for hashing public scheduling tokens. Falls back to ``APP__API_SECRET_KEY`` when empty.
    scheduling_token_secret: SecretStr = SecretStr("")
    scheduling_default_timezone: str = "Asia/Kolkata"
    #: Brain OS in-service date (first foundation commit). Pantheon ``age_in_days`` uses this.
    brain_birth_date: date = Field(default=date(2026, 3, 6))
    #: Operator IANA timezone for pipeline clock injection (IST by default).
    operator_timezone: str = "Asia/Kolkata"
    #: Inject authoritative date/time block into pipeline enrichment (step 5.5).
    brain_temporal_context_enabled: bool = True
    #: Life phase hint: awake | dream | briefing | quiet.
    brain_time_mode: Literal["awake", "dream", "briefing", "quiet"] = "awake"
    #: Tech-industry "internet years" ratio (Vint Cerf, 1999): 1 calendar year at full load
    #: ≈ this many Brain OS experience years. Default 7 → ~52.2 calendar days per Brain OS-year.
    brain_internet_years_ratio: float = Field(default=7.0, ge=1.0, le=52.0)
    #: Weight experiential age by dream/journal/runs/operator signals (else full load).
    brain_experience_activity_weighting: bool = True
    #: Minimum activity multiplier when idle (still ages slowly).
    brain_experience_idle_floor: float = Field(default=0.15, ge=0.0, le=1.0)
    #: Lookback window (days) for activity scoring.
    brain_experience_lookback_days: int = Field(default=90, ge=7, le=3650)
    scheduling_default_expiry_days: int = Field(default=7, ge=1, le=90)
    scheduling_default_buffer_minutes: int = Field(default=15, ge=0, le=180)
    scheduling_default_slot_step_minutes: int = Field(default=30, ge=5, le=240)
    #: Optional read-only iCal (.ics) feed URL (e.g. Google Calendar “secret address in iCal format”).
    #: Loaded for future busy-time / slot integration; empty disables.
    scheduling_busy_ics_url: str = ""
    #: Optional REST API key for a third-party calendar / scheduling product (iCal-related SaaS).
    #: Stored for future integration; empty disables. Pair with ``scheduling_ical_api_base_url`` when the vendor requires it.
    scheduling_ical_api_key: SecretStr = SecretStr("")
    #: Optional API base URL for the same vendor (e.g. ``https://api.vendor.example/v1``). Empty if unknown or key-only.
    scheduling_ical_api_base_url: str = ""

    #: Interconnection engine for outbound drafting/sending.
    interconnections_enabled: bool = False
    interconnections_top_k: int = Field(default=6, ge=3, le=20)
    interconnections_brief_ttl_hours: int = Field(default=72, ge=1, le=720)
    #: Require a valid brief id for campaign sends on /api/email/send.
    interconnections_require_recent_brief_on_campaign_send: bool = False

    #: When true and ``NEVERBOUNCE_API_KEY`` is set, user-initiated sends call NeverBounce ``/single/check``
    #: and block ``invalid`` / ``disposable`` results before Gmail.
    neverbounce_outbound_verify: bool = True
    #: httpx read timeout (seconds) for NeverBounce single check.
    neverbounce_request_timeout_seconds: float = Field(default=12.0, ge=3.0, le=30.0)

    #: Bounded wait for Qdrant collection ensure at API startup (hangs surface as timeout, not silent stall).
    startup_qdrant_ensure_timeout_s: float = Field(default=120.0, ge=10.0, le=600.0)
    #: When False, Qdrant ensure timeout/error logs a warning and boot continues (Railway / degraded API).
    startup_qdrant_ensure_required: bool = True
    #: Bounded wait for Postgres CRM/vendor DDL at startup (misconfig surfaces quickly).
    startup_sql_schema_timeout_s: float = Field(default=120.0, ge=10.0, le=600.0)
    #: When False, CRM create_tables timeout/error logs a warning and boot continues.
    startup_sql_schema_required: bool = True
    #: Run the heavy startup sequence in a background task and accept traffic after logging init.
    startup_defer_to_background: bool = False

    #: When True, ``brain delegate claude-code`` and MCP ``invoke_claude_code`` may spawn the Claude Code CLI.
    claude_code_delegate_enabled: bool = False
    #: Executable basename resolved via PATH (default Anthropic Claude Code CLI).
    claude_code_delegate_command: str = "claude"
    claude_code_delegate_timeout_seconds: int = Field(default=600, ge=30, le=7200)
    #: Comma-separated absolute paths; empty = Brain OS repo checkout root only.
    claude_code_delegate_allowed_roots: str = ""
    claude_code_delegate_max_prompt_chars: int = Field(default=48_000, ge=500, le=200_000)
    claude_code_delegate_max_output_chars: int = Field(default=400_000, ge=10_000, le=2_000_000)

    #: When True, ``brain git ship commit`` may run ``git add`` + ``git commit`` under allowed roots.
    git_ship_enabled: bool = False
    #: Separate gate for ``brain git ship push`` (requires ``git_ship_enabled`` as well).
    git_ship_allow_push: bool = False
    #: Comma-separated absolute dirs for git ship cwd (empty = Brain OS repo root only).
    git_ship_allowed_roots: str = ""
    git_ship_timeout_seconds: int = Field(default=120, ge=10, le=900)
    git_ship_max_commit_message_chars: int = Field(default=8000, ge=20, le=50_000)

    #: Optional HTTPS URL (Slack/Discord incoming webhook, etc.) for operator notifications.
    operator_webhook_url: SecretStr = SecretStr("")
    operator_webhook_timeout_s: float = Field(default=8.0, ge=1.0, le=60.0)
    #: Comma-separated event names to send (empty = all implemented: dream, delegate, git_ship).
    operator_webhook_events: str = ""

    #: Brain OS Agency — proactive operator suggestion deck (Sprint 1).
    agency_enabled: bool = True
    agency_max_daily_cards: int = Field(default=8, ge=1, le=50)
    agency_goals_path: str = ""
    agency_prefs_path: str = ""
    #: Accept on outreach/followup runs persuasion sprint instead of revenue draft package.
    agency_accept_use_persuasion: bool = False
    #: After agency_accept, seed Tinder queue from agency deck (agency_deck mode).
    agency_feed_tinder: bool = False
    #: Before Tinder ``right_draft``, run account-brief triangulation (KB + mail + CRM + proof).
    tinder_triangulate_before_draft: bool = True
    #: Include Argus dossier in Tinder triangulation (slower; off by default).
    tinder_triangulate_deep: bool = False
    #: Refuse ``brain tinder draft`` until ``brain tinder card`` has built company intel.
    tinder_require_company_card_before_draft: bool = True
    #: On ``brain tinder status``, build company intel when missing (may take ~60s).
    tinder_auto_fetch_company_card_on_status: bool = True
    #: Before showing a card, classify domain as industrial forming machinery buyer (website + LLM).
    tinder_icp_gate_enabled: bool = True
    #: Auto ``left`` when classifier says non-buyer at or above this confidence.
    tinder_icp_auto_skip_min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    #: Scrape homepage/services when no cached site profile exists.
    tinder_icp_scrape_on_card: bool = True
    tinder_icp_scrape_timeout_s: float = Field(default=25.0, ge=5.0, le=120.0)
    #: Max auto-skips per status/card fetch (prevents infinite loop).
    tinder_icp_max_auto_skips_per_status: int = Field(default=25, ge=1, le=100)
    #: Drop May-6-style batch spray subjects when building mailbox_oldest queue.
    tinder_exclude_batch_spray_subjects: bool = True

    #: Acme formal quote prep MCP — Manus Machine Quote Generator URL.
    demo_quote_generator_url: str = Field(
        default="https://quote-demo.example.com/",
        validation_alias=AliasChoices(
            "APP__DEMO_QUOTE_MANUS_URL",
            "DEMO_QUOTE_MANUS_URL",
        ),
    )
    #: Optional access code for the Manus quote generator (empty = not shown in workflow).
    demo_quote_access_code: str = Field(
        default="",
        validation_alias=AliasChoices(
            "APP__DEMO_QUOTE_MANUS_ACCESS_CODE",
            "DEMO_QUOTE_MANUS_ACCESS_CODE",
        ),
    )

    #: Multi-page industrial former website research (Firecrawl map + section scrape + LLM profile).
    thermoformer_research_max_pages: int = Field(default=12, ge=3, le=40)
    thermoformer_research_delay_s: float = Field(default=2.0, ge=0.0, le=60.0)
    thermoformer_research_auto_ingest: bool = True
    thermoformer_research_llm_provider: Literal["openai", "anthropic", "ollama"] = "anthropic"
    #: Write gauge_tier / ICP fields onto Neo4j :Company after classify, Tinder ICP, site ingest.
    neo4j_icp_gauge_sync_enabled: bool = True
    #: Morning digest: cards included in operator webhook payload.
    agency_digest_webhook_top_n: int = Field(default=3, ge=1, le=10)
    agency_digest_followup_idle_days: int = Field(default=15, ge=1, le=60)
    #: When False, heartbeat/CLI digest still persists deck but skips webhook POST.
    agency_digest_webhook_enabled: bool = True

    #: Gmail-backed cadence audit before Tinder/revenue catch-up (flag-only; no send block).
    outbound_touch_audit_enabled: bool = True
    #: Min days after our last outbound without their reply before another touch is advised.
    outbound_min_days_without_reply: int = Field(default=15, ge=1, le=120)
    #: Max messages per ``audit_outbound_touch`` Gmail search.
    outbound_touch_audit_sent_max: int = Field(default=25, ge=5, le=100)
    #: NA human-assisted call queue: max dial attempts per account before skip.
    na_call_max_attempts: int = Field(default=2, ge=1, le=10)
    #: Days between no_answer retries (domain cooldown in pause registry).
    na_call_retry_days: int = Field(default=3, ge=1, le=30)
    #: Cooldown days applied on no_answer (``cooldown_until`` on domain pause).
    na_call_cooldown_days: int = Field(default=14, ge=1, le=90)
    #: Account-journey mailbox scan cap (used by brief deep-mail mode and email journey tools).
    account_journey_max_messages: int = Field(default=500, ge=20, le=5000)
    #: Max unique Gmail threads loaded per account-journey run.
    account_journey_max_threads: int = Field(default=120, ge=5, le=1000)
    #: Per-message attachment extraction cap for account journey (chars).
    account_journey_max_attachment_chars: int = Field(default=50_000, ge=1000, le=500_000)
    #: Gmail list page size for account journey pagination.
    account_journey_page_size: int = Field(default=250, ge=20, le=500)
    #: Block proactive draft/send when dreamer cooldown or EXCLUDED segment applies.
    outbound_proactive_hard_block: bool = True
    #: Dreamer cooldown length after classification (days).
    outbound_dreamer_cooldown_days: int = Field(default=120, ge=30, le=365)
    #: Outbound threads with zero inbound before DREAMER segment.
    outbound_dreamer_min_outbound: int = Field(default=5, ge=3, le=20)
    #: Max proactive touches per domain per 90 days (uses stored outbound_touch_count).
    outbound_max_touches_per_domain_90d: int = Field(default=3, ge=1, le=20)
    #: Write Brain OS segment into Twenty Person jobTitle prefix (until custom field wired).
    twenty_sync_segment_in_job_title: bool = True

    #: When True, heartbeat pipeline ``query`` jobs skip until operator release session is active.
    operator_release_required: bool = False
    #: Default TTL for ``POST /api/operator/release`` (Release autonomous mode).
    operator_release_ttl_hours: float = Field(default=4.0, ge=0.25, le=168.0)

    #: Hermes-style loop: promote compiler candidates into ``ProceduralMemory`` when thresholds match.
    learned_procedure_promotion_enabled: bool = False
    #: Apply ``data/brain/learning/routing_nudges_active.json`` in :class:`DeterministicRouter`.
    learned_routing_enabled: bool = False
    #: Minimum identical agent-path repetitions before a candidate procedure is emitted.
    learning_compiler_min_evidence: int = Field(default=5, ge=1, le=10_000)
    #: Dream stage 4 writes candidates only (no direct ``learn_procedure``) when True.
    dream_stage4_candidates_only: bool = True
    #: Stage 12d: require this many distinct dream-cycle sightings before promotion.
    learning_promote_min_dream_cycles: int = Field(default=2, ge=0, le=100)
    #: Log a warning and audit line when a deprecated procedure routes.
    procedure_deprecated_warn_enabled: bool = True
    #: Auto-retire deprecated procedures after this many routed uses (0 = never).
    procedure_auto_retire_deprecated_uses: int = Field(default=5, ge=0, le=10_000)
    #: Memory-health observer (Metis companion — phase 3 Lettie lessons).
    memory_health_corrections_24h_warn: int = Field(default=3, ge=0, le=10_000)
    memory_health_low_metis_sessions_24h_warn: int = Field(default=5, ge=0, le=10_000)
    memory_health_low_metis_floor: float = Field(default=40.0, ge=0.0, le=100.0)
    memory_health_stale_procedure_ratio_warn: float = Field(default=0.25, ge=0.0, le=1.0)
    memory_health_candidate_backlog_warn: int = Field(default=20, ge=0, le=10_000)
    memory_health_deprecated_use_24h_warn: int = Field(default=10, ge=0, le=10_000)
    memory_health_metis_rolling_warn: float = Field(default=60.0, ge=0.0, le=100.0)
    #: Defer scheduled dream when memory-health score is below this and issues are non-empty.
    dream_cadence_gate_enabled: bool = True
    dream_cadence_min_health_score: int = Field(default=70, ge=0, le=100)
    dream_cadence_defer_hours: float = Field(default=6.0, ge=0.5, le=168.0)
    #: When True, defer on low memory-health score even if ``issues`` is empty (Letta cadence).
    dream_cadence_defer_on_score_only: bool = False
    #: When True, also defer on corrections spike or low Metis rolling avg (multi-signal gate).
    dream_cadence_multi_signal: bool = False
    #: Operator praise ("well done", "bravo") reinforces procedures and method traces.
    praise_learning_enabled: bool = True
    #: Min cluster size for praise-tagged Graphe rows in the learning compiler.
    praise_compiler_min_evidence: int = Field(default=1, ge=1, le=100)
    #: Dream stage 12f: compile skill candidates from praise_events.jsonl.
    skill_candidate_compile_enabled: bool = False
    #: Min praise events before writing a playbook under data/brain/skills/playbooks/.
    skill_candidate_min_evidence: int = Field(default=2, ge=1, le=100)
    #: Append gaps-only triangulation table after shape (SOUL ops).
    pipeline_triangulation_gaps_enabled: bool = True
    #: When enabled with gaps block, include hex legs (Atlas/graph/proof) not only triangle.
    pipeline_triangulation_gaps_hex: bool = False
    #: Run account brief + gap check before MCP/API draft_email and persuasion (hex optional).
    triangulation_enforce_before_outbound_draft: bool = True
    #: Hard-block outbound drafts when triangle (or hex) legs are missing.
    triangulation_block_on_gaps: bool = True
    #: Require hex legs (Atlas, graph, proof) for outbound draft / persuasion gates.
    triangulation_hex_for_outbound_draft: bool = True
    #: Reuse operator_context account_brief snapshot within this window (0 = always rebuild).
    triangulation_reuse_recent_brief_hours: float = Field(default=24.0, ge=0.0, le=168.0)
    #: ``build_account_brief`` timeout for outbound triangulation gate (seconds).
    triangulation_outbound_brief_timeout_s: float = Field(default=30.0, ge=5.0, le=120.0)
    #: Stricter Aletheia provenance for outbound drafts (numeric/date claims).
    aletheia_outbound_mode_enabled: bool = True
    #: Block HTTP/MCP email draft when claim_map has source_type=none (after Calliope).
    outbound_claim_map_enforce_on_draft: bool = False
    #: Firecrawl scrape disk cache + retry on timeout (classify / site research).
    firecrawl_scrape_cache_enabled: bool = True
    firecrawl_scrape_cache_ttl_hours: float = Field(default=168.0, ge=1.0, le=720.0)
    firecrawl_scrape_max_retries: int = Field(default=2, ge=0, le=5)
    #: Include prior conference top-3 in synthesis prompt (week-over-week diff).
    pantheon_conference_include_prior_improvements: bool = True
    #: Optional Slack incoming webhook for Sunday conference digest (operator-only).
    pantheon_conference_slack_webhook_url: str = ""
    #: ``build_account_brief`` timeout for pipeline gaps appendix (seconds).
    pipeline_triangulation_gaps_timeout_s: float = Field(default=25.0, ge=5.0, le=120.0)
    #: Skip appendix when agents already called get_account_brief / build_account_brief.
    pipeline_triangulation_gaps_skip_if_agent_ran_brief: bool = True
    #: Comma-separated channels allowed for gaps appendix (e.g. cli,api,cursor).
    pipeline_triangulation_gaps_channel_allowlist: str = "cli,api,cursor"
    #: Math Mode rollout phase 2: append scores to math_shadow.jsonl on brief (no UI change).
    math_mode_shadow_enabled: bool = False
    #: Math Mode rollout phase 3: attach math_advisory on brief JSON + inbox hints.
    math_mode_advisory_enabled: bool = False
    #: Permission to speculate (R&D mode) master switch — per-turn via metadata speculation_mode.
    speculation_mode_enabled: bool = False
    #: Phase 1 shadow: tag sentences in pipeline_trace + speculation_shadow.jsonl (no text change).
    speculation_mode_shadow_enabled: bool = True
    #: Cap promotions per ISO calendar week (0 = unlimited).
    learning_max_promotions_per_week: int = Field(default=10, ge=0, le=500)
    #: Max routing nudge ``weight_delta`` absolute value when merging pattern scores.
    learning_routing_max_weight_delta: float = Field(default=2.0, ge=0.0, le=20.0)
    #: Append-only audit of promotions (under ``BRAIN_DATA_DIR`` / ``./data``).
    learning_promotions_audit_filename: str = "brain/learning/promotions.jsonl"

    #: Dream stage 3.6: reconcile yesterday's hot-lead predictions vs today's outcomes.
    prediction_reconciliation_enabled: bool = True
    #: Max reconciliation rows sent to Sophia reflection LLM per dream cycle.
    prediction_reconciliation_max_accounts: int = Field(default=50, ge=1, le=500)
    #: Also log/reconcile revenue-desk must-act items (reply, followup, rewrite).
    prediction_record_must_act: bool = True
    #: Log open pipeline deal win-progress forecasts (Tyche-style, deterministic).
    prediction_record_tyche_deals: bool = True
    #: On hot-leads board CSV write, record HOT_NOW predictions (deduped per account).
    prediction_record_intraday_on_board_write: bool = False
    #: Heartbeat ``prediction_reconcile`` action skips Sophia LLM (faster mid-day pass).
    prediction_reconcile_heartbeat_skip_llm: bool = True

    #: GEPA-style: persist each tool outcome to SQLite (under data dir) for offline compile.
    gepa_persist_tool_invocations: bool = False
    #: Relative path under ``BRAIN_DATA_DIR`` for tool invocation SQLite.
    gepa_tool_invocations_db_path: str = "brain/tool_invocations.sqlite"
    gepa_tool_invocations_retention_days: int = Field(default=30, ge=1, le=3650)
    gepa_tool_invocations_max_rows: int = Field(default=500_000, ge=1_000, le=50_000_000)
    #: Persist pipeline run records (evidence bundles) to SQLite under ``BRAIN_DATA_DIR``.
    run_record_enabled: bool = False
    #: Relative path under ``BRAIN_DATA_DIR`` for run-record SQLite.
    run_record_db_path: str = "brain/run_records.sqlite"
    run_record_retention_days: int = Field(default=30, ge=1, le=3650)
    run_record_max_rows: int = Field(default=100_000, ge=1_000, le=50_000_000)
    #: Hash/redact sender and contact fields in stored run records.
    run_record_redact_pii: bool = True
    #: Persist operator context runs (brief / Tinder / quote prep) for precedent search.
    operator_context_enabled: bool = True
    operator_context_db_path: str = "brain/operator_context.sqlite"
    operator_context_retention_days: int = Field(default=180, ge=1, le=3650)
    operator_context_max_rows: int = Field(default=50_000, ge=100, le=50_000_000)
    operator_context_precedent_limit: int = Field(default=5, ge=1, le=20)
    #: Mirror operator context + pipeline run records into Neo4j (context graph P1).
    operator_context_neo4j_sync: bool = False
    #: 1-hop Neo4j expansion into account brief / company graph retrieval (P2).
    operator_context_graph_expand_enabled: bool = True
    #: Voyage embeddings on Company nodes for similar-account discovery (P3).
    company_similarity_enabled: bool = True
    company_similarity_limit: int = Field(default=5, ge=1, le=20)
    company_similarity_min_score: float = Field(default=0.72, ge=0.0, le=1.0)
    company_similarity_candidate_pool: int = Field(default=500, ge=10, le=5000)
    company_similarity_auto_embed_on_brief: bool = True
    #: Inject ``strategy_overlays_active.json`` fragments into agent system prompts (see ``BRAIN_DISABLE_GEPA_OVERLAYS``).
    gepa_strategy_overlay_runtime: bool = False
    #: Dream: compile ``strategy_overlays_candidate.json`` from persisted tool stats + LLM.
    gepa_dream_compile_enabled: bool = False
    #: Dream promotion: copy strategy overlay candidate to active (requires compile or manual candidate).
    gepa_promote_strategy_overlays: bool = False
    #: Before overlay promotion: denylist + optional LLM approve/reject gate.
    gepa_nemesis_gate_enabled: bool = False
    #: Minimum invocations per (agent, tool) to consider in Dream overlay compile.
    gepa_compile_min_invocations_per_pair: int = Field(default=15, ge=1, le=100_000)
    #: Flag pairs with success rate at or below this (0–1) for overlay hints.
    gepa_compile_max_success_rate: float = Field(default=0.85, ge=0.0, le=1.0)
    gepa_compile_window_days: int = Field(default=7, ge=1, le=365)
    #: Cap strategy-overlay promotions per ISO week (0 = unlimited).
    gepa_max_strategy_overlay_promotions_per_week: int = Field(default=5, ge=0, le=500)

    @field_validator("pipeline_stage_budgets", mode="before")
    @classmethod
    def _coerce_pipeline_stage_budgets(cls, v: Any) -> Any:
        if v is None or v == "":
            return v
        if isinstance(v, str):
            data = json.loads(v)
            if not isinstance(data, dict):
                raise ValueError("pipeline_stage_budgets must be a JSON object")
            return {str(k): float(data[k]) for k in data}
        return v


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",  # so APP__FAITHFULNESS_HARD_THRESHOLD loads app.faithfulness_hard_threshold
        populate_by_name=True,
    )

    #: When ``True``, outbound Gmail policy treats guardrail checker *exceptions* as hard blocks even
    #: for one-off sends. Otherwise, campaign sends (non-empty ``campaign_id``) still fail closed on exceptions.
    brain_outbound_fail_closed: bool = Field(
        default=False,
        validation_alias=AliasChoices("BRAIN_OUTBOUND_FAIL_CLOSED"),
    )

    #: Which specialist agents Pantheon loads. ``off`` = full pantheon; ``standard`` = sales-heavy subset
    #: (default, matches historical behavior); ``strict`` = minimal 10-agent revenue stack (optional).
    brain_revenue_mode: Literal["off", "standard", "strict"] = Field(
        default="standard",
        validation_alias=AliasChoices("BRAIN_REVENUE_MODE"),
    )
    brain_route_provider: str = Field(
        default="",
        validation_alias=AliasChoices("BRAIN_ROUTE_PROVIDER"),
    )
    brain_synthesis_provider: str = Field(
        default="",
        validation_alias=AliasChoices("BRAIN_SYNTHESIS_PROVIDER"),
    )
    brain_digestive_provider: str = Field(
        default="",
        validation_alias=AliasChoices("BRAIN_DIGESTIVE_PROVIDER"),
    )
    brain_embedding_provider: str = Field(
        default="",
        validation_alias=AliasChoices("BRAIN_EMBEDDING_PROVIDER"),
    )

    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    qdrant: QdrantConfig = QdrantConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    database: DatabaseConfig = DatabaseConfig()
    memory: MemoryConfig = MemoryConfig()
    pdfco: PdfCoConfig = PdfCoConfig()
    document_ai: DocumentAIConfig = DocumentAIConfig()
    redis: RedisConfig = RedisConfig()
    google: GoogleConfig = GoogleConfig()
    external_apis: ExternalAPIsConfig = ExternalAPIsConfig()
    apollo: ApolloConfig = ApolloConfig()
    vapi: VapiConfig = VapiConfig()
    demo_campaign_campaign: DemoCampaignCampaignConfig = DemoCampaignCampaignConfig()
    neverbounce: NeverBounceConfig = NeverBounceConfig()
    wolfram: WolframConfig = WolframConfig()
    search: SearchConfig = SearchConfig()
    jina: JinaConfig = JinaConfig()
    langfuse: LangfuseConfig = LangfuseConfig()
    helicone: HeliconeConfig = HeliconeConfig()
    #: Phase-2: centralised LLM / embedding endpoint URLs (defaults preserve behaviour).
    llm_endpoints: LLMEndpointsConfig = LLMEndpointsConfig()
    firecrawl: FirecrawlConfig = FirecrawlConfig()
    unstructured: UnstructuredConfig = UnstructuredConfig()
    sentry: SentryConfig = SentryConfig()
    honcho: HonchoConfig = HonchoConfig()
    slack: SlackConfig = SlackConfig()
    app: AppConfig = AppConfig()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Phase 6/8: ops snapshots in config_ops_snapshots — full re-export for backward compatibility.
from brain_os.config_ops_snapshots import (
    get_app_feature_flags_public_snapshot,
    get_app_runtime_imports_public_snapshot,
    get_crm_database_public_snapshot,
    get_email_ops_public_snapshot,
    get_email_training_public_snapshot,
    get_event_driven_ingestion_public_snapshot,
    get_governance_brain_public_snapshot,
    get_graph_store_public_snapshot,
    get_http_api_public_snapshot,
    get_ingestion_brain_public_snapshot,
    get_knowledge_store_public_snapshot,
    get_llm_stack_public_snapshot,
    get_mem0_public_snapshot,
    get_neo4j_connection_flags_public_snapshot,
    get_optional_tooling_imports_public_snapshot,
    get_pipeline_observability_public_snapshot,
    get_pipeline_runtime_public_snapshot,
    get_process_context_public_snapshot,
    get_quality_stack_imports_public_snapshot,
    get_redis_cache_public_snapshot,
    get_repo_brain_aux_public_snapshot,
    get_repo_framework_public_snapshot,
    get_repo_sources_public_snapshot,
    get_repo_truth_hints_public_snapshot,
    get_repo_vcs_public_snapshot,
    get_runtime_build_public_snapshot,
    get_runtime_capabilities_public_snapshot,
    get_sales_coaching_brain_public_snapshot,
    get_secondary_google_oauth_public_snapshot,
    get_secondary_mail_public_snapshot,
    get_skills_registry_public_snapshot,
    get_storage_clients_public_snapshot,
    get_tracing_public_snapshot,
    get_vendor_apis_public_snapshot,
    get_workspace_content_public_snapshot,
    infer_http_health_status,
)

__all__ = [
    "EmailMode",
    "Settings",
    "get_app_feature_flags_public_snapshot",
    "get_app_runtime_imports_public_snapshot",
    "get_crm_database_public_snapshot",
    "get_email_ops_public_snapshot",
    "get_email_training_public_snapshot",
    "get_event_driven_ingestion_public_snapshot",
    "get_governance_brain_public_snapshot",
    "get_graph_store_public_snapshot",
    "get_http_api_public_snapshot",
    "get_ingestion_brain_public_snapshot",
    "get_knowledge_store_public_snapshot",
    "get_llm_stack_public_snapshot",
    "get_mem0_public_snapshot",
    "get_neo4j_connection_flags_public_snapshot",
    "get_optional_tooling_imports_public_snapshot",
    "get_pipeline_observability_public_snapshot",
    "get_pipeline_runtime_public_snapshot",
    "get_process_context_public_snapshot",
    "get_quality_stack_imports_public_snapshot",
    "get_redis_cache_public_snapshot",
    "get_repo_brain_aux_public_snapshot",
    "get_repo_framework_public_snapshot",
    "get_repo_sources_public_snapshot",
    "get_repo_truth_hints_public_snapshot",
    "get_repo_vcs_public_snapshot",
    "get_runtime_build_public_snapshot",
    "get_runtime_capabilities_public_snapshot",
    "get_sales_coaching_brain_public_snapshot",
    "get_secondary_google_oauth_public_snapshot",
    "get_secondary_mail_public_snapshot",
    "get_settings",
    "get_skills_registry_public_snapshot",
    "get_storage_clients_public_snapshot",
    "get_tracing_public_snapshot",
    "get_vendor_apis_public_snapshot",
    "get_workspace_content_public_snapshot",
    "infer_http_health_status",
]
