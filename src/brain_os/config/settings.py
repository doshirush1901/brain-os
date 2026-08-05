"""Root Settings + get_settings + secrets accessor."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from brain_os.config.app import AppConfig
from brain_os.config.google import GoogleConfig
from brain_os.config.integrations import (
    ApolloConfig,
    BoardConfig,
    DocumentAIConfig,
    ExternalAPIsConfig,
    FirecrawlConfig,
    HonchoConfig,
    JinaConfig,
    LangfuseConfig,
    DemoCampaignCampaignConfig,
    PdfCoConfig,
    PeopleDataLabsConfig,
    SearchConfig,
    SentryConfig,
    SlackConfig,
    UnstructuredConfig,
    VapiConfig,
    WhatsAppConfig,
    WolframConfig,
)
from brain_os.config.llm import EmbeddingConfig, HeliconeConfig, LLMConfig, LLMEndpointsConfig
from brain_os.config.storage import (
    DatabaseConfig,
    MemoryConfig,
    Neo4jConfig,
    QdrantConfig,
    RedisConfig,
)


class Settings(BaseSettings):
    """Root pydantic-settings model — all Brain OS configuration loaded from ``.env`` / environment."""

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
    people_data_labs: PeopleDataLabsConfig = PeopleDataLabsConfig()
    vapi: VapiConfig = VapiConfig()
    whatsapp: WhatsAppConfig = WhatsAppConfig()
    demo_campaign_campaign: DemoCampaignCampaignConfig = DemoCampaignCampaignConfig()
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
    board: BoardConfig = BoardConfig()
    app: AppConfig = AppConfig()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance."""
    return Settings()


class _SecretsAccessor:
    """Single front door for runtime secret reads (Revenue A-grade Phase 7).

    Today this is a thin wrapper over :data:`os.environ` (after pydantic-settings
    has loaded ``.env``), so behaviour is unchanged. A future vault backend
    (1Password / AWS Secrets Manager) swaps the implementation of this one class
    — callers never read ``os.environ`` for secrets directly.

    Usage::

        from brain_os.config import secrets
        key = secrets.get("NEWSDATA_API_KEY")

    Enforced by ``scripts/secrets_audit.py`` (pre-commit): no module under
    ``src/brain_os`` may read secret-named environment variables except this file.
    """

    @staticmethod
    def get(name: str, default: str = "") -> str:
        """Return the secret value for ``name`` (empty string when unset)."""
        import os

        return os.environ.get(name, default)

    @staticmethod
    def require(name: str) -> str:
        """Return the secret for ``name`` or raise ``KeyError`` when missing/blank."""
        value = _SecretsAccessor.get(name)
        if not value.strip():
            raise KeyError(f"Required secret {name!r} is not configured")
        return value


secrets = _SecretsAccessor()
