"""LLM / embedding / Helicone / endpoint URL settings."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from brain_os.config.common import _COMMON


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
        """Resolve a logical model profile / tier to a concrete provider model name.

        Tiers (see ``docs/LLM_TIERS.md``): ``cheap`` → fast mini, ``standard`` →
        default workhorse, ``frontier`` → reasoning flagship.
        """
        if not profile:
            return self.openai_model
        normalized = profile.strip().lower().replace("-", "_")
        mapping = {
            "default": self.openai_model,
            "cheap": self.brain_model_fast,
            "standard": self.openai_model,
            "frontier": self.brain_model_reasoning,
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
            "cheap": fast_p,
            "standard": "",
            "frontier": reasoning_p,
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
    #: Pitch media bank Phase 2 — multimodal retrieval over photos/slide stills.
    multimodal_model: str = "voyage-multimodal-3.5"
    multimodal_output_dimension: int = Field(default=1024, ge=256, le=2048)


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
    voyage_multimodal_embeddings_url: str = "https://api.voyageai.com/v1/multimodalembeddings"


class HeliconeConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HELICONE_", **_COMMON)

    api_key: SecretStr = SecretStr("")
    # Optional Helicone dashboard dimensions (Helicone-Property-* on every proxied LLM request).
    property_environment: str = ""
    property_app: str = ""
