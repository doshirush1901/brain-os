"""Google OAuth / Gmail / Calendar / Maps settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from brain_os.config.common import _COMMON, EmailMode


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
    #: Delay before Places Text Search pagination requests (Google recommends ~2s).
    maps_pagination_sleep_seconds: float = Field(default=2.0, ge=0.0, le=10.0)
