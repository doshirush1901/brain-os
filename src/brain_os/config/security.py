"""Faithfulness / guardrails / PII / outbound safety flags."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator


class AppSecurityMixin:
    """Mixin slice of :class:`~brain_os.config.app.AppConfig` (MOVE only)."""

    faithfulness_heuristic_only: bool = False

    faithfulness_threshold: float = 0.6

    faithfulness_hard_threshold: float = 0.3

    faithfulness_mode: Literal["best_effort", "strict"] = "best_effort"

    confidence_floor: float = 0.3

    guardrails_fail_closed: bool = True

    mnemon_semantic_check: bool = False

    legacy_quarantine_strict: bool = False

    redact_pii_at_ingest: bool = False

    uncensored_local_llm_mode: bool = False

    retriever_bundle_pii_mode: str = "off"

    aletheia_outbound_mode_enabled: bool = True

    proof_before_prose_enabled: bool = True

    employment_before_mail_enabled: bool = True

    shopfloor_before_customer_enabled: bool = True

    run_record_redact_pii: bool = True
