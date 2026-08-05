"""Composed AppConfig (mixins + validators)."""

from __future__ import annotations

import json
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from brain_os.config.app_core import AppCoreMixin
from brain_os.config.common import _COMMON
from brain_os.config.gtm import AppGtmMixin
from brain_os.config.learning import AppLearningMixin
from brain_os.config.pipeline import AppPipelineMixin
from brain_os.config.security import AppSecurityMixin
from brain_os.config.stores_pg import AppStoresPgMixin


class AppConfig(
    AppStoresPgMixin,
    AppLearningMixin,
    AppGtmMixin,
    AppSecurityMixin,
    AppPipelineMixin,
    AppCoreMixin,
    BaseSettings,
):
    """Application feature flags and runtime knobs (``APP__*`` env prefix)."""

    model_config = SettingsConfigDict(**_COMMON, populate_by_name=True)

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
