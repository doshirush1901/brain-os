"""Shared pydantic-settings helpers for Brain OS config domains."""

from __future__ import annotations

from enum import Enum


class EmailMode(str, Enum):
    """Operating mode for the EmailProcessor."""

    TRAINING = "TRAINING"
    OPERATIONAL = "OPERATIONAL"


_COMMON = dict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
