"""Outbound voice LLM schemas (Insight-tight judge rubric).

Extracted from ``llm_outputs`` (L0.1 file-size peel). Re-exported from
``brain_os.schemas.llm_outputs`` for API stability.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OutboundVoiceRubric(BaseModel):
    """LLM judge scores for Insight-tight outbound (Phase 2 / 2.5).

    Each dimension is 0–1. ``send_ready`` is True only when all dimensions
    meet the configured threshold and the draft is not CapEx/AI-smooth.
    """

    crisp: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: float = Field(default=0.0, ge=0.0, le=1.0)
    human_not_ai: float = Field(default=0.0, ge=0.0, le=1.0)
    curiosity: float = Field(default=0.0, ge=0.0, le=1.0)
    one_ask: float = Field(default=0.0, ge=0.0, le=1.0)
    capex_clean: float = Field(default=0.0, ge=0.0, le=1.0)
    overall: float = Field(default=0.0, ge=0.0, le=1.0)
    send_ready: bool = False
    issues: list[str] = Field(default_factory=list, max_length=12)
    rewrite_hint: str = Field(default="", max_length=500)
