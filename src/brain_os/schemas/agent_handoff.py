"""Structured brief for ``ask_agent`` delegation (optional ``handoff_json``)."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field, field_validator

_MAX_BULLETS = 20
_MAX_CONSTRAINTS = 20
_MAX_ACCEPTANCE_CRITERIA = 10
_MAX_ITEM_LEN = 500
_MAX_GOAL_LEN = 2000
_MAX_DOMAIN_LEN = 500
_MAX_SOURCE_LEN = 64


class AgentHandoffBrief(BaseModel):
    """Machine-readable handoff prepended to delegated ``question`` strings.

    Governance: whenever Athena or a specialist forwards work with ``delegate_to_agent`` /
    ``ask_agent``, populate ``goal``, 2–5 ``bullets`` (constraints + deliverables),
    optional ``constraints``, and ``domain`` so the callee avoids rework. Keep each
    bullet under ~320 characters of operational prose.
    """

    goal: str = Field(..., min_length=1, max_length=_MAX_GOAL_LEN)
    bullets: list[str] = Field(default_factory=list, max_length=_MAX_BULLETS)
    acceptance_criteria: list[str] = Field(
        default_factory=list, max_length=_MAX_ACCEPTANCE_CRITERIA
    )
    constraints: list[str] = Field(default_factory=list, max_length=_MAX_CONSTRAINTS)
    domain: str | None = Field(default=None, max_length=_MAX_DOMAIN_LEN)
    source_agent: str | None = Field(default=None, max_length=_MAX_SOURCE_LEN)

    @field_validator("bullets", "constraints", "acceptance_criteria", mode="before")
    @classmethod
    def _coerce_str_lists(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise TypeError("expected a list of strings")
        out: list[str] = []
        for item in v:
            s = str(item).strip()
            if not s:
                continue
            if len(s) > _MAX_ITEM_LEN:
                s = s[:_MAX_ITEM_LEN]
            out.append(s)
        return out

    def domain_fingerprint(self) -> str | None:
        """Short non-reversible fingerprint for telemetry (no raw domain in logs if omitted)."""
        if not self.domain or not self.domain.strip():
            return None
        h = hashlib.sha256(self.domain.strip().lower().encode()).hexdigest()
        return h[:12]

    @staticmethod
    def compose_query(brief: AgentHandoffBrief, question: str) -> str:
        """Prepend delimiter block so callees can parse without API changes."""
        inner = brief.model_dump_json(exclude_none=True)
        return f"<<<HANDOFF_BRIEF>>>\n{inner}\n<<<END_HANDOFF>>>\n\n{question.lstrip()}"
