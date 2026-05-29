"""Structured company context-graph slice for MCP / CLI (P5)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from brain_os.schemas.operator_context import ContextPrecedent
from brain_os.schemas.similar_company import SimilarCompany


class CompanyContextGraphBundle(BaseModel):
    """Read-only context graph around one :Company node."""

    company_name: str
    company_normalized: str = ""
    company_exists: bool = False
    region: str | None = None
    industry: str | None = None
    website: str | None = None
    operator_runs: list[dict[str, object]] = Field(default_factory=list)
    pipeline_runs: list[dict[str, object]] = Field(default_factory=list)
    contacts: list[dict[str, object]] = Field(default_factory=list)
    quotes: list[dict[str, object]] = Field(default_factory=list)
    quote_documents: list[dict[str, object]] = Field(
        default_factory=list,
        description="Quote/proposal sources from Chunk-DESCRIBES (KB ingest; not QUOTED_TO).",
    )
    related_company_names: list[str] = Field(
        default_factory=list,
        description="Alias rollup — other :Company nodes matching the same search token.",
    )
    graph_context_lines: list[str] = Field(default_factory=list)
    similar_companies: list[SimilarCompany] = Field(default_factory=list)
    sqlite_precedents: list[ContextPrecedent] = Field(
        default_factory=list,
        description="Recent operator runs from SQLite (canonical precedents).",
    )
