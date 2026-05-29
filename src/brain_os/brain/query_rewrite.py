"""Deterministic query normalization for retrieval (no LLM in the default path)."""

from __future__ import annotations

import re
from typing import Any

from brain_os.brain.machine_identifiers import normalize_machine_tokens

_WS = re.compile(r"\s+")


def normalize_for_retrieval(query: str) -> str:
    """Collapse whitespace and normalize common machine tokens."""
    if not isinstance(query, str):
        query = str(query or "")
    q = _WS.sub(" ", query.strip())
    if not q:
        return q
    return normalize_machine_tokens(q)


def pick_first_distinct_subquery(original: str, subqueries: list[str]) -> str | None:
    """Return first sub-query that differs from *original* (case-folded strip)."""
    onorm = original.strip().lower()
    for s in subqueries:
        cand = (s or "").strip()
        if cand and cand.lower() != onorm:
            return cand
    return None


def summarize_for_trace(query: str, *, max_len: int = 200) -> dict[str, Any]:
    nq = normalize_for_retrieval(query)
    return {
        "query_len": len(query),
        "normalized_len": len(nq),
        "normalized_preview": nq[:max_len],
    }
