"""Lightweight guards before writing to long-term conversational memory."""

from __future__ import annotations

import re
from functools import lru_cache

_EPHEMERAL_PATTERNS = (
    r"\bignore\s+(the\s+)?(above|this)\b",
    r"\bscratch\s+(that|this)\b",
    r"\bjust\s+a\s+test\b",
    r"\bstet\b",
    r"\bdo\s+not\s+(store|remember)\b",
    r"\bdelete\s+(this\s+)?memory\b",
)


@lru_cache(maxsize=1)
def _ephemeral_regexes() -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in _EPHEMERAL_PATTERNS)


def should_skip_mem_store(content: str) -> tuple[bool, str]:
    """Return (skip, reason) when content should not be persisted to Mem0."""
    text = (content or "").strip()
    if not text:
        return True, "empty"
    if len(text) < 18:
        return True, "too_short"
    for rx in _ephemeral_regexes():
        if rx.search(text):
            return True, "ephemeral_marker"
    return False, ""
