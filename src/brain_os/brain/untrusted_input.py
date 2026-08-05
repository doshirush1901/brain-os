"""Normalize and fence untrusted text before LLM embedding.

Removes characters that break parsers, caps extreme length to bound prompt-injection
surface and token use. Does not remove or rewrite semantic content.

* :func:`sanitize_pipeline_query` — main request path (after REMEMBER).
* :func:`sanitize_for_llm_embedding` — secondary LLM prompts (sales, pricing,
  feedback disambiguation, realtime observer, knowledge discovery).
* :func:`fence_untrusted` — structural delimiters for retrieved/scraped/email
  content and ReAct tool observations (S-H1 / S-H2).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Line/paragraph separators that sometimes appear in pasted content and confuse tools.
_REPLACE_SEPARATORS = str.maketrans(
    {
        "\u2028": "\n",
        "\u2029": "\n",
    }
)

_LLM_TRUNC_SUFFIX = "\n\n[Truncated for LLM context bound.]"

# Fence markers — must stay in sync with prompts/react_system.txt SECURITY section.
_FENCE_OPEN_PREFIX = "<<<UNTRUSTED DATA source="
_FENCE_OPEN_SUFFIX = ">>>"
_FENCE_CLOSE = "<<<END UNTRUSTED DATA>>>"

# Tokens neutralized inside content to prevent fence-escape / marker spoofing.
_DELIMITER_NEUTRALIZE: tuple[tuple[str, str], ...] = (
    ("<<<UNTRUSTED DATA", "‹‹‹UNTRUSTED DATA"),
    ("<<<END UNTRUSTED DATA>>>", "‹‹‹END UNTRUSTED DATA›››"),
    ("<<<USER INPUT>>>", "‹‹‹USER INPUT›››"),
    ("<<<END INPUT>>>", "‹‹‹END INPUT›››"),
)

_SOURCE_SAFE_RE = re.compile(r"[^a-zA-Z0-9_.:\-]+")
_DEFAULT_FENCE_MAX_CHARS = 8000


def _normalize_untrusted_chars(text: str) -> str:
    return text.replace("\x00", "").translate(_REPLACE_SEPARATORS)


def sanitize_for_llm_embedding(text: str | None, *, max_chars: int) -> str:
    """Normalize and cap text embedded in LLM user messages (not the main pipeline query).

    Same NUL/separator handling as :func:`sanitize_pipeline_query`, with a compact
    truncation note suitable for secondary prompts (feedback, pricing, discovery).
    """
    if text is None or text == "":
        return ""
    t = _normalize_untrusted_chars(str(text))
    if max_chars > 0 and len(t) > max_chars:
        keep = max_chars - len(_LLM_TRUNC_SUFFIX)
        if keep < 1:
            t = t[:max_chars]
        else:
            t = t[:keep] + _LLM_TRUNC_SUFFIX
        logger.debug("LLM context snippet truncated to %d chars", max_chars)
    return t


def sanitize_pipeline_query(text: str, *, max_chars: int) -> str:
    """Prepare raw user text for the request pipeline (after coref, before routing).

    * Strips NUL bytes.
    * Normalizes Unicode line/paragraph separators to newlines.
    * Truncates to *max_chars* with a short system suffix when exceeded.
    """
    if not text:
        return ""
    t = _normalize_untrusted_chars(text)
    if max_chars > 0 and len(t) > max_chars:
        suffix = "\n\n[Truncated: input exceeded configured pipeline_query_max_chars.]"
        keep = max_chars - len(suffix)
        logger.info(
            "Pipeline query truncated from %d chars (keeping %d body + suffix)",
            len(t),
            max(0, keep),
        )
        if keep < 1:
            t = t[:max_chars]
        else:
            t = t[:keep] + suffix
    return t


def _sanitize_source_label(source: str) -> str:
    label = (source or "unknown").strip() or "unknown"
    label = _SOURCE_SAFE_RE.sub("_", label)[:80]
    return label or "unknown"


def _neutralize_delimiter_tokens(text: str) -> str:
    out = text
    for needle, replacement in _DELIMITER_NEUTRALIZE:
        if needle in out:
            out = out.replace(needle, replacement)
    return out


def is_fenced_untrusted(text: str | None) -> bool:
    """True when *text* is already a complete :func:`fence_untrusted` block."""
    if not text:
        return False
    stripped = str(text).strip()
    if not stripped.startswith(_FENCE_OPEN_PREFIX):
        return False
    if not stripped.endswith(_FENCE_CLOSE):
        return False
    # Exactly one close marker — nested/spoofed closes fail this check.
    return stripped.count(_FENCE_CLOSE) == 1


def fence_untrusted(
    text: str | None,
    source: str,
    *,
    max_chars: int = _DEFAULT_FENCE_MAX_CHARS,
) -> str:
    """Wrap untrusted content in unambiguous delimiters for LLM prompts.

    Applies :func:`sanitize_for_llm_embedding`, neutralizes any in-content
    fence/USER-INPUT delimiter tokens (fence-escape prevention), then wraps::

        <<<UNTRUSTED DATA source={source}>>>
        ...content...
        <<<END UNTRUSTED DATA>>>

    Idempotent: content that is already a complete fence is returned unchanged
    (after strip). Empty input still returns a fenced empty block so callers
    can always treat the return value as delimited data.
    """
    if is_fenced_untrusted(text):
        return str(text).strip()

    sanitized = sanitize_for_llm_embedding(text, max_chars=max_chars)
    neutralized = _neutralize_delimiter_tokens(sanitized)
    safe_source = _sanitize_source_label(source)
    return f"{_FENCE_OPEN_PREFIX}{safe_source}{_FENCE_OPEN_SUFFIX}\n{neutralized}\n{_FENCE_CLOSE}"
