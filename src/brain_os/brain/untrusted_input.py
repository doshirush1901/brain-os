"""Normalize untrusted user/query text before routing and agent execution.

Removes characters that break parsers, caps extreme length to bound prompt-injection
surface and token use. Does not remove or rewrite semantic content.

* :func:`sanitize_pipeline_query` — main request path (after REMEMBER).
* :func:`sanitize_for_llm_embedding` — secondary LLM prompts (sales, pricing,
  feedback disambiguation, realtime observer, knowledge discovery).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Line/paragraph separators that sometimes appear in pasted content and confuse tools.
_REPLACE_SEPARATORS = str.maketrans(
    {
        "\u2028": "\n",
        "\u2029": "\n",
    }
)

_LLM_TRUNC_SUFFIX = "\n\n[Truncated for LLM context bound.]"


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
