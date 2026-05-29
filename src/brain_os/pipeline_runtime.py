"""Shared runtime helpers extracted from pipeline.py.

Phase modules under ``pipeline_phases/`` MUST import from here, NEVER from
``pipeline.py`` — that path is circular once the orchestrator imports phases.
"""

from __future__ import annotations

import re
from typing import Any

# Quick pipeline: short queries that only ask for the sales pipeline summary.
_QUICK_PIPELINE_PATTERN = re.compile(
    r"^\s*"
    r"(show\s+(me\s+)?(the\s+)?|what'?s\s+(the\s+)?|what\s+is\s+(the\s+)?|"
    r"give\s+me\s+(the\s+)?|(get|fetch|generate)\s+(the\s+)?(sales\s+)?)?"
    r"(sales\s+)?pipeline(\s+summary|\s+status)?\s*[!?.]*\s*$",
    re.IGNORECASE,
)


def _is_quick_pipeline_query(text: str) -> bool:
    """True if the query is a short, simple request for the pipeline summary."""
    t = (text or "").strip()
    return len(t) <= 80 and bool(_QUICK_PIPELINE_PATTERN.match(t))


_AMBIGUITY_SENSITIVE_PATTERNS = (
    r"\b(status|update|progress|how much|cost|price|pricing)\b",
    r"\b(send|draft|reply)\b",
    r"\b(latest email|latest message|latest thread)\b",
)


def _is_ambiguity_sensitive_query(text: str) -> bool:
    """True when request likely needs scoped clarification before execution."""
    lowered = (text or "").lower()
    return any(re.search(pattern, lowered) for pattern in _AMBIGUITY_SENSITIVE_PATTERNS)


def _csv_tokens(value: str) -> list[str]:
    return [tok.strip().lower() for tok in (value or "").split(",") if tok.strip()]


_DEFAULT_CHEAP_EXIT_BYPASS_KEYWORDS = (
    "send,dispatch,draft,reply,follow-up,follow up,outreach,campaign,"
    "latest,recent,today,yesterday,just sent,unread,inbox,"
    "update,create,delete,close deal,mark as,"
    "confidential,internal,private,margin,pricing,quote,what did"
)


# ── Phase-2 hardening: dedup payload envelope ────────────────────────────────
#
# Old behaviour stored just the shaped response string in Redis / RAM dedup
# and returned ``agents_used = []`` on cache hit. That left analytics, debug
# logs, and observability blind to which agents really produced the cached
# answer. Phase-2 stores a tagged JSON envelope that also carries the
# original ``agents_used`` list and ``run_id``, while remaining backward
# compatible with legacy single-string cache entries (~5 min TTL during
# rollout). See docs/PHASE2_HARDENING_AUDIT.md §4.2.

_DEDUP_ENVELOPE_VERSION: str = "ira_dedup_v1"


def _encode_dedup_payload(shaped: str, agents_used: list[str], run_id: str) -> str:
    """Serialise a dedup cache entry to a JSON envelope.

    Falls back to the raw shaped string if JSON encoding fails for any
    reason — the cache must never silently break the user-facing response.
    """
    try:
        import json as _json

        return _json.dumps(
            {
                "v": _DEDUP_ENVELOPE_VERSION,
                "shaped": shaped,
                "agents_used": list(agents_used or []),
                "run_id": run_id or "",
            }
        )
    except Exception:  # pragma: no cover — defensive
        return shaped


def _decode_dedup_payload(raw: str | None) -> tuple[str, list[str], str] | None:
    """Decode a dedup cache entry back to ``(shaped, agents_used, run_id)``.

    Returns ``None`` when *raw* is empty. Tolerates legacy entries that are
    plain strings (TTL'd entries written before the phase-2 envelope) by
    returning them with an empty ``agents_used`` list and empty ``run_id``.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        # Defensive: Redis client should always return strings.
        try:
            raw = str(raw)
        except Exception:
            return None
    if not raw:
        # Treat empty string as no hit.
        return None
    try:
        import json as _json

        parsed = _json.loads(raw)
    except Exception:
        # Legacy plain-string entry from before the envelope existed.
        return raw, [], ""
    if not isinstance(parsed, dict) or parsed.get("v") != _DEDUP_ENVELOPE_VERSION:
        return raw, [], ""
    shaped = parsed.get("shaped")
    if not isinstance(shaped, str):
        return raw, [], ""
    agents = parsed.get("agents_used")
    if not isinstance(agents, list):
        agents = []
    rid = parsed.get("run_id") or ""
    return shaped, [str(a) for a in agents], str(rid)


def _should_bypass_cheap_exits(
    text: str,
    *,
    enabled: bool = True,
    bypass_keywords: str = _DEFAULT_CHEAP_EXIT_BYPASS_KEYWORDS,
    allowlist_keywords: str = "",
) -> bool:
    """Return True when dedup/fast shortcuts should be skipped for safety."""
    if not enabled:
        return False
    t = (text or "").strip()
    if not t:
        return False
    lowered = t.lower()
    allowlist = _csv_tokens(allowlist_keywords)
    if allowlist and any(token in lowered for token in allowlist):
        return False
    bypass = _csv_tokens(bypass_keywords)
    return any(token in lowered for token in bypass)


def _format_pipeline_summary_md(summary: dict[str, Any]) -> str:
    """Format CRM pipeline summary as markdown for chat."""
    total_count = summary.get("total_count", 0)
    total_value = summary.get("total_value", 0)
    stages = summary.get("stages") or {}
    lines = [
        "**Sales pipeline**",
        f"Total: **{total_count}** deals | **${total_value:,.0f}** value",
        "",
    ]
    if stages:
        lines.append("| Stage | Count | Value |")
        lines.append("|-------|-------|-------|")
        for stage_name, data in sorted(stages.items()):
            if isinstance(data, dict):
                count = data.get("count", 0)
                val = data.get("total_value", 0)
                lines.append(f"| {stage_name} | {count} | ${val:,.0f} |")
    else:
        lines.append("No deals in pipeline.")
    return "\n".join(lines)


_TEACHING_SIGNAL_PATTERN = re.compile(
    r"\b(remember|teach|learn|note(?:\s+this)?|we\s+need\s+to|must|should)\b",
    re.IGNORECASE,
)


def _extract_teaching_facts(raw_input: str) -> list[str]:
    """Extract explicit user-provided teachings without relying on an LLM."""
    text = (raw_input or "").strip()
    if not text:
        return []
    if not _TEACHING_SIGNAL_PATTERN.search(text):
        return []

    facts: list[str] = []
    for chunk in re.split(r"[\n.;]+", text):
        sentence = chunk.strip().strip("-").strip()
        if not sentence:
            continue
        low = sentence.lower()
        if len(sentence) < 12:
            continue
        if any(
            marker in low
            for marker in (
                "remember",
                "teach",
                "learn",
                "note",
                "we need to",
                "we should",
                "we must",
            )
        ):
            facts.append(f"User instruction: {sentence}")
        if len(facts) >= 5:
            break
    return facts


def _faithfulness_strict_intent(
    agents_used: list[str] | None,
    user_text: str,
    *,
    channel: str = "",
) -> bool:
    """Heuristic: high-risk intents need stricter verification when faithfulness_mode=strict."""
    from brain_os.config import EmailMode, get_settings

    agents = {(a or "").strip().lower() for a in (agents_used or []) if a}
    text = (user_text or "").lower()
    if agents & {"plutus", "quotebuilder"}:
        return True
    if agents & {"calliope"}:
        return True
    # Outbound/high-stakes motions when Gmail is OPERATIONAL — tighten verifier behavior.
    if get_settings().google.email_mode == EmailMode.OPERATIONAL and (
        agents & {"calliope", "hermes", "na_sales"}
        or "send email" in text
        or "send the email" in text
        or "draft email" in text
    ):
        return True
    needles = (
        "price",
        "pricing",
        "quote",
        "delivery",
        "lead time",
        "leadtime",
        "spec",
        "specification",
        "dispatch",
        "shipment",
        "send email",
        "send the email",
        "draft email",
        "outbound",
        "contract",
        "warranty",
        "guarantee",
        "nda",
        "mou",
        "commitment",
    )
    if any(n in text for n in needles):
        return True
    _ch = (channel or "").upper()
    if _ch in {"OPERATIONAL", "EMAIL_OPERATIONAL"} and (
        agents & {"calliope", "hermes", "na_sales"} or "email" in text or "outbound" in text
    ):
        return True
    return False


__all__ = [
    "_AMBIGUITY_SENSITIVE_PATTERNS",
    "_DEDUP_ENVELOPE_VERSION",
    "_DEFAULT_CHEAP_EXIT_BYPASS_KEYWORDS",
    "_QUICK_PIPELINE_PATTERN",
    "_csv_tokens",
    "_decode_dedup_payload",
    "_encode_dedup_payload",
    "_extract_teaching_facts",
    "_faithfulness_strict_intent",
    "_format_pipeline_summary_md",
    "_is_ambiguity_sensitive_query",
    "_is_quick_pipeline_query",
    "_should_bypass_cheap_exits",
]
