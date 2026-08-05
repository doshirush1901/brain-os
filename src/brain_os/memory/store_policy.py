"""Lightweight guards before writing to long-term conversational memory.

Inspired by biological memory encoding: most moments are not stored; what survives
needs salience (novelty, stakes, repetition cues). Retrieval rewrites memory
(reconsolidation) — stores after recall are tagged accordingly.

Category gate (2026-08-02 landfill repair): ingest_log / bookkeeping / status_echo
are blocked from Mem0 — they belong in the stomach ledger, not semantic memory.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

logger = logging.getLogger(__name__)

ProvenanceClass = Literal["evidence", "lore", "correction", "preference"]

MemoryCategory = Literal[
    "ingest_log",
    "bookkeeping",
    "status_echo",
    "session",
    "fact",
    "commercial_fact",
    "preference",
    "correction",
    "relationship",
    "episode",
    "pending_hint",
    "sophia_lesson",
    "session_digest",
]

BLOCKED_CATEGORIES: frozenset[str] = frozenset({"ingest_log", "bookkeeping", "status_echo"})

_VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "ingest_log",
        "bookkeeping",
        "status_echo",
        "session",
        "fact",
        "commercial_fact",
        "preference",
        "correction",
        "relationship",
        "episode",
        "pending_hint",
        "sophia_lesson",
        "session_digest",
    }
)

_EPHEMERAL_PATTERNS = (
    r"\bignore\s+(the\s+)?(above|this)\b",
    r"\bscratch\s+(that|this)\b",
    r"\bjust\s+a\s+test\b",
    r"\bstet\b",
    r"\bdo\s+not\s+(store|remember)\b",
    r"\bdelete\s+(this\s+)?memory\b",
)

_EVIDENCE_SOURCE_PREFIXES = (
    "email:",
    "takeout_email",
    "contact_history:",
    "kb:",
    "crm:",
    "quote:",
    "gmail:",
    "document:",
)

_STAKES_MARKERS = re.compile(
    r"\b(urgent|critical|complaint|deadline|lost|won|signed|quote|price|"
    r"objection|production|delivery|warranty|correction|remember that)\b",
    re.IGNORECASE,
)

_INGEST_LOG_RX = re.compile(
    r"(?:user\s+)?ingested\s+source\b",
    re.IGNORECASE,
)
_BOOKKEEPING_RX = re.compile(
    r"\b(chunks_created|ingestion_log|digest\s+complete|files?\s+processed)\b",
    re.IGNORECASE,
)
_STATUS_ECHO_RX = re.compile(
    r"^(ok|done|success|failed|skipped|status:\s*\w+)\s*$",
    re.IGNORECASE,
)

_TYPE_TO_CATEGORY: dict[str, str] = {
    "ingested_source": "ingest_log",
    "correction": "correction",
    "preference": "preference",
    "episode": "episode",
    "commercial_fact": "commercial_fact",
    "pending_memory_hint": "pending_hint",
    "sophia_prediction_reflection": "sophia_lesson",
    "session_mine_daily_digest": "session_digest",
    "relationship": "relationship",
    "fact": "fact",
}


@dataclass(frozen=True)
class StoreDecision:
    """Result of ``evaluate_mem_store`` — allow/deny plus enriched metadata."""

    allow: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    salience_score: float = 0.0
    memory_category: str = "session"


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
            _record_skip("ephemeral_marker")
            return True, "ephemeral_marker"
    return False, ""


def _record_skip(reason: str) -> None:
    """Report a memory-store block to the immune registry (never raises)."""
    from brain_os.immune.registry import record_trigger

    record_trigger("memory_store_policy", {"reason": reason})


_ECHO_OVERLAP_THRESHOLD = 0.8
_ECHO_PREFIX_RX = re.compile(r"^(user instruction|fact|note)\s*:\s*", re.IGNORECASE)


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 3}


def is_query_echo(fact: str, raw_input: str) -> bool:
    """True when *fact* merely restates the user's question instead of adding knowledge."""
    fact_tokens = _content_tokens(_ECHO_PREFIX_RX.sub("", fact or ""))
    if not fact_tokens:
        return True
    input_tokens = _content_tokens(raw_input)
    if not input_tokens:
        return False
    overlap = len(fact_tokens & input_tokens) / len(fact_tokens)
    return overlap >= _ECHO_OVERLAP_THRESHOLD


def classify_memory_category(
    content: str,
    *,
    source: str = "",
    metadata: dict[str, Any] | None = None,
    declared: str | None = None,
) -> tuple[str, bool]:
    """Return ``(category, was_explicit)``.

    Explicit declaration (``declared`` or ``metadata.memory_category``) wins.
    Do **not** trust ``metadata.category`` — that field is used by Nemesis
    correction taxonomy slugs. Unclassified → ``session`` with
    ``was_explicit=False`` (caller should warn).
    """
    meta = metadata or {}
    for candidate in (declared, meta.get("memory_category")):
        raw = str(candidate or "").strip().lower()
        if raw in _VALID_CATEGORIES:
            return raw, True

    mem_type = str(meta.get("type", "")).lower()
    if mem_type in _TYPE_TO_CATEGORY:
        return _TYPE_TO_CATEGORY[mem_type], True

    text = (content or "").strip()
    if _INGEST_LOG_RX.search(text) or mem_type == "ingested_source":
        return "ingest_log", False
    if _BOOKKEEPING_RX.search(text):
        return "bookkeeping", False
    if _STATUS_ECHO_RX.match(text):
        return "status_echo", False

    src = (source or str(meta.get("source", ""))).lower()
    if src.startswith("quote:") or "commercial_fact" in src:
        return "commercial_fact", False
    if src.startswith(("pending_queue:", "pending_queue")):
        return "pending_hint", False
    if "preference" in src:
        return "preference", False

    return "session", False


def classify_provenance(source: str, metadata: dict[str, Any] | None) -> ProvenanceClass:
    """Label memory as evidence (anchored) vs lore (synthesized/narrative)."""
    meta = metadata or {}
    mem_type = str(meta.get("type", "")).lower()
    if mem_type == "correction":
        return "correction"
    if mem_type == "preference":
        return "preference"
    if meta.get("verified") is True:
        return "evidence"
    src = (source or str(meta.get("source", ""))).lower()
    if src.startswith(_EVIDENCE_SOURCE_PREFIXES):
        return "evidence"
    if mem_type in {"fact", "episode", "commercial_fact"}:
        try:
            if float(meta.get("confidence", 0)) >= 0.8:
                return "evidence"
        except (TypeError, ValueError):
            pass
    if meta.get("reconsolidated") or meta.get("recalled_from"):
        return "lore"
    return "lore"


def score_salience(
    content: str,
    *,
    source: str = "",
    metadata: dict[str, Any] | None = None,
) -> float:
    """Heuristic 0–1 salience score (competition for encoding)."""
    text = (content or "").strip()
    meta = metadata or {}
    src = source or str(meta.get("source", ""))
    provenance = classify_provenance(src, meta)

    if provenance == "correction":
        return 1.0
    if provenance == "preference":
        return 0.9
    if str(meta.get("memory_category", "")) == "commercial_fact":
        return 0.95

    score = 0.2
    if len(text) >= 40:
        score += 0.12
    if len(text) >= 100:
        score += 0.08
    if re.search(r"\d", text):
        score += 0.12
    if re.search(r"\$|USD|INR|€|₹|EUR|GBP", text, re.IGNORECASE):
        score += 0.08
    if not text.rstrip().endswith("?"):
        score += 0.05
    if _STAKES_MARKERS.search(text):
        score += 0.15
    if provenance == "evidence":
        score += 0.2
    if meta.get("recalled_from"):
        score += 0.08
    return min(1.0, score)


def _salience_min_score() -> float:
    from brain_os.config import get_settings

    return float(getattr(get_settings().app, "mem0_salience_min_score", 0.35))


def evaluate_mem_store(
    content: str,
    *,
    source: str = "",
    metadata: dict[str, Any] | None = None,
    recalled_ids: list[str] | None = None,
    bypass_salience: bool = False,
    category: str | None = None,
) -> StoreDecision:
    """Unified gate before Mem0 writes — category + noise filter + salience + provenance."""
    skip, reason = should_skip_mem_store(content)
    if skip:
        return StoreDecision(False, reason, {}, 0.0, "session")

    meta: dict[str, Any] = dict(metadata or {})
    src = source or str(meta.get("source", ""))
    if src and not meta.get("source"):
        meta["source"] = src

    cat, explicit = classify_memory_category(content, source=src, metadata=meta, declared=category)
    if not explicit:
        logger.warning(
            "Mem0 store without explicit memory_category; defaulting to %r (source=%r)",
            cat,
            src or meta.get("source", ""),
        )
    meta["memory_category"] = cat

    if cat in BLOCKED_CATEGORIES:
        block_reason = f"category_blocked:{cat}"
        _record_skip(block_reason)
        logger.info("Mem0 store blocked by category gate: %s", block_reason)
        return StoreDecision(False, block_reason, meta, 0.0, cat)

    provenance = classify_provenance(src, meta)
    meta["provenance_class"] = provenance
    salience = score_salience(content, source=src, metadata=meta)
    meta["salience_score"] = round(salience, 3)

    clean_ids = [str(i) for i in (recalled_ids or []) if i]
    if clean_ids:
        meta["recalled_from"] = ",".join(clean_ids)
        meta["reconsolidated"] = True
        meta["provenance_class"] = "lore"

    if provenance in ("correction", "preference") or cat in {
        "correction",
        "preference",
        "commercial_fact",
    }:
        return StoreDecision(True, "protected_type", meta, salience, cat)

    if bypass_salience:
        return StoreDecision(True, "bypass", meta, salience, cat)

    if salience < _salience_min_score():
        _record_skip("low_salience")
        return StoreDecision(False, "low_salience", meta, salience, cat)

    return StoreDecision(True, "ok", meta, salience, cat)


def format_reconsolidation_warning() -> str:
    """Operator-facing note after recall — vivid ≠ accurate."""
    return (
        "Reconsolidation note: recalled memories are narrative (lore), not playback. "
        "Verify prices, stages, and dates against email/KB/corrections before acting."
    )


def is_ingest_log_memory(memory: dict[str, Any]) -> bool:
    """True when a Mem0 row is ingest-log landfill (content or metadata)."""
    meta = memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
    cat = str(meta.get("memory_category") or meta.get("type") or "").lower()
    if cat in {"ingest_log", "ingested_source"}:
        return True
    text = str(memory.get("memory") or memory.get("content") or "")
    return bool(_INGEST_LOG_RX.search(text))
