"""Unified retrieval layer for the Brain OS system.

**No agent should query Qdrant or Neo4j directly.**  Every knowledge lookup
flows through :class:`UnifiedRetriever`, which fans out across the vector
store, the knowledge graph, and (optionally) Mem0 conversational memory,
then merges and reranks the results with Voyage AI Rerank (with FlashRank
as a local fallback) before returning them.

**Store split (operational contract):**
- **Qdrant** — unstructured evidence chunks (PDFs, playbooks, email-derived text).
- **Neo4j** — entities and relationships; hits are stitched back to Qdrant text via
  :meth:`UnifiedRetriever._stitch_graph_to_vectors` so answers cite passages, not
  bare node IDs.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
from flashrank import Ranker, RerankRequest
from langfuse.decorators import observe
from neo4j.exceptions import Neo4jError

from brain_os.brain import query_rewrite as _qr
from brain_os.brain.context_graph_expand import (
    graph_context_to_retriever_hits,
    graph_expand_enabled,
)
from brain_os.brain.knowledge_graph import KnowledgeGraph, normalize_entity_name
from brain_os.brain.qdrant_manager import QdrantManager
from brain_os.brain.retrieval_context import retrieval_profile_var
from brain_os.brain.retrieval_eval import keyword_overlap_score
from brain_os.brain.retrieval_slo import merge_backend_timeouts
from brain_os.brain.retrieval_trace import emit_retrieval_trace
from brain_os.config import get_settings
from brain_os.exceptions import DatabaseError, IngestionError, BrainOSError, LLMError
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import EntityNames, SubQueries
from brain_os.services.llm_client import get_llm_client
from brain_os.services.resilience import CircuitBreaker, RetryPolicy, run_with_retry

try:
    from voyageai.error import VoyageError as _VoyageError
except ImportError:  # pragma: no cover — voyage optional in some test envs
    _VoyageError = ()  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

# Soft-fail backends: Neo4j AuthError / Voyage APIError historically bubbled out of
# search_knowledge (tool_invocations error_code AuthError/APIError, June 2026).
_BACKEND_SOFT_ERRORS: tuple[type[BaseException], ...] = (
    DatabaseError,
    Neo4jError,
    httpx.HTTPError,
    OSError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    RuntimeError,
    BrainOSError,
    LLMError,
    IngestionError,
)
if _VoyageError:
    _BACKEND_SOFT_ERRORS = (*_BACKEND_SOFT_ERRORS, _VoyageError)  # type: ignore[assignment]

_DECOMPOSE_SYSTEM_PROMPT = load_prompt("decompose_query")

#: Default Voyage rerank endpoint. Phase-2 hardening: the actual URL used
#: at request time is read from ``Settings.llm_endpoints.voyage_rerank_url``.
#: This constant remains for backward compatibility with anything importing it.
_VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"


#: Mem0 retrievals are metered; the default fan-out only includes Mem0 when
#: the query plausibly targets conversational memory rather than the KB.
_MEMORY_FLAVOR_PATTERNS = (
    r"\bremember(ed|s|ing)?\b",
    r"\brecall(ed|s)?\b",
    r"\bmemor(y|ies|ize|ized)\b",
    r"\bprefer(s|red|ence|ences)?\b",
    r"\blast\s+time\b",
    r"\bwe\s+(discussed|talked|spoke|agreed)\b",
    r"\b(you|i)\s+(said|told|mentioned|asked)\b",
    r"\bcorrect(ion|ions|ed)\b",
    r"\bepisodes?\b",
    r"\babout\s+me\b",
    r"\bconversations?\b",
    r"\brelationship\b",
)

_memory_flavor_regexes_cache: tuple[Any, ...] | None = None


def query_is_memory_flavored(query: str) -> bool:
    """Deterministic check: does *query* target conversational/personal memory?"""
    global _memory_flavor_regexes_cache
    if _memory_flavor_regexes_cache is None:
        import re

        _memory_flavor_regexes_cache = tuple(
            re.compile(p, re.IGNORECASE) for p in _MEMORY_FLAVOR_PATTERNS
        )
    text = (query or "").strip()
    if not text:
        return False
    return any(rx.search(text) for rx in _memory_flavor_regexes_cache)


def _keyword_score(query: str, content: str, source: str = "") -> float:
    """Fraction of query words (min 2 chars) that appear in content or source. 0–1."""
    words = {w.lower() for w in query.split() if len(w) >= 2}
    if not words:
        return 0.0
    text = ((content or "") + " " + (source or "")).lower()
    hits = sum(1 for w in words if w in text)
    return hits / len(words)


_TRUSTED_SOURCE_SUBSTRINGS = (
    "website_source_of_truth",
    "sleep_trainer_correction",
)
_TRUSTED_SOURCE_SCORE_BOOST = 0.08
_TEACHER_CANON_SCORE_BOOST = 0.06
_TEACHER_TARGETED_BOOST = 0.04
# Commercial-intent boost: prefer chunks flagged has_quote / has_price when the
# query is about quotes, pricing, or commercial terms (backfilled payload flags).
_QUOTE_SIGNAL_BOOST = 0.06
_PRICE_SIGNAL_BOOST = 0.03
_QUOTE_QUERY_MARKERS = (
    "quote",
    "quotation",
    "price",
    "pricing",
    "cost",
    "budget",
    "proforma",
    "commercial offer",
    "offer no",
    "rate for",
    "how much",
    "inr",
    "usd",
    "lakh",
    "crore",
)
_TEACHER_QUERY_MARKERS = (
    "constitutional ai",
    "model spec",
    "agentic",
    "llm behavior",
    "transformer",
    "context window",
    "faithfulness",
    "karpathy",
    "nanogpt",
    "safety",
    "claude character",
    # GTM / industrial forming / outbound
    "thermoform",
    "vacuum form",
    "draw ratio",
    "plug assist",
    "heavy gauge",
    "thin gauge",
    "pf1",
    "closed chamber",
    "closed-chamber",
    "incoterm",
    "fob ",
    "cif ",
    "exw ",
    "cold email",
    "outbound email",
    "drip campaign",
    "reshoring",
    "great lakes",
    "web meeting",
    "can-spam",
    "objection handling",
    "discovery call",
    "claude code",
    "prompt engineering",
)
# Query substring → teacher id for targeted boost (longest keys checked first).
_TEACHER_ID_QUERY_MARKERS: dict[str, tuple[str, ...]] = {
    "thermoforming_canon": (
        "draw ratio",
        "plug assist",
        "thermoform",
        "vacuum form",
        "heavy gauge",
        "thin gauge",
        "forming temperature",
        "sheet sag",
    ),
    "demo_product": (
        "pf1",
        "closed chamber",
        "closed-chamber",
        "acme-corp",
        "uno",
        "duo",
        "fcs",
        "atf",
    ),
    "outbound_email": (
        "cold email",
        "outbound email",
        "can-spam",
        "plain text email",
        "follow-up email",
    ),
    "b2b_sales_craft": (
        "objection handling",
        "discovery call",
        "sales coaching",
        "b2b sales",
    ),
    "global_markets": (
        "incoterm",
        "fob ",
        "cif ",
        "exw ",
        "reshoring",
        "great lakes",
        "web meeting",
        "freight lane",
    ),
    "anthropic": (
        "constitutional ai",
        "claude character",
        "model spec",
        "agentic",
        "faithfulness",
    ),
    "karpathy": ("karpathy", "nanogpt", "micrograd", "transformer", "llm behavior"),
    "claude_code": ("claude code", "prompt engineering", "mcp tool", "cursor agent"),
}


def teacher_ids_for_query(query: str) -> frozenset[str]:
    """Teacher ids whose corpus should rank higher for this query (tests / debugging)."""
    q = (query or "").lower()
    matched: set[str] = set()
    for teacher_id, markers in _TEACHER_ID_QUERY_MARKERS.items():
        if any(m in q for m in markers):
            matched.add(teacher_id)
    return frozenset(matched)


def query_triggers_teacher_boost(query: str) -> bool:
    """True when retrieval should prefer teacher_canon hits for this query."""
    q = (query or "").lower()
    return bool(teacher_ids_for_query(query)) or any(m in q for m in _TEACHER_QUERY_MARKERS)


def query_has_commercial_intent(query: str) -> bool:
    """True when the query is about quotes, prices, or commercial terms."""
    q = (query or "").lower()
    return any(m in q for m in _QUOTE_QUERY_MARKERS)


def _apply_quote_signal_boost(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Boost chunks flagged ``has_quote`` / ``has_price`` on commercial queries."""
    if not results or not query_has_commercial_intent(query):
        return results
    for r in results:
        meta = r.get("metadata") or {}
        if not isinstance(meta, dict):
            continue
        if meta.get("has_quote"):
            r["score"] = float(r.get("score", 0)) + _QUOTE_SIGNAL_BOOST
            r["_quote_signal_boost"] = True
        if meta.get("has_price"):
            r["score"] = float(r.get("score", 0)) + _PRICE_SIGNAL_BOOST
            r["_price_signal_boost"] = True
    return results


def _apply_trusted_source_boost(
    results: list[dict[str, Any]],
    *,
    query: str = "",
) -> list[dict[str, Any]]:
    """Prefer operator-maintained Acme Corp source-of-truth over ad-hoc uploads."""
    if not results:
        return results
    (query or "").lower()
    boost_teacher = query_triggers_teacher_boost(query)
    targeted_teachers = teacher_ids_for_query(query)
    for r in results:
        src = str(r.get("source") or "")
        if any(token in src for token in _TRUSTED_SOURCE_SUBSTRINGS):
            r["score"] = float(r.get("score", 0)) + _TRUSTED_SOURCE_SCORE_BOOST
            r["_trusted_source_boost"] = True
        meta = r.get("metadata") or {}
        hit_teacher = (
            str(meta.get("teacher") or "").strip().lower() if isinstance(meta, dict) else ""
        )
        if (boost_teacher and str(r.get("source_category") or "") == "teacher_canon") or (
            boost_teacher
            and isinstance(meta, dict)
            and str(meta.get("source_authority") or "").lower() == "high"
        ):
            r["score"] = float(r.get("score", 0)) + _TEACHER_CANON_SCORE_BOOST
            r["_teacher_canon_boost"] = True
            if hit_teacher and hit_teacher in targeted_teachers:
                r["score"] = float(r.get("score", 0)) + _TEACHER_TARGETED_BOOST
                r["_teacher_targeted_boost"] = True
    results.sort(key=lambda item: float(item.get("score", 0)), reverse=True)
    return results


def _apply_keyword_boost(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort results by vector score + keyword boost so exact-match candidates rank higher before rerank."""
    if not query or not results:
        return results
    kw_w = float(get_settings().app.retriever_keyword_boost_weight)
    for r in results:
        r["_keyword_score"] = _keyword_score(
            query,
            r.get("content", ""),
            r.get("source", ""),
        )

    # Blend: higher keyword score moves item up when vector scores are close.
    def _sort_key(r: dict[str, Any]) -> tuple[float, float]:
        base = float(r.get("score", 0))
        kw = float(r.get("_keyword_score", 0))
        return (-(base + kw_w * kw), -kw)

    results.sort(key=_sort_key)
    results = _apply_quote_signal_boost(query, results)
    results = _apply_trusted_source_boost(results, query=query)
    for r in results:
        r.pop("_keyword_score", None)
    return results


def _diversify_results(
    results: list[dict[str, Any]],
    limit: int,
    overlap_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Reduce near-duplicate chunks: keep items whose content is not too similar to already-selected.

    Uses word-set overlap as a cheap proxy for semantic similarity so we avoid extra embedding calls.
    """
    if overlap_threshold is None:
        overlap_threshold = float(get_settings().app.retriever_diversity_overlap_threshold)
    if not results or limit <= 0:
        return results[:limit]
    if len(results) <= limit:
        return results

    def _word_set(text: str, max_words: int = 80) -> set[str]:
        words = (text or "").strip().lower().split()
        return set(words[:max_words])

    selected: list[dict[str, Any]] = []
    selected_sigs: list[set[str]] = []

    for r in results:
        if len(selected) >= limit:
            break
        content = r.get("content", "") or ""
        sig = _word_set(content)
        if not sig:
            selected.append(r)
            selected_sigs.append(sig)
            continue
        max_overlap = 0.0
        for prev in selected_sigs:
            if not prev:
                continue
            overlap = len(sig & prev) / min(len(sig), len(prev)) if prev else 0
            max_overlap = max(max_overlap, overlap)
        if max_overlap <= overlap_threshold:
            selected.append(r)
            selected_sigs.append(sig)

    return selected


def _word_set_for_dedup(text: str, max_chars: int) -> set[str]:
    snippet = (text or "")[:max_chars].lower()
    return {w for w in snippet.split() if len(w) > 2}


def _jaccard_word_sets(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _dedup_near_duplicate_passages(
    rows: list[dict[str, Any]],
    *,
    max_jaccard: float,
    content_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    """Greedy keep-first: drop rows whose word-set Jaccard to any kept row is >= *max_jaccard*."""
    kept: list[dict[str, Any]] = []
    sigs: list[set[str]] = []
    removed = 0
    for r in rows:
        text = str(r.get("content") or "")
        sig = _word_set_for_dedup(text, content_chars)
        if not sig:
            kept.append(r)
            sigs.append(sig)
            continue
        too_similar = False
        for prev in sigs:
            if not prev:
                continue
            if _jaccard_word_sets(sig, prev) >= max_jaccard:
                too_similar = True
                break
        if too_similar:
            removed += 1
            continue
        kept.append(r)
        sigs.append(sig)
    return kept, removed


def _is_retryable_external_error(exc: Exception) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status is None or status >= 500 or status == 429


def flashrank_cache_probe() -> dict[str, Any]:
    """Resolve FlashRank ONNX cache path and count ``*.onnx`` files under it.

    Does not instantiate :class:`flashrank.Ranker` (avoids download / import side effects).
    """
    settings = get_settings()
    from brain_os.systems.data_dir_lock import coerce_config_path, get_data_dir

    try:
        cache_root = coerce_config_path(settings.app.flashrank_cache_dir)
    except ValueError:
        cache_root = ""
    if not cache_root:
        cache_root = str(get_data_dir() / ".flashrank_cache")
    root = Path(cache_root)
    onnx_files = 0
    if root.is_dir():
        onnx_files = sum(1 for _ in root.rglob("*.onnx"))
    return {"cache_dir": cache_root, "onnx_files": onnx_files}


class UnifiedRetriever:
    """Single entry-point for all knowledge retrieval in Brain OS."""

    def __init__(
        self,
        qdrant: QdrantManager,
        graph: KnowledgeGraph,
        long_term: Any | None = None,
        *,
        mem0_client: Any | None = None,
        reranker_model: str = "ms-marco-MiniLM-L-12-v2",
    ) -> None:
        self._qdrant = qdrant
        self._graph = graph
        if long_term is not None:
            self._long_term = long_term
        elif mem0_client is not None:
            logger.warning(
                "UnifiedRetriever mem0_client is deprecated; pass long_term=LongTermMemory()"
            )
            self._long_term = None
        else:
            self._long_term = None

        settings = get_settings()
        from brain_os.systems.data_dir_lock import coerce_config_path, get_data_dir

        try:
            cache_root = coerce_config_path(settings.app.flashrank_cache_dir)
        except ValueError:
            cache_root = ""
        if not cache_root:
            cache_root = str(get_data_dir() / ".flashrank_cache")
        Path(cache_root).mkdir(parents=True, exist_ok=True)
        try:
            self._flashrank = Ranker(model_name=reranker_model, cache_dir=cache_root)
        except (OSError, ImportError, RuntimeError, ValueError, TypeError):
            logger.warning(
                "FlashRank init failed (missing ONNX/cache or runtime error). "
                "Voyage rerank still works when configured; otherwise retrieval uses "
                "pre-rerank scores. Cache dir: %s",
                cache_root,
                exc_info=True,
            )
            self._flashrank = None

        self._llm = get_llm_client()
        self._voyage_key = settings.embedding.api_key.get_secret_value()
        self._voyage_rerank_model = settings.embedding.rerank_model
        self._mem0_timeout_seconds = max(1.0, float(settings.app.mem0_timeout))
        self._flashrank_timeout_seconds = 20.0
        self._backend_timeouts_seconds: dict[str, float] = {
            "qdrant": 35.0,  # Allow time for Qdrant client timeout (default 30s) so we don't cancel mid-request
            "neo4j": 12.0,
            "mem0": max(2.0, self._mem0_timeout_seconds + 1.0),
        }
        self._external_retry = RetryPolicy(max_attempts=3, base_delay_seconds=0.8)
        self._mem0_breaker = CircuitBreaker(threshold=8, window_seconds=180)
        self._voyage_breaker = CircuitBreaker(threshold=8, window_seconds=180)
        #: Last :meth:`search` health for pipeline ``degradation`` trace (best-effort).
        self._last_search_health: dict[str, Any] = {}
        self._search_entity_id: str | None = None
        self._search_company: str | None = None

    def breaker_public_snapshot(self) -> dict[str, Any]:
        """Non-secret breaker state for deep-health probes."""
        return {
            "mem0": self._mem0_breaker.public_snapshot(),
            "voyage_rerank": self._voyage_breaker.public_snapshot(),
        }

    def last_retrieval_health_public(self) -> dict[str, Any]:
        """Snapshot of retrieval health for the current task (ContextVar) or process fallback.

        Intended for ``GET /api/metrics`` and operators. Contains no secrets.
        Concurrent searches prefer the task-local ContextVar so they do not clobber
        each other mid-flight; the instance field remains a last-writer metrics fallback.
        """
        from brain_os.brain.retrieval_context import get_last_search_health

        h = get_last_search_health()
        if h:
            return h
        h2 = self._last_search_health
        return dict(h2) if isinstance(h2, dict) else {}

    def _default_fanout_sources(self, query: str) -> set[str]:
        """Backends queried when the caller does not restrict ``sources``."""
        use = {"qdrant", "neo4j"}
        try:
            scoped = bool(getattr(get_settings().app, "retriever_mem0_scoped_fanout", True))
        except (AttributeError, TypeError):
            scoped = True
        if not scoped or query_is_memory_flavored(query):
            use.add("mem0")
        return use

    # ── primary search ───────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int = 10,
        company: str | None = None,
        entity_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fan-out search across Qdrant, Neo4j, and Mem0, then rerank.

        *sources* can restrict which backends are queried.  Valid values:
        ``"qdrant"``, ``"neo4j"``, ``"mem0"``.  ``None`` means all.

        Optional *company* / *entity_id* filter Qdrant via
        ``metadata.graph_entity_ids`` (e.g. ``Company:acme-corp``).

        When the primary backends return no results, the imports fallback
        retriever (Alexandros's metadata index) is consulted automatically.

        Mem0 is metered per month, so the *default* fan-out (``sources=None``)
        includes it only for memory-flavored queries (see
        :func:`query_is_memory_flavored`). Set
        ``APP__RETRIEVER_MEM0_SCOPED_FANOUT=false`` to restore the old
        always-include behavior; explicit ``sources=["mem0"]`` always works.
        """
        use = set(sources) if sources else self._default_fanout_sources(query)
        self._search_entity_id = (entity_id or "").strip() or None
        self._search_company = (company or "").strip() or None
        if self._search_company and not self._search_entity_id:
            from brain_os.brain.knowledge_graph_text import company_name_key

            key = company_name_key(self._search_company)
            if key:
                self._search_entity_id = f"Company:{key}"

        health: dict[str, Any] = {
            "backend_failures": [],
            "used_discovery": False,
            "imports_fallback": False,
            "weak_evidence": False,
            "keyword_overlap_prerank": None,
            "second_pass_used": False,
            "second_pass_query": None,
            "rerank_path": None,
            "dedup_removed": 0,
            "retrieval_profile": None,
            "mem0_scoped_out": (
                sources is None and self._long_term is not None and "mem0" not in use
            ),
        }

        try:
            return await self._search_inner(query, use=use, limit=limit, health=health)
        except Exception:
            bf = health["backend_failures"]
            if isinstance(bf, list):
                bf.append("search_fatal")
            logger.exception("Retriever.search failed closed to empty results")
            return []
        finally:
            from brain_os.brain.retrieval_context import set_last_search_health

            set_last_search_health(health)
            # Process-level fallback for metrics / post-scope readers (last writer wins).
            self._last_search_health = health

    async def _search_inner(
        self,
        query: str,
        *,
        use: set[str],
        limit: int,
        health: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Core fan-out search; callers wrap with fail-closed empty results."""
        app_cfg = get_settings().app
        effective_query = _qr.normalize_for_retrieval(query)
        profile = (
            retrieval_profile_var.get() or app_cfg.retriever_default_profile or "default"
        ).strip()
        health["retrieval_profile"] = profile
        local_timeouts = merge_backend_timeouts(self._backend_timeouts_seconds, profile)

        emit_retrieval_trace(
            "search_start",
            sources=sorted(use),
            limit=limit,
            **_qr.summarize_for_trace(query),
        )

        ov = max(1, int(app_cfg.retriever_over_retrieve_factor))
        tasks: dict[str, asyncio.Task[Any]] = {}
        if "qdrant" in use:
            tasks["qdrant"] = asyncio.create_task(
                self._search_qdrant(effective_query, limit=limit * ov)
            )
        if "neo4j" in use:
            tasks["neo4j"] = asyncio.create_task(self._search_graph(effective_query))
        if "mem0" in use and self._long_term is not None:
            tasks["mem0"] = asyncio.create_task(self._search_mem0(effective_query, limit=limit))

        merged: list[dict[str, Any]] = []
        for source_type, task in tasks.items():
            try:
                backend_timeout = local_timeouts.get(source_type, 15.0)
                results = await asyncio.wait_for(task, timeout=backend_timeout)
                for r in results:
                    r["source_type"] = source_type
                merged.extend(results)
            except TimeoutError:
                bf = health["backend_failures"]
                if isinstance(bf, list):
                    bf.append(f"{source_type}_timeout")
                logger.warning(
                    "Retrieval from %s timed out after %.1fs",
                    source_type,
                    local_timeouts.get(source_type, 15.0),
                )
            except _BACKEND_SOFT_ERRORS:
                bf = health["backend_failures"]
                if isinstance(bf, list):
                    bf.append(f"{source_type}_error")
                logger.exception("Retrieval from %s failed", source_type)

        emit_retrieval_trace("backend_fanout_done", n_raw=len(merged))

        if not merged:
            try:
                from brain_os.brain.knowledge_discovery import KnowledgeDiscovery

                discovery = KnowledgeDiscovery(
                    retriever=self,
                    qdrant_manager=self._qdrant,
                    embedding_service=self._qdrant._embeddings,
                )
                discovered = await discovery.discover_and_store(effective_query, [])
                if discovered:
                    health["used_discovery"] = True
                    for d in discovered:
                        d["source_type"] = "discovery"
                    merged.extend(discovered)
            except (
                TimeoutError,
                BrainOSError,
                DatabaseError,
                httpx.HTTPError,
                OSError,
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
            ):
                logger.warning("Knowledge discovery fallback failed", exc_info=True)

        if not merged:
            health["imports_fallback"] = True
            return await self._imports_fallback(effective_query, limit)

        merged = await self._stitch_graph_to_vectors(merged)
        merged = _apply_keyword_boost(effective_query, merged)

        sample_cap = min(len(merged), max(limit * 3, 12))
        sample = merged[:sample_cap] if merged else []
        kw_pre = keyword_overlap_score(effective_query, sample) if sample else 1.0
        health["keyword_overlap_prerank"] = round(float(kw_pre), 4)
        thr = float(getattr(app_cfg, "retriever_second_pass_keyword_threshold", 0.22))
        weak = bool(sample) and kw_pre < thr
        health["weak_evidence"] = weak

        if (
            weak
            and bool(getattr(app_cfg, "retriever_second_pass_enabled", False))
            and "qdrant" in use
        ):
            alt_q = await self._second_pass_decomposed_query(effective_query)
            if alt_q:
                try:
                    extra = await asyncio.wait_for(
                        self._search_qdrant(alt_q, limit=limit * ov),
                        timeout=local_timeouts.get("qdrant", 35.0),
                    )
                    for r in extra:
                        r["source_type"] = "qdrant"
                    merged.extend(extra)
                    merged = await self._stitch_graph_to_vectors(merged)
                    merged = _apply_keyword_boost(effective_query, merged)
                    health["second_pass_used"] = True
                    health["second_pass_query"] = alt_q[:500]
                except TimeoutError:
                    bf = health["backend_failures"]
                    if isinstance(bf, list):
                        bf.append("second_pass_qdrant_timeout")
                    logger.warning(
                        "Retriever second-pass Qdrant timed out after %.1fs",
                        local_timeouts.get("qdrant", 35.0),
                    )
                except (
                    DatabaseError,
                    httpx.HTTPError,
                    OSError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    KeyError,
                ):
                    bf = health["backend_failures"]
                    if isinstance(bf, list):
                        bf.append("second_pass_qdrant_error")
                    logger.warning("Retriever second-pass Qdrant failed", exc_info=True)

        dedup_rm = 0
        if app_cfg.retriever_dedup_enabled and merged:
            merged, dedup_rm = _dedup_near_duplicate_passages(
                merged,
                max_jaccard=float(app_cfg.retriever_dedup_max_jaccard),
                content_chars=int(app_cfg.retriever_dedup_content_chars),
            )
        health["dedup_removed"] = dedup_rm

        emit_retrieval_trace("pre_rerank", n_hits=len(merged), dedup_removed=dedup_rm)

        await self._log_retrieval(effective_query, merged)
        return await self._rerank(effective_query, merged, limit, health=health)

    # ── graph-vector stitching ───────────────────────────────────────────

    async def _stitch_graph_to_vectors(
        self,
        merged_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Force-fetch Qdrant chunks for entities the graph knows about.

        When Neo4j returns a Quote or Machine node the metadata carries an
        identifier (``quote_id``, ``model``) but no document text.  If the
        initial Qdrant fan-out didn't happen to retrieve the matching PDF
        chunk the LLM is left to guess — causing hallucinated prices and
        specs.

        Uses (1) hybrid search by identifier and (2) Chunk–DESCRIBES links:
        for each graph entity, fetch Chunk point IDs from Neo4j and
        get_points() from Qdrant for denser network retrieval.
        """
        graph_items = [r for r in merged_results if r.get("source_type") == "neo4j"]
        if not graph_items:
            return merged_results

        stitch_attempted = 0
        stitch_misses = 0
        existing_contents: set[str] = {r.get("content", "")[:200] for r in merged_results}

        identifiers: set[str] = set()
        for r in graph_items:
            meta = r.get("metadata", {})
            if not isinstance(meta, dict):
                continue
            for key in ("quote_id", "model"):
                val = str(meta.get(key, "")).strip()
                if val:
                    identifiers.add(val)

        id_list = list(identifiers)[:10]
        stitch_attempted += len(id_list)
        stitch_tasks = [
            self._qdrant.hybrid_search(
                eid,
                limit=3,
                entity_id=self._search_entity_id,
                company=self._search_company,
            )
            for eid in id_list
        ]

        # Denser stitch: (entity_label, entity_key) → Chunk point IDs → get_points
        entity_refs: list[tuple[str, str]] = []
        for r in graph_items:
            meta = r.get("metadata", {}) or {}
            if meta.get("quote_id"):
                entity_refs.append(("Quote", str(meta["quote_id"])))
            if meta.get("model"):
                entity_refs.append(("Machine", str(meta["model"])))
            if meta.get("email"):
                entity_refs.append(("Person", str(meta["email"])))
            if meta.get("name") and not meta.get("email"):
                entity_refs.append(("Company", str(meta["name"])))
        seen_ref = set()
        unique_refs = [x for x in entity_refs if (x not in seen_ref and not seen_ref.add(x))]
        point_ids: set[str] = set()
        for label, key in unique_refs[:15]:
            stitch_attempted += 1
            try:
                ids = await self._graph.get_chunk_point_ids_for_entity(label, key, limit=5)
                if ids:
                    point_ids.update(ids)
                else:
                    stitch_misses += 1
            except (
                TimeoutError,
                DatabaseError,
                httpx.HTTPError,
                OSError,
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
            ):
                stitch_misses += 1
                logger.debug(
                    "get_chunk_point_ids_for_entity failed for %s:%s",
                    label,
                    key,
                    exc_info=True,
                )
        if point_ids:
            stitch_attempted += 1
            stitch_tasks.append(self._qdrant.get_points(list(point_ids)))

        all_results = await asyncio.gather(*stitch_tasks, return_exceptions=True)

        new_chunks: list[dict[str, Any]] = []
        for result_set in all_results:
            if isinstance(result_set, BaseException):
                stitch_misses += 1
                logger.debug("Graph-vector stitch search failed: %s", result_set)
                continue
            if not result_set:
                stitch_misses += 1
                continue
            for r in result_set:
                content_key = r.get("content", "")[:200]
                if content_key not in existing_contents:
                    existing_contents.add(content_key)
                    r["source_type"] = "qdrant_stitched"
                    new_chunks.append(r)

        stitch_resolved = len(new_chunks)
        if stitch_misses:
            logger.warning(
                "Graph-vector stitch misses: %d/%d attempted (added %d chunks)",
                stitch_misses,
                stitch_attempted,
                stitch_resolved,
            )
        elif new_chunks:
            logger.info(
                "Graph-vector stitch added %d Qdrant chunks (identifiers + Chunk links)",
                stitch_resolved,
            )

        if stitch_attempted:
            merged_results.append(
                {
                    "_stitch_stats": True,
                    "stitch_attempted": stitch_attempted,
                    "stitch_resolved": stitch_resolved,
                    "stitch_misses": stitch_misses,
                    "stitch_miss_rate": round(stitch_misses / stitch_attempted, 4),
                }
            )

        merged_results.extend(new_chunks)

        return merged_results

    # ── query decomposition ──────────────────────────────────────────────

    @observe()
    async def decompose_and_search(
        self,
        complex_query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Break a complex query into sub-queries, search each, and merge."""
        sub_queries = await self._decompose_query(complex_query)
        if not sub_queries:
            return await self.search(complex_query, limit=limit)

        tasks = [self.search(sq, limit=limit) for sq in sub_queries]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: list[dict[str, Any]] = []
        seen_contents: set[str] = set()
        for result_set in all_results:
            if isinstance(result_set, BaseException):
                logger.exception("Sub-query search failed", exc_info=result_set)
                continue
            for r in result_set:
                content_key = r.get("content", "")[:200]
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    merged.append(r)

        if not merged:
            return []

        return await self._rerank(complex_query, merged, limit, health=None)

    # ── category-filtered search ─────────────────────────────────────────

    async def search_by_category(
        self,
        query: str,
        category: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search Qdrant restricted to a single source_category."""
        ov = max(1, int(get_settings().app.retriever_over_retrieve_factor))
        results = await self._qdrant.hybrid_search(
            query,
            source_category=category,
            limit=limit * ov,
        )
        for r in results:
            r["source_type"] = "qdrant"
        return await self._rerank(query, results, limit, health=None)

    # ── backend-specific search helpers ──────────────────────────────────

    async def _search_qdrant(self, query: str, limit: int) -> list[dict[str, Any]]:
        return await self._qdrant.hybrid_search(
            query,
            limit=limit,
            entity_id=self._search_entity_id,
            company=self._search_company,
        )

    async def _search_graph(self, query: str) -> list[dict[str, Any]]:
        """Extract entities from the query via LLM, then look each up in Neo4j.

        Scores are assigned based on match quality:
        - Direct name match on the queried entity: 0.85
        - 1-hop related nodes: 0.65
        - 2+-hop related nodes: 0.45
        """
        entity_names = await self._extract_entity_names(query)
        if not entity_names:
            entity_names = [query]

        query_lower = query.lower()
        all_results: list[dict[str, Any]] = []
        seen: set[str] = set()

        for name in entity_names:
            name_lower = name.lower()
            company_key = normalize_entity_name(name.strip())
            if graph_expand_enabled() and company_key:
                try:
                    if await self._graph.company_node_exists(company_key):
                        expansion = await self._graph.expand_company_one_hop(company_key)
                        for hit in graph_context_to_retriever_hits(company_key, expansion):
                            content = str(hit.get("content") or "")
                            if content and content not in seen:
                                seen.add(content)
                                all_results.append(hit)
                        continue
                except (
                    TimeoutError,
                    Neo4jError,
                    DatabaseError,
                    httpx.HTTPError,
                    OSError,
                    ValueError,
                    TypeError,
                    AttributeError,
                    KeyError,
                ):
                    logger.debug(
                        "P2 company 1-hop expand failed for %s",
                        company_key[:80],
                        exc_info=True,
                    )

            try:
                subgraph = await self._graph.find_related_entities(name, max_hops=2)
            except Neo4jError:
                logger.debug(
                    "Neo4j find_related_entities failed for %s",
                    name[:80],
                    exc_info=True,
                )
                continue
            nodes = subgraph.get("nodes", [])
            relationships = subgraph.get("relationships", [])

            rel_strings: list[str] = []
            for rel in relationships:
                if isinstance(rel, dict):
                    rel_type = rel.get("type", "RELATED_TO")
                    from_name = rel.get("from", "?")
                    to_name = rel.get("to", "?")
                    rel_strings.append(f"{from_name} -[{rel_type}]-> {to_name}")

            for node in nodes:
                props = dict(node) if hasattr(node, "__iter__") else {"value": str(node)}
                content_parts = [f"{k}: {v}" for k, v in props.items() if v]
                content = ", ".join(content_parts)
                if content in seen:
                    continue
                seen.add(content)

                node_name = str(props.get("name", "")).lower()
                if node_name == name_lower or node_name in query_lower:
                    score = 0.85
                elif any(name_lower in str(v).lower() for v in props.values() if v):
                    score = 0.65
                else:
                    score = 0.45

                all_results.append(
                    {
                        "content": content,
                        "score": score,
                        "source": "neo4j",
                        "metadata": props,
                    }
                )

            if rel_strings:
                rel_content = "Graph relationships: " + "; ".join(rel_strings)
                if rel_content not in seen:
                    seen.add(rel_content)
                    all_results.append(
                        {
                            "content": rel_content,
                            "score": 0.75,
                            "source": "neo4j",
                            "metadata": {"type": "relationships", "entity": name},
                        }
                    )

        return all_results

    async def _extract_entity_names(self, query: str) -> list[str]:
        """Use the LLM to pull entity names (companies, people, machines) from a query."""
        system = (
            "Extract entity names from the user query. Return a JSON "
            "array of strings — company names, person names, email "
            "addresses, machine model numbers, and quote IDs. Return "
            "only the JSON array, nothing else. If no entities are "
            "found, return an empty array []."
        )
        try:
            result = await self._llm.generate_structured(
                system,
                query,
                EntityNames,
                name="retriever.extract_entities",
                model_tier="cheap",
            )
            return result.entities
        except (
            TimeoutError,
            LLMError,
            httpx.HTTPError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ):
            logger.warning(
                "Entity name extraction failed for graph search; using raw query", exc_info=True
            )
        return []

    async def _search_mem0(self, query: str, limit: int) -> list[dict[str, Any]]:
        if self._long_term is None:
            return []
        try:
            mem0_timeout = getattr(self, "_mem0_timeout_seconds", 15.0)
            external_retry = getattr(
                self,
                "_external_retry",
                RetryPolicy(max_attempts=3, base_delay_seconds=0.8),
            )
            mem0_breaker = getattr(self, "_mem0_breaker", None)

            from brain_os.brain.retrieval_context import mem0_user_id_var
            from brain_os.memory.mem0_access_tracker import get_mem0_access_tracker
            from brain_os.memory.mem0_search_cache import get_mem0_search_cache

            _uid = mem0_user_id_var.get()

            cache = get_mem0_search_cache()
            cached = await cache.get("retriever", query, _uid, limit)
            if cached is not None:
                await get_mem0_access_tracker().record_hits(
                    [r.get("metadata", {}).get("mem0_id", "") for r in cached], _uid
                )
                return cached

            async def _operation() -> Any:
                return await asyncio.wait_for(
                    self._long_term.search(query, user_id=_uid, limit=limit),
                    timeout=mem0_timeout,
                )

            memories = await run_with_retry(
                _operation,
                policy=external_retry,
                is_retryable=_is_retryable_external_error,
                circuit_breaker=mem0_breaker,
            )
            results = [
                {
                    "content": m.get("memory", m.get("text", "")),
                    "score": m.get("score", 0.5),
                    "source": "mem0",
                    # mem0_id rides along so access tracking survives caching
                    # and downstream merges that only keep metadata.
                    "metadata": {**(m.get("metadata", {}) or {}), "mem0_id": str(m.get("id", ""))},
                }
                for m in (memories if isinstance(memories, list) else [])
            ]
            await cache.put("retriever", query, _uid, limit, results)
            await get_mem0_access_tracker().record_hits(
                [r["metadata"].get("mem0_id", "") for r in results], _uid
            )
            return results
        except (
            TimeoutError,
            DatabaseError,
            httpx.HTTPError,
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ):
            logger.exception("Mem0 search failed")
            return []

    # ── retrieval logging (for graph consolidation) ────────────────────────

    async def _log_retrieval(self, query: str, results: list[dict[str, Any]]) -> None:
        try:
            from brain_os.brain.graph_consolidation import GraphConsolidation

            gc = GraphConsolidation(knowledge_graph=self._graph)
            stitch_stats: dict[str, Any] | None = None
            log_results = results
            for r in results:
                if r.get("_stitch_stats"):
                    stitch_stats = {
                        k: r[k]
                        for k in (
                            "stitch_attempted",
                            "stitch_resolved",
                            "stitch_misses",
                            "stitch_miss_rate",
                        )
                        if k in r
                    }
                    log_results = [x for x in results if not x.get("_stitch_stats")]
                    break
            source_ids = [r.get("source", r.get("id", ""))[:200] for r in log_results[:10]]
            source_types = [r.get("source_type", "") for r in log_results[:10]]
            await gc.log_retrieval(query, source_ids, source_types, stitch_stats=stitch_stats)
        except (
            TimeoutError,
            DatabaseError,
            httpx.HTTPError,
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ):
            logger.warning("Retrieval logging failed", exc_info=True)

    # ── imports fallback ───────────────────────────────────────────────────

    async def _imports_fallback(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Last-resort: search the raw imports archive via Alexandros's metadata index."""
        try:
            from brain_os.brain.imports_fallback_retriever import fallback_retrieve

            results = await fallback_retrieve(query)
            return [
                {
                    "content": r.get("content", ""),
                    "score": r.get("relevance", 0.5),
                    "source": r.get("source", ""),
                    "source_type": "imports_fallback",
                    "metadata": {
                        "filename": r.get("filename", ""),
                        "doc_type": r.get("doc_type", ""),
                    },
                }
                for r in results[:limit]
            ]
        except (
            IngestionError,
            BrainOSError,
            OSError,
            httpx.HTTPError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ):
            logger.warning("Imports fallback not available", exc_info=True)
            return []

    # ── reranking ────────────────────────────────────────────────────────

    async def _rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        limit: int,
        *,
        health: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        if self._voyage_key:
            try:
                ranked = await self._voyage_rerank(query, results, limit)
                if isinstance(health, dict):
                    health["rerank_path"] = "voyage"
                return ranked
            except (
                TimeoutError,
                BrainOSError,
                httpx.HTTPError,
                OSError,
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
            ):
                logger.warning(
                    "Voyage rerank failed — falling back to FlashRank",
                    exc_info=True,
                )

        ranked = await self._flashrank_rerank(query, results, limit)
        if isinstance(health, dict):
            if self._voyage_key:
                health["rerank_path"] = "voyage_failed_flashrank"
            else:
                health["rerank_path"] = "flashrank"
        return ranked

    _MAX_DOC_CHARS = 4000
    _MAX_RERANK_DOCS = 100

    async def _voyage_rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Rerank via the Voyage AI Rerank API."""
        filtered: list[tuple[int, str]] = []
        for i, r in enumerate(results):
            doc = r.get("content", "").strip()
            if doc:
                filtered.append((i, doc[: self._MAX_DOC_CHARS]))

        if not filtered:
            return await self._flashrank_rerank(query, results, limit)

        filtered = filtered[: self._MAX_RERANK_DOCS]
        original_indices, documents = zip(*filtered)
        external_retry = getattr(
            self,
            "_external_retry",
            RetryPolicy(max_attempts=3, base_delay_seconds=0.8),
        )
        voyage_breaker = getattr(self, "_voyage_breaker", None)

        _rrf = max(1, int(get_settings().app.retriever_rerank_candidates_factor))
        rerank_top_k = min(limit * _rrf, len(documents))

        try:
            _rerank_url = get_settings().llm_endpoints.voyage_rerank_url or _VOYAGE_RERANK_URL
        except (AttributeError, TypeError):
            _rerank_url = _VOYAGE_RERANK_URL

        async def _operation() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    _rerank_url,
                    json={
                        "query": query,
                        "documents": list(documents),
                        "model": self._voyage_rerank_model,
                        "top_k": rerank_top_k,
                    },
                    headers={
                        "Authorization": f"Bearer {self._voyage_key}",
                        "Content-Type": "application/json",
                    },
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "Voyage rerank returned %d: %s",
                        resp.status_code,
                        resp.text[:500],
                    )
                resp.raise_for_status()
                return resp.json()

        data = await run_with_retry(
            _operation,
            policy=external_retry,
            is_retryable=_is_retryable_external_error,
            circuit_breaker=voyage_breaker,
        )
        ranked_items = data.get("data", [])

        output: list[dict[str, Any]] = []
        for item in ranked_items:
            idx = item["index"]
            original = results[original_indices[idx]]
            content = original.get("content", "")
            parent = (original.get("metadata") or {}).get("parent_summary", "")
            if parent and isinstance(parent, str) and parent.strip():
                content = parent.strip() + "\n\n" + content
            output.append(
                {
                    "id": original.get("id", ""),
                    "content": content,
                    "score": item.get("relevance_score", 0.0),
                    "source": original.get("source", ""),
                    "source_type": original.get("source_type", ""),
                    "metadata": original.get("metadata", {}),
                }
            )

        output = _diversify_results(output, limit)
        output = await self._apply_access_frequency_boost(output)
        return await self._apply_learned_corrections(output)

    async def _flashrank_rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Local FlashRank fallback reranker."""
        if self._flashrank is None:
            logger.info("FlashRank unavailable — returning top results by retrieval score")
            ranked = sorted(results, key=lambda r: float(r.get("score", 0)), reverse=True)
            ranked = await self._apply_access_frequency_boost(ranked[:limit])
            return await self._apply_learned_corrections(ranked)

        passages = [
            {"id": i, "text": r.get("content", ""), "meta": r} for i, r in enumerate(results)
        ]

        try:
            flashrank_timeout = getattr(self, "_flashrank_timeout_seconds", 20.0)
            reranked = await asyncio.wait_for(
                asyncio.to_thread(
                    self._flashrank.rerank,
                    RerankRequest(query=query, passages=passages),
                ),
                timeout=flashrank_timeout,
            )
        except (
            TimeoutError,
            BrainOSError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ):
            logger.exception("FlashRank reranking failed; returning by original score")
            results.sort(key=lambda r: r.get("score", 0), reverse=True)
            ranked = await self._apply_access_frequency_boost(results[:limit])
            return await self._apply_learned_corrections(ranked)

        _rrf = max(1, int(get_settings().app.retriever_rerank_candidates_factor))
        take = min(limit * _rrf, len(reranked))
        output: list[dict[str, Any]] = []
        for item in reranked[:take]:
            meta: dict[str, Any] = item.get("meta") or item.get("metadata") or {}
            content = item.get("text", "")
            parent = (meta.get("metadata") or {}).get("parent_summary", "")
            if parent and isinstance(parent, str) and parent.strip():
                content = parent.strip() + "\n\n" + content
            output.append(
                {
                    "id": meta.get("id", ""),
                    "content": content,
                    "score": item.get("score", 0.0),
                    "source": meta.get("source", ""),
                    "source_type": meta.get("source_type", ""),
                    "metadata": meta.get("metadata", {}),
                }
            )

        output = _diversify_results(output, limit)
        output = await self._apply_access_frequency_boost(output)
        return await self._apply_learned_corrections(output)

    async def _apply_access_frequency_boost(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Blend Mem0 access-tracker usage prior into post-rerank scores (bounded)."""
        if not results:
            return results
        try:
            app = get_settings().app
            if not bool(getattr(app, "retriever_access_boost_enabled", True)):
                return results
            from brain_os.brain.retrieval_access_boost import (
                apply_access_frequency_boost,
                result_access_key,
            )
            from brain_os.memory.mem0_access_tracker import get_mem0_access_tracker

            keys = [result_access_key(r) for r in results]
            stats = await get_mem0_access_tracker().get_access_stats(keys)
            if not stats:
                return results
            return apply_access_frequency_boost(
                results,
                stats,
                weight=float(getattr(app, "retriever_access_boost_weight", 1.0)),
                max_influence=float(getattr(app, "retriever_access_boost_max_influence", 0.12)),
                half_life_days=float(getattr(app, "retriever_access_boost_half_life_days", 30.0)),
            )
        except (
            TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ):
            logger.debug("access-frequency boost skipped", exc_info=True)
            return results

    # ── learned corrections ────────────────────────────────────────────────

    async def _apply_learned_corrections(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Post-process results with Mnemon's correction ledger.

        Checks each result against the correction ledger and appends
        correction notices when stale data is detected.  Also injects
        corrected content at the top of results for matching entities.
        """
        try:
            from brain_os.agents.mnemon import _load_ledger

            ledger = await _load_ledger()
            entities = ledger.get("entities", {})
            if not entities:
                return results

            injected: list[dict[str, Any]] = []
            for r in results:
                content = r.get("content", "")
                content_lower = content.lower()
                for entity_key, entry in entities.items():
                    if entity_key not in content_lower:
                        continue
                    precedence = int(entry.get("precedence", 100))
                    correct_value = str(
                        entry.get("correct_value") or entry.get("current_status", "")
                    )
                    for stale in entry.get("stale_values", []):
                        if stale.lower() in content_lower:
                            r.setdefault("metadata", {})["mnemon_correction"] = (
                                f"{entity_key}: '{stale}' is outdated. "
                                f"Current: {correct_value[:200]}"
                            )
                            if not any(i.get("_correction_for") == entity_key for i in injected):
                                injected.append(
                                    {
                                        "content": (
                                            f"[MNEMON CORRECTION] {entity_key}: {correct_value}"
                                        ),
                                        "score": min(1.0, 0.9 + (precedence / 1000.0)),
                                        "source": "correction_ledger",
                                        "metadata": {
                                            "is_correction": True,
                                            "precedence": precedence,
                                        },
                                        "_correction_for": entity_key,
                                    }
                                )

            injected.sort(
                key=lambda item: int(item.get("metadata", {}).get("precedence", 100)),
                reverse=True,
            )
            return injected + results
        except (
            BrainOSError,
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ):
            logger.warning("Mnemon corrections overlay failed", exc_info=True)
            return results

    # ── LLM query decomposition ──────────────────────────────────────────

    async def _decompose_query(self, query: str) -> list[str]:
        try:
            result = await self._llm.generate_structured(
                _DECOMPOSE_SYSTEM_PROMPT,
                query,
                SubQueries,
                name="retriever.decompose",
                model_tier="cheap",
            )
            if result.queries:
                logger.info("Decomposed query into %d sub-queries", len(result.queries))
            return result.queries
        except (
            TimeoutError,
            LLMError,
            httpx.HTTPError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ):
            logger.exception("Query decomposition failed")
        return []

    async def _second_pass_decomposed_query(self, query: str) -> str | None:
        """Return first decomposed sub-query that differs from *query* (one LLM call)."""
        subs = await self._decompose_query(query)
        return _qr.pick_first_distinct_subquery(query, subs)
