"""Hybrid search over the imports metadata index.

Last-resort retrieval path: when Qdrant, Mem0, and Neo4j come up empty,
Clio (or any agent) falls back here.  Uses keyword scoring from the
metadata index plus Voyage semantic embeddings over file summaries,
merged via reciprocal rank fusion (RRF).

Files accessed via fallback are queued for proper deep ingestion during
the next sleep cycle so the knowledge is available instantly next time.

Design choices:
  - No per-file LLM call on retrieval: Athena already has conversation
    context, so raw extracted text is returned directly.
  - Hybrid search: keyword overlap catches exact matches (model numbers,
    company names); Voyage embeddings catch semantic matches.
  - Atomic queue writes prevent corruption on crash.
  - Summary embedding cache is rebuilt only when the metadata index changes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from brain_os.brain.document_ingestor import (
    read_csv,
    read_docx,
    read_pdf,
    read_pptx,
    read_txt,
    read_xls,
    read_xlsx,
)
from brain_os.brain.imports_intents import normalize_intent_tags
from brain_os.brain.imports_metadata_index import (
    _MACHINE_MODEL_RE,
    load_index,
    search_index,
)
from brain_os.config import get_settings
from brain_os.exceptions import IngestionError, IraError, LLMError, PathTraversalError

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFERRED_QUEUE_PATH = _PROJECT_ROOT / "data" / "brain" / "deferred_ingestion_queue.jsonl"
EMBEDDING_CACHE_PATH = _PROJECT_ROOT / "data" / "brain" / "imports_summary_embeddings.json"

_MAX_CANDIDATES = 3
_MAX_EXTRACT_CHARS = 6000
_MIN_SEMANTIC_SCORE = 0.35
_RRF_K = 60

_READERS = {
    ".pdf": read_pdf,
    ".xlsx": read_xlsx,
    ".xls": read_xls,
    ".docx": read_docx,
    ".txt": read_txt,
    ".csv": read_csv,
    ".pptx": read_pptx,
}

_embedding_cache: dict[str, list[float]] | None = None
_embedding_cache_version: str | None = None


# ── Voyage embedding helpers ─────────────────────────────────────────────


def _get_voyage_key() -> str:
    return get_settings().embedding.api_key.get_secret_value()


async def _embed_texts_voyage(
    texts: list[str],
    *,
    input_type: str = "document",
) -> list[list[float]]:
    """Embed texts via the Voyage API (async, batched)."""
    import httpx

    api_key = _get_voyage_key()
    if not api_key:
        return []

    settings = get_settings()
    model = settings.embedding.model
    # Phase-2 hardening: use centralised endpoint config; default preserves behaviour.
    voyage_url = (
        settings.llm_endpoints.voyage_embeddings_url or "https://api.voyageai.com/v1/embeddings"
    )
    all_embeddings: list[list[float]] = []
    batch_size = 64

    async with httpx.AsyncClient(timeout=60) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            resp = await client.post(
                voyage_url,
                json={"input": batch, "model": model, "input_type": input_type},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            all_embeddings.extend(item["embedding"] for item in data["data"])

    return all_embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── embedding cache ──────────────────────────────────────────────────────


async def _load_embedding_cache() -> dict[str, list[float]]:
    global _embedding_cache, _embedding_cache_version

    index = await load_index()
    index_version = index.get("built_at", "")

    if _embedding_cache is not None and _embedding_cache_version == index_version:
        return _embedding_cache

    if EMBEDDING_CACHE_PATH.exists():
        try:
            raw = await asyncio.to_thread(EMBEDDING_CACHE_PATH.read_text)
            cached = json.loads(raw)
            if cached.get("version") == index_version:
                _embedding_cache = cached.get("embeddings", {})
                _embedding_cache_version = index_version
                return _embedding_cache
        except (OSError, json.JSONDecodeError):
            pass

    _embedding_cache = {}
    _embedding_cache_version = index_version
    return _embedding_cache


async def _build_summary_embeddings(index: dict[str, Any]) -> dict[str, list[float]]:
    """Batch-embed all metadata summaries and persist to disk."""
    files = index.get("files", {})
    to_embed: dict[str, str] = {}
    for rel_path, meta in files.items():
        summary = meta.get("summary", "")
        entities = ", ".join(meta.get("entities", []))
        machines = ", ".join(meta.get("machines", []))
        topics = ", ".join(meta.get("topics", []))
        keywords = ", ".join(meta.get("keywords", []))
        intents = ", ".join(meta.get("intent_tags", []))
        counterparty = meta.get("counterparty_type", "unknown")
        role = meta.get("document_role", "other")
        text = (
            f"{meta.get('name', '')}. {summary}. "
            f"{entities}. {machines}. {topics}. {keywords}. "
            f"Intents: {intents}. Counterparty: {counterparty}. Role: {role}."
        ).strip()
        if len(text) > 10:
            to_embed[rel_path] = text

    if not to_embed:
        return {}

    paths = list(to_embed.keys())
    texts = [to_embed[p] for p in paths]

    try:
        vectors = await _embed_texts_voyage(texts, input_type="document")
    except (
        TimeoutError,
        LLMError,
        httpx.HTTPError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
    ) as exc:
        logger.warning("Failed to build summary embeddings: %s", exc)
        return {}

    embeddings = dict(zip(paths, vectors))

    global _embedding_cache, _embedding_cache_version
    _embedding_cache = embeddings
    _embedding_cache_version = index.get("built_at", "")

    try:
        EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            EMBEDDING_CACHE_PATH.write_text,
            json.dumps(
                {
                    "version": _embedding_cache_version,
                    "count": len(embeddings),
                    "built_at": datetime.now(UTC).isoformat(),
                    "embeddings": embeddings,
                }
            ),
        )
        logger.info("Built and cached %d summary embeddings", len(embeddings))
    except (
        IraError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
    ) as exc:
        logger.warning("Failed to save embedding cache: %s", exc)

    return embeddings


# ── semantic search ──────────────────────────────────────────────────────


async def _semantic_search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Voyage embedding-based semantic search over metadata summaries."""
    if not _get_voyage_key():
        return []

    index = await load_index()
    embeddings = await _load_embedding_cache()

    if not embeddings:
        logger.info("Building summary embeddings for semantic fallback search...")
        embeddings = await _build_summary_embeddings(index)
        if not embeddings:
            return []

    query_vectors = await _embed_texts_voyage([query], input_type="query")
    if not query_vectors:
        return []
    query_emb = query_vectors[0]

    scored: list[dict[str, Any]] = []
    files = index.get("files", {})
    for rel_path, emb in embeddings.items():
        sim = _cosine_similarity(query_emb, emb)
        if sim >= _MIN_SEMANTIC_SCORE and rel_path in files:
            meta = files[rel_path]
            scored.append(
                {
                    "path": meta.get("path", ""),
                    "name": meta.get("name", ""),
                    "score": round(sim, 4),
                    "summary": meta.get("summary", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "machines": meta.get("machines", []),
                    "topics": meta.get("topics", []),
                    "intent_tags": meta.get("intent_tags", []),
                    "counterparty_type": meta.get("counterparty_type", "unknown"),
                    "document_role": meta.get("document_role", "other"),
                    "search_type": "semantic",
                }
            )

    scored.sort(key=lambda x: -x["score"])
    return scored[:limit]


# ── hybrid search (keyword + semantic, RRF merge) ───────────────────────


async def hybrid_search(
    query: str,
    limit: int = _MAX_CANDIDATES,
    doc_type_filter: str = "",
    intent_filters: list[str] | None = None,
    counterparty_filter: str = "",
    role_filter: str = "",
) -> list[dict[str, Any]]:
    """Merge keyword and semantic results via weighted reciprocal rank fusion.

    When the query contains a machine model number, keyword results get
    1.5x weight (exact model matching matters more).  Otherwise semantic
    results get 1.5x weight (meaning-based matching is more useful).
    """
    intent_filters = normalize_intent_tags(intent_filters)
    kw_results = await search_index(
        query,
        limit=limit * 2,
        doc_type_filter=doc_type_filter,
        intent_filters=intent_filters,
        counterparty_filter=counterparty_filter,
        role_filter=role_filter,
    )
    sem_results = await _semantic_search(query, limit=limit * 2)

    if doc_type_filter and sem_results:
        sem_results = [r for r in sem_results if r.get("doc_type") == doc_type_filter]
    if intent_filters and sem_results:
        sem_results = [
            r
            for r in sem_results
            if any(tag in normalize_intent_tags(r.get("intent_tags", [])) for tag in intent_filters)
        ]
    if counterparty_filter and sem_results:
        sem_results = [
            r for r in sem_results if r.get("counterparty_type", "unknown") == counterparty_filter
        ]
    if role_filter and sem_results:
        sem_results = [r for r in sem_results if r.get("document_role", "other") == role_filter]

    has_model = bool(_MACHINE_MODEL_RE.search(query))
    kw_weight = 1.5 if has_model else 1.0
    sem_weight = 1.0 if has_model else 1.5

    rrf_scores: dict[str, float] = {}
    result_map: dict[str, dict[str, Any]] = {}

    for rank, r in enumerate(kw_results):
        path = r["path"]
        rrf_scores[path] = rrf_scores.get(path, 0) + kw_weight / (_RRF_K + rank + 1)
        result_map.setdefault(path, r)

    for rank, r in enumerate(sem_results):
        path = r["path"]
        rrf_scores[path] = rrf_scores.get(path, 0) + sem_weight / (_RRF_K + rank + 1)
        result_map.setdefault(path, r)

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])

    results: list[dict[str, Any]] = []
    for path, rrf in ranked[:limit]:
        entry = result_map[path].copy()
        entry["rrf_score"] = round(rrf, 6)
        results.append(entry)

    if not results and kw_results:
        results = kw_results[:limit]

    return results


# ── text extraction ──────────────────────────────────────────────────────


async def extract_file_text(filepath: str | Path, max_chars: int = _MAX_EXTRACT_CHARS) -> str:
    """Extract text from a supported file, truncated to *max_chars*."""
    fp = Path(filepath)
    data_root = _PROJECT_ROOT / "data"
    if not fp.resolve().is_relative_to(data_root):
        raise PathTraversalError(f"Path {filepath} is outside the data directory")

    reader = _READERS.get(fp.suffix.lower())
    if reader is None:
        if fp.suffix.lower() in (".txt", ".json", ".csv", ".md"):
            try:
                raw = await asyncio.to_thread(
                    lambda: fp.read_text(errors="ignore"),
                )
                return raw[:max_chars]
            except (IraError, OSError, ValueError, TypeError, UnicodeError):
                logger.debug("Plain-text read failed for %s", fp.name)
                return ""
        return ""

    try:
        text = await asyncio.to_thread(reader, fp)
        return text[:max_chars]
    except (IngestionError, OSError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
        logger.warning("Text extraction failed for %s: %s", fp.name, exc)
        return ""


# ── deferred ingestion queue ─────────────────────────────────────────────


async def queue_for_deferred_ingestion(
    filepath: str,
    filename: str,
    query: str,
    doc_type: str,
) -> None:
    """Queue a file for proper ingestion during the next sleep cycle.

    Uses atomic write (temp file + rename) to prevent corruption.
    """

    def _write() -> None:
        import fcntl

        DEFERRED_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = json.dumps(
            {
                "filepath": filepath,
                "filename": filename,
                "query_that_triggered": query,
                "doc_type": doc_type,
                "queued_at": datetime.now(UTC).isoformat(),
                "status": "pending",
            }
        )

        lock_path = DEFERRED_QUEUE_PATH.with_suffix(".lock")
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                existing = ""
                if DEFERRED_QUEUE_PATH.exists():
                    existing = DEFERRED_QUEUE_PATH.read_text()

                fd, tmp_path = tempfile.mkstemp(
                    dir=str(DEFERRED_QUEUE_PATH.parent),
                    suffix=".tmp",
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        if existing:
                            f.write(existing)
                            if not existing.endswith("\n"):
                                f.write("\n")
                        f.write(entry + "\n")
                    os.replace(tmp_path, str(DEFERRED_QUEUE_PATH))
                except (IraError, OSError, ValueError, TypeError, AttributeError):
                    os.unlink(tmp_path)
                    raise
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    try:
        await asyncio.to_thread(_write)
        logger.info("Queued %s for deferred ingestion", filename)
    except (IngestionError, OSError, ValueError, TypeError, AttributeError, IraError) as exc:
        logger.warning("Failed to queue %s: %s", filename, exc)


async def load_deferred_queue() -> list[dict[str, Any]]:
    """Load all pending entries from the deferred ingestion queue."""
    if not DEFERRED_QUEUE_PATH.exists():
        return []
    raw = await asyncio.to_thread(DEFERRED_QUEUE_PATH.read_text)
    entries: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if entry.get("status") == "pending":
                entries.append(entry)
        except json.JSONDecodeError:
            continue
    return entries


async def mark_deferred_ingested(filepath: str) -> None:
    """Mark a deferred queue entry as ingested (atomic rewrite)."""

    def _rewrite() -> None:
        if not DEFERRED_QUEUE_PATH.exists():
            return
        lines = DEFERRED_QUEUE_PATH.read_text().splitlines()
        updated: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("filepath") == filepath and entry.get("status") == "pending":
                    entry["status"] = "ingested"
                    entry["ingested_at"] = datetime.now(UTC).isoformat()
                updated.append(json.dumps(entry))
            except json.JSONDecodeError:
                updated.append(line)

        content = "\n".join(updated) + "\n"
        fd, tmp_path = tempfile.mkstemp(
            dir=str(DEFERRED_QUEUE_PATH.parent),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.replace(tmp_path, str(DEFERRED_QUEUE_PATH))
        except (IraError, OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    await asyncio.to_thread(_rewrite)


# ── main entry point ─────────────────────────────────────────────────────


async def fallback_retrieve(query: str) -> list[dict[str, Any]]:
    """Last-resort retrieval: hybrid-search the metadata index, extract raw
    text from the best matching files, and return it directly.

    Returns a list of result dicts compatible with the retriever format.
    Also queues accessed files for proper ingestion during the next sleep cycle.
    """
    start = time.time()
    candidates = await hybrid_search(query)

    if not candidates:
        logger.debug("Imports fallback: no candidates for '%s'", query[:80])
        return []

    logger.info(
        "Imports fallback: %d candidates for '%s' (top: %s, rrf=%.4f)",
        len(candidates),
        query[:60],
        candidates[0]["name"],
        candidates[0].get("rrf_score", 0),
    )

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        filepath = candidate.get("path", "")
        filename = candidate.get("name", "")
        if not filepath or not Path(filepath).exists():
            continue

        text = await extract_file_text(filepath)
        if not text or len(text.strip()) < 30:
            continue

        header = (
            f"[Document: {filename} | Type: {candidate.get('doc_type', 'unknown')} | "
            f"Summary: {candidate.get('summary', 'N/A')[:150]}]\n\n"
        )

        results.append(
            {
                "source": f"Raw Document ({filename})",
                "type": "imports_fallback",
                "content": header + text,
                "relevance": min(0.65 + candidate.get("rrf_score", 0) * 15, 0.88),
                "filename": filename,
                "doc_type": candidate.get("doc_type", "other"),
            }
        )

        await queue_for_deferred_ingestion(
            filepath,
            filename,
            query,
            candidate.get("doc_type", "other"),
        )

    elapsed = time.time() - start
    logger.info("Imports fallback: returned %d results in %.1fs", len(results), elapsed)
    return results
