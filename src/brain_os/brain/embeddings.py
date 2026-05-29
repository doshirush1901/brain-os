"""Primary embedding service for all vector operations in Ira.

Every component that needs to convert text into dense vectors — the Qdrant
manager, the retriever, the document ingestor, sales intelligence — goes
through :class:`EmbeddingService`.  It wraps the Voyage AI embeddings API,
handles batching, retries, and a tiered cache (in-memory LRU, SQLite on disk,
optional Redis TTL when injected) so repeated texts avoid duplicate Voyage calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from collections import OrderedDict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from brain_os.config import EmbeddingConfig, get_settings

logger = logging.getLogger(__name__)

#: Default Voyage embeddings endpoint. Phase-2 hardening: the actual URL
#: used at request time is read from ``Settings.llm_endpoints.voyage_embeddings_url``
#: so an alt-endpoint / proxy / local mock can be wired up by env var alone.
#: This constant remains for backward compatibility with anything that imports it.
_VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
_MAX_BATCH_SIZE = 128
_DEFAULT_CACHE_SIZE = 4096
_DEFAULT_CACHE_PATH = "data/brain/embedding_cache.db"
_SQLITE_TIMEOUT_S = 30.0

_MAX_RETRIES = 5
_INITIAL_BACKOFF_S = 0.5
_BACKOFF_MULTIPLIER = 2
_EMBED_REDIS_TTL = 7 * 86400  # 7 days


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _init_sqlite_cache(path: str) -> None:
    """Create cache directory and table if they do not exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=_SQLITE_TIMEOUT_S)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB,
                created_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _sqlite_get_sync(path: str, text_hash: str) -> list[float] | None:
    """Synchronous SQLite cache lookup."""
    try:
        conn = sqlite3.connect(path, timeout=_SQLITE_TIMEOUT_S)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
            row = conn.execute(
                "SELECT embedding FROM embedding_cache WHERE text_hash = ?",
                (text_hash,),
            ).fetchone()
            if row is None:
                return None
            blob = row[0]
            return json.loads(blob.decode("utf-8") if isinstance(blob, bytes) else blob)
        finally:
            conn.close()
    except (sqlite3.Error, json.JSONDecodeError) as e:
        logger.warning("SQLite cache read failed for %s: %s", text_hash[:16], e)
        return None


def _sqlite_put_sync(path: str, text_hash: str, embedding: list[float]) -> None:
    """Synchronous SQLite cache write."""
    try:
        conn = sqlite3.connect(path, timeout=_SQLITE_TIMEOUT_S)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.execute(
                """
                INSERT OR REPLACE INTO embedding_cache (text_hash, embedding, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    text_hash,
                    json.dumps(embedding).encode("utf-8"),
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning("SQLite cache write failed for %s: %s", text_hash[:16], e)


class EmbeddingService:
    """Async wrapper around the Voyage AI embeddings endpoint."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        cache_size: int = _DEFAULT_CACHE_SIZE,
        cache_path: str = _DEFAULT_CACHE_PATH,
    ) -> None:
        cfg = config or get_settings().embedding
        self._api_key = cfg.api_key.get_secret_value()
        self._model = cfg.model
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = cache_size
        self._cache_path = cache_path
        self._redis_cache: Any = None
        if self._sqlite_cache_enabled():
            _init_sqlite_cache(cache_path)

    def _sqlite_cache_enabled(self) -> bool:
        """Return whether sqlite embedding cache should be used."""
        try:
            app = get_settings().app
            return bool(getattr(app, "embedding_sqlite_cache_enabled", True))
        except (AttributeError, TypeError):
            return True

    def set_redis_cache(self, cache: Any) -> None:
        """Inject Redis for optional second-level embedding cache (7-day TTL)."""
        self._redis_cache = cache

    def _redis_embed_key(self, text_hash: str) -> str:
        return f"emb:{self._model}:{text_hash}"

    # ── public API ───────────────────────────────────────────────────────

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        input_type: str = "document",
    ) -> list[list[float]]:
        """Embed a list of texts, returning one vector per input.

        L1 (in-memory) is checked first, then L2 (SQLite). Uncached texts
        are sent to Voyage AI in batches of up to ``_MAX_BATCH_SIZE``.
        New embeddings are stored in both caches.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        l1_miss_indices: list[int] = []

        for i, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                results[i] = cached
            else:
                l1_miss_indices.append(i)

        if not l1_miss_indices:
            return results  # type: ignore[return-value]

        # L2 (Redis) lookup for L1 misses when available
        l2_redis_results: list[list[float] | None] = [None] * len(l1_miss_indices)
        if self._redis_cache and self._redis_cache.available:
            try:
                l2_redis_results = await asyncio.gather(
                    *[
                        self._redis_cache.get_json(self._redis_embed_key(_text_hash(texts[idx])))
                        for idx in l1_miss_indices
                    ]
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        after_redis_miss: list[int] = []
        for i, idx in enumerate(l1_miss_indices):
            vec = l2_redis_results[i] if i < len(l2_redis_results) else None
            if vec is not None:
                results[idx] = vec
                self._cache_put(texts[idx], vec)
            else:
                after_redis_miss.append(idx)

        # L3 (SQLite) lookup for remaining misses
        l2_results: list[list[float] | None] = []
        if self._sqlite_cache_enabled():
            l2_results = await asyncio.gather(
                *[
                    asyncio.to_thread(_sqlite_get_sync, self._cache_path, _text_hash(texts[idx]))
                    for idx in after_redis_miss
                ]
            )
        else:
            l2_results = [None] * len(after_redis_miss)
        l2_miss_indices: list[int] = []
        for idx, vec in zip(after_redis_miss, l2_results):
            if vec is not None:
                results[idx] = vec
                self._cache_put(texts[idx], vec)
            else:
                l2_miss_indices.append(idx)

        if l2_miss_indices:
            uncached_texts = [texts[i] for i in l2_miss_indices]
            vectors = await self._embed_batched(uncached_texts, input_type=input_type)
            for idx, vec in zip(l2_miss_indices, vectors):
                results[idx] = vec
                text = texts[idx]
                self._cache_put(text, vec)
            if self._sqlite_cache_enabled():
                await asyncio.gather(
                    *[
                        asyncio.to_thread(
                            _sqlite_put_sync,
                            self._cache_path,
                            _text_hash(texts[idx]),
                            vec,
                        )
                        for idx, vec in zip(l2_miss_indices, vectors)
                    ]
                )
            if self._redis_cache and self._redis_cache.available:
                try:
                    await asyncio.gather(
                        *[
                            self._redis_cache.set_json(
                                self._redis_embed_key(_text_hash(texts[idx])),
                                vec,
                                ttl_seconds=_EMBED_REDIS_TTL,
                            )
                            for idx, vec in zip(l2_miss_indices, vectors)
                        ]
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass

        return results  # type: ignore[return-value]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for retrieval."""
        cached = self._cache_get(query)
        if cached is not None:
            return cached

        th = _text_hash(query)
        if self._redis_cache and self._redis_cache.available:
            try:
                vec = await self._redis_cache.get_json(self._redis_embed_key(th))
                if vec is not None:
                    self._cache_put(query, vec)
                    return vec
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass

        if self._sqlite_cache_enabled():
            vec = await asyncio.to_thread(_sqlite_get_sync, self._cache_path, th)
            if vec is not None:
                self._cache_put(query, vec)
                return vec

        vectors = await self._call_api([query], input_type="query")
        self._cache_put(query, vectors[0])
        if self._sqlite_cache_enabled():
            await asyncio.to_thread(_sqlite_put_sync, self._cache_path, th, vectors[0])
        if self._redis_cache and self._redis_cache.available:
            try:
                await self._redis_cache.set_json(
                    self._redis_embed_key(th),
                    vectors[0],
                    ttl_seconds=_EMBED_REDIS_TTL,
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        return vectors[0]

    # ── batching ─────────────────────────────────────────────────────────

    async def _embed_batched(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
    ) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH_SIZE):
            batch = texts[start : start + _MAX_BATCH_SIZE]
            vectors = await self._call_api(batch, input_type=input_type)
            all_vectors.extend(vectors)
        return all_vectors

    # ── HTTP with retry ──────────────────────────────────────────────────

    async def _call_api(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
    ) -> list[list[float]]:
        if not self._api_key:
            raise RuntimeError("Voyage API key is missing (set VOYAGE_API_KEY).")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": list(texts),
            "model": self._model,
            "input_type": input_type,
        }

        backoff = _INITIAL_BACKOFF_S
        last_exc: Exception | None = None

        # Phase-2: read from centralised endpoints config; default preserves behaviour.
        try:
            _voyage_url = get_settings().llm_endpoints.voyage_embeddings_url or _VOYAGE_API_URL
        except (AttributeError, TypeError):
            _voyage_url = _VOYAGE_API_URL

        async with httpx.AsyncClient(timeout=60) as client:
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    resp = await client.post(_voyage_url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    return [item["embedding"] for item in data["data"]]
                except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                    last_exc = exc
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                        raise
                    logger.warning(
                        "Voyage API attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt,
                        _MAX_RETRIES,
                        exc,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= _BACKOFF_MULTIPLIER

        raise RuntimeError(f"Voyage API failed after {_MAX_RETRIES} retries") from last_exc

    # ── LRU cache (dict-based) ───────────────────────────────────────────

    def _cache_get(self, text: str) -> list[float] | None:
        key = _text_hash(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, text: str, vector: list[float]) -> None:
        key = _text_hash(text)
        self._cache[key] = vector
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
