"""Primary embedding service for all vector operations in Brain OS.

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

#: Module write counter for opportunistic prune cadence.
_sqlite_write_count = 0


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _ensure_last_accessed_column(conn: sqlite3.Connection) -> None:
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(embedding_cache)").fetchall()}
    if "last_accessed" not in cols:
        conn.execute("ALTER TABLE embedding_cache ADD COLUMN last_accessed TEXT")
        conn.execute(
            "UPDATE embedding_cache SET last_accessed = created_at "
            "WHERE last_accessed IS NULL OR last_accessed = ''"
        )


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
                created_at TEXT,
                last_accessed TEXT
            )
            """
        )
        _ensure_last_accessed_column(conn)
        conn.commit()
    finally:
        conn.close()


def _sqlite_get_sync(path: str, text_hash: str) -> list[float] | None:
    """Synchronous SQLite cache lookup (touches ``last_accessed`` on hit)."""
    try:
        conn = sqlite3.connect(path, timeout=_SQLITE_TIMEOUT_S)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
            _ensure_last_accessed_column(conn)
            row = conn.execute(
                "SELECT embedding FROM embedding_cache WHERE text_hash = ?",
                (text_hash,),
            ).fetchone()
            if row is None:
                return None
            now = datetime.now(UTC).isoformat()
            conn.execute(
                "UPDATE embedding_cache SET last_accessed = ? WHERE text_hash = ?",
                (now, text_hash),
            )
            conn.commit()
            blob = row[0]
            return json.loads(blob.decode("utf-8") if isinstance(blob, bytes) else blob)
        finally:
            conn.close()
    except (sqlite3.Error, json.JSONDecodeError) as e:
        logger.warning("SQLite cache read failed for %s: %s", text_hash[:16], e)
        return None


def _sqlite_cache_file_bytes(path: str | Path) -> int:
    p = Path(path)
    try:
        return int(p.stat().st_size) if p.is_file() else 0
    except OSError:
        return 0


def _embedding_cache_limits() -> tuple[int, int, int, int]:
    """Return (max_bytes, max_rows, retention_days, prune_every_n)."""
    try:
        app = get_settings().app
        max_bytes = int(getattr(app, "embedding_sqlite_cache_max_bytes", 300 * 1024 * 1024))
        max_rows = int(getattr(app, "embedding_sqlite_cache_max_rows", 100_000))
        retention_days = int(getattr(app, "embedding_sqlite_cache_retention_days", 30))
        every_n = int(getattr(app, "embedding_sqlite_cache_prune_every_n_writes", 1_000))
        return max_bytes, max_rows, retention_days, every_n
    except (AttributeError, TypeError, ValueError):
        return 300 * 1024 * 1024, 100_000, 30, 1_000


def prune_sqlite_cache(
    path: str | Path,
    *,
    max_bytes: int | None = None,
    max_rows: int | None = None,
    retention_days: int | None = None,
    vacuum: bool = True,
) -> dict[str, Any]:
    """LRU / retention / row-cap prune for the SQLite embedding cache.

    Eviction order: oldest ``COALESCE(last_accessed, created_at)`` first.
    When rows are deleted and ``vacuum`` is True, runs ``VACUUM`` so file size
    shrinks on disk.
    """
    cache_path = Path(path)
    cfg_max_bytes, cfg_max_rows, cfg_retention, _ = _embedding_cache_limits()
    budget = int(max_bytes if max_bytes is not None else cfg_max_bytes)
    row_cap = int(max_rows if max_rows is not None else cfg_max_rows)
    retain_days = int(retention_days if retention_days is not None else cfg_retention)

    before_bytes = _sqlite_cache_file_bytes(cache_path)
    result: dict[str, Any] = {
        "path": str(cache_path),
        "bytes_before": before_bytes,
        "bytes_after": before_bytes,
        "rows_before": 0,
        "rows_after": 0,
        "deleted_retention": 0,
        "deleted_row_cap": 0,
        "deleted_byte_cap": 0,
        "vacuumed": False,
        "status": "missing",
        "max_bytes": budget,
        "max_rows": row_cap,
    }
    if not cache_path.is_file():
        return result

    _init_sqlite_cache(str(cache_path))
    deleted_total = 0
    try:
        conn = sqlite3.connect(str(cache_path), timeout=_SQLITE_TIMEOUT_S)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
            _ensure_last_accessed_column(conn)
            rows_before = int(conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0])
            result["rows_before"] = rows_before

            if retain_days > 0:
                cutoff = datetime.now(UTC).timestamp() - (retain_days * 86400)
                # ISO timestamps sort lexicographically for UTC Zulu; use string cutoff.
                cutoff_iso = datetime.fromtimestamp(cutoff, tz=UTC).isoformat()
                cur = conn.execute(
                    """
                    DELETE FROM embedding_cache
                    WHERE COALESCE(last_accessed, created_at) < ?
                    """,
                    (cutoff_iso,),
                )
                result["deleted_retention"] = int(cur.rowcount or 0)
                deleted_total += result["deleted_retention"]

            row_count = int(conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0])
            if row_count > row_cap:
                overflow = row_count - row_cap
                cur = conn.execute(
                    """
                    DELETE FROM embedding_cache WHERE text_hash IN (
                        SELECT text_hash FROM embedding_cache
                        ORDER BY COALESCE(last_accessed, created_at) ASC
                        LIMIT ?
                    )
                    """,
                    (overflow,),
                )
                result["deleted_row_cap"] = int(cur.rowcount or 0)
                deleted_total += result["deleted_row_cap"]

            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        result["status"] = "error"
        result["error"] = str(exc)[:200]
        logger.warning("Embedding cache prune failed for %s: %s", cache_path, exc)
        return result

    # Byte-cap loop: delete batches of oldest rows until under budget (or empty).
    # File size only shrinks after VACUUM, so estimate pressure via main file size
    # between vacuum passes.
    try:
        while _sqlite_cache_file_bytes(cache_path) > budget:
            conn = sqlite3.connect(str(cache_path), timeout=_SQLITE_TIMEOUT_S)
            try:
                conn.execute("PRAGMA busy_timeout=30000;")
                remaining = int(conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0])
                if remaining <= 0:
                    break
                # Delete ~5% of rows per pass (min 1, max 500) then vacuum.
                batch = max(1, min(500, remaining // 20 or 1))
                cur = conn.execute(
                    """
                    DELETE FROM embedding_cache WHERE text_hash IN (
                        SELECT text_hash FROM embedding_cache
                        ORDER BY COALESCE(last_accessed, created_at) ASC
                        LIMIT ?
                    )
                    """,
                    (batch,),
                )
                n = int(cur.rowcount or 0)
                result["deleted_byte_cap"] += n
                deleted_total += n
                conn.commit()
            finally:
                conn.close()
            if vacuum:
                _vacuum_sqlite(cache_path)
                result["vacuumed"] = True
            if n == 0:
                break
    except sqlite3.Error as exc:
        result["status"] = "error"
        result["error"] = str(exc)[:200]
        logger.warning("Embedding cache byte-cap prune failed for %s: %s", cache_path, exc)
        return result

    if deleted_total > 0 and vacuum and not result["vacuumed"]:
        _vacuum_sqlite(cache_path)
        result["vacuumed"] = True

    try:
        conn = sqlite3.connect(str(cache_path), timeout=_SQLITE_TIMEOUT_S)
        try:
            result["rows_after"] = int(
                conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
            )
        finally:
            conn.close()
    except sqlite3.Error:
        pass

    result["bytes_after"] = _sqlite_cache_file_bytes(cache_path)
    if deleted_total == 0 and result["bytes_after"] <= budget:
        result["status"] = "ok_under_budget"
    elif result["bytes_after"] <= budget:
        result["status"] = "ok_pruned"
    else:
        result["status"] = "ok_still_over_budget"
    return result


def _vacuum_sqlite(path: Path) -> None:
    try:
        conn = sqlite3.connect(str(path), timeout=_SQLITE_TIMEOUT_S)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.execute("VACUUM;")
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("VACUUM failed for %s: %s", path, exc)


def prune_embedding_cache_to_budget(
    path: str | Path | None = None,
    *,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Prune the default (or given) embedding cache to the configured byte budget."""
    if path is None:
        try:
            from brain_os.systems.data_dir_lock import get_data_dir

            cache_path: Path = get_data_dir() / "brain" / "embedding_cache.db"
        except Exception:
            cache_path = Path(_DEFAULT_CACHE_PATH)
    else:
        cache_path = Path(path)
    return prune_sqlite_cache(cache_path, max_bytes=max_bytes)


def _maybe_prune_after_write(path: str) -> None:
    """Opportunistic prune every N sqlite writes."""
    global _sqlite_write_count
    _sqlite_write_count += 1
    _, _, _, every_n = _embedding_cache_limits()
    if every_n <= 0 or (_sqlite_write_count % every_n) != 0:
        return
    try:
        prune_sqlite_cache(path)
    except Exception:
        logger.warning("Opportunistic embedding cache prune failed", exc_info=True)


def _sqlite_put_sync(path: str, text_hash: str, embedding: list[float]) -> None:
    """Synchronous SQLite cache write."""
    try:
        conn = sqlite3.connect(path, timeout=_SQLITE_TIMEOUT_S)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
            _ensure_last_accessed_column(conn)
            now = datetime.now(UTC).isoformat()
            conn.execute(
                """
                INSERT OR REPLACE INTO embedding_cache
                    (text_hash, embedding, created_at, last_accessed)
                VALUES (?, ?, ?, ?)
                """,
                (
                    text_hash,
                    json.dumps(embedding).encode("utf-8"),
                    now,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        _maybe_prune_after_write(path)
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

    @property
    def model(self) -> str:
        return self._model

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
