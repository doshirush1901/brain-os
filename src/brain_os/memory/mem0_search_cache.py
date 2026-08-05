"""Short-TTL cache for Mem0 search results (in-process L1 + optional Redis L2).

Mem0 retrieval requests are metered per plan month, and a single pipeline
request can issue many near-identical searches (parallel agents, ReAct
iterations, enrichment fan-outs). This cache absorbs those repeats.

Staleness is bounded two ways:

* entries expire after ``APP__MEM0_SEARCH_CACHE_TTL_SECONDS`` (default 300s,
  ``0`` disables caching entirely);
* every successful Mem0 *write* for a user bumps that user's generation
  counter, which is part of the cache key — so "remember X" followed by
  "what do you remember about X?" sees the fresh memory immediately.

A module-level singleton is shared by every ``LongTermMemory`` instance and
the retriever fan-out. Redis (L2) is optional and injected at runtime wiring
via :func:`get_mem0_search_cache`'s ``set_redis_cache``; without it the cache
still works per-process.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any

from brain_os.config import get_settings

logger = logging.getLogger(__name__)

_L1_MAX_ENTRIES = 256
_GENERATION_TTL_SECONDS = 7 * 86_400


def _ttl_seconds() -> int:
    try:
        return max(0, int(getattr(get_settings().app, "mem0_search_cache_ttl_seconds", 300)))
    except (AttributeError, TypeError, ValueError):
        return 300


class Mem0SearchCache:
    """L1 (process LRU) + optional L2 (Redis) cache for Mem0 search calls."""

    def __init__(self) -> None:
        self._l1: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()
        self._generations: dict[str, int] = {}
        self._redis: Any = None

    def set_redis_cache(self, cache: Any) -> None:
        """Inject the shared :class:`RedisCache` so hits survive across processes."""
        self._redis = cache

    def _redis_available(self) -> bool:
        return self._redis is not None and bool(getattr(self._redis, "available", False))

    # ── write invalidation via per-user generations ───────────────────────

    async def _generation(self, user_id: str) -> int:
        if self._redis_available():
            try:
                val = await self._redis.get_int(f"mem0:gen:{user_id}")
                if val is not None:
                    return val
            except (OSError, TypeError, ValueError):
                logger.debug("Mem0 cache generation read failed", exc_info=True)
        return self._generations.get(user_id, 0)

    async def bump_generation(self, user_id: str) -> None:
        """Invalidate cached searches for *user_id* after a successful write."""
        self._generations[user_id] = self._generations.get(user_id, 0) + 1
        if self._redis_available():
            try:
                await self._redis.incrby(
                    f"mem0:gen:{user_id}", 1, ttl_seconds=_GENERATION_TTL_SECONDS
                )
            except (OSError, TypeError, ValueError):
                logger.debug("Mem0 cache generation bump failed", exc_info=True)

    # ── cache operations ──────────────────────────────────────────────────

    async def _key(self, namespace: str, query: str, user_id: str, limit: int) -> str:
        gen = await self._generation(user_id)
        digest = hashlib.sha256(
            f"{namespace}|{user_id}|{limit}|{(query or '').strip().lower()}".encode()
        ).hexdigest()[:32]
        return f"mem0:search:{gen}:{digest}"

    def _l1_put(self, key: str, results: list[dict[str, Any]], ttl: int) -> None:
        self._l1[key] = (time.monotonic() + ttl, results)
        self._l1.move_to_end(key)
        while len(self._l1) > _L1_MAX_ENTRIES:
            self._l1.popitem(last=False)

    async def get(
        self,
        namespace: str,
        query: str,
        user_id: str,
        limit: int,
    ) -> list[dict[str, Any]] | None:
        """Return cached results (top-level copies) or ``None`` on miss."""
        if _ttl_seconds() <= 0:
            return None
        key = await self._key(namespace, query, user_id, limit)

        entry = self._l1.get(key)
        if entry is not None:
            expires_at, rows = entry
            if expires_at > time.monotonic():
                self._l1.move_to_end(key)
                return [dict(r) for r in rows]
            self._l1.pop(key, None)

        if self._redis_available():
            try:
                data = await self._redis.get_json(key)
            except (OSError, TypeError, ValueError):
                logger.debug("Mem0 cache Redis read failed", exc_info=True)
                data = None
            if isinstance(data, list):
                rows = [r for r in data if isinstance(r, dict)]
                self._l1_put(key, rows, _ttl_seconds())
                return [dict(r) for r in rows]
        return None

    async def put(
        self,
        namespace: str,
        query: str,
        user_id: str,
        limit: int,
        results: list[dict[str, Any]],
    ) -> None:
        ttl = _ttl_seconds()
        if ttl <= 0:
            return
        key = await self._key(namespace, query, user_id, limit)
        rows = [dict(r) for r in results if isinstance(r, dict)]
        self._l1_put(key, rows, ttl)
        if self._redis_available():
            try:
                await self._redis.set_json(key, rows, ttl_seconds=ttl)
            except (OSError, TypeError, ValueError):
                logger.debug("Mem0 cache Redis write failed", exc_info=True)

    def clear_local(self) -> None:
        """Drop the in-process layer (tests / operator hygiene)."""
        self._l1.clear()
        self._generations.clear()


_cache: Mem0SearchCache | None = None


def get_mem0_search_cache() -> Mem0SearchCache:
    global _cache
    if _cache is None:
        _cache = Mem0SearchCache()
    return _cache
