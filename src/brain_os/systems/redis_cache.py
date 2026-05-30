"""Shared Redis client and caching helpers for Brain OS.

Provides :class:`RedisCache`, a thin async wrapper around ``redis.asyncio``
that every subsystem can use for:

* **Key-value caching** with optional TTL (LLM responses, dedup fingerprints).
* **Pub/sub** for persistent inter-agent messaging.
* **Atomic counters** for rate limiting.

The client is designed to be constructed once at startup and injected via
the service locator (``ServiceKey.REDIS``).  All operations degrade
gracefully — a missing or unreachable Redis instance logs a warning and
returns ``None`` / empty results so the rest of the system keeps running.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

from brain_os.config import RedisConfig, get_settings

logger = logging.getLogger(__name__)

_KEY_PREFIX = "brain:"


class RedisCache:
    """Async Redis client with caching, pub/sub, and rate-limit helpers."""

    def __init__(self, config: RedisConfig | None = None) -> None:
        cfg = config or get_settings().redis
        self._url = cfg.url
        self._client: aioredis.Redis | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the connection pool.  Safe to call multiple times."""
        if self._client is not None:
            return
        if not self._url:
            logger.warning("REDIS_URL not configured — Redis features disabled")
            return
        try:
            self._client = aioredis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            await self._client.ping()
            logger.info("Redis connected: %s", self._url.split("@")[-1])
        except (aioredis.RedisError, OSError) as exc:
            logger.warning("Redis connection failed (%s) — running without cache", exc)
            self._client = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("Redis connection closed")

    @property
    def available(self) -> bool:
        return self._client is not None

    # ── key-value cache ───────────────────────────────────────────────────

    async def get(self, key: str) -> str | None:
        if self._client is None:
            return None
        try:
            return await self._client.get(f"{_KEY_PREFIX}{key}")
        except aioredis.RedisError:
            logger.warning("Redis GET failed for %s", key, exc_info=True)
            return None

    async def set(
        self,
        key: str,
        value: str,
        ttl_seconds: int | None = None,
    ) -> bool:
        if self._client is None:
            return False
        try:
            await self._client.set(f"{_KEY_PREFIX}{key}", value, ex=ttl_seconds)
            return True
        except aioredis.RedisError:
            logger.warning("Redis SET failed for %s", key, exc_info=True)
            return False

    async def delete(self, key: str) -> bool:
        if self._client is None:
            return False
        try:
            await self._client.delete(f"{_KEY_PREFIX}{key}")
            return True
        except aioredis.RedisError:
            logger.warning("Redis DELETE failed for %s", key, exc_info=True)
            return False

    async def incrby(
        self,
        key: str,
        amount: int,
        *,
        ttl_seconds: int | None = None,
        ttl_only_if_unset: bool = False,
    ) -> int | None:
        """Increment a counter by *amount*. Optionally set TTL (whole-key expiry).

        When *ttl_only_if_unset* is True, ``EXPIRE`` runs only if the key
        had no TTL (``TTL == -1``), so recurring writes refresh usage but keep
        one monthly bucket from the first increment.
        """
        if self._client is None:
            return None
        full = f"{_KEY_PREFIX}{key}"
        try:
            val = await self._client.incrby(full, amount)
            if ttl_seconds is not None and ttl_seconds > 0:
                if ttl_only_if_unset:
                    t = await self._client.ttl(full)
                    if t == -1:
                        await self._client.expire(full, ttl_seconds)
                else:
                    await self._client.expire(full, ttl_seconds)
            return int(val)
        except aioredis.RedisError:
            logger.warning("Redis INCRBY failed for %s", key, exc_info=True)
            return None

    async def get_int(self, key: str) -> int | None:
        """Read an integer counter; returns ``None`` if missing or not parseable."""
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    # ── hash (for pipeline clarification store) ───────────────────────────

    async def hget(self, name: str, key: str) -> str | None:
        """Get a field from a hash. Used by pipeline for pending_clarifications."""
        if self._client is None:
            return None
        try:
            raw = await self._client.hget(name, key)
            if raw is None:
                return None
            return raw.decode("utf-8") if isinstance(raw, bytes) else raw
        except aioredis.RedisError:
            logger.warning("Redis HGET failed for %s %s", name, key, exc_info=True)
            return None

    async def hset(self, name: str, key: str, value: str) -> bool:
        """Set a field in a hash."""
        if self._client is None:
            return False
        try:
            await self._client.hset(name, key, value)
            return True
        except aioredis.RedisError:
            logger.warning("Redis HSET failed for %s %s", name, key, exc_info=True)
            return False

    async def hdel(self, name: str, key: str) -> bool:
        """Remove a field from a hash."""
        if self._client is None:
            return False
        try:
            await self._client.hdel(name, key)
            return True
        except aioredis.RedisError:
            logger.warning("Redis HDEL failed for %s %s", name, key, exc_info=True)
            return False

    # ── JSON helpers ──────────────────────────────────────────────────────

    async def get_json(self, key: str) -> Any | None:
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    async def set_json(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> bool:
        return await self.set(key, json.dumps(value, default=str), ttl_seconds)

    # ── dedup ─────────────────────────────────────────────────────────────

    async def dedup_check(self, fingerprint: str, ttl_seconds: int = 300) -> str | None:
        """Return the cached response for *fingerprint*, or ``None`` on miss."""
        return await self.get(f"dedup:{fingerprint}")

    async def dedup_store(
        self,
        fingerprint: str,
        response: str,
        ttl_seconds: int = 300,
    ) -> None:
        await self.set(f"dedup:{fingerprint}", response, ttl_seconds)

    # ── LLM response cache ───────────────────────────────────────────────

    async def get_llm_cache(self, cache_key: str) -> str | None:
        return await self.get(f"llm:{cache_key}")

    async def set_llm_cache(
        self,
        cache_key: str,
        response: str,
        ttl_seconds: int = 3600,
    ) -> None:
        await self.set(f"llm:{cache_key}", response, ttl_seconds)

    # ── pub/sub ───────────────────────────────────────────────────────────

    async def publish(self, channel: str, message: str) -> int:
        """Publish a message; returns the number of subscribers that received it."""
        if self._client is None:
            return 0
        try:
            return await self._client.publish(f"{_KEY_PREFIX}ch:{channel}", message)
        except aioredis.RedisError:
            logger.warning("Redis PUBLISH failed on %s", channel, exc_info=True)
            return 0

    def subscriber(self) -> aioredis.client.PubSub | None:
        """Return a PubSub object for subscribing, or ``None`` if unavailable."""
        if self._client is None:
            return None
        return self._client.pubsub()

    # ── rate limiting ─────────────────────────────────────────────────────

    async def rate_limit_check(
        self,
        resource: str,
        limit: int,
        window_seconds: int = 60,
    ) -> bool:
        """Return ``True`` if the request is within the rate limit."""
        if self._client is None:
            return True
        key = f"{_KEY_PREFIX}rl:{resource}"
        try:
            pipe = self._client.pipeline(transaction=True)
            pipe.incr(key)
            pipe.expire(key, window_seconds, nx=True)
            results = await pipe.execute()
            return int(results[0]) <= limit
        except aioredis.RedisError:
            logger.warning("Redis rate-limit check failed for %s", resource, exc_info=True)
            return True

    # ── health ────────────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        if self._client is None:
            return {"status": "disconnected", "available": False}
        try:
            info = await self._client.info("server")
            return {
                "status": "connected",
                "available": True,
                "redis_version": info.get("redis_version", "?"),
                "uptime_days": info.get("uptime_in_days", "?"),
            }
        except aioredis.RedisError as exc:
            return {"status": "error", "available": False, "error": str(exc)}
