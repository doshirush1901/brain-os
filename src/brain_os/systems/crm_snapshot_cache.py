"""Short-TTL in-process cache for expensive CRM read-mostly snapshots."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()
_store: dict[str, tuple[float, dict[str, Any]]] = {}


def _cache_key(filters: dict[str, Any] | None) -> str:
    try:
        return json.dumps(filters or {}, sort_keys=True, default=str)
    except TypeError:
        return str(hash(str(filters)))


async def get_pipeline_summary_cached(
    crm: Any,
    *,
    filters: dict[str, Any] | None,
    ttl_seconds: float,
) -> dict[str, Any]:
    """Return ``crm.get_pipeline_summary(filters)`` with optional in-process TTL.

    *ttl_seconds* <= 0 disables caching (always hits Postgres).
    """
    if ttl_seconds <= 0:
        return await crm.get_pipeline_summary(filters)

    key = _cache_key(filters)
    now = time.monotonic()
    async with _lock:
        hit = _store.get(key)
        if hit is not None and now - hit[0] < ttl_seconds:
            logger.debug("crm_snapshot_cache hit key=%s", key[:80])
            return dict(hit[1])

    data = await crm.get_pipeline_summary(filters)
    async with _lock:
        _store[key] = (time.monotonic(), dict(data))
    return data


def clear_pipeline_summary_cache_for_tests() -> None:
    """Test helper: drop all cached pipeline rows."""
    _store.clear()
