"""Monthly LLM usage budget (Redis-backed, Paperclip-style cost gate).

This is **best-effort**: usage is estimated per completed pipeline turn from
visible I/O size and agent count. For exact token accounting, use provider
dashboards or Langfuse; this gate blocks runaway months when Redis is up.

Enable with ``APP__LLM_MONTHLY_TOKEN_BUDGET`` > 0 and ``REDIS_URL`` set.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain_os.systems.redis_cache import RedisCache

logger = logging.getLogger(__name__)

_BUDGET_KEY_TTL_SECONDS = 35 * 24 * 3600  # roll keys naturally after ~35 days


def _year_month_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def budget_counter_key(*, scope: str, bucket: str) -> str:
    """Redis key fragment (without global ``ira:`` prefix — :class:`RedisCache` adds it)."""
    ym = _year_month_utc()
    safe_bucket = bucket.replace(":", "_")[:200]
    return f"llm_budget:{ym}:{scope}:{safe_bucket}"


def estimate_turn_tokens(
    raw_input: str,
    response: str,
    agents_used: list[str],
    *,
    per_agent_floor: int = 12_000,
    base_floor: int = 800,
) -> int:
    """Heuristic tokens for one pipeline completion (structured calls included)."""
    if not agents_used:
        return 0
    # Rough chars→tokens; multi-agent paths dominate via per-agent floor
    body = max(base_floor, (len(raw_input) + len(response)) // 3)
    n = len([a for a in agents_used if a and a not in ("timeout", "dedup")])
    agent_component = n * per_agent_floor
    return min(body + agent_component, 2_000_000)


async def get_monthly_usage(redis: RedisCache, *, scope: str, bucket: str) -> int:
    key = budget_counter_key(scope=scope, bucket=bucket)
    val = await redis.get_int(key)
    return int(val or 0)


async def add_monthly_usage(
    redis: RedisCache,
    *,
    scope: str,
    bucket: str,
    tokens: int,
) -> None:
    if tokens <= 0:
        return
    key = budget_counter_key(scope=scope, bucket=bucket)
    result = await redis.incrby(
        key,
        tokens,
        ttl_seconds=_BUDGET_KEY_TTL_SECONDS,
        ttl_only_if_unset=True,
    )
    if result is not None:
        logger.debug("LLM budget | +%s tokens → %s (key tail %s)", tokens, result, key[-40:])


def budget_block_message(*, used: int, limit: int, bucket: str) -> str:
    return (
        "**Monthly LLM budget reached** for this period. "
        f"Estimated usage is about **{used:,}** / **{limit:,}** tokens (heuristic; bucket `{bucket}`). "
        "Raise ``APP__LLM_MONTHLY_TOKEN_BUDGET``, switch scope, or wait for the next UTC month bucket."
    )


async def check_budget_allows(
    redis: RedisCache | None,
    *,
    limit: int,
    scope: str,
    bucket: str,
) -> str | None:
    """Return user-facing error markdown if over budget; else ``None``."""
    if limit <= 0 or redis is None or not redis.available:
        return None
    used = await get_monthly_usage(redis, scope=scope, bucket=bucket)
    if used >= limit:
        logger.warning("LLM budget block | used=%s limit=%s bucket=%s", used, limit, bucket)
        return budget_block_message(used=used, limit=limit, bucket=bucket)
    return None


def resolve_budget_bucket(sender_id: str, *, scope_mode: str) -> tuple[str, str]:
    """Return ``(redis_scope, bucket_id)`` for counter keys."""
    mode = (scope_mode or "user").strip().lower()
    if mode == "global":
        return "global", "all"
    return "user", (sender_id or "anonymous").strip() or "anonymous"
