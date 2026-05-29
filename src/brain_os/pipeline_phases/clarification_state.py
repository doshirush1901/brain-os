"""Clarification pending-state persistence (in-memory + optional Redis)."""

from __future__ import annotations

import json
import logging
from typing import Any

from brain_os.schemas.llm_outputs import ClarificationPayload

logger = logging.getLogger(__name__)


async def store_clarification(
    *,
    redis: Any | None,
    pending: dict[str, dict[str, Any]],
    lock: Any,
    sender_id: str,
    agent_name: str,
    original_query: str,
    payload: ClarificationPayload,
    redis_key: str,
    checkpoint: dict[str, Any] | None = None,
) -> str:
    """Persist normalized clarification state and return rendered question text."""
    question_text = "\n".join(payload.questions).strip()
    clarification_data: dict[str, Any] = {
        "agent_name": agent_name,
        "original_query": original_query,
        "clarification_question": question_text,
        "questions": list(payload.questions),
        "missing_slots": list(payload.missing_slots),
        "can_answer_partially": payload.can_answer_partially,
        "reason": payload.reason,
    }
    if checkpoint is not None:
        clarification_data["checkpoint"] = checkpoint
    async with lock:
        pending[sender_id] = clarification_data
    if redis is not None:
        try:
            await redis.hset(redis_key, sender_id, json.dumps(clarification_data))
        except Exception:
            logger.warning("Failed to persist clarification to Redis", exc_info=True)
    return question_text


async def pop_clarification(
    *,
    redis: Any | None,
    pending: dict[str, dict[str, Any]],
    lock: Any,
    sender_id: str,
    redis_key: str,
) -> dict[str, Any] | None:
    """Pop a pending clarification from memory and Redis."""
    async with lock:
        result = pending.pop(sender_id, None)
    if result is not None:
        if redis is not None:
            try:
                await redis.hdel(redis_key, sender_id)
            except Exception:
                logger.warning("Failed to remove clarification from Redis", exc_info=True)
        return result
    if redis is not None:
        try:
            raw = await redis.hget(redis_key, sender_id)
            if raw:
                await redis.hdel(redis_key, sender_id)
                return json.loads(raw)
        except Exception:
            logger.warning("Failed to load clarification from Redis", exc_info=True)
    return None
