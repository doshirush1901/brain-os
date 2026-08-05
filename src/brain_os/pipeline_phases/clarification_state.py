"""Clarification pending-state persistence (in-memory + optional Redis)."""

from __future__ import annotations

import json
import logging
from typing import Any

from brain_os.schemas.llm_outputs import ClarificationPayload

logger = logging.getLogger(__name__)


def _gate_field(gate_id: str) -> str:
    from brain_os.services.socratic_gate import gate_redis_field

    return gate_redis_field(gate_id)


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
    extra: dict[str, Any] | None = None,
) -> str:
    """Persist normalized clarification state and return rendered question text.

    Dual-key: ``sender_id`` (latest for that operator) and ``gate:{gate_id}``
    when a Socratic ``gate_id`` is present — so MCP/CLI can resume a specific
    turn without colliding across tabs.
    """
    from datetime import UTC, datetime

    question_text = "\n".join(payload.questions).strip()
    clarification_data: dict[str, Any] = {
        "agent_name": agent_name,
        "original_query": original_query,
        "clarification_question": question_text,
        "questions": list(payload.questions),
        "missing_slots": list(payload.missing_slots),
        "can_answer_partially": payload.can_answer_partially,
        "reason": payload.reason,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "pending",
        "sender_id": sender_id,
    }
    if checkpoint is not None:
        clarification_data["checkpoint"] = checkpoint
    if extra:
        for key, value in extra.items():
            if key not in clarification_data or key in {
                "socratic",
                "socratic_defaults",
                "socratic_questions",
                "socratic_slots",
                "socratic_score",
                "session_id",
                "conversation_id",
                "channel",
                "created_at",
                "status",
                "gate_id",
                "sender_id",
            }:
                clarification_data[key] = value

    gate_id = str(clarification_data.get("gate_id") or "").strip().lower()
    if gate_id:
        clarification_data["gate_id"] = gate_id

    async with lock:
        # Drop previous sender→gate index if replacing
        prev = pending.get(sender_id)
        if isinstance(prev, dict):
            prev_gid = str(prev.get("gate_id") or "").strip().lower()
            if prev_gid and prev_gid != gate_id:
                pending.pop(_gate_field(prev_gid), None)
        pending[sender_id] = clarification_data
        if gate_id:
            pending[_gate_field(gate_id)] = clarification_data

    if redis is not None:
        try:
            payload_json = json.dumps(clarification_data)
            await redis.hset(redis_key, sender_id, payload_json)
            if gate_id:
                await redis.hset(redis_key, _gate_field(gate_id), payload_json)
        except Exception:
            logger.warning("Failed to persist clarification to Redis", exc_info=True)
    return question_text


async def get_clarification(
    *,
    redis: Any | None,
    pending: dict[str, dict[str, Any]],
    lock: Any,
    redis_key: str,
    sender_id: str | None = None,
    gate_id: str | None = None,
) -> dict[str, Any] | None:
    """Peek pending clarification by gate_id (preferred) or sender_id."""
    gid = (gate_id or "").strip().lower()
    sid = (sender_id or "").strip()

    async with lock:
        if gid:
            hit = pending.get(_gate_field(gid))
            if isinstance(hit, dict):
                return dict(hit)
        if sid:
            hit = pending.get(sid)
            if isinstance(hit, dict):
                return dict(hit)

    if redis is not None:
        try:
            if gid:
                raw = await redis.hget(redis_key, _gate_field(gid))
                if raw:
                    return json.loads(raw)
            if sid:
                raw = await redis.hget(redis_key, sid)
                if raw:
                    return json.loads(raw)
        except Exception:
            logger.warning("Failed to load clarification from Redis", exc_info=True)
    return None


async def pop_clarification(
    *,
    redis: Any | None,
    pending: dict[str, dict[str, Any]],
    lock: Any,
    sender_id: str,
    redis_key: str,
    gate_id: str | None = None,
) -> dict[str, Any] | None:
    """Pop a pending clarification from memory and Redis.

    Prefer ``gate_id`` when provided; otherwise pop the sender's latest pending.
    Always removes both the sender key and the ``gate:`` twin when present.
    """
    gid = (gate_id or "").strip().lower()
    result: dict[str, Any] | None = None

    async with lock:
        if gid:
            result = pending.pop(_gate_field(gid), None)
            if result is not None:
                owner = str(result.get("sender_id") or sender_id or "").strip()
                if owner:
                    cur = pending.get(owner)
                    if isinstance(cur, dict) and str(cur.get("gate_id") or "").lower() == gid:
                        pending.pop(owner, None)
        if result is None:
            result = pending.pop(sender_id, None)
            if result is not None:
                twin = str(result.get("gate_id") or "").strip().lower()
                if twin:
                    pending.pop(_gate_field(twin), None)

    if result is not None:
        if redis is not None:
            try:
                owner = str(result.get("sender_id") or sender_id or "").strip()
                twin = str(result.get("gate_id") or "").strip().lower()
                if owner:
                    await redis.hdel(redis_key, owner)
                if twin:
                    await redis.hdel(redis_key, _gate_field(twin))
            except Exception:
                logger.warning("Failed to remove clarification from Redis", exc_info=True)
        return result

    if redis is not None:
        try:
            raw = None
            if gid:
                raw = await redis.hget(redis_key, _gate_field(gid))
            if not raw:
                raw = await redis.hget(redis_key, sender_id)
            if raw:
                data = json.loads(raw)
                twin = str(data.get("gate_id") or "").strip().lower()
                owner = str(data.get("sender_id") or sender_id or "").strip()
                if twin:
                    await redis.hdel(redis_key, _gate_field(twin))
                if owner:
                    await redis.hdel(redis_key, owner)
                return data
        except Exception:
            logger.warning("Failed to load clarification from Redis", exc_info=True)
    return None
