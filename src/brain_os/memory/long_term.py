"""Persistent semantic memory backed by the Mem0 REST API."""

from __future__ import annotations

import logging
import math
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from brain_os.config import MemoryConfig, get_settings
from brain_os.contracts.write_receipt import build_receipt, enqueue_retry, record_receipt
from brain_os.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class LongTermMemory:
    def __init__(self, config: MemoryConfig | None = None) -> None:
        cfg = config or get_settings().memory
        self._api_key = cfg.api_key.get_secret_value()
        self._base_url = "https://api.mem0.ai"
        self._timeout = get_settings().app.mem0_timeout
        self._headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "application/json",
        }

    async def store(
        self,
        content: str,
        user_id: str = "global",
        metadata: dict | None = None,
        run_id: str = "",
    ) -> list[dict]:
        meta = metadata or {}
        t0 = time.monotonic()
        if not self._api_key:
            logger.warning("Mem0 API key not configured; skipping memory store")
            receipt = build_receipt(
                store="mem0",
                attempted=False,
                success=False,
                operation="store",
                run_id=run_id,
                reason="no_api_key",
                metadata={"user_id": user_id, "source": meta.get("source", "")},
            )
            record_receipt(receipt)
            enqueue_retry(receipt)
            return []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/v1/memories/",
                    headers=self._headers,
                    json={
                        "messages": [{"role": "user", "content": content}],
                        "user_id": user_id,
                        "metadata": meta,
                        "infer": True,
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
                ids: list[str] = []
                if isinstance(payload, list):
                    ids = [
                        str(item.get("id", ""))
                        for item in payload
                        if isinstance(item, dict) and item.get("id")
                    ]
                elif isinstance(payload, dict):
                    mem_id = payload.get("id")
                    if mem_id:
                        ids = [str(mem_id)]
                    elif isinstance(payload.get("results"), list):
                        ids = [
                            str(item.get("id", ""))
                            for item in payload.get("results", [])
                            if isinstance(item, dict) and item.get("id")
                        ]
                success = len(ids) > 0
                reason = "" if success else "zero_memories_written"
                retry_meta: dict[str, Any] = {"user_id": user_id, "source": meta.get("source", "")}
                if not success:
                    retry_meta["replay_content"] = (content or "")[:16_384]
                receipt = build_receipt(
                    store="mem0",
                    attempted=True,
                    success=success,
                    operation="store",
                    run_id=run_id,
                    reason=reason,
                    status_code=int(getattr(resp, "status_code", 0) or 0),
                    latency_ms=round((time.monotonic() - t0) * 1000, 2),
                    ids=ids,
                    metadata=retry_meta,
                )
                record_receipt(receipt)
                if not success:
                    enqueue_retry(receipt)
                return (
                    payload
                    if isinstance(payload, list)
                    else ([payload] if isinstance(payload, dict) else [])
                )
        except httpx.HTTPError as e:
            logger.exception("Mem0 store failed: %s", e)
            status_code = None
            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
            retry_meta = {
                "user_id": user_id,
                "source": meta.get("source", ""),
                "replay_content": (content or "")[:16_384],
            }
            receipt = build_receipt(
                store="mem0",
                attempted=True,
                success=False,
                operation="store",
                run_id=run_id,
                reason="http_failure",
                status_code=status_code,
                latency_ms=round((time.monotonic() - t0) * 1000, 2),
                metadata=retry_meta,
            )
            record_receipt(receipt)
            enqueue_retry(receipt)
            return []

    async def search(
        self,
        query: str,
        user_id: str = "global",
        limit: int = 5,
    ) -> list[dict]:
        if not self._api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/v1/memories/search/",
                    headers=self._headers,
                    json={
                        "query": query,
                        "user_id": user_id,
                        "top_k": limit,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()
                memories = raw.get("results", raw) if isinstance(raw, dict) else raw
                results = []
                now = datetime.now(UTC)
                for m in memories:
                    result = {
                        "id": m.get("id", ""),
                        "memory": m.get("memory", ""),
                        "score": m.get("score", 0.5),
                        "metadata": m.get("metadata", {}),
                        "created_at": m.get("created_at", ""),
                    }
                    created_at = result["created_at"]
                    if created_at:
                        try:
                            if isinstance(created_at, str):
                                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                            else:
                                dt = created_at
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=UTC)
                            delta = now - dt
                            days_old = max(0, delta.days)
                            result["score"] *= self.apply_decay(days_old)
                        except (ValueError, TypeError):
                            pass
                    results.append(result)
                results.sort(key=lambda r: r["score"], reverse=True)
                return results
        except httpx.HTTPError as e:
            logger.exception("Mem0 search failed: %s", e)
            return []

    async def store_correction(
        self,
        original: str,
        corrected: str,
        context: str,
    ) -> list[dict]:
        content = (
            f"CORRECTION: Originally '{original}' was stated, but the correct "
            f"information is '{corrected}'. Context: {context}"
        )
        return await self.store(
            content,
            user_id="global",
            metadata={
                "type": "correction",
                "priority": "high",
                "original": original,
                "corrected": corrected,
            },
        )

    async def store_preference(
        self,
        user_id: str,
        preference_type: str,
        value: str,
    ) -> list[dict]:
        content = f"User preference: {preference_type} = {value}"
        return await self.store(
            content,
            user_id=user_id,
            metadata={
                "type": "preference",
                "preference_type": preference_type,
                "value": value,
            },
        )

    async def get_user_preferences(self, user_id: str) -> dict:
        try:
            results = await self.search(
                "user preferences and settings",
                user_id=user_id,
                limit=20,
            )
            filtered = [m for m in results if m.get("metadata", {}).get("type") == "preference"]
            return {
                m["metadata"]["preference_type"]: m["metadata"]["value"]
                for m in filtered
                if "preference_type" in m.get("metadata", {})
            }
        except (DatabaseError, Exception) as e:
            logger.exception("Mem0 get_user_preferences failed: %s", e)
            return {}

    async def store_fact(
        self,
        fact: str,
        source: str,
        confidence: float,
        *,
        mem0_user_id: str | None = None,
        run_id: str = "",
    ) -> list[dict]:
        meta: dict[str, Any] = {
            "type": "fact",
            "source": source,
            "confidence": confidence,
            "verified": confidence >= 0.8,
        }
        if source.startswith(("email:", "takeout_email", "contact_history:")):
            meta["source_type"] = "email"
        uid = mem0_user_id if (mem0_user_id and mem0_user_id.strip()) else "global"
        return await self.store(fact, user_id=uid, metadata=meta, run_id=run_id)

    def apply_decay(self, days_old: int, access_count: int = 0) -> float:
        base_stability = 30.0
        stability = base_stability * (1 + math.log(1 + access_count))
        retention = math.exp(-days_old / stability)
        return max(0.0, min(1.0, retention))
