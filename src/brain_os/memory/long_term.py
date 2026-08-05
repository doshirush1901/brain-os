"""Persistent semantic memory backed by the Mem0 REST API."""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from brain_os.config import MemoryConfig, get_settings
from brain_os.contracts.write_receipt import build_receipt, enqueue_retry, record_receipt
from brain_os.exceptions import DatabaseError
from brain_os.memory.mem0_access_tracker import get_mem0_access_tracker
from brain_os.memory.mem0_reconsolidation import (
    get_recall_context,
    reconsolidation_enabled,
    set_recall_context,
)
from brain_os.memory.mem0_search_cache import get_mem0_search_cache
from brain_os.memory.store_policy import classify_provenance, evaluate_mem_store

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
                if success:
                    # New memory must be visible to the next recall immediately.
                    await get_mem0_search_cache().bump_generation(user_id)
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

    @staticmethod
    def _extract_store_ids(payload: list[dict] | dict | Any) -> list[str]:
        if isinstance(payload, list):
            return [
                str(item.get("id", ""))
                for item in payload
                if isinstance(item, dict) and item.get("id")
            ]
        if isinstance(payload, dict):
            mem_id = payload.get("id")
            if mem_id:
                return [str(mem_id)]
            if isinstance(payload.get("results"), list):
                return [
                    str(item.get("id", ""))
                    for item in payload.get("results", [])
                    if isinstance(item, dict) and item.get("id")
                ]
        return []

    async def store_gated(
        self,
        content: str,
        user_id: str = "global",
        metadata: dict | None = None,
        run_id: str = "",
        *,
        source: str = "",
        category: str | None = None,
        recalled_from: list[str] | None = None,
        bypass_salience: bool = False,
    ) -> dict[str, Any]:
        """Store with category + salience + provenance gates and reconsolidation audit."""
        ctx = get_recall_context() if reconsolidation_enabled() else {}
        recalled_ids = recalled_from if recalled_from is not None else list(ctx.get("ids", []))
        meta_in = metadata or {}
        src = source or str(meta_in.get("source", ""))

        decision = evaluate_mem_store(
            content,
            source=src,
            metadata=meta_in,
            recalled_ids=recalled_ids,
            bypass_salience=bypass_salience,
            category=category,
        )
        if not decision.allow:
            return {
                "skipped": True,
                "reason": decision.reason,
                "salience_score": decision.salience_score,
                "memory_category": decision.memory_category,
            }

        payload = await self.store(
            content,
            user_id=user_id,
            metadata=decision.metadata,
            run_id=run_id,
        )
        if recalled_ids and reconsolidation_enabled():
            new_ids = self._extract_store_ids(payload)
            await get_mem0_access_tracker().link_reconsolidation(
                recalled_ids,
                new_ids,
                user_id=user_id,
                query=str(ctx.get("query", "")),
                recalled_at=str(ctx.get("at", "")),
            )
        if isinstance(payload, list):
            return {"skipped": False, "entries": payload, "metadata": decision.metadata}
        return {
            "skipped": False,
            "entries": payload if payload else [],
            "metadata": decision.metadata,
        }

    def _provenance_score_multiplier(self, metadata: dict[str, Any]) -> float:
        app = get_settings().app
        boost = float(getattr(app, "mem0_evidence_score_boost", 1.15))
        penalty = float(getattr(app, "mem0_lore_score_penalty", 0.92))
        provenance = metadata.get("provenance_class") or classify_provenance(
            str(metadata.get("source", "")), metadata
        )
        if provenance in ("correction", "evidence"):
            return boost
        if provenance == "lore":
            return penalty
        return 1.0

    async def search(
        self,
        query: str,
        user_id: str = "global",
        limit: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Semantic search over Mem0.

        When *metadata_filter* is given (e.g. ``{"type": "preference"}``),
        the v2 search endpoint filters server-side — fewer wasted retrieval
        rows than fetching broadly and filtering in Python.
        """
        if not self._api_key:
            return []
        cache_namespace = "ltm"
        if metadata_filter:
            cache_namespace = f"ltm|{json.dumps(metadata_filter, sort_keys=True)}"
        cache = get_mem0_search_cache()
        cached = await cache.get(cache_namespace, query, user_id, limit)
        if cached is not None:
            # Cached recalls still count as accesses for spaced repetition.
            await get_mem0_access_tracker().record_hits([r.get("id", "") for r in cached], user_id)
            if reconsolidation_enabled():
                set_recall_context(
                    [str(r.get("id", "")) for r in cached if r.get("id")],
                    query,
                    user_id,
                )
            return cached
        if metadata_filter:
            url = f"{self._base_url}/v2/memories/search/"
            body: dict[str, Any] = {
                "query": query,
                "filters": {
                    "AND": [
                        {"user_id": user_id},
                        {"metadata": metadata_filter},
                    ]
                },
                "top_k": limit,
            }
        else:
            url = f"{self._base_url}/v1/memories/search/"
            body = {
                "query": query,
                "user_id": user_id,
                "top_k": limit,
            }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    url,
                    headers=self._headers,
                    json=body,
                )
                resp.raise_for_status()
                raw = resp.json()
                memories = raw.get("results", raw) if isinstance(raw, dict) else raw
                tracker = get_mem0_access_tracker()
                access_counts = await tracker.get_counts(
                    [str(m.get("id", "")) for m in memories if isinstance(m, dict)]
                )
                results = []
                now = datetime.now(UTC)
                for m in memories:
                    meta = m.get("metadata", {}) if isinstance(m.get("metadata"), dict) else {}
                    result = {
                        "id": m.get("id", ""),
                        "memory": m.get("memory", ""),
                        "score": m.get("score", 0.5),
                        "metadata": meta,
                        "created_at": m.get("created_at", ""),
                        "provenance_class": meta.get("provenance_class")
                        or classify_provenance(str(meta.get("source", "")), meta),
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
                            result["score"] *= self.apply_decay(
                                days_old,
                                access_count=access_counts.get(str(result["id"]), 0),
                            )
                        except (ValueError, TypeError):
                            pass
                    result["score"] *= self._provenance_score_multiplier(meta)
                    results.append(result)
                results.sort(key=lambda r: r["score"], reverse=True)
                await cache.put(cache_namespace, query, user_id, limit, results)
                await tracker.record_hits([r.get("id", "") for r in results], user_id)
                if reconsolidation_enabled():
                    set_recall_context(
                        [str(r.get("id", "")) for r in results if r.get("id")],
                        query,
                        user_id,
                    )
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
        gated = await self.store_gated(
            content,
            user_id="global",
            metadata={
                "type": "correction",
                "priority": "high",
                "original": original,
                "corrected": corrected,
                "provenance_class": "correction",
                "memory_category": "correction",
            },
            category="correction",
            source="mnemon:correction",
            bypass_salience=True,
        )
        if gated.get("skipped"):
            return []
        entries = gated.get("entries", [])
        return entries if isinstance(entries, list) else []

    async def store_preference(
        self,
        user_id: str,
        preference_type: str,
        value: str,
    ) -> list[dict]:
        content = f"User preference: {preference_type} = {value}"
        gated = await self.store_gated(
            content,
            user_id=user_id,
            metadata={
                "type": "preference",
                "preference_type": preference_type,
                "value": value,
                "provenance_class": "preference",
                "memory_category": "preference",
            },
            category="preference",
            source="operator:preference",
            bypass_salience=True,
        )
        if gated.get("skipped"):
            return []
        entries = gated.get("entries", [])
        return entries if isinstance(entries, list) else []

    async def get_user_preferences(self, user_id: str) -> dict:
        try:
            # Server-side type filter (v2) — no fetch-broad-then-filter waste.
            results = await self.search(
                "user preferences and settings",
                user_id=user_id,
                limit=20,
                metadata_filter={"type": "preference"},
            )
            return {
                m["metadata"]["preference_type"]: m["metadata"]["value"]
                for m in results
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
        category: str = "fact",
    ) -> list[dict]:
        meta: dict[str, Any] = {
            "type": "fact" if category == "fact" else category,
            "source": source,
            "confidence": confidence,
            "verified": confidence >= 0.8,
            "memory_category": category,
        }
        if source.startswith(("email:", "takeout_email", "contact_history:")):
            meta["source_type"] = "email"
        meta["provenance_class"] = classify_provenance(source, meta)
        uid = mem0_user_id if (mem0_user_id and mem0_user_id.strip()) else "global"
        gated = await self.store_gated(
            fact,
            user_id=uid,
            metadata=meta,
            run_id=run_id,
            source=source,
            category=category,
        )
        if gated.get("skipped"):
            return []
        entries = gated.get("entries", [])
        return entries if isinstance(entries, list) else []

    async def list_memories(self, user_id: str, page: int = 1, page_size: int = 100) -> list[dict]:
        """List raw memories for *user_id* (forgetting sweep input). Empty on failure."""
        if not self._api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{self._base_url}/v1/memories/",
                    headers=self._headers,
                    params={"user_id": user_id, "page": page, "page_size": page_size},
                )
                resp.raise_for_status()
                raw = resp.json()
                if isinstance(raw, dict):
                    rows = raw.get("results", raw.get("memories", []))
                elif isinstance(raw, list):
                    rows = raw
                else:
                    rows = []
                return [r for r in rows if isinstance(r, dict)]
        except httpx.HTTPError as e:
            logger.exception("Mem0 list_memories failed for %s: %s", user_id, e)
            return []

    async def delete_memory(self, memory_id: str) -> bool:
        """Permanently delete one memory. Returns True on success.

        HTTP 404 is treated as success (already gone) so landfill sweeps stay
        idempotent when the hosted index races ahead of a local id list.
        """
        if not self._api_key or not memory_id:
            return False
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.delete(
                    f"{self._base_url}/v1/memories/{memory_id}/",
                    headers=self._headers,
                )
                if resp.status_code == 404:
                    return True
                resp.raise_for_status()
                return True
        except httpx.HTTPError as e:
            logger.warning("Mem0 delete_memory failed for %s: %s", memory_id, e)
            return False

    def apply_decay(self, days_old: int, access_count: int = 0) -> float:
        base_stability = 30.0
        stability = base_stability * (1 + math.log(1 + access_count))
        retention = math.exp(-days_old / stability)
        return max(0.0, min(1.0, retention))
