"""Append-only queue of operator hints for dream-mode Mem0 ingestion."""

from __future__ import annotations

import logging
from datetime import UTC as _UTC
from datetime import datetime as _datetime
from pathlib import Path
from typing import Any

import aiosqlite as _aiosqlite

from brain_os.config import get_settings
from brain_os.memory.pending_memory_queue_backend import (
    PendingMemoryQueueBackend,
    build_pending_memory_queue_backend,
)

logger = logging.getLogger(__name__)

# Compatibility exports used by characterization tests/monkeypatching.
datetime = _datetime
UTC = _UTC
aiosqlite = _aiosqlite


class PendingMemoryQueue:
    """FIFO queue for short free-text memory hints (SQLite + optional PG shadow)."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = get_settings().app
        path = Path(db_path or cfg.pending_memory_queue_path)
        # Compatibility: characterization tests and legacy callers inspect queue._path directly.
        self._path = path
        self._backend: PendingMemoryQueueBackend = build_pending_memory_queue_backend(path)

    async def initialize(self) -> None:
        await self._backend.initialize()

    async def enqueue(self, body: str, *, source: str = "cli") -> int:
        return await self._backend.enqueue(body, source=source)

    async def list_pending(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return await self._backend.list_pending(limit=limit)

    async def count_pending(self) -> int:
        return await self._backend.count_pending()

    async def mark_processed(self, row_id: int) -> None:
        await self._backend.mark_processed(row_id)

    async def drain_to_long_term(self, long_term: Any) -> dict[str, Any]:
        if not get_settings().memory.api_key.get_secret_value():
            return {
                "status": "skipped",
                "reason": "no_mem0_key",
                "rows": 0,
                "stored": 0,
                "failed": 0,
            }

        pending = await self.list_pending(limit=500)
        if not pending:
            return {"status": "ok", "rows": 0, "stored": 0, "failed": 0}

        stored = 0
        failed = 0
        for row in pending:
            rid = int(row["id"])
            body = str(row.get("body") or "").strip()
            src = str(row.get("source") or "pending_queue")
            if not body:
                await self.mark_processed(rid)
                continue
            try:
                gated = await long_term.store_gated(
                    body,
                    user_id="global",
                    metadata={
                        "type": "pending_memory_hint",
                        "source": src,
                        "queued_at": row.get("created_at", ""),
                        "memory_category": "pending_hint",
                    },
                    source=f"pending_queue:{src}",
                    category="pending_hint",
                )
                if gated.get("skipped"):
                    await self.mark_processed(rid)
                    continue
                ids = gated.get("entries", [])
                if not ids:
                    failed += 1
                    continue
                await self.mark_processed(rid)
                stored += 1
            except Exception:
                logger.exception("PendingMemoryQueue: store failed for id=%s", rid)
                failed += 1

        return {
            "status": "ok" if failed == 0 else "partial",
            "rows": len(pending),
            "stored": stored,
            "failed": failed,
        }
