"""Approval-gated outbound campaign drafting for governed script productization."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OutboundMessage:
    to: str
    subject: str
    body: str


class OutboundApprovalService:
    """Stores pending outbound batches and releases drafts only after approval."""

    def __init__(self, storage_path: Path) -> None:
        self._storage_path = storage_path
        self._lock = asyncio.Lock()
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._storage_path.exists():
            self._storage_path.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self._storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _save(self, rows: list[dict[str, Any]]) -> None:
        self._storage_path.write_text(
            json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    async def create_batch(
        self,
        *,
        campaign_name: str,
        created_by: str,
        messages: list[OutboundMessage],
    ) -> dict[str, Any]:
        async with self._lock:
            rows = self._load()
            batch_id = uuid.uuid4().hex[:12]
            now = datetime.now(UTC).isoformat()
            row = {
                "batch_id": batch_id,
                "campaign_name": campaign_name,
                "created_by": created_by,
                "created_at": now,
                "approved_at": None,
                "approved_by": None,
                "status": "pending_approval",
                "messages": [m.__dict__ for m in messages],
                "drafts": [],
            }
            rows.append(row)
            self._save(rows)
            return row

    async def reject_batch(
        self,
        *,
        batch_id: str,
        rejected_by: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Mark a pending batch as rejected (no Gmail drafts)."""
        async with self._lock:
            rows = self._load()
            for row in rows:
                if row.get("batch_id") != batch_id:
                    continue
                if row.get("status") == "approved":
                    raise ValueError("Cannot reject an already-approved batch")
                if row.get("status") == "rejected":
                    return row
                row["status"] = "rejected"
                row["rejected_at"] = datetime.now(UTC).isoformat()
                row["rejected_by"] = rejected_by
                row["reject_reason"] = (reason or "").strip()[:2000]
                self._save(rows)
                return row
            raise KeyError(f"Unknown batch_id: {batch_id}")

    async def approve_batch(
        self,
        *,
        batch_id: str,
        approved_by: str,
        gmail_draft_sender: Any,
    ) -> dict[str, Any]:
        async with self._lock:
            rows = self._load()
            for row in rows:
                if row.get("batch_id") != batch_id:
                    continue
                if row.get("status") == "rejected":
                    raise ValueError(f"Batch {batch_id} was rejected and cannot be approved")
                if row.get("status") == "approved":
                    return row
                drafts: list[dict[str, Any]] = []
                for msg in row.get("messages", []):
                    draft = await gmail_draft_sender.create_draft(
                        to=msg["to"],
                        subject=msg["subject"],
                        body=msg["body"],
                    )
                    drafts.append(draft)
                row["approved_at"] = datetime.now(UTC).isoformat()
                row["approved_by"] = approved_by
                row["status"] = "approved"
                row["drafts"] = drafts
                self._save(rows)
                return row
            raise KeyError(f"Unknown batch_id: {batch_id}")

    async def approve_message(
        self,
        *,
        batch_id: str,
        index: int,
        approved_by: str,
        gmail_draft_sender: Any,
    ) -> dict[str, Any]:
        """Approve one message in a pending batch; stage Gmail draft for that row only."""
        async with self._lock:
            rows = self._load()
            for row in rows:
                if row.get("batch_id") != batch_id:
                    continue
                if row.get("status") == "rejected":
                    raise ValueError(f"Batch {batch_id} was rejected and cannot be approved")
                messages: list[dict[str, Any]] = list(row.get("messages") or [])
                if index < 0 or index >= len(messages):
                    raise IndexError(f"Message index {index} out of range for batch {batch_id}")
                msg = messages[index]
                if msg.get("approved_at"):
                    return {"batch_id": batch_id, "index": index, "draft": msg.get("draft")}
                draft = await gmail_draft_sender.create_draft(
                    to=msg["to"],
                    subject=msg["subject"],
                    body=msg["body"],
                )
                msg["approved_at"] = datetime.now(UTC).isoformat()
                msg["approved_by"] = approved_by
                msg["draft"] = draft
                row["messages"] = messages
                if all(m.get("approved_at") or m.get("skipped_at") for m in messages):
                    row["status"] = "approved"
                    row["approved_at"] = datetime.now(UTC).isoformat()
                    row["approved_by"] = approved_by
                self._save(rows)
                return {"batch_id": batch_id, "index": index, "draft": draft}
            raise KeyError(f"Unknown batch_id: {batch_id}")

    async def skip_message(
        self,
        *,
        batch_id: str,
        index: int,
        skipped_by: str,
    ) -> dict[str, Any]:
        """Skip one message in a pending batch without staging a draft."""
        async with self._lock:
            rows = self._load()
            for row in rows:
                if row.get("batch_id") != batch_id:
                    continue
                messages: list[dict[str, Any]] = list(row.get("messages") or [])
                if index < 0 or index >= len(messages):
                    raise IndexError(f"Message index {index} out of range for batch {batch_id}")
                msg = messages[index]
                msg["skipped_at"] = datetime.now(UTC).isoformat()
                msg["skipped_by"] = skipped_by
                row["messages"] = messages
                if all(m.get("approved_at") or m.get("skipped_at") for m in messages):
                    row["status"] = "approved"
                    row["approved_at"] = datetime.now(UTC).isoformat()
                    row["approved_by"] = skipped_by
                self._save(rows)
                return {"batch_id": batch_id, "index": index}
            raise KeyError(f"Unknown batch_id: {batch_id}")

    async def list_batches(self, *, limit: int = 50) -> list[dict[str, Any]]:
        async with self._lock:
            rows = self._load()
            return rows[-limit:]
