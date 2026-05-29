"""Letta-style memory blocks — pinned, bounded core context per contact."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# label -> (description, char limit)
BLOCK_SPECS: dict[str, tuple[str, int]] = {
    "human": (
        "Who this contact is: role, company, preferences, memorable facts.",
        800,
    ),
    "relationship": (
        "Relationship warmth, interaction cadence, and tone guidance for replies.",
        400,
    ),
    "timeline": (
        "Rolling narrative of past interactions with this contact (compacted).",
        1200,
    ),
    "commitments": (
        "Open commitments, decisions, and follow-ups owed to or by this contact.",
        600,
    ),
}

DEFAULT_BLOCK_LABELS = tuple(BLOCK_SPECS.keys())


@dataclass(frozen=True)
class MemoryBlock:
    """One pinned context block (Letta core-memory shape)."""

    scope: str
    label: str
    description: str
    value: str
    limit: int
    updated_at: str | None = None


def truncate_to_limit(value: str, limit: int) -> str:
    """Trim value to at most ``limit`` characters, keeping a trailing ellipsis hint."""
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def render_blocks_xml(blocks: list[MemoryBlock]) -> str:
    """Render blocks for agent context (always-visible core memory)."""
    if not blocks:
        return ""
    lines = ["<memory_blocks>"]
    for block in blocks:
        if not (block.value or "").strip():
            continue
        lines.append(f"  <{block.label}>")
        lines.append(f"    <description>{block.description}</description>")
        lines.append(f"    <value>\n{block.value.strip()}\n    </value>")
        lines.append(f"  </{block.label}>")
    lines.append("</memory_blocks>")
    return "\n".join(lines)


class MemoryBlockStore:
    """SQLite-backed store for per-scope memory blocks."""

    def __init__(self, db_path: str = "data/memory_blocks.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path, timeout=30.0)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_blocks (
                scope TEXT NOT NULL,
                label TEXT NOT NULL,
                description TEXT NOT NULL,
                value TEXT NOT NULL DEFAULT '',
                block_limit INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (scope, label)
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_blocks_scope ON memory_blocks(scope)"
        )
        await self._db.commit()

    async def get_block(self, scope: str, label: str) -> MemoryBlock | None:
        assert self._db is not None
        description, limit = BLOCK_SPECS.get(label, (label, 2000))
        cursor = await self._db.execute(
            """
            SELECT scope, label, description, value, block_limit, updated_at
            FROM memory_blocks WHERE scope = ? AND label = ?
            """,
            (scope, label),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return MemoryBlock(
            scope=row[0],
            label=row[1],
            description=row[2] or description,
            value=row[3] or "",
            limit=int(row[4]) if row[4] else limit,
            updated_at=row[5],
        )

    async def get_blocks_for_scope(
        self,
        scope: str,
        labels: tuple[str, ...] | None = None,
    ) -> list[MemoryBlock]:
        wanted = labels or DEFAULT_BLOCK_LABELS
        blocks: list[MemoryBlock] = []
        for label in wanted:
            spec = BLOCK_SPECS.get(label)
            if spec is None:
                continue
            description, limit = spec
            existing = await self.get_block(scope, label)
            if existing is not None:
                blocks.append(existing)
            else:
                blocks.append(
                    MemoryBlock(
                        scope=scope,
                        label=label,
                        description=description,
                        value="",
                        limit=limit,
                    )
                )
        return blocks

    async def set_block(
        self,
        scope: str,
        label: str,
        value: str,
        *,
        description: str | None = None,
        limit: int | None = None,
    ) -> MemoryBlock:
        assert self._db is not None
        default_desc, default_limit = BLOCK_SPECS.get(label, (label, 2000))
        desc = description or default_desc
        char_limit = limit if limit is not None else default_limit
        trimmed = truncate_to_limit(value, char_limit)
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            """
            INSERT INTO memory_blocks (scope, label, description, value, block_limit, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope, label) DO UPDATE SET
                description = excluded.description,
                value = excluded.value,
                block_limit = excluded.block_limit,
                updated_at = excluded.updated_at
            """,
            (scope, label, desc, trimmed, char_limit, now),
        )
        await self._db.commit()
        return MemoryBlock(
            scope=scope,
            label=label,
            description=desc,
            value=trimmed,
            limit=char_limit,
            updated_at=now,
        )

    async def render_for_scope(
        self,
        scope: str,
        labels: tuple[str, ...] | None = None,
    ) -> str:
        blocks = await self.get_blocks_for_scope(scope, labels)
        non_empty = [b for b in blocks if (b.value or "").strip()]
        return render_blocks_xml(non_empty)

    async def apply_correction(self, scope: str, label: str, corrected_value: str) -> MemoryBlock:
        """Mnemon / correction ledger may override a block value."""
        return await self.set_block(scope, label, corrected_value)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> MemoryBlockStore:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
