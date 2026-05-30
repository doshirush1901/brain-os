"""SQLite-backed store for corrections that Brain OS receives from users.

Corrections flow in from Nemesis (adversarial feedback) and direct corrections.  They accumulate here until the next dream-mode
sleep-training cycle processes them into the vector store and long-term memory.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/brain/corrections.db")


class CorrectionCategory(str, Enum):
    PRICING = "PRICING"
    SPECS = "SPECS"
    CUSTOMER = "CUSTOMER"
    COMPETITOR = "COMPETITOR"
    GENERAL = "GENERAL"


class CorrectionSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CorrectionStore:
    """Async SQLite store for pending and processed corrections."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path), timeout=30.0)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS corrections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entity      TEXT NOT NULL,
                category    TEXT NOT NULL,
                severity    TEXT NOT NULL DEFAULT 'MEDIUM',
                old_value   TEXT NOT NULL DEFAULT '',
                new_value   TEXT NOT NULL,
                source      TEXT NOT NULL DEFAULT 'unknown',
                created_at  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending'
            )
            """
        )
        await self._db.commit()
        logger.info("CorrectionStore initialised at %s", self._db_path)

    async def add_correction(
        self,
        entity: str,
        new_value: str,
        *,
        category: CorrectionCategory = CorrectionCategory.GENERAL,
        severity: CorrectionSeverity = CorrectionSeverity.MEDIUM,
        old_value: str = "",
        source: str = "unknown",
    ) -> int:
        """Insert a new correction and return its row id."""
        assert self._db is not None, "Call initialize() first"
        now = datetime.now(UTC).isoformat()
        cursor = await self._db.execute(
            """
            INSERT INTO corrections (entity, category, severity, old_value, new_value, source, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (entity, category.value, severity.value, old_value, new_value, source, now),
        )
        await self._db.commit()
        row_id = cursor.lastrowid or 0
        logger.info(
            "Correction #%d added: entity=%s category=%s severity=%s",
            row_id,
            entity,
            category.value,
            severity.value,
        )
        return row_id

    async def get_pending_corrections(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return corrections that have not yet been processed by sleep training."""
        assert self._db is not None, "Call initialize() first"
        cursor = await self._db.execute(
            "SELECT id, entity, category, severity, old_value, new_value, source, created_at, status "
            "FROM corrections WHERE status = 'pending' ORDER BY created_at ASC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_row_to_dict(r) for r in rows]

    async def mark_processed(self, correction_id: int) -> None:
        """Mark a single correction as processed after sleep training."""
        assert self._db is not None, "Call initialize() first"
        await self._db.execute(
            "UPDATE corrections SET status = 'processed' WHERE id = ?",
            (correction_id,),
        )
        await self._db.commit()

    async def get_corrections_by_entity(self, entity: str) -> list[dict[str, Any]]:
        """Return all corrections (any status) for a given entity."""
        assert self._db is not None, "Call initialize() first"
        cursor = await self._db.execute(
            "SELECT id, entity, category, severity, old_value, new_value, source, created_at, status "
            "FROM corrections WHERE entity = ? ORDER BY created_at DESC",
            (entity,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_row_to_dict(r) for r in rows]

    async def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics about stored corrections."""
        assert self._db is not None, "Call initialize() first"
        cursor = await self._db.execute("SELECT status, COUNT(*) FROM corrections GROUP BY status")
        status_counts = dict(await cursor.fetchall())
        await cursor.close()

        cursor = await self._db.execute(
            "SELECT category, COUNT(*) FROM corrections GROUP BY category"
        )
        category_counts = dict(await cursor.fetchall())
        await cursor.close()

        cursor = await self._db.execute(
            "SELECT severity, COUNT(*) FROM corrections GROUP BY severity"
        )
        severity_counts = dict(await cursor.fetchall())
        await cursor.close()

        return {
            "total": sum(status_counts.values()),
            "by_status": status_counts,
            "by_category": category_counts,
            "by_severity": severity_counts,
        }

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> CorrectionStore:
        await self.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


def _row_to_dict(row: tuple) -> dict[str, Any]:
    return {
        "id": row[0],
        "entity": row[1],
        "category": row[2],
        "severity": row[3],
        "old_value": row[4],
        "new_value": row[5],
        "source": row[6],
        "created_at": row[7],
        "status": row[8],
    }
