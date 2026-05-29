"""Async SQLite log of verifiable predictions and reconciled outcomes."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.contracts.learning_paths import predictions_db_path

logger = logging.getLogger(__name__)

HOT_LEAD_BOARD_PATTERN = "hot_lead_board"
MUST_ACT_TODAY_PATTERN = "must_act_today"
TYCHE_DEAL_FORECAST_PATTERN = "tyche_deal_forecast"


@dataclass
class Prediction:
    prediction_id: str
    timestamp: str
    pattern_id: str
    predicted_outcome: str
    actual_outcome: str | None
    was_correct: bool | None
    context_json: str = "{}"
    reconciled_at: str | None = None

    @property
    def context(self) -> dict[str, Any]:
        try:
            out = json.loads(self.context_json)
            return out if isinstance(out, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}


class PredictionLog:
    """SQLite-backed log of verifiable predictions and their outcomes."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else predictions_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(str(self._db_path), timeout=30.0)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                pattern_id TEXT NOT NULL,
                predicted_outcome TEXT NOT NULL,
                actual_outcome TEXT,
                was_correct INTEGER,
                context_json TEXT NOT NULL DEFAULT '{}',
                reconciled_at TEXT
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_pattern ON predictions(pattern_id)"
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pred_unreconciled
            ON predictions(was_correct) WHERE was_correct IS NULL
            """
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def has_recent_unreconciled(
        self,
        pattern_id: str,
        *,
        account: str,
        within_hours: float = 20.0,
    ) -> bool:
        """True if an unreconciled prediction exists for ``account`` in context (dedupe)."""
        assert self._db is not None
        cutoff = (datetime.now(UTC) - timedelta(hours=within_hours)).isoformat()
        cursor = await self._db.execute(
            """
            SELECT 1 FROM predictions
            WHERE pattern_id = ?
              AND was_correct IS NULL
              AND timestamp >= ?
              AND json_extract(context_json, '$.account') = ?
            LIMIT 1
            """,
            (pattern_id, cutoff, account),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row is not None

    async def record_prediction(
        self,
        pattern_id: str,
        predicted_outcome: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Log a new verifiable prediction; returns ``prediction_id``."""
        assert self._db is not None
        pred_id = f"pred_{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC).isoformat()
        ctx = json.dumps(context or {}, ensure_ascii=True)
        await self._db.execute(
            """
            INSERT INTO predictions
                (prediction_id, timestamp, pattern_id, predicted_outcome, context_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (pred_id, now, pattern_id, predicted_outcome, ctx),
        )
        await self._db.commit()
        logger.info(
            "Prediction logged: %s pattern=%s — %s",
            pred_id,
            pattern_id,
            predicted_outcome[:60],
        )
        return pred_id

    async def record_outcome(
        self,
        prediction_id: str,
        actual_outcome: str,
        was_correct: bool,
    ) -> bool:
        """Fill in outcome for a previously logged prediction."""
        assert self._db is not None
        now = datetime.now(UTC).isoformat()
        cursor = await self._db.execute(
            """
            UPDATE predictions
            SET actual_outcome = ?,
                was_correct = ?,
                reconciled_at = ?
            WHERE prediction_id = ? AND was_correct IS NULL
            """,
            (actual_outcome, int(was_correct), now, prediction_id),
        )
        await self._db.commit()
        if cursor.rowcount == 0:
            logger.warning(
                "record_outcome: %s not found or already reconciled",
                prediction_id,
            )
            return False
        return True

    async def get_unreconciled(
        self,
        *,
        pattern_id: str | None = None,
        max_age_days: int = 30,
    ) -> list[Prediction]:
        """Predictions without ``was_correct`` within the age window."""
        assert self._db is not None
        cutoff = (datetime.now(UTC) - timedelta(days=max_age_days)).isoformat()
        if pattern_id:
            cursor = await self._db.execute(
                """
                SELECT * FROM predictions
                WHERE was_correct IS NULL
                  AND timestamp >= ?
                  AND pattern_id = ?
                ORDER BY timestamp DESC
                """,
                (cutoff, pattern_id),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT * FROM predictions
                WHERE was_correct IS NULL AND timestamp >= ?
                ORDER BY timestamp DESC
                """,
                (cutoff,),
            )
        rows = await cursor.fetchall()
        await cursor.close()
        return [self._row_to_prediction(r) for r in rows]

    async def get_by_pattern(self, pattern_id: str, *, limit: int = 100) -> list[Prediction]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM predictions
            WHERE pattern_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (pattern_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [self._row_to_prediction(r) for r in rows]

    async def accuracy_by_pattern(self, min_predictions: int = 3) -> dict[str, float]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT pattern_id,
                   COUNT(*) as total,
                   SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct
            FROM predictions
            WHERE was_correct IS NOT NULL
            GROUP BY pattern_id
            HAVING total >= ?
            """,
            (min_predictions,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        result: dict[str, float] = {}
        for row in rows:
            total = int(row[1])
            correct = int(row[2] or 0)
            result[str(row[0])] = correct / total if total > 0 else 0.0
        return result

    async def overall_accuracy(self) -> dict[str, Any]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                   SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as incorrect,
                   SUM(CASE WHEN was_correct IS NULL THEN 1 ELSE 0 END) as pending
            FROM predictions
            """
        )
        row = await cursor.fetchone()
        await cursor.close()
        if not row:
            return {
                "total_predictions": 0,
                "reconciled": 0,
                "pending": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy": None,
            }
        total = int(row[0] or 0)
        correct = int(row[1] or 0)
        incorrect = int(row[2] or 0)
        pending = int(row[3] or 0)
        reconciled = correct + incorrect
        return {
            "total_predictions": total,
            "reconciled": reconciled,
            "pending": pending,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": correct / reconciled if reconciled > 0 else None,
        }

    @staticmethod
    def _row_to_prediction(row: tuple[Any, ...]) -> Prediction:
        was_correct = row[5]
        if was_correct is not None:
            was_correct = bool(was_correct)
        return Prediction(
            prediction_id=str(row[0]),
            timestamp=str(row[1]),
            pattern_id=str(row[2]),
            predicted_outcome=str(row[3]),
            actual_outcome=row[4] if row[4] is not None else None,
            was_correct=was_correct,
            context_json=str(row[6] or "{}"),
            reconciled_at=row[7] if row[7] is not None else None,
        )

    async def __aenter__(self) -> PredictionLog:
        await self.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
