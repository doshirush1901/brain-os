"""Musculoskeletal system — action tracking and learning signal extraction.

Records every action Ira takes (emails sent, quotes generated, leads qualified)
with their outcomes, then periodically analyzes the records to extract
"myokines" — learning signals about what's working and what isn't.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Uuid,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.ext.asyncio import create_async_engine

from brain_os.config import get_settings

logger = logging.getLogger(__name__)


# ── enums ─────────────────────────────────────────────────────────────────


class ActionType(str, Enum):
    EMAIL_SENT = "EMAIL_SENT"
    QUOTE_GENERATED = "QUOTE_GENERATED"
    DEAL_UPDATED = "DEAL_UPDATED"
    LEAD_QUALIFIED = "LEAD_QUALIFIED"
    MEETING_SCHEDULED = "MEETING_SCHEDULED"
    RESEARCH_COMPLETED = "RESEARCH_COMPLETED"
    CAMPAIGN_STEP_SENT = "CAMPAIGN_STEP_SENT"
    KNOWLEDGE_INGESTED = "KNOWLEDGE_INGESTED"


class Outcome(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    NO_RESPONSE = "NO_RESPONSE"


# ── Pydantic transfer model ──────────────────────────────────────────────


class ActionRecord(BaseModel):
    """Transfer object for recording an action."""

    id: UUID = Field(default_factory=uuid4)
    action_type: ActionType
    target: str
    details: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    outcome: Outcome = Outcome.PENDING
    outcome_details: dict = Field(default_factory=dict)
    outcome_at: datetime | None = None


# ── SQLAlchemy Core table ─────────────────────────────────────────────────

metadata = MetaData()

action_records_table = Table(
    "action_records",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("action_type", String(50), nullable=False, index=True),
    Column("target", String(500), nullable=False),
    Column("details", JSON, nullable=False, server_default="{}"),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("outcome", String(20), nullable=False, server_default="PENDING", index=True),
    Column("outcome_details", JSON, nullable=False, server_default="{}"),
    Column("outcome_at", DateTime(timezone=True), nullable=True),
)


# ── system class ──────────────────────────────────────────────────────────


class MusculoskeletalSystem:
    """Tracks actions and extracts learning signals (myokines)."""

    def __init__(self, database_url: str | None = None) -> None:
        url = database_url or get_settings().database.url
        self._engine = create_async_engine(url)
        self._metadata = metadata

    async def create_tables(self) -> None:
        """Create the action_records table if it doesn't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(self._metadata.create_all)
        logger.info("MusculoskeletalSystem tables ensured")

    async def record_action(self, action: ActionRecord) -> UUID:
        """Insert an action record and return its id."""
        stmt = action_records_table.insert().values(
            id=action.id,
            action_type=action.action_type.value,
            target=action.target,
            details=action.details,
            created_at=action.timestamp,
            outcome=action.outcome.value,
            outcome_details=action.outcome_details,
            outcome_at=action.outcome_at,
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
        logger.info("ACTION RECORDED: %s -> %s", action.action_type.value, action.target)
        return action.id

    async def update_outcome(
        self,
        action_id: UUID,
        outcome: str,
        outcome_details: dict | None = None,
    ) -> bool:
        """Update an action's outcome. Returns True if a row was updated."""
        Outcome(outcome)  # validate

        stmt = (
            update(action_records_table)
            .where(action_records_table.c.id == action_id)
            .values(
                outcome=outcome,
                outcome_details=outcome_details or {},
                outcome_at=datetime.now(UTC),
            )
        )
        async with self._engine.begin() as conn:
            result = await conn.execute(stmt)
        updated = result.rowcount > 0  # type: ignore[union-attr]
        if updated:
            logger.info("OUTCOME UPDATED: %s -> %s", action_id, outcome)
        return updated

    async def get_actions(
        self,
        *,
        action_type: str | None = None,
        outcome: str | None = None,
        since_days: int = 30,
    ) -> list[dict[str, Any]]:
        """Query recent actions with optional filters."""
        cutoff = datetime.now(UTC) - timedelta(days=since_days)
        stmt = select(action_records_table).where(action_records_table.c.created_at >= cutoff)
        if action_type:
            stmt = stmt.where(action_records_table.c.action_type == action_type)
        if outcome:
            stmt = stmt.where(action_records_table.c.outcome == outcome)

        stmt = stmt.order_by(action_records_table.c.created_at.desc())

        async with self._engine.connect() as conn:
            rows = await conn.execute(stmt)
            return [dict(row._mapping) for row in rows]

    async def extract_myokines(self, period_days: int = 7) -> dict[str, Any]:
        """Analyze recent actions and extract learning signals."""
        actions = await self.get_actions(since_days=period_days)

        if not actions:
            return {
                "period_days": period_days,
                "total_actions": 0,
                "outcomes": {},
                "email_metrics": {},
                "quote_metrics": {},
                "lead_metrics": {},
                "top_strategies": [],
                "recommendations": [],
            }

        outcome_counts = Counter(a["outcome"] for a in actions)

        # Email metrics
        email_actions = [
            a
            for a in actions
            if a["action_type"]
            in (ActionType.EMAIL_SENT.value, ActionType.CAMPAIGN_STEP_SENT.value)
        ]
        email_metrics = self._compute_email_metrics(email_actions)

        # Quote metrics
        quote_actions = [a for a in actions if a["action_type"] == ActionType.QUOTE_GENERATED.value]
        quote_metrics = self._compute_quote_metrics(quote_actions)

        # Lead metrics
        lead_actions = [a for a in actions if a["action_type"] == ActionType.LEAD_QUALIFIED.value]
        lead_metrics = self._compute_lead_metrics(lead_actions)

        # Top strategies
        top_strategies = self._rank_strategies(actions)

        # Recommendations
        recommendations = self._generate_recommendations(email_metrics, quote_metrics, lead_metrics)

        return {
            "period_days": period_days,
            "total_actions": len(actions),
            "outcomes": dict(outcome_counts),
            "email_metrics": email_metrics,
            "quote_metrics": quote_metrics,
            "lead_metrics": lead_metrics,
            "top_strategies": top_strategies,
            "recommendations": recommendations,
        }

    @staticmethod
    def _compute_email_metrics(actions: list[dict]) -> dict[str, Any]:
        if not actions:
            return {"total_sent": 0, "reply_rate": 0.0, "by_hour": {}}

        total = len(actions)
        successes = sum(1 for a in actions if a["outcome"] == Outcome.SUCCESS.value)

        by_hour: dict[int, dict[str, int]] = {}
        for a in actions:
            hour = a["created_at"].hour if hasattr(a["created_at"], "hour") else 0
            bucket = by_hour.setdefault(hour, {"total": 0, "success": 0})
            bucket["total"] += 1
            if a["outcome"] == Outcome.SUCCESS.value:
                bucket["success"] += 1

        hour_rates = {
            h: round(v["success"] / v["total"], 2) if v["total"] else 0.0
            for h, v in sorted(by_hour.items())
        }

        return {
            "total_sent": total,
            "reply_rate": round(successes / total, 3) if total else 0.0,
            "by_hour": hour_rates,
        }

    @staticmethod
    def _compute_quote_metrics(actions: list[dict]) -> dict[str, Any]:
        if not actions:
            return {"total_generated": 0, "conversion_rate": 0.0, "by_region": {}}

        total = len(actions)
        successes = sum(1 for a in actions if a["outcome"] == Outcome.SUCCESS.value)

        by_region: dict[str, dict[str, int]] = {}
        for a in actions:
            region = (a.get("details") or {}).get("region", "unknown")
            bucket = by_region.setdefault(region, {"total": 0, "success": 0})
            bucket["total"] += 1
            if a["outcome"] == Outcome.SUCCESS.value:
                bucket["success"] += 1

        region_rates = {
            r: round(v["success"] / v["total"], 2) if v["total"] else 0.0
            for r, v in by_region.items()
        }

        return {
            "total_generated": total,
            "conversion_rate": round(successes / total, 3) if total else 0.0,
            "by_region": region_rates,
        }

    @staticmethod
    def _compute_lead_metrics(actions: list[dict]) -> dict[str, Any]:
        if not actions:
            return {"total_qualified": 0, "accuracy": 0.0}

        total = len(actions)
        correct = 0
        for a in actions:
            details = a.get("details") or {}
            predicted = details.get("predicted_score", 0)
            actual_outcome = a["outcome"]
            if (predicted >= 50 and actual_outcome == Outcome.SUCCESS.value) or (
                predicted < 50 and actual_outcome != Outcome.SUCCESS.value
            ):
                correct += 1

        return {
            "total_qualified": total,
            "accuracy": round(correct / total, 3) if total else 0.0,
        }

    @staticmethod
    def _rank_strategies(actions: list[dict]) -> list[str]:
        type_stats: dict[str, dict[str, int]] = {}
        for a in actions:
            at = a["action_type"]
            bucket = type_stats.setdefault(at, {"total": 0, "success": 0})
            bucket["total"] += 1
            if a["outcome"] == Outcome.SUCCESS.value:
                bucket["success"] += 1

        ranked = sorted(
            type_stats.items(),
            key=lambda kv: kv[1]["success"] / kv[1]["total"] if kv[1]["total"] else 0,
            reverse=True,
        )
        return [name for name, _ in ranked[:5]]

    @staticmethod
    def _generate_recommendations(
        email_metrics: dict,
        quote_metrics: dict,
        lead_metrics: dict,
    ) -> list[str]:
        recs: list[str] = []

        by_hour = email_metrics.get("by_hour", {})
        if by_hour:
            best_hour = max(
                by_hour,
                key=lambda h: (
                    by_hour[h]
                    if isinstance(by_hour[h], (int, float))
                    else by_hour[h].get("success", 0) / max(by_hour[h].get("total", 1), 1)
                ),
            )
            recs.append(
                f"Email reply rates are highest at {best_hour}:00 — schedule outreach accordingly."
            )

        by_region = quote_metrics.get("by_region", {})
        if len(by_region) >= 2:
            best_region = max(
                by_region,
                key=lambda r: (
                    by_region[r]
                    if isinstance(by_region[r], (int, float))
                    else by_region[r].get("success", 0) / max(by_region[r].get("total", 1), 1)
                ),
            )
            recs.append(
                f"Quote conversion is highest in {best_region} — consider prioritizing leads there."
            )

        accuracy = lead_metrics.get("accuracy", 0)
        if accuracy and accuracy < 0.5:
            recs.append("Lead qualification accuracy is below 50% — review scoring criteria.")

        return recs

    async def close(self) -> None:
        """Dispose the database engine."""
        await self._engine.dispose()
