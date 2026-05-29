"""Agent journaling — daily action logs and nightly reflections per agent.

Gives each agent temporal continuity and self-awareness: actions are logged
during the pipeline, and Dream Mode writes first-person journal entries.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/brain/agent_journals.db")


class AgentJournal:
    """SQLite-backed store for agent daily actions and journal entries."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create DB and tables if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path), timeout=30.0)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                date TEXT NOT NULL,
                action_text TEXT NOT NULL,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                date TEXT NOT NULL,
                reflection_text TEXT NOT NULL,
                mood TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_actions_agent_date "
            "ON daily_actions(agent_name, date)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_journal_entries_agent_date "
            "ON journal_entries(agent_name, date)"
        )
        await self._db.commit()
        logger.info("AgentJournal initialised at %s", self._db_path)

    async def log_action(
        self,
        agent_name: str,
        action_text: str,
        outcome: str,
        *,
        at_date: date | None = None,
    ) -> None:
        """Log one action for an agent on the given date (default today UTC)."""
        if self._db is None:
            return
        d = at_date or date.today()
        now = datetime.now(UTC).isoformat()
        try:
            await self._db.execute(
                """
                INSERT INTO daily_actions (agent_name, date, action_text, outcome, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (agent_name, d.isoformat(), action_text, outcome, now),
            )
            await self._db.commit()
        except Exception as exc:
            logger.warning("AgentJournal.log_action failed for %s", agent_name, exc_info=True)

    async def get_todays_actions(self, agent_name: str) -> list[dict[str, Any]]:
        """Return all actions logged for this agent today (UTC)."""
        return await self.get_actions_for_date(agent_name, date.today())

    async def get_actions_for_date(self, agent_name: str, d: date) -> list[dict[str, Any]]:
        """Return all actions for this agent on the given date."""
        if self._db is None:
            return []
        try:
            cursor = await self._db.execute(
                """
                SELECT action_text, outcome, created_at
                FROM daily_actions
                WHERE agent_name = ? AND date = ?
                ORDER BY created_at ASC
                """,
                (agent_name, d.isoformat()),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return [{"action_text": r[0], "outcome": r[1], "created_at": r[2]} for r in rows]
        except Exception as exc:
            logger.warning("AgentJournal.get_actions_for_date failed", exc_info=True)
            return []

    async def get_agents_with_actions_for_date(self, d: date) -> list[str]:
        """Return distinct agent names that have at least one action on the given date."""
        if self._db is None:
            return []
        try:
            cursor = await self._db.execute(
                "SELECT DISTINCT agent_name FROM daily_actions WHERE date = ?",
                (d.isoformat(),),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return [r[0] for r in rows]
        except Exception as exc:
            logger.warning("AgentJournal.get_agents_with_actions_for_date failed", exc_info=True)
            return []

    async def get_agents_with_actions_since_hours(self, hours: float = 24.0) -> list[str]:
        """Return distinct agent names that have at least one action in the last N hours (by created_at)."""
        if self._db is None:
            return []
        since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        try:
            cursor = await self._db.execute(
                "SELECT DISTINCT agent_name FROM daily_actions WHERE created_at >= ? ORDER BY agent_name",
                (since,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return [r[0] for r in rows]
        except Exception as exc:
            logger.warning("AgentJournal.get_agents_with_actions_since_hours failed", exc_info=True)
            return []

    async def get_actions_since_hours(
        self, agent_name: str, hours: float = 24.0
    ) -> list[dict[str, Any]]:
        """Return all actions for this agent in the last N hours (by created_at)."""
        if self._db is None:
            return []
        since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        try:
            cursor = await self._db.execute(
                """
                SELECT action_text, outcome, created_at
                FROM daily_actions
                WHERE agent_name = ? AND created_at >= ?
                ORDER BY created_at ASC
                """,
                (agent_name, since),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return [{"action_text": r[0], "outcome": r[1], "created_at": r[2]} for r in rows]
        except Exception as exc:
            logger.warning("AgentJournal.get_actions_since_hours failed", exc_info=True)
            return []

    async def save_journal_entry(
        self,
        agent_name: str,
        reflection_text: str,
        mood: str = "",
        *,
        at_date: date | None = None,
    ) -> None:
        """Save a journal entry for an agent (default today UTC)."""
        if self._db is None:
            return
        d = at_date or date.today()
        now = datetime.now(UTC).isoformat()
        try:
            await self._db.execute(
                """
                INSERT INTO journal_entries (agent_name, date, reflection_text, mood, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (agent_name, d.isoformat(), reflection_text, mood, now),
            )
            await self._db.commit()
        except Exception as exc:
            logger.warning(
                "AgentJournal.save_journal_entry failed for %s", agent_name, exc_info=True
            )

    async def search_past_journals(
        self,
        agent_name: str,
        query: str = "",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return past journal entries for this agent, optionally filtered by query.

        If query is empty, returns most recent entries. Otherwise filters by
        reflection_text containing the query (case-insensitive).
        """
        if self._db is None:
            return []
        try:
            if query.strip():
                cursor = await self._db.execute(
                    """
                    SELECT date, reflection_text, mood, created_at
                    FROM journal_entries
                    WHERE agent_name = ? AND reflection_text LIKE ?
                    ORDER BY date DESC, created_at DESC
                    LIMIT ?
                    """,
                    (agent_name, f"%{query.strip()}%", limit),
                )
            else:
                cursor = await self._db.execute(
                    """
                    SELECT date, reflection_text, mood, created_at
                    FROM journal_entries
                    WHERE agent_name = ?
                    ORDER BY date DESC, created_at DESC
                    LIMIT ?
                    """,
                    (agent_name, limit),
                )
            rows = await cursor.fetchall()
            await cursor.close()
            return [
                {
                    "date": r[0],
                    "reflection_text": r[1],
                    "mood": r[2],
                    "created_at": r[3],
                }
                for r in rows
            ]
        except Exception as exc:
            logger.warning("AgentJournal.search_past_journals failed", exc_info=True)
            return []

    async def get_latest_journal_entry(self, agent_name: str) -> str | None:
        """Return the most recent journal entry text for this agent, or None."""
        if self._db is None:
            return None
        try:
            cursor = await self._db.execute(
                """
                SELECT reflection_text
                FROM journal_entries
                WHERE agent_name = ?
                ORDER BY date DESC, created_at DESC
                LIMIT 1
                """,
                (agent_name,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            return row[0] if row else None
        except Exception as exc:
            logger.warning("AgentJournal.get_latest_journal_entry failed", exc_info=True)
            return None

    async def get_latest_journal_created_at(self, agent_name: str) -> datetime | None:
        """Return the created_at of the most recent journal entry for this agent, or None."""
        if self._db is None:
            return None
        try:
            cursor = await self._db.execute(
                """
                SELECT created_at
                FROM journal_entries
                WHERE agent_name = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (agent_name,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if not row or not row[0]:
                return None
            # created_at is ISO string; parse to datetime (assume UTC if naive)
            dt = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except Exception as exc:
            logger.warning("AgentJournal.get_latest_journal_created_at failed", exc_info=True)
            return None

    async def get_actions_since_datetime(
        self, agent_name: str, since: datetime
    ) -> list[dict[str, Any]]:
        """Return all actions for this agent with created_at >= since."""
        if self._db is None:
            return []
        since_iso = since.isoformat()
        try:
            cursor = await self._db.execute(
                """
                SELECT action_text, outcome, created_at
                FROM daily_actions
                WHERE agent_name = ? AND created_at >= ?
                ORDER BY created_at ASC
                """,
                (agent_name, since_iso),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return [{"action_text": r[0], "outcome": r[1], "created_at": r[2]} for r in rows]
        except Exception as exc:
            logger.warning("AgentJournal.get_actions_since_datetime failed", exc_info=True)
            return []

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
