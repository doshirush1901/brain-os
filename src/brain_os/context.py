"""Unified cross-channel context manager.

Tracks conversation state for each user across every channel (Email,
CLI, API) so that Brain OS maintains continuity regardless of where the
user interacts.

Storage is an in-memory dictionary for now; a persistent backend (Redis,
SQLite) can be swapped in later by subclassing or replacing the store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from brain_os.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class UserContext:
    """Mutable state bag for a single user across all channels."""

    user_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    active_goals: list[dict[str, Any]] = field(default_factory=list)
    last_channel: str = ""
    last_interaction_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )
    metadata: dict[str, Any] = field(default_factory=dict)


class UnifiedContextManager:
    """In-memory, cross-channel context store keyed by ``user_id``.

    ``user_id`` is a stable identifier for the human -- typically an email
    address or any opaque key the caller chooses.  The same ``user_id`` must be used across channels for
    continuity to work.

    Limits default from :class:`~ira.config.AppConfig` (``unified_context_max_*``)
    and can be overridden per instance for tests.
    """

    def __init__(
        self,
        *,
        max_history: int | None = None,
        max_users: int | None = None,
    ) -> None:
        app = get_settings().app
        self._max_history = (
            int(max_history) if max_history is not None else app.unified_context_max_history
        )
        self._max_users = int(max_users) if max_users is not None else app.unified_context_max_users
        self._store: dict[str, UserContext] = {}

    # ── read / write ─────────────────────────────────────────────────────

    def get(self, user_id: str) -> UserContext:
        """Return the context for *user_id*, creating one if absent.

        When the store exceeds ``max_users`` the least-recently
        interacted user is evicted to bound memory growth.
        """
        if user_id not in self._store:
            if len(self._store) >= self._max_users:
                oldest_key = min(
                    self._store,
                    key=lambda k: self._store[k].last_interaction_at,
                )
                del self._store[oldest_key]
            self._store[user_id] = UserContext(user_id=user_id)
        return self._store[user_id]

    def save(self, ctx: UserContext) -> None:
        """Persist an updated context back into the store."""
        self._store[ctx.user_id] = ctx

    # ── convenience mutators ─────────────────────────────────────────────

    def record_turn(
        self,
        user_id: str,
        channel: str,
        user_message: str,
        assistant_message: str,
    ) -> UserContext:
        """Append a user/assistant exchange and update timestamps.

        Returns the updated :class:`UserContext` so callers can inspect it
        without a second ``get`` call.
        """
        ctx = self.get(user_id)
        now = datetime.now(UTC)

        ctx.history.append(
            {
                "role": "user",
                "content": user_message,
                "channel": channel,
                "timestamp": now.isoformat(),
            }
        )
        ctx.history.append(
            {
                "role": "assistant",
                "content": assistant_message,
                "channel": channel,
                "timestamp": now.isoformat(),
            }
        )

        if len(ctx.history) > self._max_history:
            ctx.history = ctx.history[-self._max_history :]

        ctx.last_channel = channel
        ctx.last_interaction_at = now
        self.save(ctx)
        return ctx

    def set_active_goal(
        self,
        user_id: str,
        goal: dict[str, Any],
    ) -> None:
        """Replace or add an active goal for the user."""
        ctx = self.get(user_id)
        ctx.active_goals = [g for g in ctx.active_goals if g.get("id") != goal.get("id")]
        ctx.active_goals.append(goal)
        self.save(ctx)

    def clear_goal(self, user_id: str, goal_id: str) -> None:
        ctx = self.get(user_id)
        ctx.active_goals = [g for g in ctx.active_goals if g.get("id") != goal_id]
        self.save(ctx)

    # ── query helpers ────────────────────────────────────────────────────

    def recent_history(
        self,
        user_id: str,
        limit: int = 10,
        channel: str | None = None,
    ) -> list[dict[str, str]]:
        """Return the last *limit* messages, optionally filtered by channel."""
        ctx = self.get(user_id)
        if channel is not None:
            filtered = [m for m in ctx.history if m.get("channel") == channel]
        else:
            filtered = ctx.history
        return filtered[-limit:]

    def all_users(self) -> list[str]:
        return list(self._store.keys())
