"""Postgres dual-write store feature flags."""

from __future__ import annotations

from pydantic import Field


class AppStoresPgMixin:
    """Mixin slice of :class:`~brain_os.config.app.AppConfig` (MOVE only)."""

    pg_store_ingest_enabled: bool = False

    pg_store_ingest_read: bool = False

    pg_store_corrections_enabled: bool = False

    pg_store_corrections_read: bool = False

    pg_store_agent_journal_enabled: bool = False

    pg_store_agent_journal_read: bool = False

    pg_store_atlas_logbook_enabled: bool = False

    pg_store_atlas_logbook_read: bool = False

    pg_store_asclepius_punch_enabled: bool = False

    pg_store_asclepius_punch_read: bool = False

    pg_store_learning_feedback_enabled: bool = False

    pg_store_learning_feedback_read: bool = False

    pg_store_conversation_enabled: bool = False

    pg_store_conversation_read: bool = False

    pg_store_relationship_enabled: bool = False

    pg_store_relationship_read: bool = False

    pg_store_pending_memory_enabled: bool = False

    pg_store_pending_memory_read: bool = False

    pg_store_tool_invocations_enabled: bool = False

    pg_store_tool_invocations_read: bool = False

    pg_store_procedural_enabled: bool = False

    pg_store_procedural_read: bool = False

    pg_store_cursor_sessions_enabled: bool = False

    pg_store_cursor_sessions_read: bool = False

    pg_store_episodes_enabled: bool = False

    pg_store_episodes_read: bool = False

    pg_store_goals_enabled: bool = False

    pg_store_goals_read: bool = False

    pg_store_operator_context_enabled: bool = False

    pg_store_operator_context_read: bool = False

    pg_store_run_records_enabled: bool = False

    pg_store_run_records_read: bool = False
