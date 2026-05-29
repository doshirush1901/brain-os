"""Wave 2 memory stores (tool invocations, pending memory, conversation, etc.).

Revision ID: 018
Revises: 017
Create Date: 2026-05-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tool_invocations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ts", sa.Float(), nullable=False),
        sa.Column("agent", sa.String(length=128), nullable=False),
        sa.Column("tool", sa.String(length=256), nullable=False),
        sa.Column("success", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("run_id", sa.String(length=128), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_tool_inv_ts", "tool_invocations", ["ts"])
    op.create_index("idx_tool_inv_agent_tool", "tool_invocations", ["agent", "tool"])

    op.create_table(
        "pending_memory_hints",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=80), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="pending", nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("channel", sa.String(length=64), nullable=False),
        sa.Column("started_at", sa.Text(), nullable=False),
        sa.Column("last_message_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_conversations_user_channel", "conversations", ["user_id", "channel"])

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_messages_conversation", "messages", ["conversation_id"])

    op.create_table(
        "relationships",
        sa.Column("contact_id", sa.String(length=256), nullable=False),
        sa.Column("warmth_level", sa.String(length=32), server_default="STRANGER", nullable=False),
        sa.Column("interaction_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("memorable_moments", sa.Text(), server_default="[]", nullable=False),
        sa.Column("learned_preferences", sa.Text(), server_default="{}", nullable=False),
        sa.Column("first_interaction", sa.Text(), nullable=True),
        sa.Column("last_interaction", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("contact_id"),
    )

    op.create_table(
        "goals",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("goal_type", sa.String(length=64), nullable=False),
        sa.Column("contact_id", sa.String(length=256), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="ACTIVE", nullable=False),
        sa.Column("required_slots", sa.Text(), nullable=False),
        sa.Column("progress", sa.Float(), server_default="0", nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("completed_at", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_goals_contact_status", "goals", ["contact_id", "status"])

    op.create_table(
        "procedures",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trigger_pattern", sa.Text(), nullable=False),
        sa.Column("steps", sa.Text(), nullable=False),
        sa.Column("success_rate", sa.Float(), server_default="1", nullable=False),
        sa.Column("times_used", sa.Integer(), server_default="1", nullable=False),
        sa.Column("last_used", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_procedures_pattern", "procedures", ["trigger_pattern"])

    op.create_table(
        "episodes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("narrative", sa.Text(), nullable=False),
        sa.Column("key_topics", sa.Text(), nullable=False),
        sa.Column("decisions", sa.Text(), nullable=False),
        sa.Column("commitments", sa.Text(), nullable=False),
        sa.Column("emotional_tone", sa.Text(), nullable=False),
        sa.Column("relationship_impact", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_episodes_user", "episodes", ["user_id"])

    op.create_table(
        "cursor_sessions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("timestamp", sa.Float(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("agents_used", sa.Text(), nullable=False),
        sa.Column("response_summary", sa.Text(), nullable=False),
        sa.Column("tool_calls", sa.Text(), server_default="[]", nullable=False),
        sa.Column("sources", sa.Text(), server_default="[]", nullable=False),
        sa.Column("email_threads", sa.Text(), server_default="[]", nullable=False),
        sa.Column("user_feedback", sa.Text(), nullable=True),
        sa.Column("run_id", sa.String(length=128), nullable=True),
        sa.Column("work_context_json", sa.Text(), nullable=True),
        sa.Column("learning_meta_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_cursor_sessions_run_id", "cursor_sessions", ["run_id"])
    op.create_index("idx_cursor_sessions_timestamp", "cursor_sessions", ["timestamp"])


def downgrade() -> None:
    op.drop_index("idx_cursor_sessions_timestamp", table_name="cursor_sessions")
    op.drop_index("idx_cursor_sessions_run_id", table_name="cursor_sessions")
    op.drop_table("cursor_sessions")
    op.drop_index("idx_episodes_user", table_name="episodes")
    op.drop_table("episodes")
    op.drop_index("idx_procedures_pattern", table_name="procedures")
    op.drop_table("procedures")
    op.drop_index("idx_goals_contact_status", table_name="goals")
    op.drop_table("goals")
    op.drop_table("relationships")
    op.drop_index("idx_messages_conversation", table_name="messages")
    op.drop_table("messages")
    op.drop_index("idx_conversations_user_channel", table_name="conversations")
    op.drop_table("conversations")
    op.drop_table("pending_memory_hints")
    op.drop_index("idx_tool_inv_agent_tool", table_name="tool_invocations")
    op.drop_index("idx_tool_inv_ts", table_name="tool_invocations")
    op.drop_table("tool_invocations")
