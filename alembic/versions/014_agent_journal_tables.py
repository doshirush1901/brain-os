"""Agent journal daily actions and reflections.

Revision ID: 014
Revises: 013
Create Date: 2026-05-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_journal_daily_actions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("agent_name", sa.String(length=64), nullable=False),
        sa.Column("date", sa.String(length=16), nullable=False),
        sa.Column("action_text", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_agent_journal_daily_actions_agent_date",
        "agent_journal_daily_actions",
        ["agent_name", "date"],
    )
    op.create_table(
        "agent_journal_entries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("agent_name", sa.String(length=64), nullable=False),
        sa.Column("date", sa.String(length=16), nullable=False),
        sa.Column("reflection_text", sa.Text(), nullable=False),
        sa.Column("mood", sa.Text(), server_default="", nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_agent_journal_entries_agent_date",
        "agent_journal_entries",
        ["agent_name", "date"],
    )


def downgrade() -> None:
    op.drop_index("idx_agent_journal_entries_agent_date", table_name="agent_journal_entries")
    op.drop_table("agent_journal_entries")
    op.drop_index(
        "idx_agent_journal_daily_actions_agent_date", table_name="agent_journal_daily_actions"
    )
    op.drop_table("agent_journal_daily_actions")
