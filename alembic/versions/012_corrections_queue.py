"""Nemesis / feedback correction queue.

Revision ID: 012
Revises: 011
Create Date: 2026-05-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "corrections",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("entity", sa.String(length=512), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("severity", sa.String(length=16), server_default="MEDIUM", nullable=False),
        sa.Column("old_value", sa.Text(), server_default="", nullable=False),
        sa.Column("new_value", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=128), server_default="unknown", nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="pending", nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_corrections_status", "corrections", ["status"])
    op.create_index("idx_corrections_entity", "corrections", ["entity"])


def downgrade() -> None:
    op.drop_index("idx_corrections_entity", table_name="corrections")
    op.drop_index("idx_corrections_status", table_name="corrections")
    op.drop_table("corrections")
