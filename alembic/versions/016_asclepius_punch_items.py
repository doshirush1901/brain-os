"""Asclepius punch-list items.

Revision ID: 016
Revises: 015
Create Date: 2026-05-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "asclepius_punch_items",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("project_name", sa.String(length=512), nullable=False),
        sa.Column("phase", sa.String(length=32), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="OPEN", nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("closed_at", sa.Text(), nullable=True),
        sa.Column("resolution", sa.Text(), server_default="", nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_asclepius_punch_items_project_name",
        "asclepius_punch_items",
        ["project_name"],
    )


def downgrade() -> None:
    op.drop_index("idx_asclepius_punch_items_project_name", table_name="asclepius_punch_items")
    op.drop_table("asclepius_punch_items")
