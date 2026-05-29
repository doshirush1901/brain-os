"""Atlas project logbook (projects + events).

Revision ID: 015
Revises: 014
Create Date: 2026-05-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "atlas_projects",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("customer", sa.Text(), server_default="", nullable=False),
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "atlas_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["atlas_projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_atlas_events_project_id", "atlas_events", ["project_id"])


def downgrade() -> None:
    op.drop_index("idx_atlas_events_project_id", table_name="atlas_events")
    op.drop_table("atlas_events")
    op.drop_table("atlas_projects")
