"""Learning hub feedback records.

Revision ID: 017
Revises: 016
Create Date: 2026-05-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "learning_feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("interaction_id", sa.String(length=512), nullable=False),
        sa.Column("feedback_score", sa.Integer(), nullable=False),
        sa.Column("correction", sa.Text(), nullable=True),
        sa.Column("correction_analysis", sa.Text(), server_default="{}", nullable=False),
        sa.Column("gap_analysis", sa.Text(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_learning_feedback_interaction_id",
        "learning_feedback",
        ["interaction_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_learning_feedback_interaction_id", table_name="learning_feedback")
    op.drop_table("learning_feedback")
