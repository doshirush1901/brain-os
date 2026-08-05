"""ERP Stage 3 Wave 3 — relationship-aware payment chase ladder on milestones.

Revision ID: 033
Revises: 032
Create Date: 2026-08-02

- wo_payment_milestones.chase_stage (none|nudge_in_update|reminder|founder_call|dispatch_hold)
- wo_payment_milestones.chase_stage_at
- wo_payment_milestones.chase_proposal_id (outbound batch / inbox card id)
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None

_CHASE_STAGES = (
    "none",
    "nudge_in_update",
    "reminder",
    "founder_call",
    "dispatch_hold",
)


def upgrade() -> None:
    op.add_column(
        "wo_payment_milestones",
        sa.Column(
            "chase_stage",
            sa.String(length=32),
            nullable=False,
            server_default="none",
        ),
    )
    op.add_column(
        "wo_payment_milestones",
        sa.Column("chase_stage_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "wo_payment_milestones",
        sa.Column("chase_proposal_id", sa.String(length=128), nullable=True),
    )
    op.create_check_constraint(
        "ck_wo_payment_milestones_chase_stage",
        "wo_payment_milestones",
        "chase_stage IN (" + ", ".join(f"'{s}'" for s in _CHASE_STAGES) + ")",
    )
    op.create_index(
        "ix_wo_payment_milestones_chase_stage",
        "wo_payment_milestones",
        ["chase_stage"],
    )


def downgrade() -> None:
    op.drop_index("ix_wo_payment_milestones_chase_stage", table_name="wo_payment_milestones")
    op.drop_constraint(
        "ck_wo_payment_milestones_chase_stage",
        "wo_payment_milestones",
        type_="check",
    )
    op.drop_column("wo_payment_milestones", "chase_proposal_id")
    op.drop_column("wo_payment_milestones", "chase_stage_at")
    op.drop_column("wo_payment_milestones", "chase_stage")
