"""ERP Stage 2 Wave 4 — warranty_months on work_orders (handover ritual).

Revision ID: 035
Revises: 034
Create Date: 2026-08-02

Per-deal warranty length (months), mined 12–24 mo range. Used at handover to
stamp machine_assets.warranty_end.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "035"
down_revision = "034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "work_orders",
        sa.Column("warranty_months", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("work_orders", "warranty_months")
