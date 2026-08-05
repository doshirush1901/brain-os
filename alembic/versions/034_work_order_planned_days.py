"""ERP Stage 2 Wave 3 — per-stage planned_days on work_orders.

Revision ID: 034
Revises: 033
Create Date: 2026-08-02

JSON map of stage → planned calendar days (NRC-mined defaults applied in app).
Null planned_days ⇒ delay detection falls back to event-staleness only.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "work_orders",
        sa.Column("planned_days", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("work_orders", "planned_days")
