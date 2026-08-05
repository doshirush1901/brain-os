"""Widen aftermarket_events.kind — warranty_notice exceeds varchar(14).

Revision ID: 036
Revises: 035
Create Date: 2026-08-02

Live PG had kind as VARCHAR(14) from early Enum create_all (max name
``warranty_claim``). Wave-4 ``WARRANTY_NOTICE`` / ``warranty_notice`` need ≥15.
Canonical width matches alembic 022 (String(40)).
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "036"
down_revision = "035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "aftermarket_events",
        "kind",
        existing_type=sa.String(length=14),
        type_=sa.String(length=40),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "aftermarket_events",
        "kind",
        existing_type=sa.String(length=40),
        type_=sa.String(length=14),
        existing_nullable=False,
    )
