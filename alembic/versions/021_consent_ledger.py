"""Per-contact consent ledger table (Revenue A-grade Phase 5).

Durable mirror of ``data/revenue_mode/consent_ledger.jsonl`` — opt-out
enforcement reads the file; this table is the auditable record.

Revision ID: 021
Revises: 020
Create Date: 2026-06-12
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None

_TABLE = "consent_events"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _TABLE in inspector.get_table_names():
        return
    op.create_table(
        _TABLE,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("event", sa.String(length=20), nullable=False),
        sa.Column("channel", sa.String(length=20), nullable=False, server_default="email"),
        sa.Column(
            "basis",
            sa.String(length=40),
            nullable=False,
            server_default="legitimate_interest",
        ),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("campaign_id", sa.String(length=255), nullable=True),
        sa.Column("opt_out_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_consent_events_email", _TABLE, ["email"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _TABLE not in inspector.get_table_names():
        return
    op.drop_index("ix_consent_events_email", table_name=_TABLE)
    op.drop_table(_TABLE)
