"""CPQ Wave 4 — quote validity, lost_reason, confidence, HISTORICAL.

Revision ID: 031
Revises: 030
Create Date: 2026-08-02
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "quotes",
        sa.Column("validity_days", sa.Integer(), nullable=False, server_default="30"),
    )
    op.add_column("quotes", sa.Column("valid_until", sa.DateTime(), nullable=True))
    op.add_column("quotes", sa.Column("lost_reason", sa.String(length=64), nullable=True))
    op.add_column("quotes", sa.Column("confidence", sa.Float(), nullable=True))
    op.create_index("ix_quotes_valid_until", "quotes", ["valid_until"])
    op.create_index("ix_quotes_status_sent_at", "quotes", ["status", "sent_at"])


def downgrade() -> None:
    op.drop_index("ix_quotes_status_sent_at", table_name="quotes")
    op.drop_index("ix_quotes_valid_until", table_name="quotes")
    op.drop_column("quotes", "confidence")
    op.drop_column("quotes", "lost_reason")
    op.drop_column("quotes", "valid_until")
    op.drop_column("quotes", "validity_days")
