"""Document ingest idempotency ledger.

Revision ID: 013
Revises: 012
Create Date: 2026-05-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingested_files",
        sa.Column("path", sa.String(length=2048), nullable=False),
        sa.Column("hash", sa.String(length=128), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("ingested_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("path"),
    )


def downgrade() -> None:
    op.drop_table("ingested_files")
