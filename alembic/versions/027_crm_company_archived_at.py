"""Add companies.archived_at for soft-archive merges (Wave 2).

Revision ID: 027
Revises: 026
Create Date: 2026-08-02
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "027"
down_revision = "026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("companies", sa.Column("archived_at", sa.DateTime(), nullable=True))
    op.create_index("ix_companies_archived_at", "companies", ["archived_at"])


def downgrade() -> None:
    op.drop_index("ix_companies_archived_at", table_name="companies")
    op.drop_column("companies", "archived_at")
