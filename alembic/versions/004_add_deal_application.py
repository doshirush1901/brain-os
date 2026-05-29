"""Add application to deals.

Revision ID: 004
Revises: 003
Create Date: 2026-03-14

"""

import sqlalchemy as sa
from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("deals", sa.Column("application", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("deals", "application")
