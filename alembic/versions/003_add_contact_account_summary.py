"""Add account_summary to contacts.

Revision ID: 003
Revises: 002
Create Date: 2026-03-13

"""

import sqlalchemy as sa
from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("contacts", sa.Column("account_summary", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("contacts", "account_summary")
