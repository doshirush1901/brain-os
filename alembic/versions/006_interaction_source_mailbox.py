"""Add interactions.source_mailbox for dual Gmail CRM attribution.

Revision ID: 006
Revises: 005
Create Date: 2026-04-13

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = {c.get("name") for c in inspector.get_columns("interactions")}
    if "source_mailbox" not in existing_cols:
        op.add_column(
            "interactions",
            sa.Column("source_mailbox", sa.String(length=320), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = {c.get("name") for c in inspector.get_columns("interactions")}
    if "source_mailbox" in existing_cols:
        op.drop_column("interactions", "source_mailbox")
