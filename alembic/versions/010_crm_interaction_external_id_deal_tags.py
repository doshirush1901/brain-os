"""CRM: interaction external_message_id + deal tags JSON.

Revision ID: 010
Revises: 009
Create Date: 2026-04-29
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    ix_cols = {c.get("name") for c in inspector.get_columns("interactions")}
    if "external_message_id" not in ix_cols:
        op.add_column(
            "interactions",
            sa.Column("external_message_id", sa.String(length=255), nullable=True),
        )
        op.create_index(
            "ix_interactions_external_message_id",
            "interactions",
            ["external_message_id"],
            unique=False,
        )
    deal_cols = {c.get("name") for c in inspector.get_columns("deals")}
    if "tags" not in deal_cols:
        op.add_column("deals", sa.Column("tags", sa.JSON(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    deal_cols = {c.get("name") for c in inspector.get_columns("deals")}
    if "tags" in deal_cols:
        op.drop_column("deals", "tags")
    ix_cols = {c.get("name") for c in inspector.get_columns("interactions")}
    if "external_message_id" in ix_cols:
        op.drop_index("ix_interactions_external_message_id", table_name="interactions")
        op.drop_column("interactions", "external_message_id")
