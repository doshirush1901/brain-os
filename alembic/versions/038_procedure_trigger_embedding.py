"""Alembic: cache Voyage embeddings on procedural trigger_patterns."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision = "038"
down_revision = "037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    cols = {c["name"] for c in inspect(bind).get_columns("procedures")}
    if "trigger_embedding" not in cols:
        op.add_column(
            "procedures",
            sa.Column("trigger_embedding", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    cols = {c["name"] for c in inspect(bind).get_columns("procedures")}
    if "trigger_embedding" in cols:
        op.drop_column("procedures", "trigger_embedding")
