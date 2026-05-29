"""Entity score history for interconnection analytics.

Revision ID: 008
Revises: 007
Create Date: 2026-04-27

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())
    if "entity_score_history" not in existing_tables:
        op.create_table(
            "entity_score_history",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("entity_type", sa.String(length=80), nullable=False),
            sa.Column("entity_id", sa.String(length=255), nullable=False),
            sa.Column("score_type", sa.String(length=80), nullable=False),
            sa.Column("score_value", sa.Float(), nullable=False),
            sa.Column("score_breakdown_json", sa.JSON(), nullable=True),
            sa.Column("source", sa.String(length=255), nullable=True),
            sa.Column("run_id", sa.String(length=128), nullable=True),
            sa.Column("model_version", sa.String(length=80), nullable=True),
            sa.Column("computed_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        )
        inspector = sa.inspect(bind)
    indexes = {i.get("name") for i in inspector.get_indexes("entity_score_history")}
    if "ix_entity_score_history_entity_time" not in indexes:
        op.create_index(
            "ix_entity_score_history_entity_time",
            "entity_score_history",
            ["entity_type", "entity_id", "computed_at"],
        )
    if "ix_entity_score_history_score_time" not in indexes:
        op.create_index(
            "ix_entity_score_history_score_time",
            "entity_score_history",
            ["score_type", "computed_at"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())
    if "entity_score_history" in existing_tables:
        indexes = {i.get("name") for i in inspector.get_indexes("entity_score_history")}
        if "ix_entity_score_history_score_time" in indexes:
            op.drop_index("ix_entity_score_history_score_time", table_name="entity_score_history")
        if "ix_entity_score_history_entity_time" in indexes:
            op.drop_index("ix_entity_score_history_entity_time", table_name="entity_score_history")
        op.drop_table("entity_score_history")
