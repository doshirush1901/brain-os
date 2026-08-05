"""Operator context runs, pipeline run records, usage metric events.

Revision ID: 023
Revises: 022
Create Date: 2026-08-01
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "023"
down_revision = "022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "operator_context_runs",
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("ts", sa.Float(), nullable=False),
        sa.Column("company_key", sa.String(length=256), nullable=False),
        sa.Column("domain", sa.String(length=128), nullable=True),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("machine_model", sa.String(length=128), nullable=True),
        sa.Column("outcome", sa.String(length=32), nullable=True),
        sa.Column("success", sa.Integer(), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index(
        "idx_opctx_company_ts",
        "operator_context_runs",
        ["company_key", "ts"],
    )
    op.create_index(
        "idx_opctx_domain_ts",
        "operator_context_runs",
        ["domain", "ts"],
    )

    op.create_table(
        "run_records",
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("ts", sa.Float(), nullable=False),
        sa.Column("channel", sa.String(length=64), nullable=True),
        sa.Column("outcome", sa.String(length=32), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("idx_run_records_ts", "run_records", ["ts"])
    op.create_index(
        "idx_run_records_channel_ts",
        "run_records",
        ["channel", "ts"],
    )

    # usage_metrics.py is a reader over cursor_sessions; scaffold event counters.
    op.create_table(
        "usage_metric_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("payload_json", sa.Text(), server_default="{}", nullable=False),
        sa.Column("ts", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_usage_metric_events_name_ts", "usage_metric_events", ["name", "ts"])


def downgrade() -> None:
    op.drop_index("idx_usage_metric_events_name_ts", table_name="usage_metric_events")
    op.drop_table("usage_metric_events")
    op.drop_index("idx_run_records_channel_ts", table_name="run_records")
    op.drop_index("idx_run_records_ts", table_name="run_records")
    op.drop_table("run_records")
    op.drop_index("idx_opctx_domain_ts", table_name="operator_context_runs")
    op.drop_index("idx_opctx_company_ts", table_name="operator_context_runs")
    op.drop_table("operator_context_runs")
