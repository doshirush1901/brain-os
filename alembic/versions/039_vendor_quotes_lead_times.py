"""ERP Stage 6 Wave 1 — vendor_quotes, component_lead_times, wo_critical_items.

Revision ID: 039
Revises: 038
Create Date: 2026-08-03
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision = "039"
down_revision = "038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = set(inspect(bind).get_table_names())

    if "vendor_quotes" not in existing:
        op.create_table(
            "vendor_quotes",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("vendor_id", sa.String(length=36), nullable=True),
            sa.Column("company_id", sa.String(length=36), nullable=True),
            sa.Column("vendor_name", sa.String(length=255), nullable=False),
            sa.Column("domain", sa.String(length=255), nullable=True),
            sa.Column("wo_id", sa.String(length=36), nullable=True),
            sa.Column("wo_number", sa.String(length=64), nullable=True),
            sa.Column("thread_id", sa.String(length=128), nullable=True),
            sa.Column("message_id", sa.String(length=128), nullable=True),
            sa.Column("subject", sa.String(length=500), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="parsed"),
            sa.Column("kind", sa.String(length=32), nullable=False, server_default="vendor_quote"),
            sa.Column("currency", sa.String(length=10), nullable=True),
            sa.Column("total", sa.Float(), nullable=True),
            sa.Column("production_days", sa.Integer(), nullable=True),
            sa.Column("freight_days", sa.Integer(), nullable=True),
            sa.Column("freight_mode", sa.String(length=64), nullable=True),
            sa.Column("payment_terms", sa.String(length=128), nullable=True),
            sa.Column("validity", sa.String(length=128), nullable=True),
            sa.Column("line_items", sa.JSON(), nullable=False),
            sa.Column("field_confidence", sa.JSON(), nullable=True),
            sa.Column("payload", sa.JSON(), nullable=True),
            sa.Column("source", sa.String(length=64), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("received_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.create_index("ix_vendor_quotes_vendor_id", "vendor_quotes", ["vendor_id"])
        op.create_index("ix_vendor_quotes_company_id", "vendor_quotes", ["company_id"])
        op.create_index("ix_vendor_quotes_wo_number", "vendor_quotes", ["wo_number"])
        op.create_index("ix_vendor_quotes_thread_id", "vendor_quotes", ["thread_id"])
        op.create_index("ix_vendor_quotes_message_id", "vendor_quotes", ["message_id"])
        op.create_index("ix_vendor_quotes_kind", "vendor_quotes", ["kind"])

    if "component_lead_times" not in existing:
        op.create_table(
            "component_lead_times",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("vendor_id", sa.String(length=36), nullable=True),
            sa.Column("company_id", sa.String(length=36), nullable=True),
            sa.Column("component_class", sa.String(length=128), nullable=False),
            sa.Column("quoted_days_min", sa.Integer(), nullable=True),
            sa.Column("quoted_days_max", sa.Integer(), nullable=True),
            sa.Column("observed_days", sa.Integer(), nullable=True),
            sa.Column(
                "source", sa.String(length=32), nullable=False, server_default="vendor_quote"
            ),
            sa.Column("observed_at", sa.DateTime(), nullable=True),
            sa.Column("wo_id", sa.String(length=36), nullable=True),
            sa.Column("wo_number", sa.String(length=64), nullable=True),
            sa.Column("vendor_quote_id", sa.String(length=36), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.create_index(
            "ix_component_lead_times_class", "component_lead_times", ["component_class"]
        )
        op.create_index("ix_component_lead_times_vendor", "component_lead_times", ["vendor_id"])
        op.create_index("ix_component_lead_times_wo", "component_lead_times", ["wo_number"])

    if "wo_critical_items" not in existing:
        op.create_table(
            "wo_critical_items",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("wo_id", sa.String(length=36), nullable=False),
            sa.Column("wo_number", sa.String(length=64), nullable=False),
            sa.Column("company_id", sa.String(length=36), nullable=True),
            sa.Column("component_class", sa.String(length=128), nullable=False),
            sa.Column("vendor", sa.String(length=255), nullable=True),
            sa.Column("vendor_id", sa.String(length=36), nullable=True),
            sa.Column(
                "needed_by_stage",
                sa.String(length=32),
                nullable=False,
                server_default="procurement_build",
            ),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="needed"),
            sa.Column("vendor_quote_id", sa.String(length=36), nullable=True),
            sa.Column("eta_days_min", sa.Integer(), nullable=True),
            sa.Column("eta_days_max", sa.Integer(), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.create_index("ix_wo_critical_items_wo_id", "wo_critical_items", ["wo_id"])
        op.create_index("ix_wo_critical_items_wo_number", "wo_critical_items", ["wo_number"])
        op.create_index("ix_wo_critical_items_class", "wo_critical_items", ["component_class"])
        op.create_index("ix_wo_critical_items_status", "wo_critical_items", ["status"])


def downgrade() -> None:
    bind = op.get_bind()
    existing = set(inspect(bind).get_table_names())
    if "wo_critical_items" in existing:
        op.drop_index("ix_wo_critical_items_status", table_name="wo_critical_items")
        op.drop_index("ix_wo_critical_items_class", table_name="wo_critical_items")
        op.drop_index("ix_wo_critical_items_wo_number", table_name="wo_critical_items")
        op.drop_index("ix_wo_critical_items_wo_id", table_name="wo_critical_items")
        op.drop_table("wo_critical_items")
    if "component_lead_times" in existing:
        op.drop_index("ix_component_lead_times_wo", table_name="component_lead_times")
        op.drop_index("ix_component_lead_times_vendor", table_name="component_lead_times")
        op.drop_index("ix_component_lead_times_class", table_name="component_lead_times")
        op.drop_table("component_lead_times")
    if "vendor_quotes" in existing:
        op.drop_index("ix_vendor_quotes_kind", table_name="vendor_quotes")
        op.drop_index("ix_vendor_quotes_message_id", table_name="vendor_quotes")
        op.drop_index("ix_vendor_quotes_thread_id", table_name="vendor_quotes")
        op.drop_index("ix_vendor_quotes_wo_number", table_name="vendor_quotes")
        op.drop_index("ix_vendor_quotes_company_id", table_name="vendor_quotes")
        op.drop_index("ix_vendor_quotes_vendor_id", table_name="vendor_quotes")
        op.drop_table("vendor_quotes")
