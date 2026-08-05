"""Alembic 041 — PO lifecycle extensions + wo_costs + po_number_counters.

Revision ID: 041
Revises: 040
Create Date: 2026-08-03
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision = "041"
down_revision = "040"
branch_labels = None
depends_on = None


def _add_col_if_missing(table: str, column: sa.Column) -> None:
    bind = op.get_bind()
    cols = {c["name"] for c in inspect(bind).get_columns(table)}
    if column.name not in cols:
        op.add_column(table, column)


def upgrade() -> None:
    bind = op.get_bind()
    existing = set(inspect(bind).get_table_names())

    if "po_number_counters" not in existing:
        op.create_table(
            "po_number_counters",
            sa.Column("year", sa.Integer(), primary_key=True),
            sa.Column("last_n", sa.Integer(), nullable=False, server_default="0"),
        )

    if "purchase_orders" in existing:
        _add_col_if_missing(
            "purchase_orders", sa.Column("vendor_quote_id", sa.String(length=36), nullable=True)
        )
        _add_col_if_missing(
            "purchase_orders", sa.Column("wo_id", sa.String(length=36), nullable=True)
        )
        _add_col_if_missing(
            "purchase_orders", sa.Column("wo_number", sa.String(length=64), nullable=True)
        )
        _add_col_if_missing(
            "purchase_orders", sa.Column("company_id", sa.String(length=36), nullable=True)
        )
        _add_col_if_missing(
            "purchase_orders",
            sa.Column("component_class", sa.String(length=128), nullable=True),
        )
        _add_col_if_missing(
            "purchase_orders", sa.Column("legacy_alias", sa.String(length=32), nullable=True)
        )
        _add_col_if_missing(
            "purchase_orders", sa.Column("pdf_path", sa.String(length=512), nullable=True)
        )
        _add_col_if_missing("purchase_orders", sa.Column("sent_at", sa.DateTime(), nullable=True))
        _add_col_if_missing(
            "purchase_orders", sa.Column("source", sa.String(length=64), nullable=True)
        )

    if "wo_costs" not in existing:
        op.create_table(
            "wo_costs",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("wo_id", sa.String(length=36), nullable=False),
            sa.Column("wo_number", sa.String(length=64), nullable=False),
            sa.Column("company_id", sa.String(length=36), nullable=True),
            sa.Column("kind", sa.String(length=32), nullable=False, server_default="component"),
            sa.Column("amount", sa.Float(), nullable=False, server_default="0"),
            sa.Column("currency", sa.String(length=10), nullable=False, server_default="INR"),
            sa.Column("amount_inr", sa.Float(), nullable=True),
            sa.Column("source_ref", sa.String(length=128), nullable=True),
            sa.Column("source_kind", sa.String(length=32), nullable=True),
            sa.Column("vendor_id", sa.String(length=36), nullable=True),
            sa.Column("vendor_name", sa.String(length=255), nullable=True),
            sa.Column("component_class", sa.String(length=128), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.create_index("ix_wo_costs_wo_id", "wo_costs", ["wo_id"])
        op.create_index("ix_wo_costs_wo_number", "wo_costs", ["wo_number"])
        op.create_index("ix_wo_costs_source_ref", "wo_costs", ["source_ref"])


def downgrade() -> None:
    bind = op.get_bind()
    existing = set(inspect(bind).get_table_names())
    if "wo_costs" in existing:
        op.drop_index("ix_wo_costs_source_ref", table_name="wo_costs")
        op.drop_index("ix_wo_costs_wo_number", table_name="wo_costs")
        op.drop_index("ix_wo_costs_wo_id", table_name="wo_costs")
        op.drop_table("wo_costs")
    if "po_number_counters" in existing:
        op.drop_table("po_number_counters")
    # leave purchase_orders extension columns (non-destructive downgrade)
