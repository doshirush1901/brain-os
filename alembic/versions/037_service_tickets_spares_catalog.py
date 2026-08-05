"""ERP Stage 4 Wave 2 — service_tickets + spares_catalog.

Revision ID: 037
Revises: 036
Create Date: 2026-08-02
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision = "037"
down_revision = "036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = set(inspect(bind).get_table_names())

    if "service_tickets" not in existing:
        op.create_table(
            "service_tickets",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("company_name", sa.String(length=255), nullable=False),
            sa.Column("company_id", sa.String(length=36), nullable=True),
            sa.Column("machine_asset_id", sa.String(length=36), nullable=True),
            sa.Column("machine_model", sa.String(length=255), nullable=True),
            sa.Column("kind", sa.String(length=32), nullable=False),
            sa.Column("severity", sa.String(length=32), nullable=False, server_default="routine"),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="open"),
            sa.Column("subject", sa.String(length=500), nullable=True),
            sa.Column("thread_ref", sa.String(length=128), nullable=True),
            sa.Column("contact_email", sa.String(length=320), nullable=True),
            sa.Column("quote_id", sa.String(length=36), nullable=True),
            sa.Column("quote_number", sa.String(length=64), nullable=True),
            sa.Column("payload", sa.JSON(), nullable=True),
            sa.Column("manan_task_line", sa.Text(), nullable=True),
            sa.Column("notify_message_id", sa.String(length=128), nullable=True),
            sa.Column("source", sa.String(length=64), nullable=True),
            sa.Column("opened_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.create_index("ix_service_tickets_company_name", "service_tickets", ["company_name"])
        op.create_index("ix_service_tickets_kind", "service_tickets", ["kind"])
        op.create_index("ix_service_tickets_severity", "service_tickets", ["severity"])
        op.create_index("ix_service_tickets_thread_ref", "service_tickets", ["thread_ref"])
        op.create_index("ix_service_tickets_asset", "service_tickets", ["machine_asset_id"])

    if "spares_catalog" not in existing:
        op.create_table(
            "spares_catalog",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("sku", sa.String(length=120), nullable=True),
            sa.Column("description", sa.String(length=500), nullable=False),
            sa.Column("part_description", sa.Text(), nullable=True),
            sa.Column("make", sa.String(length=120), nullable=True),
            sa.Column("category", sa.String(length=64), nullable=True),
            sa.Column("machine_model", sa.String(length=120), nullable=True),
            sa.Column("machine_family", sa.String(length=40), nullable=True),
            sa.Column("unit_price", sa.Float(), nullable=True),
            sa.Column("currency", sa.String(length=10), nullable=False, server_default="EUR"),
            sa.Column("qty_default", sa.Integer(), nullable=True),
            sa.Column("source", sa.String(length=255), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        )
        op.create_index("ix_spares_catalog_sku", "spares_catalog", ["sku"])
        op.create_index("ix_spares_catalog_family", "spares_catalog", ["machine_family"])
        op.create_index("ix_spares_catalog_desc", "spares_catalog", ["description"])


def downgrade() -> None:
    bind = op.get_bind()
    existing = set(inspect(bind).get_table_names())
    if "spares_catalog" in existing:
        op.drop_index("ix_spares_catalog_desc", table_name="spares_catalog")
        op.drop_index("ix_spares_catalog_family", table_name="spares_catalog")
        op.drop_index("ix_spares_catalog_sku", table_name="spares_catalog")
        op.drop_table("spares_catalog")
    if "service_tickets" in existing:
        op.drop_index("ix_service_tickets_asset", table_name="service_tickets")
        op.drop_index("ix_service_tickets_thread_ref", table_name="service_tickets")
        op.drop_index("ix_service_tickets_severity", table_name="service_tickets")
        op.drop_index("ix_service_tickets_kind", table_name="service_tickets")
        op.drop_index("ix_service_tickets_company_name", table_name="service_tickets")
        op.drop_table("service_tickets")
