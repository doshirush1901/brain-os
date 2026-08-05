"""ERP Stage 6 Wave 2 — vendor_rfqs outbound tracker.

Revision ID: 040
Revises: 039
Create Date: 2026-08-03
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision = "040"
down_revision = "039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = set(inspect(bind).get_table_names())
    if "vendor_rfqs" in existing:
        return
    op.create_table(
        "vendor_rfqs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("wo_id", sa.String(length=36), nullable=True),
        sa.Column("wo_number", sa.String(length=64), nullable=True),
        sa.Column("company_id", sa.String(length=36), nullable=True),
        sa.Column("component_class", sa.String(length=128), nullable=False),
        sa.Column("vendors_contacted", sa.JSON(), nullable=False),
        sa.Column("quoted_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="sent"),
        sa.Column("chosen_vendor_id", sa.String(length=36), nullable=True),
        sa.Column("chosen_vendor_name", sa.String(length=255), nullable=True),
        sa.Column("vendor_quote_ids", sa.JSON(), nullable=False),
        sa.Column("subject", sa.String(length=500), nullable=True),
        sa.Column("thread_id", sa.String(length=128), nullable=True),
        sa.Column("message_id", sa.String(length=128), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("sent_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_vendor_rfqs_wo_id", "vendor_rfqs", ["wo_id"])
    op.create_index("ix_vendor_rfqs_wo_number", "vendor_rfqs", ["wo_number"])
    op.create_index("ix_vendor_rfqs_component_class", "vendor_rfqs", ["component_class"])
    op.create_index("ix_vendor_rfqs_status", "vendor_rfqs", ["status"])


def downgrade() -> None:
    bind = op.get_bind()
    existing = set(inspect(bind).get_table_names())
    if "vendor_rfqs" not in existing:
        return
    op.drop_index("ix_vendor_rfqs_status", table_name="vendor_rfqs")
    op.drop_index("ix_vendor_rfqs_component_class", table_name="vendor_rfqs")
    op.drop_index("ix_vendor_rfqs_wo_number", table_name="vendor_rfqs")
    op.drop_index("ix_vendor_rfqs_wo_id", table_name="vendor_rfqs")
    op.drop_table("vendor_rfqs")
