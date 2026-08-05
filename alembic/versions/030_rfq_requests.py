"""CPQ Wave 3 — rfq_requests intake table.

Revision ID: 030
Revises: 029
Create Date: 2026-08-02
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rfq_requests",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("company_name", sa.String(length=255), nullable=False),
        sa.Column("company_id", sa.String(length=36), nullable=True),
        sa.Column("contact_email", sa.String(length=320), nullable=True),
        sa.Column("contact_name", sa.String(length=255), nullable=True),
        sa.Column("deal_id", sa.String(length=36), nullable=True),
        sa.Column("quote_id", sa.String(length=36), nullable=True),
        sa.Column("thread_id", sa.String(length=128), nullable=True),
        sa.Column("message_id", sa.String(length=128), nullable=True),
        sa.Column("subject", sa.String(length=500), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="parsed"),
        sa.Column("machine_type", sa.String(length=128), nullable=True),
        sa.Column("machine_series", sa.String(length=64), nullable=True),
        sa.Column("machine_hint", sa.String(length=255), nullable=True),
        sa.Column("requirements", sa.JSON(), nullable=False),
        sa.Column("field_confidence", sa.JSON(), nullable=True),
        sa.Column("missing_fields", sa.JSON(), nullable=True),
        sa.Column("attachments", sa.JSON(), nullable=True),
        sa.Column("gap_questions", sa.JSON(), nullable=True),
        sa.Column("gap_draft_path", sa.String(length=1024), nullable=True),
        sa.Column("notify_message_id", sa.String(length=128), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("received_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_rfq_requests_company_name", "rfq_requests", ["company_name"])
    op.create_index("ix_rfq_requests_deal_id", "rfq_requests", ["deal_id"])
    op.create_index("ix_rfq_requests_thread_id", "rfq_requests", ["thread_id"])
    op.create_index("ix_rfq_requests_message_id", "rfq_requests", ["message_id"])
    op.create_index("ix_rfq_requests_status", "rfq_requests", ["status"])


def downgrade() -> None:
    op.drop_index("ix_rfq_requests_status", table_name="rfq_requests")
    op.drop_index("ix_rfq_requests_message_id", table_name="rfq_requests")
    op.drop_index("ix_rfq_requests_thread_id", table_name="rfq_requests")
    op.drop_index("ix_rfq_requests_deal_id", table_name="rfq_requests")
    op.drop_index("ix_rfq_requests_company_name", table_name="rfq_requests")
    op.drop_table("rfq_requests")
