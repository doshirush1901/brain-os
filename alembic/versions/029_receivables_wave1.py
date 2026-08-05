"""ERP Stage 3 Wave 1 — invoices registry + credit-terms + FX-at-booking.

Revision ID: 029
Revises: 028
Create Date: 2026-08-02

- invoices (commercial PI / tax invoice registry; Tally stays bookkeeping)
- customer_credit_terms (mined from customer_orders_history.md)
- wo_payment_milestones.fx_rate_at_booking
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "029"
down_revision = "028"
branch_labels = None
depends_on = None

_INVOICE_KINDS = ("proforma", "tax_invoice")
_INVOICE_STATUSES = ("draft", "sent", "acknowledged", "paid", "cancelled")
_INSTRUMENTS = ("LC", "PBG", "cheque", "TT")


def upgrade() -> None:
    op.add_column(
        "wo_payment_milestones",
        sa.Column("fx_rate_at_booking", sa.Float(), nullable=True),
    )

    op.create_table(
        "customer_credit_terms",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "company_id",
            sa.String(length=36),
            sa.ForeignKey("companies.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("company_name", sa.String(length=255), nullable=False),
        sa.Column("company_name_norm", sa.String(length=255), nullable=False),
        sa.Column("instrument", sa.String(length=16), nullable=False),
        sa.Column("advance_pct", sa.Float(), nullable=True),
        sa.Column("tail_pct", sa.Float(), nullable=True),
        sa.Column("tail_security_months", sa.Integer(), nullable=True),
        sa.Column("evidence_ref", sa.String(length=512), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.7"),
        sa.Column("needs_review", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("raw_terms", sa.Text(), nullable=True),
        sa.Column("source_year", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.CheckConstraint(
            "instrument IN (" + ", ".join(f"'{s}'" for s in _INSTRUMENTS) + ")",
            name="ck_customer_credit_terms_instrument",
        ),
    )
    op.create_index(
        "ix_customer_credit_terms_company_norm",
        "customer_credit_terms",
        ["company_name_norm"],
    )
    op.create_index(
        "ix_customer_credit_terms_company_id",
        "customer_credit_terms",
        ["company_id"],
    )
    op.create_index(
        "uq_customer_credit_terms_norm_instrument_evidence",
        "customer_credit_terms",
        ["company_name_norm", "instrument", "evidence_ref"],
        unique=True,
    )

    op.create_table(
        "invoices",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("invoice_no", sa.String(length=64), nullable=False),
        sa.Column(
            "series_key",
            sa.String(length=32),
            nullable=False,
            server_default="export_e_fy",
        ),
        sa.Column(
            "work_order_id",
            sa.String(length=36),
            sa.ForeignKey("work_orders.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "milestone_id",
            sa.String(length=36),
            sa.ForeignKey("wo_payment_milestones.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("kind", sa.String(length=32), nullable=False, server_default="proforma"),
        sa.Column("amount", sa.Numeric(15, 2), nullable=True),
        sa.Column("currency", sa.String(length=10), nullable=True),
        sa.Column("raised_at", sa.DateTime(), nullable=True),
        sa.Column("sent_message_id", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="draft"),
        sa.Column("file_path", sa.String(length=1024), nullable=True),
        sa.Column("markdown_path", sa.String(length=1024), nullable=True),
        sa.Column(
            "company_id",
            sa.String(length=36),
            sa.ForeignKey("companies.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("customer_name", sa.String(length=255), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.CheckConstraint(
            "kind IN (" + ", ".join(f"'{s}'" for s in _INVOICE_KINDS) + ")",
            name="ck_invoices_kind",
        ),
        sa.CheckConstraint(
            "status IN (" + ", ".join(f"'{s}'" for s in _INVOICE_STATUSES) + ")",
            name="ck_invoices_status",
        ),
    )
    op.create_index("uq_invoices_invoice_no", "invoices", ["invoice_no"], unique=True)
    op.create_index("ix_invoices_work_order_id", "invoices", ["work_order_id"])
    op.create_index("ix_invoices_milestone_id", "invoices", ["milestone_id"])
    op.create_index("ix_invoices_status", "invoices", ["status"])
    op.create_index("ix_invoices_company_id", "invoices", ["company_id"])


def downgrade() -> None:
    op.drop_index("ix_invoices_company_id", table_name="invoices")
    op.drop_index("ix_invoices_status", table_name="invoices")
    op.drop_index("ix_invoices_milestone_id", table_name="invoices")
    op.drop_index("ix_invoices_work_order_id", table_name="invoices")
    op.drop_index("uq_invoices_invoice_no", table_name="invoices")
    op.drop_table("invoices")

    op.drop_index(
        "uq_customer_credit_terms_norm_instrument_evidence",
        table_name="customer_credit_terms",
    )
    op.drop_index("ix_customer_credit_terms_company_id", table_name="customer_credit_terms")
    op.drop_index("ix_customer_credit_terms_company_norm", table_name="customer_credit_terms")
    op.drop_table("customer_credit_terms")

    op.drop_column("wo_payment_milestones", "fx_rate_at_booking")
