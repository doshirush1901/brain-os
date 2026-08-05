"""ERP Stage 3 Wave 2 — payment match candidates + realized FX.

Revision ID: 032
Revises: 031
Create Date: 2026-08-02

- wo_payment_milestones.fx_rate_realized (stamped on operator confirm)
- wo_payment_milestones.due_at (optional dated due basis for aging)
- customer_credit_terms.instrument_expires_at (LC/PBG expiry alerts)
- payment_match_candidates (email/remittance proposals — never auto-confirm)
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None

_CANDIDATE_STATUSES = ("pending", "confirmed", "rejected", "expired")


def upgrade() -> None:
    op.add_column(
        "wo_payment_milestones",
        sa.Column("fx_rate_realized", sa.Float(), nullable=True),
    )
    op.add_column(
        "wo_payment_milestones",
        sa.Column("due_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "customer_credit_terms",
        sa.Column("instrument_expires_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "payment_match_candidates",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("amount", sa.Numeric(15, 2), nullable=True),
        sa.Column("currency", sa.String(length=10), nullable=True),
        sa.Column("paid_at", sa.DateTime(), nullable=True),
        sa.Column("order_ref", sa.String(length=255), nullable=True),
        sa.Column("customer_hint", sa.String(length=255), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="email"),
        sa.Column("evidence_ref", sa.String(length=512), nullable=True),
        sa.Column("message_id", sa.String(length=255), nullable=True),
        sa.Column("thread_id", sa.String(length=255), nullable=True),
        sa.Column("raw_excerpt", sa.Text(), nullable=True),
        sa.Column(
            "proposed_milestone_id",
            sa.String(length=36),
            sa.ForeignKey("wo_payment_milestones.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "proposed_work_order_id",
            sa.String(length=36),
            sa.ForeignKey("work_orders.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("match_score", sa.Float(), nullable=True),
        sa.Column("match_reasons", sa.JSON(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("confirmed_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "status IN (" + ", ".join(f"'{s}'" for s in _CANDIDATE_STATUSES) + ")",
            name="ck_payment_match_candidates_status",
        ),
    )
    op.create_index(
        "ix_payment_match_candidates_status",
        "payment_match_candidates",
        ["status"],
    )
    op.create_index(
        "ix_payment_match_candidates_proposed_ms",
        "payment_match_candidates",
        ["proposed_milestone_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_payment_match_candidates_proposed_ms", table_name="payment_match_candidates")
    op.drop_index("ix_payment_match_candidates_status", table_name="payment_match_candidates")
    op.drop_table("payment_match_candidates")
    op.drop_column("customer_credit_terms", "instrument_expires_at")
    op.drop_column("wo_payment_milestones", "due_at")
    op.drop_column("wo_payment_milestones", "fx_rate_realized")
