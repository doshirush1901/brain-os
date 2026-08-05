"""CPQ Wave 2 — price_book + price_book_options.

Revision ID: 026
Revises: 025
Create Date: 2026-08-02
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "026"
down_revision = "025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "price_book",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("series", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("base_price_inr", sa.Numeric(15, 2), nullable=True),
        sa.Column("cost_inr", sa.Numeric(15, 2), nullable=True),
        sa.Column("margin_floor_pct", sa.Float(), nullable=False, server_default="15"),
        sa.Column("region_multipliers", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="todo"),
        sa.Column("source", sa.String(length=128), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("uq_price_book_model", "price_book", ["model"], unique=True)
    op.create_index("ix_price_book_series", "price_book", ["series"])
    op.create_index("ix_price_book_status", "price_book", ["status"])

    op.create_table(
        "price_book_options",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "price_book_id",
            sa.String(length=36),
            sa.ForeignKey("price_book.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("option_key", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("price_inr", sa.Numeric(15, 2), nullable=True),
        sa.Column("cost_inr", sa.Numeric(15, 2), nullable=True),
        sa.Column("constraints", sa.JSON(), nullable=True),
        sa.Column("margin_floor_pct", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="todo"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index(
        "uq_price_book_options_key_book",
        "price_book_options",
        ["option_key", "price_book_id"],
        unique=True,
    )
    op.create_index("ix_price_book_options_key", "price_book_options", ["option_key"])


def downgrade() -> None:
    op.drop_index("ix_price_book_options_key", table_name="price_book_options")
    op.drop_index("uq_price_book_options_key_book", table_name="price_book_options")
    op.drop_table("price_book_options")
    op.drop_index("ix_price_book_status", table_name="price_book")
    op.drop_index("ix_price_book_series", table_name="price_book")
    op.drop_index("uq_price_book_model", table_name="price_book")
    op.drop_table("price_book")
