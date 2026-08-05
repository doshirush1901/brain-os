"""CPQ Wave 1 — quotes registry: numbering, versioning, ledger keys, aliases.

Revision ID: 025
Revises: 024
Create Date: 2026-08-02

Adds canonical MCT-YYYY-NNNN counter, quote_number/version/revision_of,
send-ledger join keys, provenance, and quote_aliases for legacy ids.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import text

from alembic import op

revision = "025"
down_revision = "024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quote_number_counters",
        sa.Column("year", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("last_n", sa.Integer(), nullable=False, server_default="0"),
    )

    op.add_column("quotes", sa.Column("quote_number", sa.String(length=64), nullable=True))
    op.add_column(
        "quotes",
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "quotes",
        sa.Column(
            "revision_of",
            sa.String(length=36),
            sa.ForeignKey("quotes.id"),
            nullable=True,
        ),
    )
    op.add_column("quotes", sa.Column("thread_id", sa.String(length=128), nullable=True))
    op.add_column("quotes", sa.Column("message_id", sa.String(length=128), nullable=True))
    op.add_column("quotes", sa.Column("provenance", sa.String(length=64), nullable=True))
    op.add_column("quotes", sa.Column("pdf_path", sa.String(length=1024), nullable=True))
    op.add_column("quotes", sa.Column("pdf_sha256", sa.String(length=64), nullable=True))

    op.create_index("ix_quotes_quote_number", "quotes", ["quote_number"], unique=True)
    op.create_index("ix_quotes_thread_id", "quotes", ["thread_id"])
    op.create_index("ix_quotes_message_id", "quotes", ["message_id"])

    op.create_table(
        "quote_aliases",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "quote_id",
            sa.String(length=36),
            sa.ForeignKey("quotes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("alias", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("uq_quote_aliases_alias", "quote_aliases", ["alias"], unique=True)
    op.create_index("ix_quote_aliases_quote_id", "quote_aliases", ["quote_id"])

    # Seed counter for current UTC year so allocate starts at 1.
    op.execute(
        text(
            """
            INSERT INTO quote_number_counters (year, last_n)
            VALUES (CAST(EXTRACT(YEAR FROM (NOW() AT TIME ZONE 'UTC')) AS INTEGER), 0)
            ON CONFLICT (year) DO NOTHING
            """
        )
    )


def downgrade() -> None:
    op.drop_index("ix_quote_aliases_quote_id", table_name="quote_aliases")
    op.drop_index("uq_quote_aliases_alias", table_name="quote_aliases")
    op.drop_table("quote_aliases")

    op.drop_index("ix_quotes_message_id", table_name="quotes")
    op.drop_index("ix_quotes_thread_id", table_name="quotes")
    op.drop_index("ix_quotes_quote_number", table_name="quotes")

    op.drop_column("quotes", "pdf_sha256")
    op.drop_column("quotes", "pdf_path")
    op.drop_column("quotes", "provenance")
    op.drop_column("quotes", "message_id")
    op.drop_column("quotes", "thread_id")
    op.drop_column("quotes", "revision_of")
    op.drop_column("quotes", "version")
    op.drop_column("quotes", "quote_number")

    op.drop_table("quote_number_counters")
