"""Machine passports + aftermarket events (installed-base Phase 1).

Revision ID: 022
Revises: 021
Create Date: 2026-06-12
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "022"
down_revision = "021"
branch_labels = None
depends_on = None

_ASSETS = "machine_assets"
_EVENTS = "aftermarket_events"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _ASSETS in inspector.get_table_names():
        return
    op.create_table(
        _ASSETS,
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "company_id", sa.String(length=36), sa.ForeignKey("companies.id"), nullable=False
        ),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("model_family", sa.String(length=40), nullable=True),
        sa.Column("serial", sa.String(length=120), nullable=True),
        sa.Column("configuration", sa.JSON(), nullable=True),
        sa.Column("quote_ref", sa.String(length=120), nullable=True),
        sa.Column("sold_date", sa.DateTime(), nullable=True),
        sa.Column("installed_date", sa.DateTime(), nullable=True),
        sa.Column("warranty_end", sa.DateTime(), nullable=True),
        sa.Column(
            "verification", sa.String(length=20), nullable=False, server_default="unverified"
        ),
        sa.Column("evidence", sa.JSON(), nullable=True),
        sa.Column("candidate_id", sa.String(length=255), nullable=True),
        sa.Column("last_aftermarket_touch", sa.DateTime(), nullable=True),
        sa.Column("health_notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_machine_assets_company_id", _ASSETS, ["company_id"])
    op.create_index("ix_machine_assets_verification", _ASSETS, ["verification"])
    op.create_index(
        "uq_machine_assets_company_model",
        _ASSETS,
        ["company_id", "model"],
        unique=True,
    )

    op.create_table(
        _EVENTS,
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "asset_id",
            sa.String(length=36),
            sa.ForeignKey(f"{_ASSETS}.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("event_at", sa.DateTime(), nullable=False),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_aftermarket_events_asset_id", _EVENTS, ["asset_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _EVENTS in inspector.get_table_names():
        op.drop_index("ix_aftermarket_events_asset_id", table_name=_EVENTS)
        op.drop_table(_EVENTS)
    if _ASSETS in inspector.get_table_names():
        op.drop_index("uq_machine_assets_company_model", table_name=_ASSETS)
        op.drop_index("ix_machine_assets_verification", table_name=_ASSETS)
        op.drop_index("ix_machine_assets_company_id", table_name=_ASSETS)
        op.drop_table(_ASSETS)
