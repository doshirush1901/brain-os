"""Add CRM motivation-fit and buyer-role fields.

Revision ID: 009
Revises: 008
Create Date: 2026-04-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = {c.get("name") for c in inspector.get_columns("contacts")}
    additions = [
        ("buyer_role", sa.String(length=40)),
        ("motivation_signal", sa.JSON()),
        ("motivation_signal_at", sa.DateTime()),
        ("motivation_signal_source", sa.String(length=255)),
        ("job_to_be_done", sa.Text()),
        ("channel_friction_score", sa.Float()),
        ("motivation_fit_score", sa.Float()),
        ("motivation_confidence", sa.Float()),
        ("motivation_freshness_days", sa.Integer()),
    ]
    for name, col_type in additions:
        if name not in existing_cols:
            op.add_column("contacts", sa.Column(name, col_type, nullable=True))

    inspector = sa.inspect(bind)
    indexes = {i.get("name") for i in inspector.get_indexes("contacts")}
    if "ix_contacts_buyer_role" not in indexes:
        op.create_index("ix_contacts_buyer_role", "contacts", ["buyer_role"])
    if "ix_contacts_motivation_fit_score" not in indexes:
        op.create_index("ix_contacts_motivation_fit_score", "contacts", ["motivation_fit_score"])
    if "ix_contacts_motivation_signal_at" not in indexes:
        op.create_index("ix_contacts_motivation_signal_at", "contacts", ["motivation_signal_at"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    indexes = {i.get("name") for i in inspector.get_indexes("contacts")}
    if "ix_contacts_motivation_signal_at" in indexes:
        op.drop_index("ix_contacts_motivation_signal_at", table_name="contacts")
    if "ix_contacts_motivation_fit_score" in indexes:
        op.drop_index("ix_contacts_motivation_fit_score", table_name="contacts")
    if "ix_contacts_buyer_role" in indexes:
        op.drop_index("ix_contacts_buyer_role", table_name="contacts")

    existing_cols = {c.get("name") for c in inspector.get_columns("contacts")}
    for col in [
        "motivation_freshness_days",
        "motivation_confidence",
        "motivation_fit_score",
        "channel_friction_score",
        "job_to_be_done",
        "motivation_signal_source",
        "motivation_signal_at",
        "motivation_signal",
        "buyer_role",
    ]:
        if col in existing_cols:
            op.drop_column("contacts", col)
