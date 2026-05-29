"""Add ira_segment and outbound cadence fields to CRM contacts/companies.

Revision ID: 019
Revises: 018
Create Date: 2026-05-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None


def _add_contact_columns() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {c.get("name") for c in inspector.get_columns("contacts")}
    additions = [
        ("ira_segment", sa.String(length=40)),
        ("segment_updated_at", sa.DateTime()),
        ("segment_evidence", sa.Text()),
        ("last_inbound_at", sa.DateTime()),
        ("last_outbound_at", sa.DateTime()),
        ("outbound_touch_count", sa.Integer()),
        ("cooldown_until", sa.DateTime()),
    ]
    for name, col_type in additions:
        if name not in existing:
            op.add_column("contacts", sa.Column(name, col_type, nullable=True))


def _add_company_columns() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {c.get("name") for c in inspector.get_columns("companies")}
    if "ira_segment" not in existing:
        op.add_column("companies", sa.Column("ira_segment", sa.String(length=40), nullable=True))


def upgrade() -> None:
    _add_contact_columns()
    _add_company_columns()
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    contact_indexes = {i.get("name") for i in inspector.get_indexes("contacts")}
    if "ix_contacts_ira_segment" not in contact_indexes:
        op.create_index("ix_contacts_ira_segment", "contacts", ["ira_segment"])
    if "ix_contacts_cooldown_until" not in contact_indexes:
        op.create_index("ix_contacts_cooldown_until", "contacts", ["cooldown_until"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    contact_indexes = {i.get("name") for i in inspector.get_indexes("contacts")}
    if "ix_contacts_cooldown_until" in contact_indexes:
        op.drop_index("ix_contacts_cooldown_until", table_name="contacts")
    if "ix_contacts_ira_segment" in contact_indexes:
        op.drop_index("ix_contacts_ira_segment", table_name="contacts")

    for table, cols in (
        (
            "contacts",
            [
                "cooldown_until",
                "outbound_touch_count",
                "last_outbound_at",
                "last_inbound_at",
                "segment_evidence",
                "segment_updated_at",
                "ira_segment",
            ],
        ),
        ("companies", ["ira_segment"]),
    ):
        existing = {c.get("name") for c in inspector.get_columns(table)}
        for col in cols:
            if col in existing:
                op.drop_column(table, col)
