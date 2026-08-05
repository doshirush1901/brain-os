"""Add WhatsApp number + opt-in fields to CRM contacts.

``Channel.WHATSAPP`` needs no DB change (interactions.channel is stored as a
plain string — ``native_enum=False``); only the contact columns land here.

Revision ID: 020
Revises: 019
Create Date: 2026-06-12
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None

_COLUMNS = [
    ("whatsapp_number", sa.String(length=32)),
    ("whatsapp_opt_in", sa.Boolean()),
]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {c.get("name") for c in inspector.get_columns("contacts")}
    for name, col_type in _COLUMNS:
        if name not in existing:
            op.add_column("contacts", sa.Column(name, col_type, nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {c.get("name") for c in inspector.get_columns("contacts")}
    for name, _ in reversed(_COLUMNS):
        if name in existing:
            op.drop_column("contacts", name)
