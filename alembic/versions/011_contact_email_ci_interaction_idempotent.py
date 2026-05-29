"""Case-insensitive unique email + idempotent interaction external_message_id per contact.

Revision ID: 011
Revises: 010
Create Date: 2026-05-01
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # Normalize existing emails to lowercase (application also enforces on write).
    op.execute(sa.text("UPDATE contacts SET email = lower(trim(email))"))

    for uc in list(insp.get_unique_constraints("contacts") or []):
        cols = set(uc.get("column_names") or [])
        if cols == {"email"} and uc.get("name"):
            op.drop_constraint(uc["name"], "contacts", type_="unique")

    op.execute(
        sa.text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_contacts_email_lower "
            "ON contacts (lower(trim(email)))"
        ),
    )

    # Replace non-unique external_message_id index with partial unique (contact + id).
    ix_names = {ix["name"] for ix in insp.get_indexes("interactions") or []}
    if "ix_interactions_external_message_id" in ix_names:
        op.drop_index("ix_interactions_external_message_id", table_name="interactions")

    op.execute(
        sa.text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_interactions_contact_external_mid "
            "ON interactions (contact_id, external_message_id) "
            "WHERE external_message_id IS NOT NULL AND trim(external_message_id) <> ''"
        ),
    )


def downgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS uq_interactions_contact_external_mid"))
    op.execute(sa.text("DROP INDEX IF EXISTS uq_contacts_email_lower"))

    op.create_index(
        "ix_interactions_external_message_id",
        "interactions",
        ["external_message_id"],
        unique=False,
    )

    bind = op.get_bind()
    insp = sa.inspect(bind)
    ucs = {
        (u.get("name"), tuple(u.get("column_names") or ()))
        for u in (insp.get_unique_constraints("contacts") or [])
    }
    if ("contacts_email_key", ("email",)) not in ucs:
        op.create_unique_constraint("contacts_email_key", "contacts", ["email"])
