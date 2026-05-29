"""Scheduling requests, slots, and calendar bookings.

Revision ID: 007
Revises: 006
Create Date: 2026-04-26

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    def _index_exists(table: str, index: str) -> bool:
        return any(i.get("name") == index for i in inspector.get_indexes(table))

    if "scheduling_requests" not in existing_tables:
        op.create_table(
            "scheduling_requests",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("contact_id", sa.String(36), sa.ForeignKey("contacts.id"), nullable=True),
            sa.Column("contact_email", sa.String(320), nullable=False),
            sa.Column("contact_name", sa.String(255), nullable=True),
            sa.Column("thread_id", sa.String(255), nullable=True),
            sa.Column("owner_mailbox", sa.String(320), nullable=True),
            sa.Column("timezone", sa.String(80), nullable=False, server_default="UTC"),
            sa.Column("duration_minutes", sa.Integer, nullable=False, server_default="30"),
            sa.Column("buffer_minutes", sa.Integer, nullable=False, server_default="15"),
            sa.Column("purpose", sa.Text, nullable=True),
            sa.Column("status", sa.String(40), nullable=False, server_default="pending"),
            sa.Column("expires_at", sa.DateTime, nullable=False),
            sa.Column("booked_at", sa.DateTime, nullable=True),
            sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        )
        inspector = sa.inspect(bind)
    if not _index_exists("scheduling_requests", "ix_scheduling_requests_contact_email"):
        op.create_index(
            "ix_scheduling_requests_contact_email", "scheduling_requests", ["contact_email"]
        )
    if not _index_exists("scheduling_requests", "ix_scheduling_requests_status"):
        op.create_index("ix_scheduling_requests_status", "scheduling_requests", ["status"])

    if "scheduling_slots" not in existing_tables:
        op.create_table(
            "scheduling_slots",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "request_id",
                sa.String(36),
                sa.ForeignKey("scheduling_requests.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("start_at", sa.DateTime, nullable=False),
            sa.Column("end_at", sa.DateTime, nullable=False),
            sa.Column("label", sa.String(255), nullable=False),
            sa.Column("token_hash", sa.String(128), unique=True, nullable=False),
            sa.Column("status", sa.String(40), nullable=False, server_default="pending"),
            sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        )
        inspector = sa.inspect(bind)
    if not _index_exists("scheduling_slots", "ix_scheduling_slots_request_id"):
        op.create_index("ix_scheduling_slots_request_id", "scheduling_slots", ["request_id"])
    if not _index_exists("scheduling_slots", "ix_scheduling_slots_token_hash"):
        op.create_index(
            "ix_scheduling_slots_token_hash", "scheduling_slots", ["token_hash"], unique=True
        )
    if not _index_exists("scheduling_slots", "ix_scheduling_slots_status"):
        op.create_index("ix_scheduling_slots_status", "scheduling_slots", ["status"])

    if "calendar_bookings" not in existing_tables:
        op.create_table(
            "calendar_bookings",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "request_id",
                sa.String(36),
                sa.ForeignKey("scheduling_requests.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "slot_id",
                sa.String(36),
                sa.ForeignKey("scheduling_slots.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("google_event_id", sa.String(255), nullable=True),
            sa.Column("booked_email", sa.String(320), nullable=False),
            sa.Column("metadata_json", sa.JSON, nullable=True),
            sa.Column("booked_at", sa.DateTime, server_default=sa.func.now()),
        )
        inspector = sa.inspect(bind)
    if not _index_exists("calendar_bookings", "ix_calendar_bookings_request_id"):
        op.create_index("ix_calendar_bookings_request_id", "calendar_bookings", ["request_id"])
    if not _index_exists("calendar_bookings", "ix_calendar_bookings_slot_id"):
        op.create_index("ix_calendar_bookings_slot_id", "calendar_bookings", ["slot_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())
    if "calendar_bookings" in existing_tables:
        op.drop_table("calendar_bookings")
    if "scheduling_slots" in existing_tables:
        op.drop_table("scheduling_slots")
    if "scheduling_requests" in existing_tables:
        op.drop_table("scheduling_requests")
