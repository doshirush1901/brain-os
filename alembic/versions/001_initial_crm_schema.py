"""Initial CRM schema.

Creates the core CRM tables: companies, contacts, deals, interactions,
drip_campaigns, drip_steps, and quotes.

Revision ID: 001
Revises:
Create Date: 2026-03-07
"""

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "companies",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), unique=True, nullable=False),
        sa.Column("region", sa.String(100), nullable=True),
        sa.Column("industry", sa.String(255), nullable=True),
        sa.Column("website", sa.String(500), nullable=True),
        sa.Column("employee_count", sa.Integer, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "contacts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "company_id",
            sa.String(36),
            sa.ForeignKey("companies.id"),
            nullable=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(320), unique=True, nullable=False),
        sa.Column("phone", sa.String(50), nullable=True),
        sa.Column("role", sa.String(255), nullable=True),
        sa.Column("source", sa.String(100), nullable=True),
        sa.Column("lead_score", sa.Float, default=0.0),
        sa.Column("contact_type", sa.String(50), nullable=True),
        sa.Column("warmth_level", sa.String(50), nullable=True),
        sa.Column("tags", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "deals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "contact_id",
            sa.String(36),
            sa.ForeignKey("contacts.id"),
            nullable=False,
        ),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("value", sa.Numeric(15, 2), default=0),
        sa.Column("currency", sa.String(10), default="USD"),
        sa.Column("stage", sa.String(50), default="NEW"),
        sa.Column("machine_model", sa.String(255), nullable=True),
        sa.Column("expected_close_date", sa.DateTime, nullable=True),
        sa.Column("actual_close_date", sa.DateTime, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "interactions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "contact_id",
            sa.String(36),
            sa.ForeignKey("contacts.id"),
            nullable=False,
        ),
        sa.Column(
            "deal_id",
            sa.String(36),
            sa.ForeignKey("deals.id"),
            nullable=True,
        ),
        sa.Column("channel", sa.String(50)),
        sa.Column("direction", sa.String(50)),
        sa.Column("subject", sa.String(500), nullable=True),
        sa.Column("content", sa.Text, nullable=True),
        sa.Column("sentiment", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "drip_campaigns",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("target_segment", sa.JSON, nullable=True),
        sa.Column("status", sa.String(50), default="ACTIVE"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "drip_steps",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "campaign_id",
            sa.String(36),
            sa.ForeignKey("drip_campaigns.id"),
            nullable=False,
        ),
        sa.Column(
            "contact_id",
            sa.String(36),
            sa.ForeignKey("contacts.id"),
            nullable=False,
        ),
        sa.Column("step_number", sa.Integer, nullable=False),
        sa.Column("email_subject", sa.String(500), nullable=True),
        sa.Column("email_body", sa.Text, nullable=True),
        sa.Column("scheduled_at", sa.DateTime, nullable=True),
        sa.Column("sent_at", sa.DateTime, nullable=True),
        sa.Column("reply_received", sa.Boolean, default=False),
        sa.Column("reply_content", sa.Text, nullable=True),
        sa.Column("opened", sa.Boolean, default=False),
    )

    op.create_table(
        "quotes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "contact_id",
            sa.String(36),
            sa.ForeignKey("contacts.id"),
            nullable=False,
        ),
        sa.Column(
            "deal_id",
            sa.String(36),
            sa.ForeignKey("deals.id"),
            nullable=True,
        ),
        sa.Column("company_name", sa.String(255), nullable=True),
        sa.Column("machine_model", sa.String(255), nullable=True),
        sa.Column("configuration", sa.JSON, nullable=True),
        sa.Column("estimated_value", sa.Numeric(15, 2), nullable=True),
        sa.Column("currency", sa.String(10), default="USD"),
        sa.Column("status", sa.String(50), default="DRAFT"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("sent_at", sa.DateTime, nullable=True),
        sa.Column("last_follow_up_at", sa.DateTime, nullable=True),
        sa.Column("closed_at", sa.DateTime, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("quotes")
    op.drop_table("drip_steps")
    op.drop_table("drip_campaigns")
    op.drop_table("interactions")
    op.drop_table("deals")
    op.drop_table("contacts")
    op.drop_table("companies")
