"""Add performance indexes for common query patterns.

Revision ID: 002
Revises: 001
Create Date: 2026-03-07
"""

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_contacts_company_id", "contacts", ["company_id"])
    op.create_index("ix_deals_contact_id", "deals", ["contact_id"])
    op.create_index("ix_deals_stage", "deals", ["stage"])
    op.create_index("ix_interactions_contact_id", "interactions", ["contact_id"])
    op.create_index("ix_interactions_created_at", "interactions", ["created_at"])
    op.create_index("ix_quotes_status", "quotes", ["status"])
    op.create_index("ix_quotes_contact_id", "quotes", ["contact_id"])
    op.create_index("ix_drip_steps_scheduled_at", "drip_steps", ["scheduled_at"])
    op.create_index("ix_drip_steps_campaign_id", "drip_steps", ["campaign_id"])


def downgrade() -> None:
    op.drop_index("ix_drip_steps_campaign_id", "drip_steps")
    op.drop_index("ix_drip_steps_scheduled_at", "drip_steps")
    op.drop_index("ix_quotes_contact_id", "quotes")
    op.drop_index("ix_quotes_status", "quotes")
    op.drop_index("ix_interactions_created_at", "interactions")
    op.drop_index("ix_interactions_contact_id", "interactions")
    op.drop_index("ix_deals_stage", "deals")
    op.drop_index("ix_deals_contact_id", "deals")
    op.drop_index("ix_contacts_company_id", "contacts")
