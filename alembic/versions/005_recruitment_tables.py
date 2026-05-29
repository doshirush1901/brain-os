"""Recruitment tables for Anu (candidates + stage events).

Revision ID: 005
Revises: 004
Create Date: 2026-03-16

"""

import sqlalchemy as sa
from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "recruitment_candidates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(320), unique=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("phone", sa.String(50), nullable=True),
        sa.Column("role_applied", sa.String(255), nullable=True),
        sa.Column("profile_json", sa.JSON, nullable=True),
        sa.Column("cv_parsed_json", sa.JSON, nullable=True),
        sa.Column("score_json", sa.JSON, nullable=True),
        sa.Column("ctc_current", sa.String(100), nullable=True),
        sa.Column("source_type", sa.String(100), nullable=True),
        sa.Column("source_id", sa.String(255), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_recruitment_candidates_email",
        "recruitment_candidates",
        ["email"],
        unique=True,
    )

    op.create_table(
        "recruitment_stage_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "candidate_id",
            sa.String(36),
            sa.ForeignKey("recruitment_candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage", sa.String(80), nullable=False),
        sa.Column("event_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("metadata_json", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_recruitment_stage_events_stage",
        "recruitment_stage_events",
        ["stage"],
        unique=False,
    )
    op.create_index(
        "ix_recruitment_stage_events_candidate_id",
        "recruitment_stage_events",
        ["candidate_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("recruitment_stage_events")
    op.drop_table("recruitment_candidates")
