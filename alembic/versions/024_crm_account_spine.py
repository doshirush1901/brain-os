"""CRM account spine — domain identity, aliases, deal company_id + stages.

Revision ID: 024
Revises: 023
Create Date: 2026-08-02

Wave 1 schema only (typed writes land in application code). Reversible.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import text

from alembic import op

revision = "024"
down_revision = "023"
branch_labels = None
depends_on = None

_DEAL_STAGES = (
    "NEW",
    "CONTACTED",
    "ENGAGED",
    "QUALIFIED",
    "PROPOSAL",
    "NEGOTIATION",
    "WON",
    "IN_PRODUCTION",
    "AFTERMARKET",
    "LOST",
)


def upgrade() -> None:
    # ── companies: domain + provenance ───────────────────────────────────
    op.add_column("companies", sa.Column("domain", sa.String(length=255), nullable=True))
    op.add_column("companies", sa.Column("source", sa.String(length=100), nullable=True))
    op.add_column("companies", sa.Column("source_ref", sa.String(length=255), nullable=True))
    op.create_index(
        "uq_companies_domain_not_null",
        "companies",
        ["domain"],
        unique=True,
        postgresql_where=sa.text("domain IS NOT NULL"),
    )

    # Light domain backfill from website (unique hosts only). Region backfill → C3.
    op.execute(
        text(
            """
            WITH extracted AS (
              SELECT
                id,
                lower(
                  regexp_replace(
                    regexp_replace(
                      regexp_replace(
                        coalesce(website, ''),
                        '^https?://',
                        '',
                        'i'
                      ),
                      '/.*$',
                      ''
                    ),
                    '^www\\.',
                    ''
                  )
                ) AS dom
              FROM companies
              WHERE website IS NOT NULL AND trim(website) <> ''
            ),
            unique_doms AS (
              SELECT dom
              FROM extracted
              WHERE dom <> '' AND position('@' in dom) = 0 AND position('.' in dom) > 0
              GROUP BY dom
              HAVING count(*) = 1
            )
            UPDATE companies c
            SET domain = e.dom
            FROM extracted e
            JOIN unique_doms u ON u.dom = e.dom
            WHERE c.id = e.id
              AND c.domain IS NULL
            """
        )
    )

    # Trivial region normalize: 2-letter alpha → uppercase (in place).
    op.execute(
        text(
            """
            UPDATE companies
            SET region = upper(trim(region))
            WHERE region IS NOT NULL
              AND length(trim(region)) = 2
              AND region ~ '^[A-Za-z]{2}$'
            """
        )
    )

    # ── company_aliases ───────────────────────────────────────────────────
    op.create_table(
        "company_aliases",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "company_id",
            sa.String(length=36),
            sa.ForeignKey("companies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("alias", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False, server_default="name_variant"),
        sa.Column("source", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "kind IN ('name_variant','former_name','email_domain','operator_pin_alias')",
            name="ck_company_aliases_kind",
        ),
    )
    op.create_index("ix_company_aliases_company_id", "company_aliases", ["company_id"])
    op.execute(
        text(
            """
            CREATE UNIQUE INDEX uq_company_aliases_alias_lower
            ON company_aliases (lower(trim(alias)))
            """
        )
    )
    # Seed email_domain aliases from companies.domain where present.
    op.execute(
        text(
            """
            INSERT INTO company_aliases (id, company_id, alias, kind, source)
            SELECT
              gen_random_uuid()::text,
              id,
              domain,
              'email_domain',
              '024_crm_account_spine'
            FROM companies
            WHERE domain IS NOT NULL
            ON CONFLICT DO NOTHING
            """
        )
    )

    # ── deals: company_id, provenance, stage ladder + CHECK ───────────────
    op.add_column(
        "deals",
        sa.Column("company_id", sa.String(length=36), nullable=True),
    )
    op.create_foreign_key(
        "fk_deals_company_id",
        "deals",
        "companies",
        ["company_id"],
        ["id"],
    )
    op.create_index("ix_deals_company_id", "deals", ["company_id"])
    # expected_close_date already exists (001); add provenance only.
    op.add_column("deals", sa.Column("source", sa.String(length=100), nullable=True))
    op.add_column("deals", sa.Column("source_ref", sa.String(length=255), nullable=True))

    op.alter_column(
        "deals",
        "stage",
        existing_type=sa.String(length=11),
        type_=sa.String(length=32),
        existing_nullable=False,
    )
    stage_list = ", ".join(f"'{s}'" for s in _DEAL_STAGES)
    op.create_check_constraint(
        "ck_deals_stage_valid",
        "deals",
        f"stage IN ({stage_list})",
    )

    # Backfill deal.company_id from contact.company_id where available.
    op.execute(
        text(
            """
            UPDATE deals d
            SET company_id = c.company_id
            FROM contacts c
            WHERE d.contact_id = c.id
              AND d.company_id IS NULL
              AND c.company_id IS NOT NULL
            """
        )
    )

    # ── interactions: timeline columns ────────────────────────────────────
    op.add_column(
        "interactions",
        sa.Column("company_id", sa.String(length=36), nullable=True),
    )
    op.create_foreign_key(
        "fk_interactions_company_id",
        "interactions",
        "companies",
        ["company_id"],
        ["id"],
    )
    op.add_column("interactions", sa.Column("type", sa.String(length=40), nullable=True))
    op.add_column("interactions", sa.Column("ref", sa.String(length=255), nullable=True))
    op.add_column("interactions", sa.Column("occurred_at", sa.DateTime(), nullable=True))
    op.create_check_constraint(
        "ck_interactions_type",
        "interactions",
        "type IS NULL OR type IN ('email','call','meeting','quote','pin','web','other')",
    )
    op.create_index(
        "ix_interactions_company_occurred",
        "interactions",
        ["company_id", "occurred_at"],
    )
    op.create_index(
        "ix_interactions_contact_occurred",
        "interactions",
        ["contact_id", "occurred_at"],
    )
    op.execute(
        text(
            """
            UPDATE interactions
            SET occurred_at = created_at
            WHERE occurred_at IS NULL
            """
        )
    )
    op.execute(
        text(
            """
            UPDATE interactions i
            SET company_id = c.company_id
            FROM contacts c
            WHERE i.contact_id = c.id
              AND i.company_id IS NULL
              AND c.company_id IS NOT NULL
            """
        )
    )
    # Widen channel for longer enum values (e.g. WHATSAPP / SLACK_CHANNEL).
    op.alter_column(
        "interactions",
        "channel",
        existing_type=sa.String(length=7),
        type_=sa.String(length=32),
        existing_nullable=False,
    )

    # ── updated_at triggers (Postgres) — raw SQL writers can't skip ───────
    op.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION brain_set_updated_at()
            RETURNS trigger AS $$
            BEGIN
              NEW.updated_at := now();
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
            """
        )
    )
    for table in ("companies", "contacts", "deals"):
        op.execute(text(f"DROP TRIGGER IF EXISTS trg_{table}_set_updated_at ON {table}"))
        op.execute(
            text(
                f"""
                CREATE TRIGGER trg_{table}_set_updated_at
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE FUNCTION brain_set_updated_at()
                """
            )
        )


def downgrade() -> None:
    for table in ("companies", "contacts", "deals"):
        op.execute(text(f"DROP TRIGGER IF EXISTS trg_{table}_set_updated_at ON {table}"))
    op.execute(text("DROP FUNCTION IF EXISTS brain_set_updated_at()"))

    op.drop_index("ix_interactions_contact_occurred", table_name="interactions")
    op.drop_index("ix_interactions_company_occurred", table_name="interactions")
    op.drop_constraint("ck_interactions_type", "interactions", type_="check")
    op.drop_column("interactions", "occurred_at")
    op.drop_column("interactions", "ref")
    op.drop_column("interactions", "type")
    op.drop_constraint("fk_interactions_company_id", "interactions", type_="foreignkey")
    op.drop_column("interactions", "company_id")
    op.alter_column(
        "interactions",
        "channel",
        existing_type=sa.String(length=32),
        type_=sa.String(length=7),
        existing_nullable=False,
    )

    op.drop_constraint("ck_deals_stage_valid", "deals", type_="check")
    # Reject extended stages before narrowing column.
    op.execute(
        text(
            """
            UPDATE deals
            SET stage = 'WON'
            WHERE stage IN ('IN_PRODUCTION', 'AFTERMARKET')
            """
        )
    )
    op.alter_column(
        "deals",
        "stage",
        existing_type=sa.String(length=32),
        type_=sa.String(length=11),
        existing_nullable=False,
    )
    op.drop_column("deals", "source_ref")
    op.drop_column("deals", "source")
    op.drop_index("ix_deals_company_id", table_name="deals")
    op.drop_constraint("fk_deals_company_id", "deals", type_="foreignkey")
    op.drop_column("deals", "company_id")

    op.execute(text("DROP INDEX IF EXISTS uq_company_aliases_alias_lower"))
    op.drop_index("ix_company_aliases_company_id", table_name="company_aliases")
    op.drop_table("company_aliases")

    op.drop_index("uq_companies_domain_not_null", table_name="companies")
    op.drop_column("companies", "source_ref")
    op.drop_column("companies", "source")
    op.drop_column("companies", "domain")
