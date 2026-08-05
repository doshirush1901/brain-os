"""ERP Stage 2 Wave 1 — work_orders spine + milestones + atlas_events FK.

Revision ID: 028
Revises: 027
Create Date: 2026-08-02

- work_orders (canonical YY+seq + alias, stage ladder, promised_dates)
- wo_payment_milestone_templates (region picks at WO creation)
- wo_payment_milestones (per-deal rows)
- atlas_events.work_order_id / evidence_ref / to_stage
- PG trigger: stage updates only when ira.allow_wo_stage_update=1
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import text

from alembic import op

revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None

_WO_STAGES = (
    "draft",
    "po_received",
    "wo_opened",
    "drawing_approval",
    "procurement_build",
    "trials",
    "fat",
    "dispatch",
    "installation",
    "handover",
    "warranty",
)


def upgrade() -> None:
    op.create_table(
        "wo_payment_milestone_templates",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("region_key", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("milestones", sa.JSON(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index(
        "uq_wo_payment_milestone_templates_region",
        "wo_payment_milestone_templates",
        ["region_key"],
        unique=True,
    )

    op.create_table(
        "work_orders",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("wo_number", sa.String(length=64), nullable=False),
        sa.Column("wo_number_alias", sa.String(length=64), nullable=True),
        sa.Column(
            "company_id",
            sa.String(length=36),
            sa.ForeignKey("companies.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "deal_id",
            sa.String(length=36),
            sa.ForeignKey("deals.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "quote_id",
            sa.String(length=36),
            sa.ForeignKey("quotes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("customer_name", sa.String(length=255), nullable=False),
        sa.Column("machine_model", sa.String(length=255), nullable=True),
        sa.Column("serial_number", sa.String(length=128), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=False, server_default="draft"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="draft"),
        sa.Column("promised_dates", sa.JSON(), nullable=True),
        sa.Column("project_dir", sa.String(length=1024), nullable=True),
        sa.Column("region_template", sa.String(length=32), nullable=True),
        sa.Column("quote_value", sa.Numeric(15, 2), nullable=True),
        sa.Column("currency", sa.String(length=10), nullable=True),
        sa.Column(
            "portfolio_bucket",
            sa.String(length=32),
            nullable=False,
            server_default="current",
        ),
        sa.Column("status_summary", sa.Text(), nullable=True),
        sa.Column("next_milestone", sa.Text(), nullable=True),
        sa.Column("risk", sa.Text(), nullable=True),
        sa.Column("is_sister_company", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_stock", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("low_confidence", sa.JSON(), nullable=True),
        sa.Column("atlas_project_name", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("confirmed_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "stage IN (" + ", ".join(f"'{s}'" for s in _WO_STAGES) + ")",
            name="ck_work_orders_stage",
        ),
    )
    op.create_index("uq_work_orders_wo_number", "work_orders", ["wo_number"], unique=True)
    op.create_index("ix_work_orders_alias", "work_orders", ["wo_number_alias"])
    op.create_index("ix_work_orders_company_id", "work_orders", ["company_id"])
    op.create_index("ix_work_orders_stage", "work_orders", ["stage"])
    op.create_index("ix_work_orders_status", "work_orders", ["status"])
    op.create_index("ix_work_orders_quote_id", "work_orders", ["quote_id"])

    op.create_table(
        "wo_payment_milestones",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "work_order_id",
            sa.String(length=36),
            sa.ForeignKey("work_orders.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("description", sa.String(length=512), nullable=False),
        sa.Column("pct", sa.Float(), nullable=False),
        sa.Column("gate_stage", sa.String(length=32), nullable=True),
        sa.Column("amount", sa.Numeric(15, 2), nullable=True),
        sa.Column("currency", sa.String(length=10), nullable=True),
        sa.Column("due_event", sa.String(length=128), nullable=True),
        sa.Column("received_at", sa.DateTime(), nullable=True),
        sa.Column("received_ref", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_wo_payment_milestones_wo",
        "wo_payment_milestones",
        ["work_order_id"],
    )
    op.create_index(
        "uq_wo_payment_milestones_wo_seq",
        "wo_payment_milestones",
        ["work_order_id", "seq"],
        unique=True,
    )

    # Extend atlas logbook events → work_orders (complete PG migration path).
    op.add_column(
        "atlas_events",
        sa.Column(
            "work_order_id",
            sa.String(length=36),
            sa.ForeignKey("work_orders.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column("atlas_events", sa.Column("evidence_ref", sa.Text(), nullable=True))
    op.add_column("atlas_events", sa.Column("to_stage", sa.String(length=32), nullable=True))
    op.create_index("ix_atlas_events_work_order_id", "atlas_events", ["work_order_id"])

    # Event-derived stage: block direct UPDATE of stage unless session flag set.
    op.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION brain_prevent_wo_stage_direct_update()
            RETURNS trigger AS $$
            BEGIN
              IF NEW.stage IS DISTINCT FROM OLD.stage THEN
                IF COALESCE(current_setting('brain_os.allow_wo_stage_update', true), '') <> '1' THEN
                  RAISE EXCEPTION
                    'work_orders.stage is event-derived; use stage_advance via ira wo advance'
                    USING ERRCODE = 'check_violation';
                END IF;
              END IF;
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
    )
    op.execute(text("DROP TRIGGER IF EXISTS trg_work_orders_stage_guard ON work_orders"))
    op.execute(
        text(
            """
            CREATE TRIGGER trg_work_orders_stage_guard
            BEFORE UPDATE OF stage ON work_orders
            FOR EACH ROW
            EXECUTE FUNCTION brain_prevent_wo_stage_direct_update()
            """
        )
    )

    # Seed region payment templates (operator picks + edits at WO creation).
    op.execute(
        text(
            """
            INSERT INTO wo_payment_milestone_templates (id, region_key, name, milestones, notes)
            VALUES
            (
              'tmpl-india-std',
              'india',
              'India standard 25-30 / 65-70 / 5-10',
              '[
                {"seq": 1, "description": "Advance with PO", "pct": 30, "gate_stage": "po_received"},
                {"seq": 2, "description": "Before dispatch / after FAT", "pct": 65, "gate_stage": "fat"},
                {"seq": 3, "description": "After installation", "pct": 5, "gate_stage": "installation"}
              ]'::json,
              'Mined: IAC/DEMO-CO-class India & standard pattern 25-30/65-70/5-10'
            ),
            (
              'tmpl-japan-std',
              'japan',
              'Japan 30 / 60 / 10',
              '[
                {"seq": 1, "description": "Advance with PO", "pct": 30, "gate_stage": "po_received"},
                {"seq": 2, "description": "Before dispatch", "pct": 60, "gate_stage": "fat"},
                {"seq": 3, "description": "After installation", "pct": 10, "gate_stage": "installation"}
              ]'::json,
              'Operator template: Japan 30/60/10'
            ),
            (
              'tmpl-eu-staged',
              'eu',
              'EU staged 30/25/25/15/5 (+PBG option)',
              '[
                {"seq": 1, "description": "Advance with PO", "pct": 30, "gate_stage": "po_received"},
                {"seq": 2, "description": "Progress / major items", "pct": 25, "gate_stage": "procurement_build"},
                {"seq": 3, "description": "Pre-FAT / readiness", "pct": 25, "gate_stage": "trials"},
                {"seq": 4, "description": "After FAT before dispatch", "pct": 15, "gate_stage": "fat"},
                {"seq": 5, "description": "After installation (PBG may apply)", "pct": 5, "gate_stage": "installation"}
              ]'::json,
              'Mined: AcmeDemo/EU pattern 30/25/25/15/5; PBG on final often 12-13 months'
            ),
            (
              'tmpl-na-export',
              'na',
              'NA / export 30 / 60 / 10',
              '[
                {"seq": 1, "description": "Advance with PO / LC", "pct": 30, "gate_stage": "po_received"},
                {"seq": 2, "description": "Before dispatch after FAT", "pct": 60, "gate_stage": "fat"},
                {"seq": 3, "description": "After commissioning", "pct": 10, "gate_stage": "installation"}
              ]'::json,
              'NRC/Canada-class export default'
            )
            ON CONFLICT (id) DO NOTHING
            """
        )
    )


def downgrade() -> None:
    op.execute(text("DROP TRIGGER IF EXISTS trg_work_orders_stage_guard ON work_orders"))
    op.execute(text("DROP FUNCTION IF EXISTS brain_prevent_wo_stage_direct_update()"))

    op.drop_index("ix_atlas_events_work_order_id", table_name="atlas_events")
    op.drop_column("atlas_events", "to_stage")
    op.drop_column("atlas_events", "evidence_ref")
    op.drop_column("atlas_events", "work_order_id")

    op.drop_index("uq_wo_payment_milestones_wo_seq", table_name="wo_payment_milestones")
    op.drop_index("ix_wo_payment_milestones_wo", table_name="wo_payment_milestones")
    op.drop_table("wo_payment_milestones")

    op.drop_index("ix_work_orders_quote_id", table_name="work_orders")
    op.drop_index("ix_work_orders_status", table_name="work_orders")
    op.drop_index("ix_work_orders_stage", table_name="work_orders")
    op.drop_index("ix_work_orders_company_id", table_name="work_orders")
    op.drop_index("ix_work_orders_alias", table_name="work_orders")
    op.drop_index("uq_work_orders_wo_number", table_name="work_orders")
    op.drop_table("work_orders")

    op.drop_index(
        "uq_wo_payment_milestone_templates_region",
        table_name="wo_payment_milestone_templates",
    )
    op.drop_table("wo_payment_milestone_templates")
