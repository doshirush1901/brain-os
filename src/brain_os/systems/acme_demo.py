"""Load public Acme demo pack into BRAIN_DATA_DIR (proof registry + optional CRM)."""

from __future__ import annotations

import json
import logging
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "examples" / "acme" / "crm_seed.json").is_file():
        return cwd
    candidate = Path(__file__).resolve().parents[4]
    if (candidate / "examples" / "acme" / "crm_seed.json").is_file():
        return candidate
    return cwd


def _data_dir() -> Path:
    from brain_os.systems.data_dir_lock import get_data_dir

    return get_data_dir()


def seed_acme_demo(*, force: bool = False, crm: bool = True) -> dict[str, Any]:
    """Copy proof registry; optionally seed Postgres CRM from examples/acme/crm_seed.json."""
    root = _repo_root()
    acme = root / "examples" / "acme"
    report: dict[str, Any] = {"ok": True, "acme_dir": str(acme), "actions": []}

    if not acme.is_dir():
        report["ok"] = False
        report["error"] = f"Acme pack not found: {acme}"
        return report

    data = _data_dir()
    data.mkdir(parents=True, exist_ok=True)

    proof_src = acme / "proof_registry.json"
    proof_dest = data / "knowledge" / "outbound_proof_artifacts.json"
    if proof_src.is_file():
        proof_dest.parent.mkdir(parents=True, exist_ok=True)
        if force or not proof_dest.is_file():
            shutil.copy2(proof_src, proof_dest)
            report["actions"].append(f"proof_registry → {proof_dest}")

    marker = data / "brain" / "acme_demo_seeded.json"
    if marker.is_file() and not force:
        report["actions"].append("marker present — skip CRM (use --force to re-seed)")
        return report

    if not crm:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"proof_only": True}) + "\n", encoding="utf-8")
        return report

    seed_path = acme / "crm_seed.json"
    if not seed_path.is_file():
        return report

    try:
        payload = json.loads(seed_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["ok"] = False
        report["error"] = str(exc)
        return report

    import asyncio

    async def _seed_crm() -> int:
        from brain_os.data.crm import CRMDatabase
        from brain_os.data.models import DealStage

        db = CRMDatabase()
        await db.create_tables()
        companies_by_name: dict[str, Any] = {}

        for row in payload.get("companies", []):
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            try:
                comp = await db.create_company(
                    name=name,
                    domain=str(row.get("domain", "")).strip() or None,
                    industry=str(row.get("industry", "")).strip() or None,
                )
                companies_by_name[name] = comp
            except (OSError, RuntimeError, ValueError):
                existing = await db.list_companies({"industry": row.get("industry")})
                for c in existing:
                    if c.name == name:
                        companies_by_name[name] = c
                        break

        contacts_by_email: dict[str, Any] = {}
        for row in payload.get("contacts", []):
            email = str(row.get("email", "")).strip().lower()
            if not email:
                continue
            contact, _ = await db.get_or_create_contact_by_email(
                email,
                name=str(row.get("name", "")).strip() or None,
                source="acme_demo",
            )
            contacts_by_email[email] = contact

        deals = 0
        for row in payload.get("deals", []):
            company_name = str(row.get("company", "")).strip()
            title = str(row.get("title", "")).strip()
            if not title:
                continue
            contact = next(iter(contacts_by_email.values()), None)
            for c_row in payload.get("contacts", []):
                if str(c_row.get("company", "")).strip() == company_name:
                    em = str(c_row.get("email", "")).strip().lower()
                    contact = contacts_by_email.get(em) or contact
                    break
            if contact is None:
                continue
            stage_raw = str(row.get("stage", "proposal")).strip().upper().replace("-", "_")
            try:
                stage = DealStage(stage_raw)
            except ValueError:
                stage = DealStage.ENGAGED
            value = row.get("value_usd")
            await db.create_deal(
                contact_id=contact.id,
                title=title,
                stage=stage,
                value=Decimal(str(value)) if value is not None else Decimal(0),
                notes=str(row.get("next_step", "")).strip() or None,
            )
            deals += 1
        return deals

    try:
        deal_count = asyncio.run(_seed_crm())
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"seeded_from": str(seed_path), "deals": deal_count}, indent=2) + "\n",
            encoding="utf-8",
        )
        report["actions"].append(f"crm seeded ({deal_count} deals)")
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Acme CRM seed skipped: %s", exc)
        report["actions"].append(f"crm skipped: {exc}")

    return report
