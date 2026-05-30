"""CRM Enrichment Pipeline -- multi-agent data enrichment.

Runs 5 enrichment passes over CRM contacts using KB evidence:

1. **Clio** (Research): company region/industry, contact role
2. **Hephaestus** (Production): machine model, deal creation
3. **Plutus** (Finance): deal value, currency
4. **Hermes** (Marketing): lead score, warmth level, tags
5. **Delphi** (Verify): re-verify contact type with enriched data

Each pass is idempotent and safe to re-run.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal
from typing import Any

import httpx
from qdrant_client.http.exceptions import UnexpectedResponse
from sqlalchemy.exc import SQLAlchemyError

from brain_os.brain.qdrant_manager import QdrantManager
from brain_os.data.crm import CRMDatabase
from brain_os.data.models import DealStage, WarmthLevel
from brain_os.exceptions import DatabaseError, LLMError

logger = logging.getLogger(__name__)

_KB_SEARCH_ERRORS = (
    DatabaseError,
    LLMError,
    httpx.HTTPError,
    OSError,
    ValueError,
    TypeError,
    UnexpectedResponse,
)

_ENRICH_CONTACT_ERRORS = (
    DatabaseError,
    LLMError,
    SQLAlchemyError,
    httpx.HTTPError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    UnexpectedResponse,
    RuntimeError,
)


class CRMEnricher:
    """Orchestrates multi-agent CRM enrichment from KB evidence."""

    def __init__(
        self,
        crm: CRMDatabase,
        qdrant: QdrantManager,
        *,
        dry_run: bool = False,
    ) -> None:
        self._crm = crm
        self._qdrant = qdrant
        self._dry_run = dry_run
        self._stats: dict[str, int] = {
            "contacts_processed": 0,
            "companies_enriched": 0,
            "roles_found": 0,
            "deals_created": 0,
            "deals_valued": 0,
            "scores_set": 0,
            "types_changed": 0,
            "errors": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    async def enrich_all(
        self,
        contact_type_filter: str | None = None,
    ) -> dict[str, Any]:
        """Run the full enrichment pipeline."""
        filters = {}
        if contact_type_filter:
            filters["contact_type"] = contact_type_filter

        contacts = await self._crm.list_contacts(filters or None)
        logger.info("Enriching %d contacts", len(contacts))

        company_cache: dict[str, dict[str, Any]] = {}

        for i, contact in enumerate(contacts):
            try:
                company_name = ""
                if contact.company_id:
                    comp = await self._crm.get_company(str(contact.company_id))
                    company_name = comp.name if comp else ""

                kb_evidence = await self._search_kb(contact.email, company_name, contact.name)

                await self._enrich_company(contact, company_name, kb_evidence, company_cache)
                await self._enrich_contact_role(contact, kb_evidence)
                await self._enrich_deals(contact, company_name, kb_evidence)
                await self._enrich_score(contact, company_name, kb_evidence)

                self._stats["contacts_processed"] += 1

                if (i + 1) % 25 == 0:
                    logger.info(
                        "Progress: %d/%d (deals=%d, scores=%d, companies=%d)",
                        i + 1,
                        len(contacts),
                        self._stats["deals_created"],
                        self._stats["scores_set"],
                        self._stats["companies_enriched"],
                    )
            except _ENRICH_CONTACT_ERRORS:
                self._stats["errors"] += 1
                logger.exception("Failed to enrich %s", contact.email)

        logger.info("Enrichment complete: %s", self._stats)
        return {"stats": self._stats, "dry_run": self._dry_run}

    # ── KB Search ────────────────────────────────────────────────────────

    async def _search_kb(
        self,
        email: str,
        company: str,
        name: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search KB for all evidence about this contact."""
        evidence: dict[str, list[dict[str, Any]]] = {
            "company_info": [],
            "machine_orders": [],
            "quotes": [],
            "interactions": [],
        }

        search_term = company or name or email.split("@")[0]
        if not search_term or len(search_term) < 2:
            return evidence

        try:
            results = await self._qdrant.search(
                f"{search_term} region country industry customer",
                limit=5,
            )
            evidence["company_info"] = results

            results = await self._qdrant.search(
                f"{search_term} machine DEMO order delivery forming area project",
                limit=5,
            )
            evidence["machine_orders"] = results

            results = await self._qdrant.search(
                f"{search_term} quote price value order amount payment",
                limit=5,
            )
            evidence["quotes"] = results

        except _KB_SEARCH_ERRORS:
            logger.warning("KB search failed for %s", search_term, exc_info=True)

        return evidence

    # ── Pass 1: Company Enrichment (Clio) ────────────────────────────────

    async def _enrich_company(
        self,
        contact: Any,
        company_name: str,
        evidence: dict[str, list[dict[str, Any]]],
        cache: dict[str, dict[str, Any]],
    ) -> None:
        if not contact.company_id:
            return

        cid = str(contact.company_id)
        if cid in cache:
            return

        comp = await self._crm.get_company(cid)
        if not comp:
            return

        if comp.region and comp.industry:
            cache[cid] = {"region": comp.region, "industry": comp.industry}
            return

        region = comp.region or ""
        industry = comp.industry or ""

        for r in evidence.get("company_info", []):
            content = r.get("content", "").lower()
            if company_name and company_name.lower() not in content:
                continue
            if not company_name:
                continue

            if not region:
                region = self._extract_region(content, company_name)
            if not industry:
                industry = self._extract_industry(content)

        if (region or industry) and (
            region != (comp.region or "") or industry != (comp.industry or "")
        ):
            if not self._dry_run:
                updates: dict[str, Any] = {}
                if region and not comp.region:
                    updates["region"] = region
                if industry and not comp.industry:
                    updates["industry"] = industry
                if updates:
                    await self._crm.update_company(cid, **updates)
                    self._stats["companies_enriched"] += 1
            else:
                self._stats["companies_enriched"] += 1
            logger.debug("Company %s: region=%s, industry=%s", company_name, region, industry)

        cache[cid] = {"region": region, "industry": industry}

    @staticmethod
    def _extract_region(content: str, company: str) -> str:
        region_map = {
            "india": "India",
            "germany": "Germany",
            "france": "France",
            "netherlands": "Netherlands",
            "uk": "UK",
            "sweden": "Sweden",
            "hungary": "Hungary",
            "belgium": "Belgium",
            "denmark": "Denmark",
            "japan": "Japan",
            "canada": "Canada",
            "usa": "USA",
            "uae": "UAE",
            "dubai": "UAE",
            "russia": "Russia",
            "italy": "Italy",
            "czech": "Czech Republic",
            "turkey": "Turkey",
            "south africa": "South Africa",
            "brazil": "Brazil",
            "portugal": "Portugal",
            "spain": "Spain",
            "austria": "Austria",
            "switzerland": "Switzerland",
            "norway": "Norway",
            "israel": "Israel",
            "china": "China",
            "korea": "South Korea",
            "oman": "Oman",
            "serbia": "Serbia",
            "romania": "Romania",
            "estonia": "Estonia",
            "lithuania": "Lithuania",
            "faroe": "Faroe Islands",
            "ireland": "Ireland",
        }
        for keyword, region in region_map.items():
            if keyword in content:
                return region
        return ""

    @staticmethod
    def _extract_industry(content: str) -> str:
        industry_map = {
            "automotive": "Automotive",
            "luggage": "Luggage",
            "sanitaryware": "Sanitaryware",
            "bath": "Sanitaryware",
            "packaging": "Packaging",
            "signage": "Signage",
            "construction": "Construction",
            "medical": "Medical",
            "food": "Food & Beverage",
            "refriger": "Refrigeration",
            "aerospace": "Aerospace",
            "marine": "Marine",
            "electronics": "Electronics",
            "furniture": "Furniture",
            "thermoform": "Thermoforming",
            "plastic": "Plastics",
        }
        for keyword, industry in industry_map.items():
            if keyword in content:
                return industry
        return ""

    # ── Pass 2: Contact Role ─────────────────────────────────────────────

    async def _enrich_contact_role(
        self,
        contact: Any,
        evidence: dict[str, list[dict[str, Any]]],
    ) -> None:
        if contact.role:
            return

        name = (contact.name or "").lower()
        for r in evidence.get("company_info", []) + evidence.get("machine_orders", []):
            content = r.get("content", "")
            role = self._extract_role(content, name, contact.email)
            if role:
                if not self._dry_run:
                    await self._crm.update_contact(str(contact.id), role=role)
                self._stats["roles_found"] += 1
                logger.debug("Role for %s: %s", contact.email, role)
                return

    @staticmethod
    def _extract_role(content: str, name: str, email: str) -> str:
        role_patterns = [
            r"(?:director|manager|head|chief|vp|president|ceo|cto|owner|founder|engineer|buyer|purchas)",
        ]
        local_part = email.split("@")[0].lower()

        for line in content.split("\n"):
            line_lower = line.lower()
            if name in line_lower or local_part in line_lower:
                for pattern in role_patterns:
                    match = re.search(pattern, line_lower)
                    if match:
                        words = line.strip().split()
                        for i, w in enumerate(words):
                            if re.search(pattern, w.lower()):
                                role_words = words[max(0, i - 1) : i + 3]
                                return " ".join(role_words).strip("- ,;:")
        return ""

    # ── Pass 3: Deal Creation (Hephaestus + Plutus) ──────────────────────

    async def _enrich_deals(
        self,
        contact: Any,
        company_name: str,
        evidence: dict[str, list[dict[str, Any]]],
    ) -> None:
        ct = contact.contact_type.value if contact.contact_type else ""
        if ct not in ("LIVE_CUSTOMER", "PAST_CUSTOMER", "LEAD_WITH_INTERACTIONS"):
            return

        existing_deals = await self._crm.get_deals_for_contact(str(contact.id))
        if existing_deals:
            return

        machine_model = ""
        deal_value = 0.0
        currency = "USD"
        forming_area = ""
        if ct == "LEAD_WITH_INTERACTIONS":
            stage = DealStage.ENGAGED
        elif ct == "PAST_CUSTOMER":
            stage = DealStage.WON
        else:
            stage = DealStage.NEGOTIATION
        notes_parts: list[str] = []

        for r in evidence.get("machine_orders", []):
            content = r.get("content", "")
            if company_name.lower() not in content.lower():
                continue

            if not machine_model:
                machine_model = self._extract_machine_model(content)
            if not forming_area:
                forming_area = self._extract_forming_area(content)

            source = r.get("source", "")
            if "order book" in source.lower() or "order" in source.lower():
                status = self._extract_project_status(content, company_name)
                if status:
                    notes_parts.append(f"Status: {status}")
                    if any(
                        w in status.lower()
                        for w in ("fabrication", "assembly", "design", "ordering")
                    ):
                        stage = DealStage.NEGOTIATION

        for r in evidence.get("quotes", []):
            content = r.get("content", "")
            if company_name.lower() not in content.lower():
                continue
            val, cur = self._extract_value(content)
            if val > deal_value:
                deal_value = val
                currency = cur

        if not machine_model:
            return

        title = f"{machine_model} for {company_name}"
        if forming_area:
            title += f" ({forming_area})"
            notes_parts.append(f"Forming area: {forming_area}")

        if not self._dry_run:
            await self._crm.create_deal(
                contact_id=str(contact.id),
                title=title,
                value=Decimal(str(deal_value)) if deal_value else Decimal("0"),
                currency=currency,
                stage=stage,
                machine_model=machine_model,
                notes="\n".join(notes_parts) if notes_parts else None,
            )
        self._stats["deals_created"] += 1
        if deal_value > 0:
            self._stats["deals_valued"] += 1
        logger.debug(
            "Deal: %s | %s | %s %s | %s",
            title,
            stage.value,
            currency,
            deal_value,
            contact.email,
        )

    @staticmethod
    def _extract_machine_model(content: str) -> str:
        patterns = [
            r"(DEMO[-\s]?[A-Z]?[-\s]?\d{3,5}(?:[-\s]?[A-Z]{1,3})?)",
            r"(PF[12][-\s]?[XCAPS]?[-\s]?\d{3,5})",
            r"(AM[-\s]?[A-Z]?[-\s]?\d{3,5})",
            r"(FCS[-\s]?\w+)",
            r"(IMG\s+\w+)",
            r"(RT[-\s]?\d[A-Z][-\s]?\d{4})",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_forming_area(content: str) -> str:
        patterns = [
            r"(\d{3,5}\s*[x×*]\s*\d{3,5})\s*(?:mm)?",
            r"forming\s+area[:\s]+(\d{3,5}\s*[x×*]\s*\d{3,5})",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).replace("×", "x").replace("*", "x")
        return ""

    @staticmethod
    def _extract_value(content: str) -> tuple[float, str]:
        patterns = [
            (r"(?:order\s+)?value[:\s]+(\d+(?:\.\d+)?)", "USD"),
            (r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:k|K|USD)?", "USD"),
            (r"€\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", "EUR"),
            (r"INR\s*(\d{1,3}(?:,\d{2})*(?:,\d{3})*(?:\.\d+)?)", "INR"),
            (r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs)", "INR"),
        ]
        for pattern, default_cur in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                val_str = match.group(1).replace(",", "")
                try:
                    val = float(val_str)
                    if "lakh" in content.lower():
                        val *= 100000
                    if val > 0:
                        return val, default_cur
                except ValueError:
                    pass
        return 0.0, "USD"

    @staticmethod
    def _extract_project_status(content: str, company: str) -> str:
        company_lower = company.lower()

        for line in content.split("\n"):
            if company_lower in line.lower():
                statuses = [
                    "complete",
                    "packing",
                    "shipping",
                    "fabrication",
                    "assembly",
                    "design",
                    "ordering",
                    "installation",
                    "paint",
                    "programming",
                    "wet trial",
                ]
                for status in statuses:
                    if status in line.lower():
                        return status.title()
        return ""

    # ── Pass 4: Lead Scoring (Hermes) ────────────────────────────────────

    async def _enrich_score(
        self,
        contact: Any,
        company_name: str,
        evidence: dict[str, list[dict[str, Any]]],
    ) -> None:
        if (contact.lead_score or 0) > 0:
            return

        ct = contact.contact_type.value if contact.contact_type else ""
        score = 0.0
        warmth = WarmthLevel.STRANGER
        tags: list[str] = []

        if ct == "LIVE_CUSTOMER":
            score = 100.0
            warmth = WarmthLevel.TRUSTED
            tags.append("customer:active")
        elif ct == "PAST_CUSTOMER":
            score = 70.0
            warmth = WarmthLevel.WARM
            tags.append("customer:past")
        else:
            total_evidence = sum(len(v) for v in evidence.values())
            has_machine_mention = any(
                self._extract_machine_model(r.get("content", ""))
                for r in evidence.get("machine_orders", [])
                if company_name.lower() in r.get("content", "").lower()
            )
            has_quote = any(
                company_name.lower() in r.get("content", "").lower()
                for r in evidence.get("quotes", [])
            )

            if has_machine_mention:
                score += 30
                tags.append("interest:specific_machine")
            if has_quote:
                score += 20
                tags.append("has_quote")
            if total_evidence >= 5:
                score += 15
                tags.append("high_kb_presence")
            elif total_evidence >= 2:
                score += 5

            source = contact.source or ""
            if "gmail" in source:
                score += 10
                tags.append("source:email")
            if "k20" in source.lower() or "k show" in source.lower():
                tags.append("source:k_show")
                score += 5
            if "demo_show" in source.lower():
                tags.append("source:demo_show")
                score += 5

            score = min(score, 65.0)

            if score >= 40:
                warmth = WarmthLevel.FAMILIAR
            elif score >= 20:
                warmth = WarmthLevel.ACQUAINTANCE
            else:
                warmth = WarmthLevel.STRANGER

        if not self._dry_run:
            updates: dict[str, Any] = {"lead_score": score}
            if not contact.warmth_level:
                updates["warmth_level"] = warmth
            if tags and not contact.tags:
                updates["tags"] = tags
            await self._crm.update_contact(str(contact.id), **updates)

        self._stats["scores_set"] += 1
        logger.debug("Score %s: %.0f (%s) tags=%s", contact.email, score, warmth.value, tags)
