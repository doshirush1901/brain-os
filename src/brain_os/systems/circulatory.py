"""Circulatory System -- keeps CRM, Neo4j, and Qdrant in sync.

Subscribes to :class:`~brain_os.systems.data_event_bus.DataEventBus` events
and propagates changes across stores.  Also maintains a persistent
change ledger for auditability and replay.

Named after the biological circulatory system: it circulates data
between organs (stores) the way blood carries nutrients between
body systems.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from email.utils import parseaddr
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from sqlalchemy.exc import SQLAlchemyError

from brain_os.brain.write_contract import build_receipt, enqueue_retry, record_receipt
from brain_os.data.models import KnowledgeItem
from brain_os.systems.data_event_bus import (
    DataEvent,
    DataEventBus,
    EventType,
    SourceStore,
)

logger = logging.getLogger(__name__)

_NEO4J_SYNC_ERRORS = (RuntimeError, ValueError, TypeError, AttributeError, OSError)
_QDRANT_SYNC_ERRORS = (httpx.HTTPError, OSError, ValueError, TypeError, RuntimeError)
_LEDGER_WRITE_ERRORS = (OSError, TypeError, ValueError)
_LEDGER_READ_ERRORS = (OSError, json.JSONDecodeError, ValueError, TypeError)
_CRM_GRAPH_BACK_SYNC_ERRORS = (
    SQLAlchemyError,
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    OSError,
)

_LEDGER_PATH = Path("data/brain/data_ledger.jsonl")

# Addr-spec inside name-addr (``Last, First <u@h>``); ``parseaddr`` often returns ("", "") for commas in the display name.
_ANGLE_EMAIL = re.compile(r"<\s*([^<>\n\r]+@[^<>\n\r]+)\s*>", re.IGNORECASE)
# Last resort: tail mailbox in ``First Last user@domain.tld`` (no angle brackets).
_LOOSE_EMAIL = re.compile(
    r"(?:^|[^\w.+-])([A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,})\b",
    re.IGNORECASE,
)


def _addr_spec_sane(em: str) -> bool:
    """True if ``em`` is a single mailbox-shaped addr-spec (ASCII, CRM-safe)."""
    em = em.strip()
    if em.count("@") != 1:
        return False
    local, _, domain = em.partition("@")
    if not local or not domain:
        return False
    if re.search(r"[\s<>]", domain):
        return False
    if local.startswith('"') and local.endswith('"') and len(local) >= 2:
        return True
    if re.search(r"[\s,<>;]", local):
        return False
    return True


def _normalize_person_email(raw: str) -> str:
    """Return lowercased addr-spec for CRM lookup (RFC5322-ish name-addr and loose forms)."""
    s = (raw or "").strip()
    if not s:
        return ""
    for m in reversed(list(_ANGLE_EMAIL.finditer(s))):
        em = m.group(1).strip().lower()
        if _addr_spec_sane(em):
            return em
    _name, addr = parseaddr(s)
    if addr:
        em = addr.strip().lower()
        if _addr_spec_sane(em):
            return em
    em = s.lower()
    if "<" not in em and ">" not in em and _addr_spec_sane(em):
        return em
    for m in reversed(list(_LOOSE_EMAIL.finditer(s))):
        em = m.group(1).lower()
        if _addr_spec_sane(em):
            return em
    return ""


class CirculatorySystem:
    """Wires sync handlers to the DataEventBus and maintains the change ledger."""

    def __init__(
        self,
        event_bus: DataEventBus,
        *,
        crm: Any = None,
        graph: Any = None,
        qdrant: Any = None,
        embedding: Any = None,
    ) -> None:
        self._bus = event_bus
        self._crm = crm
        self._graph = graph
        self._qdrant = qdrant
        self._embedding = embedding

        self._ledger_path = _LEDGER_PATH
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)

        DataEventBus.bind(event_bus)
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Subscribe all sync handlers to the event bus."""
        self._bus.subscribe_all(self._ledger_handler)

        if self._graph:
            self._bus.subscribe(EventType.CONTACT_CREATED, self._crm_to_neo4j_contact)
            self._bus.subscribe(EventType.CONTACT_UPDATED, self._crm_to_neo4j_contact)
            self._bus.subscribe(EventType.CONTACT_CLASSIFIED, self._crm_to_neo4j_contact)
            self._bus.subscribe(EventType.COMPANY_CREATED, self._crm_to_neo4j_company)
            self._bus.subscribe(EventType.DEAL_CREATED, self._crm_to_neo4j_deal)
            self._bus.subscribe(EventType.DEAL_UPDATED, self._crm_to_neo4j_deal)
            self._bus.subscribe(EventType.QUOTE_UPSERTED, self._erp_quote_to_neo4j)
            self._bus.subscribe(EventType.WORK_ORDER_UPSERTED, self._erp_wo_to_neo4j)
            self._bus.subscribe(EventType.MACHINE_ASSET_UPSERTED, self._erp_asset_to_neo4j)
            self._bus.subscribe(EventType.ACCOUNT_EVENT_RECORDED, self._erp_event_to_neo4j)
            self._bus.subscribe(EventType.RELATIONSHIP_DISCOVERED, self._relationship_to_neo4j)

        if self._qdrant and self._embedding:
            self._bus.subscribe(EventType.CONTACT_CREATED, self._crm_to_qdrant)
            self._bus.subscribe(EventType.CONTACT_UPDATED, self._crm_to_qdrant)
            self._bus.subscribe(EventType.CONTACT_CLASSIFIED, self._crm_to_qdrant)
            self._bus.subscribe(EventType.COMPANY_CREATED, self._company_to_qdrant)
            self._bus.subscribe(EventType.DEAL_CREATED, self._deal_to_qdrant)

        if self._crm:
            self._bus.subscribe(EventType.ENTITY_ADDED, self._neo4j_to_crm)

        # Campaign goal wait-steps — planner sequences; never sends
        self._bus.subscribe(EventType.EMAIL_CLASSIFIED, self._campaign_goal_event)
        self._bus.subscribe(EventType.EMAIL_RECEIVED, self._campaign_goal_event)
        self._bus.subscribe(EventType.DEAL_UPDATED, self._campaign_goal_event)
        self._bus.subscribe(EventType.DEAL_CREATED, self._campaign_goal_event)

        logger.info(
            "CirculatorySystem registered handlers (graph=%s, qdrant=%s, crm=%s)",
            self._graph is not None,
            self._qdrant is not None,
            self._crm is not None,
        )

    async def _campaign_goal_event(self, event: DataEvent) -> None:
        """Advance wait_for_event steps on matching bus events (gate-safe)."""
        try:
            from brain_os.systems.campaign_goals import apply_event_to_goals

            body = dict(event.payload or {})
            body["type"] = event.event_type.value
            body["event_type"] = event.event_type.value
            body.setdefault("evidence_ref", event.entity_id)
            body.setdefault("entity_id", event.entity_id)
            apply_event_to_goals(body)
        except Exception:
            logger.debug("campaign goal event hook failed", exc_info=True)

    # ── Change Ledger ────────────────────────────────────────────────────

    def _record_sync_receipt(
        self,
        *,
        store: str,
        success: bool,
        operation: str,
        run_id: str,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        receipt = build_receipt(
            store=store,
            attempted=True,
            success=success,
            operation=operation,
            run_id=run_id,
            reason=reason,
            metadata=metadata or {},
        )
        record_receipt(receipt)
        if not success:
            enqueue_retry(receipt)

    async def _ledger_handler(self, event: DataEvent) -> None:
        """Append every event to the persistent JSONL ledger."""
        entry = {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "entity_type": event.entity_type,
            "entity_id": event.entity_id,
            "source_store": event.source_store.value,
            "payload_keys": list(event.payload.keys()),
        }
        try:
            line = json.dumps(entry, default=str) + "\n"
            await asyncio.to_thread(self._append_ledger_line, line)
        except _LEDGER_WRITE_ERRORS:
            logger.warning("Ledger write failed", exc_info=True)

    # ── CRM → Neo4j ─────────────────────────────────────────────────────

    async def _crm_to_neo4j_contact(self, event: DataEvent) -> None:
        if event.source_store == SourceStore.NEO4J:
            return
        p = event.payload
        email = p.get("email", "")
        if not email:
            return

        try:
            await self._graph.add_person(
                name=p.get("name", ""),
                email=email,
                company_name=p.get("company", ""),
                role=p.get("role", ""),
                source_id=f"crm_contact::{email.lower()}",
                crm_contact_id=str(p.get("contact_id") or p.get("id") or "").strip(),
                crm_company_id=str(p.get("company_id") or "").strip(),
            )

            contact_type = p.get("contact_type", "")
            if contact_type:
                await self._graph._run_cypher_write(
                    "MATCH (p:Person {email: $email}) "
                    "SET p.contact_type = $ct, p.lead_score = $score",
                    {"email": email, "ct": contact_type, "score": p.get("lead_score", 0)},
                )

            logger.debug("Synced contact %s to Neo4j", email)
            self._record_sync_receipt(
                store="neo4j",
                success=True,
                operation="circulatory.crm_to_neo4j_contact",
                run_id=event.entity_id,
                metadata={"email": email, "event_type": event.event_type.value},
            )
        except _NEO4J_SYNC_ERRORS:
            logger.exception("CRM→Neo4j sync failed for %s", email)
            self._record_sync_receipt(
                store="neo4j",
                success=False,
                operation="circulatory.crm_to_neo4j_contact",
                run_id=event.entity_id,
                reason="sync_failed",
                metadata={"email": email, "event_type": event.event_type.value},
            )

    async def _crm_to_neo4j_company(self, event: DataEvent) -> None:
        if event.source_store == SourceStore.NEO4J:
            return
        p = event.payload
        name = p.get("name", "")
        if not name:
            return

        try:
            crm_id = str(p.get("company_id") or p.get("id") or "").strip()
            await self._graph.add_company(
                name=name,
                region=p.get("region", "") or "",
                industry=p.get("industry", "") or "",
                website=(p.get("website") or "") or "",
                crm_company_id=crm_id,
                source_id=f"crm_company::{name.lower()}",
            )
            logger.debug("Synced company %s to Neo4j", name)
        except _NEO4J_SYNC_ERRORS:
            logger.exception("CRM→Neo4j sync failed for company %s", name)
            self._record_sync_receipt(
                store="neo4j",
                success=False,
                operation="circulatory.crm_to_neo4j_company",
                run_id=event.entity_id,
                reason="sync_failed",
                metadata={"name": name, "event_type": event.event_type.value},
            )
        else:
            self._record_sync_receipt(
                store="neo4j",
                success=True,
                operation="circulatory.crm_to_neo4j_company",
                run_id=event.entity_id,
                metadata={"name": name, "event_type": event.event_type.value},
            )

    async def _crm_to_neo4j_deal(self, event: DataEvent) -> None:
        if event.source_store == SourceStore.NEO4J:
            return
        p = event.payload
        try:
            from brain_os.brain.graph_erp_spine import upsert_deal_node

            deal_id = str(p.get("id") or event.entity_id)
            company = str(p.get("company") or p.get("company_name") or "").strip()
            machine = str(p.get("machine_model") or "").strip()
            crm_company_id = str(p.get("company_id") or "").strip()
            stage = str(p.get("stage") or "")
            if hasattr(p.get("stage"), "value"):
                stage = str(p["stage"].value)

            if company:
                await self._graph.add_company(
                    name=company,
                    crm_company_id=crm_company_id,
                    source_id=f"crm_deal::{deal_id.lower()}",
                )
                await upsert_deal_node(
                    self._graph,
                    deal_id=deal_id,
                    company_name=company,
                    stage=stage,
                    value=float(p.get("value") or 0),
                    currency=str(p.get("currency") or ""),
                    machine_model=machine,
                    title=str(p.get("title") or ""),
                    crm_company_id=crm_company_id,
                    source_id=f"crm_deal::{deal_id}",
                )

            logger.debug("Synced deal %s to Neo4j (:Deal + INTERESTED_IN)", deal_id)
        except _NEO4J_SYNC_ERRORS:
            logger.exception("CRM→Neo4j deal sync failed for %s", event.entity_id)
            self._record_sync_receipt(
                store="neo4j",
                success=False,
                operation="circulatory.crm_to_neo4j_deal",
                run_id=event.entity_id,
                reason="sync_failed",
                metadata={"event_type": event.event_type.value},
            )
        else:
            self._record_sync_receipt(
                store="neo4j",
                success=True,
                operation="circulatory.crm_to_neo4j_deal",
                run_id=event.entity_id,
                metadata={"event_type": event.event_type.value},
            )

    async def _erp_quote_to_neo4j(self, event: DataEvent) -> None:
        if event.source_store == SourceStore.NEO4J or self._graph is None:
            return
        p = event.payload
        try:
            from brain_os.brain.graph_erp_spine import upsert_quote_node

            await upsert_quote_node(
                self._graph,
                quote_number=str(p.get("quote_number") or event.entity_id),
                company_name=str(p.get("company_name") or p.get("company") or ""),
                machine_model=str(p.get("machine_model") or ""),
                value=float(p.get("estimated_value") or p.get("value") or 0),
                currency=str(p.get("currency") or ""),
                status=str(p.get("status") or "OPEN"),
                sent_at=str(p.get("sent_at") or ""),
                crm_company_id=str(p.get("company_id") or ""),
                source_id=f"quote_registry::{p.get('quote_number') or event.entity_id}",
            )
            self._record_sync_receipt(
                store="neo4j",
                success=True,
                operation="circulatory.erp_quote_to_neo4j",
                run_id=event.entity_id,
            )
        except _NEO4J_SYNC_ERRORS:
            logger.exception("ERP quote→Neo4j failed for %s", event.entity_id)
            self._record_sync_receipt(
                store="neo4j",
                success=False,
                operation="circulatory.erp_quote_to_neo4j",
                run_id=event.entity_id,
                reason="sync_failed",
            )

    async def _erp_wo_to_neo4j(self, event: DataEvent) -> None:
        if event.source_store == SourceStore.NEO4J or self._graph is None:
            return
        p = event.payload
        try:
            from brain_os.brain.graph_erp_spine import upsert_account_event, upsert_work_order_node

            await upsert_work_order_node(
                self._graph,
                wo_number=str(p.get("wo_number") or event.entity_id),
                company_name=str(p.get("customer_name") or p.get("company_name") or ""),
                machine_model=str(p.get("machine_model") or ""),
                stage=str(p.get("stage") or ""),
                status=str(p.get("status") or ""),
                confirmed_at=str(p.get("confirmed_at") or ""),
                created_at=str(p.get("created_at") or ""),
                crm_company_id=str(p.get("company_id") or ""),
                source_id=f"work_order::{p.get('wo_number') or event.entity_id}",
            )
            if p.get("event_type") and p.get("event_at"):
                await upsert_account_event(
                    self._graph,
                    event_id=str(
                        p.get("event_id") or f"woevt:{event.entity_id}:{p.get('event_at')}"
                    ),
                    company_name=str(p.get("customer_name") or p.get("company_name") or ""),
                    event_type=str(p.get("event_type")),
                    at=str(p.get("event_at")),
                    stage=str(p.get("to_stage") or p.get("stage") or ""),
                    description=str(p.get("description") or "")[:500],
                    crm_company_id=str(p.get("company_id") or ""),
                    work_order_id=str(p.get("id") or ""),
                )
            self._record_sync_receipt(
                store="neo4j",
                success=True,
                operation="circulatory.erp_wo_to_neo4j",
                run_id=event.entity_id,
            )
        except _NEO4J_SYNC_ERRORS:
            logger.exception("ERP WO→Neo4j failed for %s", event.entity_id)
            self._record_sync_receipt(
                store="neo4j",
                success=False,
                operation="circulatory.erp_wo_to_neo4j",
                run_id=event.entity_id,
                reason="sync_failed",
            )

    async def _erp_asset_to_neo4j(self, event: DataEvent) -> None:
        if event.source_store == SourceStore.NEO4J or self._graph is None:
            return
        p = event.payload
        try:
            from brain_os.brain.graph_erp_spine import upsert_machine_asset_node

            await upsert_machine_asset_node(
                self._graph,
                asset_id=str(p.get("id") or event.entity_id),
                company_name=str(p.get("company_name") or ""),
                machine_model=str(p.get("model") or p.get("machine_model") or ""),
                installed_date=str(p.get("installed_date") or ""),
                warranty_end=str(p.get("warranty_end") or ""),
                serial=str(p.get("serial") or ""),
                crm_company_id=str(p.get("company_id") or ""),
                source_id=f"machine_asset::{p.get('id') or event.entity_id}",
            )
            self._record_sync_receipt(
                store="neo4j",
                success=True,
                operation="circulatory.erp_asset_to_neo4j",
                run_id=event.entity_id,
            )
        except _NEO4J_SYNC_ERRORS:
            logger.exception("ERP asset→Neo4j failed for %s", event.entity_id)
            self._record_sync_receipt(
                store="neo4j",
                success=False,
                operation="circulatory.erp_asset_to_neo4j",
                run_id=event.entity_id,
                reason="sync_failed",
            )

    async def _erp_event_to_neo4j(self, event: DataEvent) -> None:
        if event.source_store == SourceStore.NEO4J or self._graph is None:
            return
        p = event.payload
        try:
            from brain_os.brain.graph_erp_spine import upsert_account_event

            await upsert_account_event(
                self._graph,
                event_id=str(p.get("event_id") or event.entity_id),
                company_name=str(p.get("company_name") or p.get("customer_name") or ""),
                event_type=str(p.get("event_type") or "account_event"),
                at=str(p.get("at") or p.get("event_at") or ""),
                stage=str(p.get("stage") or ""),
                amount=p.get("amount"),
                currency=str(p.get("currency") or ""),
                description=str(p.get("description") or "")[:500],
                crm_company_id=str(p.get("company_id") or ""),
                work_order_id=str(p.get("work_order_id") or ""),
            )
            self._record_sync_receipt(
                store="neo4j",
                success=True,
                operation="circulatory.erp_event_to_neo4j",
                run_id=event.entity_id,
            )
        except _NEO4J_SYNC_ERRORS:
            logger.exception("ERP event→Neo4j failed for %s", event.entity_id)
            self._record_sync_receipt(
                store="neo4j",
                success=False,
                operation="circulatory.erp_event_to_neo4j",
                run_id=event.entity_id,
                reason="sync_failed",
            )

    async def _relationship_to_neo4j(self, event: DataEvent) -> None:
        """Write an agent-discovered relationship to Neo4j."""
        p = event.payload
        try:
            await self._graph.add_relationship(
                from_type=p.get("from_type", ""),
                from_key=p.get("from_key", ""),
                rel_type=p.get("rel", ""),
                to_type=p.get("to_type", ""),
                to_key=p.get("to_key", ""),
                properties=p.get("properties"),
                source_id=str(p.get("source_id") or f"circulatory::{event.entity_id}"),
            )
            logger.debug(
                "Synced relationship %s-[%s]->%s to Neo4j",
                p.get("from_key"),
                p.get("rel"),
                p.get("to_key"),
            )
            self._record_sync_receipt(
                store="neo4j",
                success=True,
                operation="circulatory.relationship_to_neo4j",
                run_id=event.entity_id,
                metadata={"event_type": event.event_type.value, "rel": p.get("rel", "")},
            )
        except _NEO4J_SYNC_ERRORS:
            logger.exception("Relationship→Neo4j sync failed for %s", event.entity_id)
            self._record_sync_receipt(
                store="neo4j",
                success=False,
                operation="circulatory.relationship_to_neo4j",
                run_id=event.entity_id,
                reason="sync_failed",
                metadata={"event_type": event.event_type.value, "rel": p.get("rel", "")},
            )

    # ── CRM → Qdrant ────────────────────────────────────────────────────

    async def _crm_to_qdrant(self, event: DataEvent) -> None:
        if event.source_store == SourceStore.QDRANT:
            return
        p = event.payload
        email = p.get("email", "")
        if not email:
            return

        contact_type = p.get("contact_type", "unknown")
        company = p.get("company", "")
        name = p.get("name", "")

        content = (
            f"CRM Contact: {name} ({email})\n"
            f"Company: {company}\n"
            f"Type: {contact_type}\n"
            f"Role: {p.get('role', 'N/A')}\n"
            f"Lead Score: {p.get('lead_score', 0)}\n"
            f"Source: {p.get('source', 'N/A')}"
        )

        item = KnowledgeItem(
            id=uuid4(),
            source=f"crm:contact:{email}",
            source_category="crm_contact",
            content=content,
            metadata={
                "email": email,
                "name": name,
                "company": company,
                "contact_type": contact_type,
                "lead_score": p.get("lead_score", 0),
                "entity_type": "contact",
            },
        )

        try:
            await self._qdrant.upsert_items([item])
            logger.debug("Synced contact %s to Qdrant", email)
        except _QDRANT_SYNC_ERRORS:
            logger.exception("CRM→Qdrant sync failed for %s", email)
            self._record_sync_receipt(
                store="qdrant",
                success=False,
                operation="circulatory.crm_to_qdrant_contact",
                run_id=event.entity_id,
                reason="sync_failed",
                metadata={"email": email, "event_type": event.event_type.value},
            )
        else:
            self._record_sync_receipt(
                store="qdrant",
                success=True,
                operation="circulatory.crm_to_qdrant_contact",
                run_id=event.entity_id,
                metadata={
                    "email": email,
                    "event_type": event.event_type.value,
                    "point_id": item.id.hex,
                },
            )

    async def _company_to_qdrant(self, event: DataEvent) -> None:
        if event.source_store == SourceStore.QDRANT:
            return
        p = event.payload
        name = p.get("name", "")
        if not name:
            return

        content = (
            f"CRM Company: {name}\n"
            f"Region: {p.get('region', 'N/A')}\n"
            f"Industry: {p.get('industry', 'N/A')}"
        )

        item = KnowledgeItem(
            id=uuid4(),
            source=f"crm:company:{name}",
            source_category="crm_company",
            content=content,
            metadata={
                "name": name,
                "region": p.get("region", ""),
                "industry": p.get("industry", ""),
                "entity_type": "company",
            },
        )

        try:
            await self._qdrant.upsert_items([item])
            logger.debug("Synced company %s to Qdrant", name)
        except _QDRANT_SYNC_ERRORS:
            logger.exception("CRM→Qdrant sync failed for company %s", name)
            self._record_sync_receipt(
                store="qdrant",
                success=False,
                operation="circulatory.crm_to_qdrant_company",
                run_id=event.entity_id,
                reason="sync_failed",
                metadata={"name": name, "event_type": event.event_type.value},
            )
        else:
            self._record_sync_receipt(
                store="qdrant",
                success=True,
                operation="circulatory.crm_to_qdrant_company",
                run_id=event.entity_id,
                metadata={
                    "name": name,
                    "event_type": event.event_type.value,
                    "point_id": item.id.hex,
                },
            )

    async def _deal_to_qdrant(self, event: DataEvent) -> None:
        if event.source_store == SourceStore.QDRANT:
            return
        p = event.payload

        content = (
            f"CRM Deal: {p.get('title', 'Untitled')}\n"
            f"Company: {p.get('company', 'N/A')}\n"
            f"Machine: {p.get('machine_model', 'N/A')}\n"
            f"Value: {p.get('currency', 'USD')} {p.get('value', 0)}\n"
            f"Stage: {p.get('stage', 'NEW')}"
        )

        item = KnowledgeItem(
            id=uuid4(),
            source=f"crm:deal:{event.entity_id}",
            source_category="crm_deal",
            content=content,
            metadata={
                "deal_id": event.entity_id,
                "title": p.get("title", ""),
                "machine_model": p.get("machine_model", ""),
                "stage": p.get("stage", "NEW"),
                "value": p.get("value", 0),
                "entity_type": "deal",
            },
        )

        try:
            await self._qdrant.upsert_items([item])
            logger.debug("Synced deal %s to Qdrant", event.entity_id)
        except _QDRANT_SYNC_ERRORS:
            logger.exception("CRM→Qdrant deal sync failed for %s", event.entity_id)
            self._record_sync_receipt(
                store="qdrant",
                success=False,
                operation="circulatory.crm_to_qdrant_deal",
                run_id=event.entity_id,
                reason="sync_failed",
                metadata={"event_type": event.event_type.value},
            )
        else:
            self._record_sync_receipt(
                store="qdrant",
                success=True,
                operation="circulatory.crm_to_qdrant_deal",
                run_id=event.entity_id,
                metadata={"event_type": event.event_type.value, "point_id": item.id.hex},
            )

    # ── Neo4j → CRM ─────────────────────────────────────────────────────

    async def _neo4j_to_crm(self, event: DataEvent) -> None:
        """When a new entity is extracted, ensure it exists in CRM.

        Handles person and company entities.  Machine entities are
        intentionally skipped -- they are product catalog items stored
        in Neo4j only, not CRM records.
        """
        if event.source_store == SourceStore.CRM:
            return
        p = event.payload
        entity_type = p.get("entity_type", event.entity_type)

        if entity_type == "machine":
            return

        try:
            if entity_type == "person":
                email = _normalize_person_email(p.get("email", ""))
                if not email:
                    return
                _contact, created = await self._crm.get_or_create_contact_by_email(
                    email,
                    name=(p.get("name") or "").strip() or None,
                    source="neo4j_sync",
                )
                if created:
                    logger.debug("Created CRM contact from Neo4j entity: %s", email)

            elif entity_type == "company":
                name = p.get("name", "")
                if not name:
                    return
                companies = await self._crm.list_companies()
                if not any(c.name.lower() == name.lower() for c in companies):
                    await self._crm.create_company(
                        name=name,
                        region=p.get("region", ""),
                        industry=p.get("industry", ""),
                    )
                    logger.debug("Created CRM company from Neo4j entity: %s", name)

        except _CRM_GRAPH_BACK_SYNC_ERRORS:
            logger.exception("Neo4j→CRM sync failed for %s", event.entity_id)

    def _append_ledger_line(self, line: str) -> None:
        with open(self._ledger_path, "a", encoding="utf-8") as f:
            f.write(line)

    # ── Ledger queries ───────────────────────────────────────────────────

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Read the most recent events from the ledger."""
        if not self._ledger_path.exists():
            return []
        try:
            lines = self._ledger_path.read_text(encoding="utf-8").strip().split("\n")
            events = [json.loads(line) for line in lines[-limit:] if line.strip()]
            events.reverse()
            return events
        except _LEDGER_READ_ERRORS:
            logger.warning("Ledger read failed", exc_info=True)
            return []

    def event_count(self) -> int:
        if not self._ledger_path.exists():
            return 0
        try:
            with open(self._ledger_path, encoding="utf-8") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0
