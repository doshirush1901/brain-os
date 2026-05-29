"""Neo4j knowledge-graph manager for Ira.

Stores and queries structured entity relationships — companies, people,
machines, and quotes — as a property graph.  Every write uses ``MERGE`` to
guarantee idempotent upserts.  An LLM-powered extraction method can
populate the graph from unstructured text.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any, ClassVar

import httpx
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError

from brain_os.brain.knowledge_graph_extraction import (
    EntityExtractionConfig,
)
from brain_os.brain.knowledge_graph_extraction import (
    extract_entities_from_text as _extract_entities_from_text,
)
from brain_os.brain.knowledge_graph_text import (
    company_search_token,
    normalize_entity_name,
    normalize_source_id,
    sanitize_relationship_props,
    valid_graph_label,
    valid_prop_key,
)
from brain_os.config import Neo4jConfig, get_settings
from brain_os.exceptions import DatabaseError, IraError
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_GRAPH_STORE_ERRORS = (
    DatabaseError,
    Neo4jError,
    httpx.HTTPError,
    asyncio.TimeoutError,
    OSError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
)


class KnowledgeGraph:
    """Async wrapper around the Neo4j graph database."""

    def __init__(
        self,
        config: Neo4jConfig | None = None,
        event_bus: Any | None = None,
    ) -> None:
        cfg = config or get_settings().neo4j
        app_cfg = get_settings().app
        neo4j_user, neo4j_password = cfg.resolved_auth()
        self._driver = AsyncGraphDatabase.driver(
            cfg.uri,
            auth=(neo4j_user, neo4j_password),
            max_connection_pool_size=app_cfg.neo4j_max_pool_size,
            connection_acquisition_timeout=60.0,
        )
        self._cloud_driver: Any = None
        cloud_uri = cfg.cloud_uri.strip()
        cloud_auth = cfg.resolved_cloud_auth()
        if cloud_uri and cloud_auth:
            cu, cpw = cloud_auth
            self._cloud_driver = AsyncGraphDatabase.driver(
                cloud_uri,
                auth=(cu, cpw),
                max_connection_pool_size=app_cfg.neo4j_max_pool_size,
                connection_acquisition_timeout=60.0,
            )
            safe = cloud_uri.split("@")[-1] if "@" in cloud_uri else cloud_uri
            logger.info("Neo4j cloud mirror enabled (secondary Bolt: %s)", safe[:120])
        elif cloud_uri and not cloud_auth:
            logger.warning(
                "NEO4J_CLOUD_URI is set but cloud credentials are missing "
                "(set NEO4J_CLOUD_PASSWORD or NEO4J_CLOUD_AUTH); dual-write disabled"
            )
        self._llm = get_llm_client()
        self._entity_fallback_provider: str = app_cfg.digestive_llm_provider
        self._digestive_anthropic_model: str = (app_cfg.digestive_anthropic_model or "").strip()
        self._event_bus = event_bus

    def _entity_extraction_config(self) -> EntityExtractionConfig:
        return EntityExtractionConfig(
            entity_fallback_provider=self._entity_fallback_provider,
            digestive_anthropic_model=self._digestive_anthropic_model,
        )

    async def cloud_health_check(self) -> dict[str, Any] | None:
        """When Neo4j Aura/cloud mirror is configured, verify Bolt connectivity.

        Returns ``None`` when ``NEO4J_CLOUD_URI`` is not set or credentials are missing.
        """
        if self._cloud_driver is None:
            return None
        start = time.monotonic()
        try:
            async with self._cloud_driver.session() as session:
                result = await session.run("RETURN 1 AS ok")
                record = await result.single()
            latency = (time.monotonic() - start) * 1000
            if not record or record.get("ok") != 1:
                return {
                    "status": "unhealthy",
                    "latency_ms": round(latency, 1),
                    "error": "empty or unexpected result from cloud Neo4j",
                }
            return {"status": "healthy", "latency_ms": round(latency, 1), "error": None}
        except (TimeoutError, Neo4jError, OSError, httpx.HTTPError, ValueError, TypeError) as exc:
            latency = (time.monotonic() - start) * 1000
            logger.warning("Neo4j cloud health check failed: %s", exc)
            return {"status": "unhealthy", "latency_ms": round(latency, 1), "error": str(exc)}

    async def _mirror_execute_write(self, work: Any, *args: Any) -> None:
        if self._cloud_driver is None:
            return
        for attempt in range(2):
            try:
                async with self._cloud_driver.session() as session:
                    await session.execute_write(work, *args)
                return
            except (
                TimeoutError,
                Neo4jError,
                OSError,
                httpx.HTTPError,
                ValueError,
                TypeError,
            ) as exc:
                if attempt == 0:
                    logger.warning(
                        "Neo4j cloud mirror execute_write failed, retrying once", exc_info=True
                    )
                else:
                    logger.warning(
                        "Neo4j cloud mirror execute_write failed after retry", exc_info=True
                    )

    async def _dual_execute_write(self, work: Any, *args: Any) -> None:
        async with self._driver.session() as session:
            await session.execute_write(work, *args)
        await self._mirror_execute_write(work, *args)

    async def _mirror_run(self, query: str, **params: Any) -> None:
        if self._cloud_driver is None:
            return
        for attempt in range(2):
            try:
                async with self._cloud_driver.session() as session:
                    await session.run(query, **params)
                return
            except (
                TimeoutError,
                Neo4jError,
                OSError,
                httpx.HTTPError,
                ValueError,
                TypeError,
            ) as exc:
                if attempt == 0:
                    logger.warning("Neo4j cloud mirror run failed, retrying once", exc_info=True)
                else:
                    logger.warning("Neo4j cloud mirror run failed after retry", exc_info=True)

    async def _dual_run_write(self, query: str, **params: Any) -> None:
        async with self._driver.session() as session:
            await session.run(query, **params)
        await self._mirror_run(query, **params)

    def set_event_bus(self, event_bus: Any) -> None:
        self._event_bus = event_bus

    async def _emit(self, entity_type: str, entity_id: str, payload: dict[str, Any]) -> None:
        if self._event_bus is None:
            return
        from brain_os.systems.data_event_bus import DataEvent, EventType, SourceStore

        try:
            await self._event_bus.emit(
                DataEvent(
                    event_type=EventType.ENTITY_ADDED,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    payload={**payload, "entity_type": entity_type},
                    source_store=SourceStore.NEO4J,
                )
            )
        except (TimeoutError, IraError, OSError, TypeError, ValueError, AttributeError, KeyError):
            logger.debug("Neo4j event emission failed", exc_info=True)

    # ── schema / indexes ─────────────────────────────────────────────────

    async def ensure_indexes(self) -> None:
        """Create uniqueness constraints and indexes for core node types."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.company_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.email IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Machine) REQUIRE m.model IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (q:Quote) REQUIRE q.quote_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.qdrant_point_id IS UNIQUE",
        ]
        async with self._driver.session() as session:
            for stmt in constraints:
                await session.run(stmt)
        if self._cloud_driver is not None:
            async with self._cloud_driver.session() as cloud_session:
                for stmt in constraints:
                    try:
                        await cloud_session.run(stmt)
                    except (
                        TimeoutError,
                        Neo4jError,
                        OSError,
                        httpx.HTTPError,
                        ValueError,
                        TypeError,
                    ) as exc:
                        logger.warning(
                            "Neo4j cloud constraint/index ensure failed: %s",
                            stmt[:80],
                            exc_info=True,
                        )
        logger.info("Neo4j indexes and constraints ensured")

    # ── entity creation ──────────────────────────────────────────────────

    async def add_company(
        self,
        name: str,
        region: str = "",
        industry: str = "",
        website: str = "",
        company_id: str = "",
        source_id: str = "",
    ) -> None:
        name = normalize_entity_name(name)
        company_id = (
            company_id or ""
        ).strip() or f"company::{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}"
        source_id = normalize_source_id(source_id)
        await self._dual_execute_write(
            self._merge_company, name, region, industry, website, company_id, source_id
        )
        await self._emit(
            "company",
            company_id or name,
            {
                "name": name,
                "company_id": company_id,
                "region": region,
                "industry": industry,
                "website": website,
                "source_id": source_id,
            },
        )

    @staticmethod
    async def _merge_company(
        tx: Any,
        name: str,
        region: str,
        industry: str,
        website: str,
        company_id: str,
        source_id: str,
    ) -> None:
        """Upsert by stable ``company_id`` (not display name).

        MERGE-on-name created a second node when LLM/Firecrawl used a variant
        label (e.g. "Kiefel GmbH" vs "Kiefel") while both slug to the same
        ``company_id``, violating the unique ``company_id`` constraint.
        """
        now = datetime.now(UTC).isoformat()
        await tx.run(
            """
            OPTIONAL MATCH (legacy:Company {name: $name})
            WHERE legacy.company_id IS NULL
              AND NOT EXISTS { MATCH (:Company {company_id: $company_id}) }
            SET legacy.company_id = $company_id

            MERGE (c:Company {company_id: $company_id})
            ON CREATE SET
                c.name = $name,
                c.region = $region,
                c.industry = $industry,
                c.website = $website,
                c.first_seen_source = CASE WHEN $source_id = '' THEN null ELSE $source_id END,
                c.last_seen_source = CASE WHEN $source_id = '' THEN null ELSE $source_id END,
                c.updated_at = $now
            ON MATCH SET
                c.name = CASE WHEN c.name IS NULL OR c.name = '' THEN $name ELSE c.name END,
                c.region = CASE WHEN $region <> '' THEN $region ELSE c.region END,
                c.industry = CASE WHEN $industry <> '' THEN $industry ELSE c.industry END,
                c.website = CASE WHEN $website <> '' THEN $website ELSE c.website END,
                c.last_seen_source = CASE WHEN $source_id = '' THEN c.last_seen_source ELSE $source_id END,
                c.first_seen_source = CASE
                    WHEN $source_id = '' THEN c.first_seen_source
                    ELSE COALESCE(c.first_seen_source, $source_id)
                END,
                c.updated_at = $now
            """,
            name=name,
            region=region,
            industry=industry,
            website=website,
            company_id=company_id,
            source_id=source_id,
            now=now,
        )

    async def set_company_icp_profile(
        self,
        *,
        name: str = "",
        website: str = "",
        company_id: str = "",
        gauge_tier: str = "",
        gauge_confidence: float | None = None,
        is_thermoformer: bool | None = None,
        icp_category: str = "",
        source_id: str = "",
    ) -> bool:
        """Set gauge / ICP fields on :Company matched by company_id, name, or website host."""
        name_norm = normalize_entity_name(name) if name.strip() else ""
        cid = (company_id or "").strip()
        if not cid and name_norm:
            cid = f"company::{re.sub(r'[^a-z0-9]+', '_', name_norm.lower()).strip('_')}"
        web = (website or "").strip()
        tier = (gauge_tier or "").strip().lower()
        now = datetime.now(UTC).isoformat()
        src = normalize_source_id(source_id)
        icp_cat = (icp_category or "").strip().lower()
        updated = False
        try:
            async with self._driver.session() as session:
                updated = bool(
                    await session.execute_write(
                        self._set_company_icp_profile_tx,
                        name_norm,
                        web,
                        cid,
                        tier,
                        gauge_confidence,
                        is_thermoformer,
                        icp_cat,
                        src,
                        now,
                    )
                )
            await self._mirror_execute_write(
                self._set_company_icp_profile_tx,
                name_norm,
                web,
                cid,
                tier,
                gauge_confidence,
                is_thermoformer,
                icp_cat,
                src,
                now,
            )
        except _GRAPH_STORE_ERRORS:
            logger.warning("set_company_icp_profile failed for %s", name_norm or cid, exc_info=True)
            return False
        if updated:
            await self._emit(
                "company_icp",
                cid or name_norm or web,
                {
                    "name": name_norm,
                    "company_id": cid,
                    "gauge_tier": tier,
                    "gauge_confidence": gauge_confidence,
                    "is_thermoformer": is_thermoformer,
                    "icp_category": icp_category,
                },
            )
        return bool(updated)

    @staticmethod
    async def _set_company_icp_profile_tx(
        tx: Any,
        name: str,
        website: str,
        company_id: str,
        gauge_tier: str,
        gauge_confidence: float | None,
        is_thermoformer: bool | None,
        icp_category: str,
        source_id: str,
        now: str,
    ) -> bool:
        domain = ""
        if website:
            from urllib.parse import urlparse

            try:
                domain = (urlparse(website).netloc or "").lower()
            except ValueError:
                domain = ""
            if domain.startswith("www."):
                domain = domain[4:]

        result = await tx.run(
            """
            MATCH (c:Company)
            WHERE ($company_id <> '' AND c.company_id = $company_id)
               OR ($name <> '' AND c.name = $name)
               OR ($domain <> '' AND toLower(coalesce(c.website, '')) CONTAINS $domain)
            SET
              c.gauge_tier = CASE WHEN $gauge_tier <> '' THEN $gauge_tier ELSE c.gauge_tier END,
              c.gauge_confidence = CASE
                WHEN $gauge_confidence IS NULL THEN c.gauge_confidence
                ELSE $gauge_confidence
              END,
              c.is_thermoformer = CASE
                WHEN $is_thermoformer IS NULL THEN c.is_thermoformer
                ELSE $is_thermoformer
              END,
              c.icp_category = CASE WHEN $icp_category <> '' THEN $icp_category ELSE c.icp_category END,
              c.gauge_tier_updated_at = $now,
              c.last_seen_source = CASE
                WHEN $source_id = '' THEN c.last_seen_source
                ELSE $source_id
              END
            RETURN count(c) AS n
            """,
            name=name,
            website=website,
            company_id=company_id,
            domain=domain,
            gauge_tier=gauge_tier,
            gauge_confidence=gauge_confidence,
            is_thermoformer=is_thermoformer,
            icp_category=icp_category,
            source_id=source_id,
            now=now,
        )
        record = await result.single()
        return bool(record and int(record.get("n") or 0) > 0)

    async def get_company_icp_profile(
        self,
        *,
        name: str = "",
        website: str = "",
        company_id: str = "",
    ) -> dict[str, Any] | None:
        """Return ICP / gauge fields for a company if a node exists."""
        name_norm = normalize_entity_name(name) if name.strip() else ""
        cid = (company_id or "").strip()
        domain = ""
        web = (website or "").strip()
        if web:
            from urllib.parse import urlparse

            try:
                domain = (urlparse(web).netloc or "").lower()
            except ValueError:
                domain = ""
            if domain.startswith("www."):
                domain = domain[4:]
        rows = await self._read(
            """
            MATCH (c:Company)
            WHERE ($company_id <> '' AND c.company_id = $company_id)
               OR ($name <> '' AND c.name = $name)
               OR ($domain <> '' AND toLower(coalesce(c.website, '')) CONTAINS $domain)
            RETURN c.name AS name,
                   c.company_id AS company_id,
                   c.website AS website,
                   c.gauge_tier AS gauge_tier,
                   c.gauge_confidence AS gauge_confidence,
                   c.is_thermoformer AS is_thermoformer,
                   c.icp_category AS icp_category,
                   c.gauge_tier_updated_at AS gauge_tier_updated_at
            LIMIT 1
            """,
            name=name_norm,
            company_id=cid,
            domain=domain,
        )
        if not rows:
            return None
        row = rows[0]
        if not row.get("name") and not row.get("company_id"):
            return None
        return dict(row)

    async def add_person(
        self,
        name: str,
        email: str,
        company_name: str = "",
        role: str = "",
        source_id: str = "",
    ) -> None:
        company_name = normalize_entity_name(company_name) if company_name else ""
        source_id = normalize_source_id(source_id)
        await self._dual_execute_write(
            self._merge_person, name, email, company_name, role, source_id
        )
        await self._emit(
            "person",
            email,
            {
                "name": name,
                "email": email,
                "company": company_name,
                "role": role,
                "source_id": source_id,
            },
        )

    @staticmethod
    async def _merge_person(
        tx: Any, name: str, email: str, company_name: str, role: str, source_id: str
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await tx.run(
            """
            MERGE (p:Person {email: $email})
            SET p.name = $name, p.role = $role,
                p.last_seen_source = CASE WHEN $source_id = '' THEN p.last_seen_source ELSE $source_id END,
                p.first_seen_source = CASE
                    WHEN $source_id = '' THEN p.first_seen_source
                    ELSE COALESCE(p.first_seen_source, $source_id)
                END,
                p.updated_at = $now
            """,
            name=name,
            email=email,
            role=role,
            source_id=source_id,
            now=now,
        )
        if company_name:
            await tx.run(
                """
                MERGE (p:Person {email: $email})
                MERGE (c:Company {name: $company})
                MERGE (p)-[r:WORKS_AT]->(c)
                SET r.role = $role,
                    r.updated_at = $now,
                    r.strength = COALESCE(r.strength, 1)
                """,
                email=email,
                company=company_name,
                role=role,
                now=now,
            )

    async def add_machine(
        self,
        model: str,
        category: str = "",
        description: str = "",
        source_id: str = "",
    ) -> None:
        source_id = normalize_source_id(source_id)
        await self._dual_execute_write(self._merge_machine, model, category, description, source_id)
        await self._emit(
            "machine",
            model,
            {
                "model": model,
                "category": category,
                "description": description,
                "source_id": source_id,
            },
        )

    @staticmethod
    async def _merge_machine(
        tx: Any, model: str, category: str, description: str, source_id: str
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await tx.run(
            """
            MERGE (m:Machine {model: $model})
            SET m.category = $category, m.description = $description,
                m.last_seen_source = CASE WHEN $source_id = '' THEN m.last_seen_source ELSE $source_id END,
                m.first_seen_source = CASE
                    WHEN $source_id = '' THEN m.first_seen_source
                    ELSE COALESCE(m.first_seen_source, $source_id)
                END,
                m.updated_at = $now
            """,
            model=model,
            category=category,
            description=description,
            source_id=source_id,
            now=now,
        )

    async def add_project(
        self,
        project_id: str,
        customer: str = "",
        machine_model: str = "",
        status: str = "",
    ) -> None:
        await self._dual_run_write(
            "MERGE (p:Project {project_id: $pid}) "
            "SET p.customer = $customer, p.machine_model = $machine_model, p.status = $status",
            pid=project_id,
            customer=customer,
            machine_model=machine_model,
            status=status,
        )
        await self._emit(
            "project",
            project_id,
            {
                "project_id": project_id,
                "customer": customer,
                "machine_model": machine_model,
                "status": status,
            },
        )

    async def add_application(self, name: str, description: str = "") -> None:
        await self._dual_run_write(
            "MERGE (a:Application {name: $name}) SET a.description = $description",
            name=name,
            description=description,
        )

    async def add_material(self, name: str, category: str = "") -> None:
        await self._dual_run_write(
            "MERGE (m:Material {name: $name}) SET m.category = $category",
            name=name,
            category=category,
        )

    async def add_exhibition(self, name: str, location: str = "", year: str = "") -> None:
        await self._dual_run_write(
            "MERGE (e:Exhibition {name: $name}) SET e.location = $location, e.year = $year",
            name=name,
            location=location,
            year=year,
        )

    async def add_quote(
        self,
        quote_id: str,
        company_name: str,
        machine_model: str,
        value: float,
        date: str,
        status: str = "OPEN",
        source_id: str = "",
    ) -> None:
        company_name = normalize_entity_name(company_name)
        source_id = normalize_source_id(source_id)
        await self._dual_execute_write(
            self._merge_quote, quote_id, company_name, machine_model, value, date, status, source_id
        )

    @staticmethod
    async def _merge_quote(
        tx: Any,
        quote_id: str,
        company_name: str,
        machine_model: str,
        value: float,
        date: str,
        status: str,
        source_id: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await tx.run(
            """
            MERGE (q:Quote {quote_id: $qid})
            SET q.value = $value, q.date = $date, q.status = $status,
                q.last_seen_source = CASE WHEN $source_id = '' THEN q.last_seen_source ELSE $source_id END,
                q.first_seen_source = CASE
                    WHEN $source_id = '' THEN q.first_seen_source
                    ELSE COALESCE(q.first_seen_source, $source_id)
                END,
                q.updated_at = $now
            WITH q
            MERGE (c:Company {name: $company})
            MERGE (q)-[qc:QUOTED_TO]->(c)
            SET qc.updated_at = $now, qc.strength = COALESCE(qc.strength, 1)
            WITH q
            MERGE (m:Machine {model: $machine})
            MERGE (q)-[qm:QUOTES_MACHINE]->(m)
            SET qm.updated_at = $now, qm.strength = COALESCE(qm.strength, 1)
            """,
            qid=quote_id,
            company=company_name,
            machine=machine_model,
            value=value,
            date=date,
            status=status,
            source_id=source_id,
            now=now,
        )

    # ── generic relationship creation ────────────────────────────────────

    # SECURITY: _KEY_FIELDS acts as the allowlist for node labels.
    # Only these four labels may be used in dynamic Cypher queries.
    # from_type / to_type are validated against this dict before
    # interpolation — never add entries without reviewing the Cypher
    # injection implications.
    _KEY_FIELDS: ClassVar[dict[str, str]] = {
        "Company": "name",
        "Person": "email",
        "Machine": "model",
        "Quote": "quote_id",
        "Project": "project_id",
        "Application": "name",
        "Material": "name",
        "Exhibition": "name",
        "OperatorContextRun": "run_id",
        "PipelineRun": "run_id",
        "ReasoningEvidence": "evidence_id",
    }

    _ALLOWED_REL_TYPES = frozenset(
        {
            "WORKS_AT",
            "INTERESTED_IN",
            "QUOTED_FOR",
            "SUPPLIES",
            "MANUFACTURES",
            "COMPETES_WITH",
            "CONTACTED_BY",
            "REFERRED_BY",
            "QUOTED_TO",
            "QUOTES_MACHINE",
            "CO_RELEVANT",
            "DESCRIBES",
            "FROM_SOURCE",
            "REFERS_TO",
            "IN_CLUSTER",
            "CUSTOMER_OF",
            "SUPPLIES_PARTS_TO",
            "DISTRIBUTES_FOR",
            "EXHIBITED_AT",
            "USES_MATERIAL",
            "FOR_APPLICATION",
            "PART_OF_PROJECT",
            "PROJECT_FOR",
            "HAS_CONTEXT_RUN",
            "HAS_PIPELINE_RUN",
            "FROM_PIPELINE",
            "HAS_EVIDENCE",
            "CONTEXT_FOR_PIPELINE",
        }
    )

    async def merge_operator_context_run(
        self,
        *,
        run_id: str,
        kind: str,
        outcome: str,
        ts: float,
        company_name: str,
        summary: str = "",
        domain: str | None = None,
        machine_model: str | None = None,
        crm_stage: str | None = None,
        pipeline_run_id: str | None = None,
        success: bool | None = None,
        source_id: str = "",
    ) -> bool:
        """Upsert an operator context run and link it to the account company."""
        rid = (run_id or "").strip()[:128]
        company = normalize_entity_name((company_name or "").strip())
        if not rid or not company:
            return False
        props = sanitize_relationship_props(
            {
                "kind": (kind or "")[:32],
                "outcome": (outcome or "")[:32],
                "ts": float(ts),
                "summary": (summary or "")[:400],
                "domain": (domain or "")[:128] or None,
                "machine_model": (machine_model or "")[:64] or None,
                "crm_stage": (crm_stage or "")[:64] or None,
                "success": success,
                "source_id": normalize_source_id(source_id) or None,
            }
        )
        try:
            await self._run_cypher_write(
                """
                MERGE (c:Company {name: $company})
                MERGE (r:OperatorContextRun {run_id: $run_id})
                SET r += $props
                MERGE (c)-[hr:HAS_CONTEXT_RUN]->(r)
                SET hr.ts = $ts, hr.kind = $kind
                """,
                params={
                    "company": company,
                    "run_id": rid,
                    "props": props,
                    "ts": float(ts),
                    "kind": (kind or "")[:32],
                },
            )
            prid = (pipeline_run_id or "").strip()[:128]
            if prid:
                await self.add_relationship(
                    "OperatorContextRun",
                    rid,
                    "FROM_PIPELINE",
                    "PipelineRun",
                    prid,
                    properties={"ts": float(ts)},
                    source_id=source_id,
                )
            return True
        except _GRAPH_STORE_ERRORS:
            logger.debug("merge_operator_context_run failed run_id=%s", rid, exc_info=True)
            return False

    async def merge_pipeline_run(
        self,
        *,
        run_id: str,
        outcome: str,
        channel: str,
        ts_end: float,
        route_method: str = "",
        agents_count: int = 0,
        input_summary: str = "",
        response_summary: str = "",
        company_name: str | None = None,
        evidence: list[dict[str, Any]] | None = None,
        source_id: str = "",
    ) -> bool:
        """Upsert a pipeline run, optional company link, and reasoning evidence nodes."""
        rid = (run_id or "").strip()[:128]
        if not rid:
            return False
        props = sanitize_relationship_props(
            {
                "outcome": (outcome or "")[:32],
                "channel": (channel or "")[:64],
                "ts_end": float(ts_end),
                "route_method": (route_method or "")[:64],
                "agents_count": int(agents_count),
                "input_summary": (input_summary or "")[:500],
                "response_summary": (response_summary or "")[:500],
                "source_id": normalize_source_id(source_id) or None,
            }
        )
        try:
            await self._run_cypher_write(
                """
                MERGE (p:PipelineRun {run_id: $run_id})
                SET p += $props
                """,
                params={"run_id": rid, "props": props},
            )
            company = normalize_entity_name((company_name or "").strip()) if company_name else ""
            if company:
                await self.add_relationship(
                    "Company",
                    company,
                    "HAS_PIPELINE_RUN",
                    "PipelineRun",
                    rid,
                    properties={"ts_end": float(ts_end), "outcome": (outcome or "")[:32]},
                    source_id=source_id,
                )
            for row in evidence or []:
                eid = str(row.get("evidence_id") or "").strip()[:180]
                ref = str(row.get("ref") or "").strip()
                if not eid or not ref:
                    continue
                ev_props = sanitize_relationship_props(
                    {
                        "kind": str(row.get("kind") or "")[:32],
                        "ref": ref[:512],
                        "score": row.get("score"),
                        "ordinal": row.get("ordinal"),
                        "source_id": normalize_source_id(source_id) or None,
                    }
                )
                await self._run_cypher_write(
                    """
                    MATCH (p:PipelineRun {run_id: $run_id})
                    MERGE (e:ReasoningEvidence {evidence_id: $evidence_id})
                    SET e += $ev_props
                    MERGE (p)-[he:HAS_EVIDENCE]->(e)
                    SET he.kind = $kind
                    """,
                    params={
                        "run_id": rid,
                        "evidence_id": eid,
                        "ev_props": ev_props,
                        "kind": str(row.get("kind") or "")[:32],
                    },
                )
            return True
        except _GRAPH_STORE_ERRORS:
            logger.debug("merge_pipeline_run failed run_id=%s", rid, exc_info=True)
            return False

    async def link_operator_context_to_pipeline(
        self,
        operator_context_run_id: str,
        pipeline_run_id: str,
        *,
        source_id: str = "",
    ) -> bool:
        """Link a prior operator context run to the pipeline run that produced it."""
        return await self.add_relationship(
            "OperatorContextRun",
            operator_context_run_id.strip()[:128],
            "CONTEXT_FOR_PIPELINE",
            "PipelineRun",
            pipeline_run_id.strip()[:128],
            properties={"linked": True},
            source_id=source_id,
        )

    async def add_relationship(
        self,
        from_type: str,
        from_key: str,
        rel_type: str,
        to_type: str,
        to_key: str,
        properties: dict[str, Any] | None = None,
        source_id: str = "",
    ) -> bool:
        """Create a relationship between two nodes, merging idempotently.

        ``from_type`` / ``to_type`` must be labels in ``_KEY_FIELDS``.
        ``rel_type`` is validated against ``_ALLOWED_REL_TYPES``.
        Returns True if the relationship was written, False if skipped.
        """
        if not from_key or not to_key:
            return False
        if from_type == "Company":
            from_key = normalize_entity_name(from_key)
        if to_type == "Company":
            to_key = normalize_entity_name(to_key)
        if rel_type not in self._ALLOWED_REL_TYPES:
            logger.warning("Ignoring unknown relationship type: %s", rel_type)
            return False

        # Strict allowlist: only known node types to prevent Cypher injection
        if from_type not in self._KEY_FIELDS:
            logger.warning(
                "Ignoring unknown from_type node label (not in allowlist): %s", from_type
            )
            return False
        if to_type not in self._KEY_FIELDS:
            logger.warning("Ignoring unknown to_type node label (not in allowlist): %s", to_type)
            return False
        from_field = self._KEY_FIELDS[from_type]
        to_field = self._KEY_FIELDS[to_type]

        if not valid_graph_label(from_type) or not valid_graph_label(to_type):
            raise ValueError(f"Invalid node label format: {from_type!r}, {to_type!r}")
        if not valid_graph_label(rel_type):
            raise ValueError(f"Invalid relationship type format: {rel_type!r}")

        props = sanitize_relationship_props(properties or {})
        source_id = normalize_source_id(source_id)
        if source_id:
            props["source_id"] = source_id
        for k in props:
            if not valid_prop_key(k):
                raise ValueError(
                    f"Invalid relationship property key (unsafe for Cypher SET): {k!r}"
                )
        set_clause = "SET r += $props" if props else ""

        query = (
            f"MERGE (a:{from_type} {{{from_field}: $from_key}}) "
            f"MERGE (b:{to_type} {{{to_field}: $to_key}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            f"{set_clause}"
        )
        params: dict[str, Any] = {"from_key": from_key, "to_key": to_key, "props": props}

        try:
            async with self._driver.session() as session:
                await session.run(query, params)
            await self._mirror_run(query, **params)
            logger.debug(
                "Relationship created: (%s:%s)-[%s]->(%s:%s)",
                from_type,
                from_key,
                rel_type,
                to_type,
                to_key,
            )
            return True
        except _GRAPH_STORE_ERRORS:
            logger.exception(
                "Failed to create relationship (%s:%s)-[%s]->(%s:%s)",
                from_type,
                from_key,
                rel_type,
                to_type,
                to_key,
            )
            return False

    # ── Chunk ↔ entity linking (Qdrant ↔ Neo4j) ───────────────────────────

    async def add_chunk_and_describes(
        self,
        qdrant_point_id: str,
        source: str,
        source_category: str,
        content_preview: str,
        entity_refs: list[tuple[str, str]],
        source_id: str = "",
    ) -> int:
        """Create a Chunk node linked to Qdrant and DESCRIBES edges to entities.

        *entity_refs* is a list of (node_label, key_value), e.g.
        [("Company", "Acme Corp"), ("Person", "john@acme.com"), ("Machine", "DEMO-A-1234")].
        Only Company, Person, Machine, Quote are allowed. Returns the number of
        DESCRIBES edges created.
        """
        if not qdrant_point_id or not qdrant_point_id.strip():
            return 0
        preview = (content_preview or "")[:500]
        created = 0
        source_id = normalize_source_id(source_id)
        await self._dual_execute_write(
            self._merge_chunk, qdrant_point_id, source, source_category, preview, source_id
        )
        for label, key in entity_refs:
            if not key or label not in self._KEY_FIELDS:
                continue
            try:
                async with self._driver.session() as session:
                    n = await session.execute_write(
                        self._merge_describes_tx, qdrant_point_id, label, key
                    )
                created += n
                await self._mirror_execute_write(
                    self._merge_describes_tx, qdrant_point_id, label, key
                )
            except _GRAPH_STORE_ERRORS:
                logger.debug("DESCRIBES merge failed for %s:%s", label, key, exc_info=True)
        return created

    @staticmethod
    async def _merge_chunk(
        tx: Any, point_id: str, source: str, source_category: str, preview: str, source_id: str
    ) -> None:
        await tx.run(
            """
            MERGE (ch:Chunk {qdrant_point_id: $point_id})
            SET ch.source = $source, ch.source_category = $source_category,
                ch.content_preview = $preview,
                ch.source_id = CASE WHEN $source_id = '' THEN ch.source_id ELSE $source_id END
            """,
            point_id=point_id,
            source=source or "",
            source_category=source_category or "",
            preview=preview,
            source_id=source_id,
        )

    @staticmethod
    async def _merge_describes_tx(tx: Any, point_id: str, label: str, key: str) -> int:
        field = KnowledgeGraph._KEY_FIELDS.get(label)
        if not field:
            return 0
        result = await tx.run(
            f"""
            MATCH (ch:Chunk {{qdrant_point_id: $point_id}})
            MERGE (n:{label} {{{field}: $key}})
            MERGE (ch)-[:DESCRIBES]->(n)
            RETURN count(*) AS c
            """,
            point_id=point_id,
            key=key,
        )
        record = await result.single()
        return int(record["c"]) if record else 0

    async def get_chunk_point_ids_for_entity(
        self, entity_label: str, entity_key: str, limit: int = 20
    ) -> list[str]:
        """Return Qdrant point IDs of Chunk nodes that DESCRIBE the given entity.

        Used by retrieval to fetch vector chunks for graph entities (denser stitch).
        """
        if entity_label not in self._KEY_FIELDS or not entity_key:
            return []
        field = self._KEY_FIELDS[entity_label]
        records = await self._read(
            f"""
            MATCH (ch:Chunk)-[:DESCRIBES]->(n:{entity_label} {{{field}: $key}})
            RETURN ch.qdrant_point_id AS point_id
            LIMIT {min(limit, 100)}
            """,
            key=entity_key,
        )
        out = [r["point_id"] for r in records if r.get("point_id")]
        return out

    # ── relationship helpers ─────────────────────────────────────────────

    async def link_person_to_company(
        self, person_email: str, company_name: str, role: str = ""
    ) -> None:
        await self._dual_execute_write(self._link_person_company, person_email, company_name, role)

    @staticmethod
    async def _link_person_company(tx: Any, email: str, company: str, role: str) -> None:
        await tx.run(
            """
            MERGE (p:Person {email: $email})
            MERGE (c:Company {name: $company})
            MERGE (p)-[r:WORKS_AT]->(c)
            SET r.role = $role
            """,
            email=email,
            company=company,
            role=role,
        )

    async def link_quote_to_company(self, quote_id: str, company_name: str) -> None:
        await self._dual_execute_write(self._link_quote_company, quote_id, company_name)

    @staticmethod
    async def _link_quote_company(tx: Any, quote_id: str, company: str) -> None:
        await tx.run(
            """
            MERGE (q:Quote {quote_id: $qid})
            MERGE (c:Company {name: $company})
            MERGE (q)-[:QUOTED_TO]->(c)
            """,
            qid=quote_id,
            company=company,
        )

    async def link_quote_to_machine(self, quote_id: str, machine_model: str) -> None:
        await self._dual_execute_write(self._link_quote_machine, quote_id, machine_model)

    @staticmethod
    async def _link_quote_machine(tx: Any, quote_id: str, model: str) -> None:
        await tx.run(
            """
            MERGE (q:Quote {quote_id: $qid})
            MERGE (m:Machine {model: $model})
            MERGE (q)-[:QUOTES_MACHINE]->(m)
            """,
            qid=quote_id,
            model=model,
        )

    # ── queries ──────────────────────────────────────────────────────────

    async def find_company_contacts(self, company_name: str) -> list[dict[str, Any]]:
        """Return all Person nodes linked to a company."""
        return await self._read(
            """
            MATCH (p:Person)-[:WORKS_AT]->(c:Company {name: $name})
            RETURN p.name AS name, p.email AS email, p.role AS role
            """,
            name=company_name,
        )

    async def find_company_quotes(self, company_name: str) -> list[dict[str, Any]]:
        """Return all Quote nodes linked to a company."""
        return await self._read(
            """
            MATCH (q:Quote)-[:QUOTED_TO]->(c:Company {name: $name})
            OPTIONAL MATCH (q)-[:QUOTES_MACHINE]->(m:Machine)
            RETURN q.quote_id AS quote_id, q.value AS value, q.date AS date,
                   q.status AS status, m.model AS machine
            """,
            name=company_name,
        )

    async def find_company_name_variants(
        self,
        company_name: str,
        *,
        limit: int = 8,
    ) -> list[str]:
        """Related :Company names sharing a search token (alias rollup for context graph)."""
        name = normalize_entity_name((company_name or "").strip())
        if not name:
            return []
        lim = max(1, min(int(limit), 15))
        token = company_search_token(name)
        if len(token) < 4:
            return [name] if await self.company_node_exists(name) else []
        rows = await self._read(
            """
            MATCH (c:Company)
            WHERE toLower(replace(c.name, ' ', '')) CONTAINS $token
            RETURN DISTINCT c.name AS name
            ORDER BY name
            LIMIT $lim
            """,
            token=token,
            lim=lim,
        )
        names = [str(r["name"]) for r in rows if r.get("name")]
        if name not in names and await self.company_node_exists(name):
            names.insert(0, name)
        return names

    async def find_company_quote_documents(
        self,
        company_names: list[str],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Quote/proposal PDF paths linked via Chunk-[:DESCRIBES]->Company (KB ingest layer)."""
        names = [
            normalize_entity_name(n)
            for n in company_names
            if normalize_entity_name((n or "").strip())
        ]
        names = list(dict.fromkeys(names))
        if not names:
            return []
        lim = max(1, min(int(limit), 25))
        return await self._read(
            """
            MATCH (ch:Chunk)-[:DESCRIBES]->(c:Company)
            WHERE c.name IN $names AND ch.source IS NOT NULL
              AND (
                toLower(ch.source) CONTAINS 'quotes_and_proposals'
                OR toLower(ch.source) CONTAINS '/quote'
                OR toLower(ch.source) CONTAINS 'proposal'
              )
            RETURN DISTINCT c.name AS company, ch.source AS source
            ORDER BY source DESC
            LIMIT $lim
            """,
            names=names,
            lim=lim,
        )

    async def company_node_exists(self, company_name: str) -> bool:
        """True when a :Company node exists for the normalized name."""
        name = normalize_entity_name((company_name or "").strip())
        if not name:
            return False
        rows = await self._read(
            "MATCH (c:Company {name: $name}) RETURN c.name AS name LIMIT 1",
            name=name,
        )
        return bool(rows)

    async def expand_company_one_hop(
        self,
        company_name: str,
        *,
        context_run_limit: int = 5,
        pipeline_run_limit: int = 3,
        contact_limit: int = 5,
        quote_limit: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Structured 1-hop neighborhood for account brief / retrieval (P2)."""
        name = normalize_entity_name((company_name or "").strip())
        empty: dict[str, list[dict[str, Any]]] = {
            "operator_runs": [],
            "pipeline_runs": [],
            "contacts": [],
            "quotes": [],
            "quote_documents": [],
            "related_companies": [],
        }
        if not name:
            return empty
        ctx_lim = max(1, min(int(context_run_limit), 20))
        pipe_lim = max(1, min(int(pipeline_run_limit), 10))
        contact_lim = max(1, min(int(contact_limit), 20))
        quote_lim = max(1, min(int(quote_limit), 20))
        try:
            variants = await self.find_company_name_variants(name, limit=8)
            operator_runs, pipeline_runs, contacts, quotes, quote_documents = await asyncio.gather(
                self._read(
                    """
                    MATCH (c:Company {name: $name})-[:HAS_CONTEXT_RUN]->(r:OperatorContextRun)
                    RETURN r.run_id AS run_id, r.kind AS kind, r.outcome AS outcome,
                           r.summary AS summary, r.ts AS ts, r.success AS success,
                           r.crm_stage AS crm_stage, r.machine_model AS machine_model
                    ORDER BY r.ts DESC
                    LIMIT $lim
                    """,
                    name=name,
                    lim=ctx_lim,
                ),
                self._read(
                    """
                    MATCH (c:Company {name: $name})-[:HAS_PIPELINE_RUN]->(p:PipelineRun)
                    OPTIONAL MATCH (p)-[:HAS_EVIDENCE]->(e:ReasoningEvidence)
                    WITH p, count(e) AS evidence_count
                    RETURN p.run_id AS run_id, p.outcome AS outcome, p.channel AS channel,
                           p.ts_end AS ts_end, p.route_method AS route_method,
                           p.agents_count AS agents_count, evidence_count
                    ORDER BY p.ts_end DESC
                    LIMIT $lim
                    """,
                    name=name,
                    lim=pipe_lim,
                ),
                self._read(
                    """
                    MATCH (p:Person)-[:WORKS_AT]->(c:Company {name: $name})
                    RETURN p.name AS name, p.email AS email, p.role AS role
                    LIMIT $lim
                    """,
                    name=name,
                    lim=contact_lim,
                ),
                self._read(
                    """
                    MATCH (q:Quote)-[:QUOTED_TO]->(c:Company)
                    WHERE c.name IN $names
                    OPTIONAL MATCH (q)-[:QUOTES_MACHINE]->(m:Machine)
                    RETURN q.quote_id AS quote_id, q.value AS value, q.date AS date,
                           q.status AS status, m.model AS machine, c.name AS company
                    ORDER BY q.date DESC
                    LIMIT $lim
                    """,
                    names=variants or [name],
                    lim=quote_lim,
                ),
                self.find_company_quote_documents(
                    variants or [name],
                    limit=quote_lim,
                ),
            )
        except _GRAPH_STORE_ERRORS:
            logger.debug("expand_company_one_hop failed name=%s", name[:80], exc_info=True)
            return empty
        return {
            "operator_runs": operator_runs,
            "pipeline_runs": pipeline_runs,
            "contacts": contacts,
            "quotes": quotes,
            "quote_documents": quote_documents,
            "related_companies": variants,
        }

    async def set_company_embedding(
        self,
        company_name: str,
        embedding: list[float],
        *,
        text_hash: str = "",
        source_id: str = "",
    ) -> bool:
        """Store Voyage embedding vector on :Company (P3 similar accounts)."""
        name = normalize_entity_name((company_name or "").strip())
        if not name or not embedding:
            return False
        props = sanitize_relationship_props(
            {
                "embedding": [float(x) for x in embedding],
                "embedding_text_hash": (text_hash or "")[:64],
                "embedding_updated_at": datetime.now(UTC).isoformat(),
                "embedding_source": normalize_source_id(source_id) or "voyage",
            }
        )
        try:
            await self._run_cypher_write(
                """
                MATCH (c:Company {name: $name})
                SET c += $props
                """,
                params={"name": name, "props": props},
            )
            return True
        except _GRAPH_STORE_ERRORS:
            logger.debug("set_company_embedding failed name=%s", name[:80], exc_info=True)
            return False

    async def get_company_embedding(self, company_name: str) -> list[float] | None:
        """Return stored embedding list for a company, if any."""
        name = normalize_entity_name((company_name or "").strip())
        if not name:
            return None
        rows = await self._read(
            """
            MATCH (c:Company {name: $name})
            RETURN c.embedding AS embedding
            LIMIT 1
            """,
            name=name,
        )
        if not rows:
            return None
        raw = rows[0].get("embedding")
        if not isinstance(raw, list):
            return None
        try:
            return [float(x) for x in raw]
        except (TypeError, ValueError):
            return None

    async def get_company_embedding_meta(self, company_name: str) -> dict[str, Any] | None:
        name = normalize_entity_name((company_name or "").strip())
        if not name:
            return None
        rows = await self._read(
            """
            MATCH (c:Company {name: $name})
            RETURN c.embedding_text_hash AS embedding_text_hash,
                   c.embedding_updated_at AS embedding_updated_at
            LIMIT 1
            """,
            name=name,
        )
        return rows[0] if rows else None

    async def list_companies_with_embeddings(
        self,
        *,
        exclude_name: str = "",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Companies with ``embedding`` set (for cosine peer search)."""
        ex = normalize_entity_name((exclude_name or "").strip())
        lim = max(1, min(int(limit), 5000))
        if ex:
            return await self._read(
                """
                MATCH (c:Company)
                WHERE c.embedding IS NOT NULL AND c.name <> $exclude
                RETURN c.name AS name, c.embedding AS embedding,
                       c.region AS region, c.industry AS industry
                LIMIT $lim
                """,
                exclude=ex,
                lim=lim,
            )
        return await self._read(
            """
            MATCH (c:Company)
            WHERE c.embedding IS NOT NULL
            RETURN c.name AS name, c.embedding AS embedding,
                   c.region AS region, c.industry AS industry
            LIMIT $lim
            """,
            lim=lim,
        )

    async def list_company_names_missing_embedding(self, *, limit: int = 200) -> list[str]:
        lim = max(1, min(int(limit), 5000))
        rows = await self._read(
            """
            MATCH (c:Company)
            WHERE c.name IS NOT NULL AND trim(toString(c.name)) <> ''
              AND c.embedding IS NULL
            RETURN c.name AS name
            ORDER BY c.updated_at DESC
            LIMIT $lim
            """,
            lim=lim,
        )
        return [str(r.get("name") or "") for r in rows if r.get("name")]

    async def find_machine_customers(self, machine_model: str) -> list[dict[str, Any]]:
        """Return companies that have received quotes for a machine model."""
        return await self._read(
            """
            MATCH (q:Quote)-[:QUOTES_MACHINE]->(m:Machine {model: $model})
            MATCH (q)-[:QUOTED_TO]->(c:Company)
            RETURN DISTINCT c.name AS company, c.region AS region,
                   c.industry AS industry
            """,
            model=machine_model,
        )

    async def find_related_entities(
        self,
        entity_name: str,
        max_hops: int = 2,
    ) -> dict[str, Any]:
        """Return a subgraph of nodes within *max_hops* of the named entity.

        Anchors the walk on ``name``, ``email``, or ``model`` only. (Neo4j emits
        ``UnknownPropertyKeyWarning`` if we OR on ``quote_id`` before any
        ``Quote`` node has that property in the catalog; use
        :meth:`find_company_quotes` for quote-id lookups.)

        Tries APOC ``subgraphAll`` first for efficiency; falls back to a
        standard variable-length MATCH if APOC is not installed.
        """
        try:
            records = await self._read(
                f"""
                MATCH (start)
                WHERE start.name = $name OR start.email = $name OR start.model = $name
                CALL apoc.path.subgraphAll(start, {{maxLevel: {max_hops}}})
                YIELD nodes, relationships
                RETURN nodes, relationships
                """,
                name=entity_name,
            )
            if not records:
                return {"nodes": [], "relationships": []}
            row = records[0]
            return {
                "nodes": row.get("nodes", []),
                "relationships": row.get("relationships", []),
            }
        except _GRAPH_STORE_ERRORS:
            logger.debug("APOC not available, falling back to MATCH path query")

        records = await self._read(
            f"""
            MATCH (start)
            WHERE start.name = $name OR start.email = $name OR start.model = $name
            OPTIONAL MATCH path = (start)-[r*1..{max_hops}]-(related)
            WITH start,
                 collect(DISTINCT related) AS related_nodes,
                 [rel IN collect(DISTINCT r) | head(rel)] AS flat_rels
            UNWIND (CASE WHEN size(flat_rels) = 0 THEN [null] ELSE flat_rels END) AS rel
            WITH start, related_nodes,
                 collect(CASE WHEN rel IS NOT NULL THEN
                   {{type: type(rel), from: startNode(rel).name, to: endNode(rel).name}}
                 END) AS rels
            RETURN [start] + related_nodes AS nodes,
                   [r IN rels WHERE r IS NOT NULL] AS relationships
            """,
            name=entity_name,
        )
        if not records:
            return {"nodes": [], "relationships": []}
        return {
            "nodes": records[0].get("nodes", []),
            "relationships": records[0].get("relationships", []),
        }

    _WRITE_KEYWORDS = frozenset({"CREATE", "DELETE", "DETACH", "SET", "REMOVE", "DROP"})
    _ALLOWED_READ_PREFIXES = frozenset({"MATCH", "OPTIONAL", "WITH", "RETURN", "CALL", "UNWIND"})

    async def run_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a **read-only** Cypher query and return the result rows.

        Only read-style operations are allowed (MATCH, OPTIONAL MATCH, WITH,
        RETURN, etc.).  Write operations (CREATE, DELETE, SET, etc.) are
        rejected.  For internal write operations use :meth:`_run_cypher_write`.
        """
        stripped = query.strip().upper()
        first_word = stripped.split()[0] if stripped else ""
        if first_word and first_word not in self._ALLOWED_READ_PREFIXES:
            raise ValueError(
                "Write operations not allowed via run_cypher: "
                f"query must start with one of {sorted(self._ALLOWED_READ_PREFIXES)}; got {first_word!r}"
            )
        tokens = set(stripped.split())
        if tokens & self._WRITE_KEYWORDS:
            raise ValueError(
                f"Write operations not allowed via run_cypher: "
                f"{sorted(tokens & self._WRITE_KEYWORDS)}"
            )
        if any(c in query for c in (";", "//", "/*")):
            raise ValueError("Query contains disallowed characters")
        return await self._read(query, **(params or {}))

    async def _run_cypher_write(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher write query.  Internal use only."""
        p = params or {}
        async with self._driver.session() as session:
            result = await session.run(query, **p)
            rows = [dict(record) async for record in result]
        if self._cloud_driver is not None:
            for attempt in range(2):
                try:
                    async with self._cloud_driver.session() as csession:
                        cresult = await csession.run(query, **p)
                        async for _ in cresult:
                            pass
                    break
                except (
                    TimeoutError,
                    Neo4jError,
                    OSError,
                    httpx.HTTPError,
                    ValueError,
                    TypeError,
                ) as exc:
                    if attempt == 0:
                        logger.warning(
                            "Neo4j cloud mirror _run_cypher_write failed, retrying once",
                            exc_info=True,
                        )
                    else:
                        logger.warning(
                            "Neo4j cloud mirror _run_cypher_write failed after retry",
                            exc_info=True,
                        )
        return rows

    # ── LLM entity extraction ────────────────────────────────────────────

    async def extract_entities_from_text(self, text: str) -> dict[str, Any]:
        """Extract entities from free text (GraphRAG or legacy LLM). See ``knowledge_graph_extraction``."""
        return await _extract_entities_from_text(
            llm=self._llm,
            config=self._entity_extraction_config(),
            text=text,
        )

    # ── bulk enrichment ────────────────────────────────────────────────

    async def enrich_interested_in_from_quotes(self) -> int:
        """Infer Company-[INTERESTED_IN]->Machine from existing quote chains."""
        result = await self._read(
            """
            MATCH (q:Quote)-[:QUOTED_TO]->(c:Company),
                  (q)-[:QUOTES_MACHINE]->(m:Machine)
            WHERE NOT (c)-[:INTERESTED_IN]->(m)
            MERGE (c)-[:INTERESTED_IN]->(m)
            RETURN count(*) AS created
            """
        )
        created = result[0].get("created", 0) if result else 0
        logger.info("Enrichment: created %d INTERESTED_IN from quotes", created)
        return created

    async def enrich_manufactures(self) -> int:
        """Link Acme Corp -[MANUFACTURES]-> all Machine nodes."""
        result = await self._read(
            """
            MERGE (mc:Company {name: 'Acme Corp'})
            WITH mc
            MATCH (m:Machine) WHERE NOT (mc)-[:MANUFACTURES]->(m)
            MERGE (mc)-[:MANUFACTURES]->(m)
            RETURN count(*) AS created
            """
        )
        created = result[0].get("created", 0) if result else 0
        logger.info("Enrichment: created %d MANUFACTURES relationships", created)
        return created

    async def cleanup_labelless_orphans(self) -> int:
        """Delete nodes that have no labels and no relationships."""
        result = await self._read(
            "MATCH (n) WHERE size(labels(n)) = 0 AND NOT (n)--() "
            "DELETE n RETURN count(n) AS deleted"
        )
        deleted = result[0].get("deleted", 0) if result else 0
        logger.info("Cleanup: deleted %d label-less orphan nodes", deleted)
        return deleted

    async def graph_stats(self) -> dict[str, Any]:
        """Return summary statistics about the graph."""
        node_rows = await self._read("MATCH (n) RETURN count(n) AS nodes")
        nodes = node_rows[0]["nodes"] if node_rows else 0
        rel_rows = await self._read("MATCH ()-[r]->() RETURN count(r) AS rels")
        rels = rel_rows[0]["rels"] if rel_rows else 0
        orphan_rows = await self._read("MATCH (n) WHERE NOT (n)--() RETURN count(n) AS orphans")
        orphans = orphan_rows[0]["orphans"] if orphan_rows else 0

        label_rows = await self._read(
            "CALL db.labels() YIELD label "
            "CALL { WITH label MATCH (n) WHERE label IN labels(n) "
            "RETURN count(n) AS c } RETURN label, c ORDER BY c DESC"
        )
        labels = {r["label"]: r["c"] for r in label_rows}

        rel_rows = await self._read(
            "CALL db.relationshipTypes() YIELD relationshipType AS type "
            "CALL { WITH type MATCH ()-[r]->() WHERE type(r) = type "
            "RETURN count(r) AS c } RETURN type, c ORDER BY c DESC"
        )
        rel_types = {r["type"]: r["c"] for r in rel_rows}

        return {
            "nodes": nodes,
            "relationships": rels,
            "orphans": orphans,
            "ratio": round(rels / max(nodes, 1), 3),
            "labels": labels,
            "relationship_types": rel_types,
        }

    # ── safe graph-consolidation helpers ─────────────────────────────────

    async def find_active_node_names(self, since_days: int = 30) -> list[str]:
        """Return names of nodes accessed within the last *since_days*."""
        rows = await self._read(
            "MATCH (n) WHERE n.last_accessed IS NOT NULL "
            "AND n.last_accessed > datetime() - duration({days: $days}) "
            "RETURN n.name AS name",
            days=since_days,
        )
        return [r["name"] for r in rows if r.get("name")]

    async def mark_nodes_stale(self, names: list[str]) -> int:
        """Mark the given nodes as stale."""
        result = await self._run_cypher_write(
            "UNWIND $names AS name "
            "MATCH (n) WHERE n.name = name "
            "SET n._stale = true "
            "RETURN count(n) AS marked",
            params={"names": names},
        )
        return result[0].get("marked", 0) if result else 0

    async def create_co_relevant_edge(self, name_a: str, name_b: str) -> int:
        """Create a CO_RELEVANT relationship between two named nodes."""
        result = await self._run_cypher_write(
            "MATCH (a) WHERE a.name = $a "
            "MATCH (b) WHERE b.name = $b "
            "MERGE (a)-[r:CO_RELEVANT]->(b) "
            "RETURN count(r) AS created",
            params={"a": name_a, "b": name_b},
        )
        return result[0].get("created", 0) if result else 0

    # ── internals ────────────────────────────────────────────────────────

    async def _read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            result = await session.run(query, **params)
            return [dict(record) async for record in result]

    async def close(self) -> None:
        await self._driver.close()
        if self._cloud_driver is not None:
            await self._cloud_driver.close()
            self._cloud_driver = None

    async def __aenter__(self) -> KnowledgeGraph:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
