"""LLM and GraphRAG entity extraction for the knowledge graph (Phase 9 split)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from langfuse.decorators import observe

from brain_os.brain.knowledge_graph_text import normalize_entity_name
from brain_os.config import get_settings
from brain_os.exceptions import LLMError
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import GraphEntities
from brain_os.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = load_prompt("extract_entities")

_EMPTY_EXTRACTION: dict[str, Any] = {
    "companies": [],
    "people": [],
    "machines": [],
    "relationships": [],
}


@dataclass(frozen=True)
class EntityExtractionConfig:
    """Provider settings for unstructured-text entity extraction."""

    entity_fallback_provider: str
    digestive_anthropic_model: str = ""


@observe()
async def extract_entities_from_text(
    *,
    llm: LLMClient,
    config: EntityExtractionConfig,
    text: str,
) -> dict[str, Any]:
    """Extract entities from free text, preferring GraphRAG when available.

    Tries the Neo4j GraphRAG schema-bound extractor first for better entity
    resolution when ``entity_fallback_provider`` is ``openai``.  Falls back to
    the legacy LLM prompt approach if GraphRAG is unavailable or fails.

    Returns a dict with keys ``companies``, ``people``, ``machines``,
    ``relationships`` — each a list of dicts (plus optional project/material keys
    from GraphRAG).
    """
    if config.entity_fallback_provider == "openai":
        try:
            return await extract_entities_graphrag(text)
        except (
            TimeoutError,
            ImportError,
            LLMError,
            httpx.HTTPError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ) as exc:
            logger.debug("GraphRAG extraction unavailable, using legacy LLM", exc_info=True)

    try:
        model_kw: dict[str, str] = {}
        if config.entity_fallback_provider == "anthropic" and config.digestive_anthropic_model:
            model_kw["model"] = config.digestive_anthropic_model
        result = await llm.generate_structured(
            EXTRACTION_SYSTEM_PROMPT,
            text[:12_000],
            GraphEntities,
            name="knowledge_graph.extract",
            provider=config.entity_fallback_provider,
            **model_kw,
        )
        return result.model_dump()
    except (
        TimeoutError,
        LLMError,
        httpx.HTTPError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
    ):
        logger.exception("Entity extraction failed for source text (%d chars)", len(text))
        return dict(_EMPTY_EXTRACTION)


async def extract_entities_graphrag(text: str) -> dict[str, Any]:
    """Schema-bound entity extraction using neo4j-graphrag."""
    from neo4j_graphrag.experimental.components.entity_relation_extractor import (
        LLMEntityRelationExtractor,
    )
    from neo4j_graphrag.experimental.components.schema import (
        SchemaBuilder,
        SchemaEntity,
        SchemaRelation,
    )
    from neo4j_graphrag.experimental.components.types import (
        TextChunk,
        TextChunks,
    )
    from neo4j_graphrag.llm import OpenAILLM

    schema_builder = SchemaBuilder()
    schema_builder.add_entity(
        SchemaEntity(
            label="Company",
            properties=[
                "name",
                "region",
                "industry",
                "website",
                "gauge_tier",
                "gauge_confidence",
                "is_thermoformer",
                "icp_category",
                "gauge_tier_updated_at",
            ],
        )
    )
    schema_builder.add_entity(SchemaEntity(label="Person", properties=["name", "email", "role"]))
    schema_builder.add_entity(
        SchemaEntity(label="Machine", properties=["model", "category", "description"])
    )
    schema_builder.add_entity(
        SchemaEntity(label="Quote", properties=["quote_id", "value", "status"])
    )
    schema_builder.add_entity(
        SchemaEntity(
            label="Project", properties=["project_id", "customer", "machine_model", "status"]
        )
    )
    schema_builder.add_entity(SchemaEntity(label="Application", properties=["name", "description"]))
    schema_builder.add_entity(SchemaEntity(label="Material", properties=["name", "category"]))
    schema_builder.add_entity(
        SchemaEntity(label="Exhibition", properties=["name", "location", "year"])
    )

    for rel_label, source, target in [
        ("WORKS_AT", "Person", "Company"),
        ("INTERESTED_IN", "Company", "Machine"),
        ("QUOTED_TO", "Quote", "Company"),
        ("QUOTES_MACHINE", "Quote", "Machine"),
        ("SUPPLIES", "Company", "Company"),
        ("MANUFACTURES", "Company", "Machine"),
        ("COMPETES_WITH", "Company", "Company"),
        ("CONTACTED_BY", "Company", "Person"),
        ("REFERRED_BY", "Person", "Person"),
        ("CUSTOMER_OF", "Company", "Company"),
        ("SUPPLIES_PARTS_TO", "Company", "Company"),
        ("DISTRIBUTES_FOR", "Company", "Company"),
        ("EXHIBITED_AT", "Company", "Exhibition"),
        ("USES_MATERIAL", "Machine", "Material"),
        ("FOR_APPLICATION", "Machine", "Application"),
        ("PART_OF_PROJECT", "Machine", "Project"),
        ("PROJECT_FOR", "Project", "Company"),
    ]:
        schema_builder.add_relation(
            SchemaRelation(label=rel_label, source_type=source, target_type=target)
        )

    schema = schema_builder.build()

    cfg = get_settings()
    openai_key = cfg.llm.openai_api_key.get_secret_value()
    llm = OpenAILLM(
        model_name=cfg.llm.openai_model,
        api_key=openai_key,
    )

    extractor = LLMEntityRelationExtractor(
        llm=llm,
        create_lexical_graph=False,
    )
    chunks = TextChunks(chunks=[TextChunk(text=text[:12_000])])
    graph_result = await extractor.run(chunks=chunks, schema=schema)

    key_fields = {
        "Company": "name",
        "Person": "email",
        "Machine": "model",
        "Quote": "quote_id",
        "Project": "project_id",
        "Application": "name",
        "Material": "name",
        "Exhibition": "name",
    }

    companies: list[dict[str, Any]] = []
    people: list[dict[str, Any]] = []
    machines: list[dict[str, Any]] = []
    projects: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    exhibitions: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    node_by_id: dict[str, Any] = {}
    for node in graph_result.nodes:
        node_by_id[node.id] = node

    for node in graph_result.nodes:
        props = node.properties or {}
        label = node.label

        if label == "Company":
            companies.append(
                {
                    "name": normalize_entity_name(props.get("name", "")),
                    "region": props.get("region", ""),
                    "industry": props.get("industry", ""),
                    "website": props.get("website", ""),
                }
            )
        elif label == "Person":
            people.append(
                {
                    "name": props.get("name", ""),
                    "email": props.get("email", ""),
                    "company": "",
                    "role": props.get("role", ""),
                }
            )
        elif label == "Machine":
            machines.append(
                {
                    "model": props.get("model", ""),
                    "category": props.get("category", ""),
                    "description": props.get("description", ""),
                }
            )
        elif label == "Project":
            projects.append(
                {
                    "project_id": props.get("project_id", ""),
                    "customer": props.get("customer", ""),
                    "machine_model": props.get("machine_model", ""),
                    "status": props.get("status", ""),
                }
            )
        elif label == "Application":
            applications.append(
                {
                    "name": props.get("name", ""),
                    "description": props.get("description", ""),
                }
            )
        elif label == "Material":
            materials.append(
                {
                    "name": props.get("name", ""),
                    "category": props.get("category", ""),
                }
            )
        elif label == "Exhibition":
            exhibitions.append(
                {
                    "name": props.get("name", ""),
                    "location": props.get("location", ""),
                    "year": props.get("year", ""),
                }
            )

    for rel in graph_result.relationships:
        start_node = node_by_id.get(rel.start_node_id)
        end_node = node_by_id.get(rel.end_node_id)

        def _node_key(n: Any) -> str:
            if n is None:
                return ""
            props = n.properties or {}
            key_field = key_fields.get(n.label, "name")
            raw = props.get(key_field, props.get("name", ""))
            if n.label == "Company":
                return normalize_entity_name(raw)
            return raw

        relationships.append(
            {
                "from_type": start_node.label if start_node else "",
                "from_key": _node_key(start_node),
                "rel": rel.type,
                "to_type": end_node.label if end_node else "",
                "to_key": _node_key(end_node),
            }
        )

    return {
        "companies": companies,
        "people": people,
        "machines": machines,
        "projects": projects,
        "applications": applications,
        "materials": materials,
        "exhibitions": exhibitions,
        "relationships": relationships,
    }
