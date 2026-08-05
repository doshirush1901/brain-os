"""Backfill Neo4j from existing Qdrant chunks: scroll → entity extraction → graph write.

Use for one-time or occasional sync of graph data from chunk content that was
already stored in Qdrant (e.g. after restore or to re-run extraction with a
better model). Uses MERGE so duplicates are idempotent.

Resume: use --resume to continue from the last saved offset (state in data/.graph_backfill_state.json).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from neo4j.exceptions import (
    IncompleteCommit,
    Neo4jError,
    ServiceUnavailable,
    SessionError,
    SessionExpired,
)

from brain_os.brain.graph_qdrant_chunk_link import extract_graph_entity_ids
from brain_os.brain.knowledge_graph import KnowledgeGraph
from brain_os.brain.knowledge_graph_extraction import resolve_digestive_openai_model
from brain_os.brain.qdrant_manager import QdrantManager
from brain_os.config import get_settings
from brain_os.exceptions import DatabaseError, LLMError
from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)

_BACKFILL_STATE_FILENAME = ".graph_backfill_state.json"

_BACKFILL_STORE_ERRORS = (
    DatabaseError,
    LLMError,
    Neo4jError,
    ServiceUnavailable,
    SessionError,
    SessionExpired,
    IncompleteCommit,
    httpx.HTTPError,
    asyncio.TimeoutError,
    OSError,
    RuntimeError,
    AssertionError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
)


def _source_to_id(source: str) -> str:
    return re.sub(r"[^a-z0-9:_-]+", "_", (source or "").strip().lower()).strip("_")[:180]


def _backfill_state_path() -> Path:
    return get_data_dir() / _BACKFILL_STATE_FILENAME


def _read_backfill_state() -> dict[str, Any] | None:
    path = _backfill_state_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data.get("last_point_id"), str):
            return data
    except (OSError, json.JSONDecodeError):
        logger.debug("Could not read backfill state from %s", path, exc_info=True)
    return None


def _write_backfill_state(
    last_point_id: str,
    chunks_processed: int,
    *,
    only_missing_entity_ids: bool = False,
) -> None:
    path = _backfill_state_path()
    try:
        path.write_text(
            json.dumps(
                {
                    "last_point_id": last_point_id,
                    "chunks_processed": chunks_processed,
                    "only_missing_entity_ids": only_missing_entity_ids,
                }
            )
        )
    except OSError:
        logger.warning("Could not write backfill state to %s", path, exc_info=True)


def _clear_backfill_state() -> None:
    path = _backfill_state_path()
    if path.exists():
        try:
            path.unlink()
        except OSError:
            logger.debug("Could not remove backfill state %s", path, exc_info=True)


def _entity_refs_from_extracted(extracted: dict[str, Any]) -> list[tuple[str, str]]:
    entity_refs: list[tuple[str, str]] = []
    for c in extracted.get("companies", []):
        if c.get("name"):
            entity_refs.append(("Company", c["name"]))
    for p in extracted.get("people", []):
        if p.get("email"):
            entity_refs.append(("Person", p["email"]))
    for m in extracted.get("machines", []):
        if m.get("model"):
            entity_refs.append(("Machine", m["model"]))
    for q in extracted.get("quotes", []):
        if q.get("quote_id"):
            entity_refs.append(("Quote", q["quote_id"]))
    return entity_refs


def _relationship_props(rel: dict[str, Any]) -> dict[str, Any] | None:
    _skip = {"from_type", "from_key", "rel", "to_type", "to_key", "properties"}
    props = {
        k: v for k, v in rel.items() if k not in _skip and isinstance(v, (str, int, float, bool))
    }
    nested = rel.get("properties")
    if isinstance(nested, dict):
        for pk, pv in nested.items():
            if isinstance(pv, (str, int, float, bool)):
                props[pk] = pv
    return props or None


async def _apply_extracted_to_graph(
    graph: KnowledgeGraph,
    extracted: dict[str, Any],
    *,
    point_id: str,
    content: str,
    source: str,
    source_id: str,
    source_category_item: str,
) -> tuple[dict[str, int], dict[str, Any] | None]:
    """Write extracted entities to Neo4j; return stat deltas and optional chunk batch row."""
    deltas = {
        "companies": 0,
        "people": 0,
        "machines": 0,
        "relationships": 0,
        "chunks_processed": 1,
    }

    for company in extracted.get("companies", []):
        if company.get("name"):
            try:
                await graph.add_company(
                    name=company["name"],
                    region=company.get("region", ""),
                    industry=company.get("industry", ""),
                    website=company.get("website", ""),
                    source_id=source_id,
                )
                deltas["companies"] += 1
            except _BACKFILL_STORE_ERRORS:
                logger.debug("Failed to add company %s", company.get("name"), exc_info=True)

    for person in extracted.get("people", []):
        if person.get("name"):
            try:
                await graph.add_person(
                    name=person["name"],
                    email=person.get("email", ""),
                    company_name=person.get("company", ""),
                    role=person.get("role", ""),
                    source_id=source_id,
                )
                deltas["people"] += 1
            except _BACKFILL_STORE_ERRORS:
                logger.debug("Failed to add person %s", person.get("name"), exc_info=True)

    for machine in extracted.get("machines", []):
        if machine.get("model"):
            try:
                await graph.add_machine(
                    model=machine["model"],
                    category=machine.get("category", ""),
                    description=machine.get("description", ""),
                    source_id=source_id,
                )
                deltas["machines"] += 1
            except _BACKFILL_STORE_ERRORS:
                logger.debug("Failed to add machine %s", machine.get("model"), exc_info=True)

    for project in extracted.get("projects", []):
        if project.get("project_id"):
            try:
                await graph.add_project(
                    project_id=project["project_id"],
                    customer=project.get("customer", ""),
                    machine_model=project.get("machine_model", ""),
                    status=project.get("status", ""),
                )
            except _BACKFILL_STORE_ERRORS:
                logger.debug("Failed to add project %s", project.get("project_id"), exc_info=True)

    for app in extracted.get("applications", []):
        if app.get("name"):
            try:
                await graph.add_application(
                    name=app["name"],
                    description=app.get("description", ""),
                )
            except _BACKFILL_STORE_ERRORS:
                logger.debug("Failed to add application %s", app.get("name"), exc_info=True)

    for mat in extracted.get("materials", []):
        if mat.get("name"):
            try:
                await graph.add_material(
                    name=mat["name"],
                    category=mat.get("category", ""),
                )
            except _BACKFILL_STORE_ERRORS:
                logger.debug("Failed to add material %s", mat.get("name"), exc_info=True)

    for exh in extracted.get("exhibitions", []):
        if exh.get("name"):
            try:
                await graph.add_exhibition(
                    name=exh["name"],
                    location=exh.get("location", ""),
                    year=exh.get("year", ""),
                )
            except _BACKFILL_STORE_ERRORS:
                logger.debug("Failed to add exhibition %s", exh.get("name"), exc_info=True)

    for rel in extracted.get("relationships", []):
        try:
            ok = await graph.add_relationship(
                from_type=rel.get("from_type", ""),
                from_key=rel.get("from_key", ""),
                rel_type=rel.get("rel", ""),
                to_type=rel.get("to_type", ""),
                to_key=rel.get("to_key", ""),
                properties=_relationship_props(rel),
                source_id=source_id,
            )
            if ok:
                deltas["relationships"] += 1
        except _BACKFILL_STORE_ERRORS:
            logger.debug("Failed to add relationship %s", rel, exc_info=True)

    chunk_row: dict[str, Any] | None = None
    entity_refs = _entity_refs_from_extracted(extracted)
    if point_id and entity_refs:
        chunk_row = {
            "point_id": point_id,
            "source": source,
            "source_category": source_category_item,
            "content_preview": content[:500],
            "entity_refs": entity_refs,
            "source_id": source_id,
        }

    return deltas, chunk_row


async def run_backfill_from_qdrant(
    qdrant: QdrantManager,
    graph: KnowledgeGraph,
    *,
    max_chunks: int | None = None,
    batch_size: int = 200,
    source_category: str | None = None,
    resume: bool = False,
    only_missing_entity_ids: bool = False,
    concurrency: int = 5,
    neo4j_batch_size: int = 50,
    on_batch: Callable[[dict[str, int]], None] | None = None,
) -> dict[str, Any]:
    """Scroll Qdrant chunks, extract entities from content, write to Neo4j.

    Same entity/relationship logic as DigestiveSystem._extract_entities.
    Returns stats: chunks_processed, companies, people, machines, relationships.
    If resume=True and state exists, continues from last saved point and clears state on success.

    When ``only_missing_entity_ids`` is true, skips points that already have
    ``graph_entity_ids`` in the Qdrant payload (fast-link cohort).
    """
    cfg = get_settings()
    concurrency = max(1, min(concurrency, 32))
    if neo4j_batch_size < 1:
        neo4j_batch_size = 1
    logger.info(
        "Backfill entity extraction provider: %s (APP__DIGESTIVE_LLM_PROVIDER); "
        "openai_model: %s; concurrency=%d neo4j_batch_size=%d",
        cfg.app.digestive_llm_provider,
        resolve_digestive_openai_model(
            digestive_openai_model=cfg.app.digestive_openai_model,
        )
        if cfg.app.digestive_llm_provider == "openai"
        else "n/a",
        concurrency,
        neo4j_batch_size,
    )
    stats: dict[str, int] = {
        "points_scrolled": 0,
        "chunks_processed": 0,
        "chunks_skipped_has_entity_ids": 0,
        "companies": 0,
        "people": 0,
        "machines": 0,
        "relationships": 0,
        "errors": 0,
        "neo4j_batches": 0,
    }
    start_after_point_id: str | None = None
    chunks_already_done = 0
    if resume:
        state = _read_backfill_state()
        if state:
            state_only_missing = bool(state.get("only_missing_entity_ids", False))
            if state_only_missing != only_missing_entity_ids:
                raise ValueError(
                    "Backfill resume state filter mode mismatch: "
                    f"checkpoint only_missing_entity_ids={state_only_missing} but "
                    f"current run only_missing_entity_ids={only_missing_entity_ids}. "
                    f"Delete {_backfill_state_path()} or rerun without --resume."
                )
            start_after_point_id = state["last_point_id"]
            chunks_already_done = int(state.get("chunks_processed", 0))
            logger.info(
                "Resuming backfill after point %s (%s chunks already processed)",
                start_after_point_id,
                chunks_already_done,
            )
        else:
            logger.info("Resume requested but no state file found; starting from beginning.")

    max_points_this_run: int | None = None
    if max_chunks is not None:
        max_points_this_run = max(0, max_chunks - chunks_already_done)
        if max_points_this_run == 0:
            logger.info(
                "Already processed %s chunks (max_chunks=%s); nothing left to do.",
                chunks_already_done,
                max_chunks,
            )
            return stats

    extract_sem = asyncio.Semaphore(concurrency)
    pending_neo4j: list[dict[str, Any]] = []

    async def _flush_neo4j_batch() -> None:
        if not pending_neo4j:
            return
        batch_rows = list(pending_neo4j)
        pending_neo4j.clear()
        try:
            await graph.add_chunks_and_describes_batch(batch_rows)
            stats["neo4j_batches"] += 1
        except _BACKFILL_STORE_ERRORS:
            logger.debug("Batch chunk link failed (%d rows)", len(batch_rows), exc_info=True)

    async def _extract_one(work: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        async with extract_sem:
            try:
                extracted = await graph.extract_entities_from_text(work["content"])
                return work, extracted
            except _BACKFILL_STORE_ERRORS:
                logger.debug(
                    "Entity extraction failed for chunk from %s",
                    work.get("source"),
                    exc_info=True,
                )
                return work, None

    async for batch in qdrant.scroll_collection_payloads(
        batch_size=batch_size,
        max_points=max_points_this_run,
        source_category=source_category,
        start_after_point_id=start_after_point_id,
        exclude_graph_entity_ids=only_missing_entity_ids,
    ):
        work_items: list[dict[str, Any]] = []
        for item in batch:
            stats["points_scrolled"] += 1
            if only_missing_entity_ids:
                payload = item.get("payload") or {}
                if extract_graph_entity_ids(payload):
                    stats["chunks_skipped_has_entity_ids"] += 1
                    continue
            content = (item.get("content") or "").strip()
            if not content:
                continue
            work_items.append(
                {
                    "point_id": item.get("point_id") or "",
                    "content": content,
                    "source": item.get("source", ""),
                    "source_id": _source_to_id(item.get("source", "")),
                    "source_category": item.get("source_category", ""),
                }
            )

        if work_items:
            extract_results = await asyncio.gather(*[_extract_one(w) for w in work_items])
            for work, extracted in extract_results:
                if extracted is None:
                    stats["errors"] += 1
                    continue
                deltas, chunk_row = await _apply_extracted_to_graph(
                    graph,
                    extracted,
                    point_id=work["point_id"],
                    content=work["content"],
                    source=work["source"],
                    source_id=work["source_id"],
                    source_category_item=work["source_category"],
                )
                for key in ("companies", "people", "machines", "relationships", "chunks_processed"):
                    stats[key] += deltas[key]
                if chunk_row is not None:
                    pending_neo4j.append(chunk_row)
                    if len(pending_neo4j) >= neo4j_batch_size:
                        await _flush_neo4j_batch()

                if stats["chunks_processed"] % 10 == 0:
                    logger.info(
                        "Backfill progress: %d chunks, companies=%d people=%d machines=%d rels=%d",
                        stats["chunks_processed"],
                        stats["companies"],
                        stats["people"],
                        stats["machines"],
                        stats["relationships"],
                    )

        await _flush_neo4j_batch()

        # Persist resume state after each batch so --resume can continue after interrupt
        if batch:
            last_id = batch[-1].get("point_id") or ""
            if last_id:
                _write_backfill_state(
                    last_id,
                    chunks_already_done + stats["chunks_processed"],
                    only_missing_entity_ids=only_missing_entity_ids,
                )

        if on_batch is not None:
            batch_stats = dict(stats)
            batch_stats["chunks_total_done"] = chunks_already_done + stats["chunks_processed"]
            on_batch(batch_stats)

        # Final batch progress
        if stats["chunks_processed"] and stats["chunks_processed"] % 10 != 0:
            logger.info(
                "Backfill progress: %d chunks, companies=%d people=%d machines=%d rels=%d",
                stats["chunks_processed"],
                stats["companies"],
                stats["people"],
                stats["machines"],
                stats["relationships"],
            )

    # Success: clear state so next run starts fresh unless --resume is used with new state
    _clear_backfill_state()
    return stats
