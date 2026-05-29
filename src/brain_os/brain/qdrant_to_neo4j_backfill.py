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
from pathlib import Path
from typing import Any

import httpx

from brain_os.brain.knowledge_graph import KnowledgeGraph
from brain_os.brain.qdrant_manager import QdrantManager
from brain_os.exceptions import DatabaseError
from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)

_BACKFILL_STATE_FILENAME = ".graph_backfill_state.json"

_BACKFILL_STORE_ERRORS = (
    DatabaseError,
    httpx.HTTPError,
    asyncio.TimeoutError,
    OSError,
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


def _write_backfill_state(last_point_id: str, chunks_processed: int) -> None:
    path = _backfill_state_path()
    try:
        path.write_text(
            json.dumps({"last_point_id": last_point_id, "chunks_processed": chunks_processed})
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


async def run_backfill_from_qdrant(
    qdrant: QdrantManager,
    graph: KnowledgeGraph,
    *,
    max_chunks: int | None = None,
    batch_size: int = 200,
    source_category: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Scroll Qdrant chunks, extract entities from content, write to Neo4j.

    Same entity/relationship logic as DigestiveSystem._extract_entities.
    Returns stats: chunks_processed, companies, people, machines, relationships.
    If resume=True and state exists, continues from last saved point and clears state on success.
    """
    stats: dict[str, int] = {
        "chunks_processed": 0,
        "companies": 0,
        "people": 0,
        "machines": 0,
        "relationships": 0,
        "errors": 0,
    }
    start_after_point_id: str | None = None
    chunks_already_done = 0
    if resume:
        state = _read_backfill_state()
        if state:
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

    async for batch in qdrant.scroll_collection_payloads(
        batch_size=batch_size,
        max_points=max_points_this_run,
        source_category=source_category,
        start_after_point_id=start_after_point_id,
    ):
        for item in batch:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            point_id = item.get("point_id") or ""
            source = item.get("source", "")
            source_id = _source_to_id(source)
            source_category_item = item.get("source_category", "")
            try:
                extracted = await graph.extract_entities_from_text(content)
            except _BACKFILL_STORE_ERRORS:
                logger.debug("Entity extraction failed for chunk from %s", source, exc_info=True)
                stats["errors"] += 1
                continue

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
                        stats["companies"] += 1
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
                        stats["people"] += 1
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
                        stats["machines"] += 1
                    except _BACKFILL_STORE_ERRORS:
                        logger.debug(
                            "Failed to add machine %s", machine.get("model"), exc_info=True
                        )

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
                        logger.debug(
                            "Failed to add project %s", project.get("project_id"), exc_info=True
                        )

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
                    _skip = {"from_type", "from_key", "rel", "to_type", "to_key", "properties"}
                    props = {
                        k: v
                        for k, v in rel.items()
                        if k not in _skip and isinstance(v, (str, int, float, bool))
                    }
                    nested = rel.get("properties")
                    if isinstance(nested, dict):
                        for pk, pv in nested.items():
                            if isinstance(pv, (str, int, float, bool)):
                                props[pk] = pv
                    ok = await graph.add_relationship(
                        from_type=rel.get("from_type", ""),
                        from_key=rel.get("from_key", ""),
                        rel_type=rel.get("rel", ""),
                        to_type=rel.get("to_type", ""),
                        to_key=rel.get("to_key", ""),
                        properties=props if props else None,
                        source_id=source_id,
                    )
                    if ok:
                        stats["relationships"] += 1
                except _BACKFILL_STORE_ERRORS:
                    logger.debug("Failed to add relationship %s", rel, exc_info=True)

            # Link this Qdrant chunk to Neo4j (Chunk node + DESCRIBES edges) for denser network
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
            if point_id and entity_refs:
                try:
                    await graph.add_chunk_and_describes(
                        qdrant_point_id=point_id,
                        source=source,
                        source_category=source_category_item,
                        content_preview=content[:500],
                        entity_refs=entity_refs,
                        source_id=source_id,
                    )
                except _BACKFILL_STORE_ERRORS:
                    logger.debug(
                        "Failed to add Chunk/DESCRIBES for point %s", point_id, exc_info=True
                    )

            stats["chunks_processed"] += 1

            # Progress every 10 chunks so the run doesn't look stuck
            if stats["chunks_processed"] % 10 == 0:
                logger.info(
                    "Backfill progress: %d chunks, companies=%d people=%d machines=%d rels=%d",
                    stats["chunks_processed"],
                    stats["companies"],
                    stats["people"],
                    stats["machines"],
                    stats["relationships"],
                )

        # Persist resume state after each batch so --resume can continue after interrupt
        if batch:
            last_id = batch[-1].get("point_id") or ""
            if last_id:
                _write_backfill_state(last_id, stats["chunks_processed"])

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
