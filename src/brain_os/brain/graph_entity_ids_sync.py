"""Sync Neo4j Chunk→entity links into Qdrant payload ``graph_entity_ids``.

After ``brain graph backfill-from-qdrant``, chunks exist in Neo4j with DESCRIBES edges but
Qdrant points often lack ``graph_entity_ids``. This module backfills that payload field
so retrieval can stitch graph hits to vectors without re-embedding.
"""

from __future__ import annotations

import logging
from typing import Any

from brain_os.brain.knowledge_graph import KnowledgeGraph
from brain_os.brain.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)

_FETCH_CHUNK_ENTITIES_CYPHER = """
MATCH (ch:Chunk)-[:DESCRIBES]->(n)
WHERE ch.qdrant_point_id IS NOT NULL
WITH ch.qdrant_point_id AS point_id, n
WITH point_id,
     head([lbl IN labels(n) WHERE lbl IN [
         'Company', 'Person', 'Machine', 'Quote',
         'Project', 'Application', 'Material', 'Exhibition'
     ] | lbl]) AS label,
     n
WITH point_id, label,
     CASE label
       WHEN 'Company' THEN n.name
       WHEN 'Person' THEN coalesce(
           CASE WHEN n.email IS NULL OR trim(toString(n.email)) = '' THEN null ELSE n.email END,
           n.name
       )
       WHEN 'Machine' THEN n.model
       WHEN 'Quote' THEN n.quote_id
       WHEN 'Project' THEN n.project_id
       WHEN 'Application' THEN n.name
       WHEN 'Material' THEN n.name
       WHEN 'Exhibition' THEN n.name
       ELSE null
     END AS key
WHERE label IS NOT NULL AND key IS NOT NULL AND trim(toString(key)) <> ''
WITH point_id, label + ':' + trim(toString(key)) AS entity_id
RETURN point_id, collect(DISTINCT entity_id) AS graph_entity_ids
"""


def format_entity_id(label: str, key: str) -> str:
    """Canonical ``Label:key`` form used in Qdrant payloads and digestive ingest."""
    return f"{label}:{key.strip()}"


async def fetch_chunk_entity_id_map(graph: KnowledgeGraph) -> dict[str, list[str]]:
    """Return ``qdrant_point_id`` → sorted unique ``graph_entity_ids`` from DESCRIBES edges."""
    rows = await graph._read(_FETCH_CHUNK_ENTITIES_CYPHER)
    out: dict[str, list[str]] = {}
    for row in rows:
        pid = str(row.get("point_id") or "").strip()
        if not pid:
            continue
        ids = row.get("graph_entity_ids") or []
        if not isinstance(ids, list):
            continue
        cleaned = sorted({str(x).strip() for x in ids if x and str(x).strip()})
        if cleaned:
            out[pid] = cleaned
    return out


async def run_sync_entity_ids_to_qdrant(
    qdrant: QdrantManager,
    graph: KnowledgeGraph,
    *,
    dry_run: bool = False,
    batch_size: int = 100,
    max_points: int | None = None,
    skip_unchanged: bool = True,
) -> dict[str, int]:
    """Set ``graph_entity_ids`` on Qdrant points that have matching Chunk nodes in Neo4j."""
    stats: dict[str, int] = {
        "chunks_in_graph": 0,
        "points_updated": 0,
        "points_skipped_unchanged": 0,
        "points_failed": 0,
        "dry_run": int(dry_run),
    }
    entity_map = await fetch_chunk_entity_id_map(graph)
    stats["chunks_in_graph"] = len(entity_map)
    if not entity_map:
        logger.info("No Chunk→entity mappings found in Neo4j; nothing to sync")
        return stats

    point_ids = list(entity_map.keys())
    if max_points is not None:
        point_ids = point_ids[: max(0, max_points)]

    for i in range(0, len(point_ids), batch_size):
        batch = point_ids[i : i + batch_size]
        if dry_run:
            stats["points_updated"] += len(batch)
            continue
        for pid in batch:
            # Digestive stores graph_entity_ids under payload.metadata (see _canonicalize_payload).
            payload: dict[str, Any] = {"metadata": {"graph_entity_ids": entity_map[pid]}}
            if skip_unchanged:
                try:
                    existing = await qdrant.get_points([pid])
                    if existing:
                        cur = (existing[0].get("metadata") or {}).get("graph_entity_ids")
                        if cur == entity_map[pid]:
                            stats["points_skipped_unchanged"] += 1
                            continue
                except (OSError, ValueError, TypeError):
                    logger.debug("Could not read point %s before set_payload", pid, exc_info=True)
            try:
                await qdrant.set_payload(pid, payload)
                stats["points_updated"] += 1
            except (OSError, ValueError, TypeError):
                stats["points_failed"] += 1
                logger.debug("set_payload failed for %s", pid, exc_info=True)
        if (i + batch_size) % 500 == 0 or i + batch_size >= len(point_ids):
            logger.info(
                "graph_entity_ids sync progress: %d / %d points",
                min(i + batch_size, len(point_ids)),
                len(point_ids),
            )

    return stats
