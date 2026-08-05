"""Sync Neo4j Chunk→entity links into Qdrant payload ``graph_entity_ids``.

After ``brain graph backfill-from-qdrant``, chunks exist in Neo4j with DESCRIBES edges but
Qdrant points often lack ``graph_entity_ids``. This module backfills that payload field
so retrieval can stitch graph hits to vectors without re-embedding.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from brain_os.brain.knowledge_graph import KnowledgeGraph
from brain_os.brain.qdrant_manager import QdrantManager
from brain_os.services.resilience import RetryPolicy, run_with_retry

logger = logging.getLogger(__name__)

_FETCH_CHUNK_ENTITIES_CYPHER = """
MATCH (ch:Chunk)-[:DESCRIBES]->(n)
WHERE ch.qdrant_point_id IS NOT NULL
  AND coalesce(ch.point_missing, false) = false
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

_SET_PAYLOAD_RETRY = RetryPolicy(max_attempts=5, base_delay_seconds=2.0, max_delay_seconds=30.0)


def format_entity_id(label: str, key: str) -> str:
    """Canonical ``Label:key`` form used in Qdrant payloads and digestive ingest."""
    return f"{label}:{key.strip()}"


def _is_qdrant_transport_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            ConnectionResetError,
            BrokenPipeError,
        ),
    ):
        return True
    if isinstance(exc, OSError) and exc.errno is not None:
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and isinstance(cause, BaseException):
        return _is_qdrant_transport_error(cause)
    if type(exc).__name__ == "ResponseHandlingException":
        inner = getattr(exc, "__cause__", None)
        if inner is not None and isinstance(inner, BaseException):
            return _is_qdrant_transport_error(inner)
    low = str(exc).lower()
    return any(
        s in low
        for s in (
            "connection reset",
            "broken pipe",
            "connection refused",
            "timed out",
            "timeout",
            "server disconnected",
            "without sending a response",
        )
    )


def _current_graph_entity_ids(point: dict[str, Any]) -> list[str] | None:
    cur = (point.get("metadata") or {}).get("graph_entity_ids")
    if cur is None:
        return None
    if not isinstance(cur, list):
        return None
    return sorted({str(x).strip() for x in cur if x and str(x).strip()})


async def _fetch_existing_entity_ids(
    qdrant: QdrantManager,
    point_ids: list[str],
) -> dict[str, list[str] | None]:
    """Batch-read current ``graph_entity_ids`` for skip-unchanged checks."""
    out: dict[str, list[str] | None] = {pid: None for pid in point_ids}
    try:
        rows = await run_with_retry(
            lambda: qdrant.get_points(point_ids),
            policy=_SET_PAYLOAD_RETRY,
            is_retryable=_is_qdrant_transport_error,
        )
    except Exception:
        logger.debug("Batch get_points failed for %d ids", len(point_ids), exc_info=True)
        return out
    for row in rows:
        pid = str(row.get("id") or row.get("point_id") or "").strip()
        if pid:
            out[pid] = _current_graph_entity_ids(row)
    return out


async def _set_payload_with_retry(
    qdrant: QdrantManager,
    point_id: str,
    payload: dict[str, Any],
) -> None:
    await run_with_retry(
        lambda: qdrant.set_payload(point_id, payload, wait=False),
        policy=_SET_PAYLOAD_RETRY,
        is_retryable=_is_qdrant_transport_error,
    )


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
    write_concurrency: int = 1,
    sleep_seconds: float = 0.05,
) -> dict[str, int]:
    """Set ``graph_entity_ids`` on Qdrant points that have matching Chunk nodes in Neo4j."""
    stats: dict[str, int] = {
        "chunks_in_graph": 0,
        "points_updated": 0,
        "points_skipped_unchanged": 0,
        "points_skipped_missing": 0,
        "points_failed": 0,
        "dry_run": int(dry_run),
    }
    await graph.ensure_connected()
    entity_map = await fetch_chunk_entity_id_map(graph)
    stats["chunks_in_graph"] = len(entity_map)
    if not entity_map:
        logger.info("No Chunk→entity mappings found in Neo4j; nothing to sync")
        return stats

    point_ids = list(entity_map.keys())
    if max_points is not None:
        point_ids = point_ids[: max(0, max_points)]

    write_semaphore = asyncio.Semaphore(max(1, write_concurrency))

    async def _write_one(pid: str, target_ids: list[str]) -> bool:
        payload: dict[str, Any] = {"metadata": {"graph_entity_ids": target_ids}}
        async with write_semaphore:
            try:
                await _set_payload_with_retry(qdrant, pid, payload)
                return True
            except Exception:
                logger.warning("set_payload failed for %s after retries", pid, exc_info=True)
                return False

    for i in range(0, len(point_ids), batch_size):
        batch = point_ids[i : i + batch_size]
        if dry_run:
            stats["points_updated"] += len(batch)
            continue

        existing_by_pid: dict[str, list[str] | None] = {}
        if skip_unchanged:
            existing_by_pid = await _fetch_existing_entity_ids(qdrant, batch)

        live_pids: set[str] = set()
        if not dry_run:
            try:
                rows = await qdrant.get_points(batch)
                live_pids = {str(r.get("id") or r.get("point_id") or "").strip() for r in rows}
                live_pids.discard("")
            except Exception:
                logger.debug("get_points failed for entity-id sync batch", exc_info=True)
                live_pids = set(batch)

        write_tasks: list[asyncio.Task[bool]] = []
        for pid in batch:
            if live_pids and pid not in live_pids:
                stats["points_skipped_missing"] += 1
                continue
            target_ids = entity_map[pid]
            if skip_unchanged and existing_by_pid.get(pid) == target_ids:
                stats["points_skipped_unchanged"] += 1
                continue
            write_tasks.append(asyncio.create_task(_write_one(pid, target_ids)))

        if write_tasks:
            results = await asyncio.gather(*write_tasks)
            stats["points_updated"] += sum(1 for ok in results if ok)
            stats["points_failed"] += sum(1 for ok in results if not ok)

        done = min(i + batch_size, len(point_ids))
        if done % 500 == 0 or done >= len(point_ids):
            logger.info(
                "graph_entity_ids sync progress: %d / %d (updated=%d skipped=%d missing=%d failed=%d)",
                done,
                len(point_ids),
                stats["points_updated"],
                stats["points_skipped_unchanged"],
                stats["points_skipped_missing"],
                stats["points_failed"],
            )
        # Brief pause between batches to avoid hammering Qdrant Cloud connections.
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)

    return stats
