"""Cross-store ingestion audit for imports, KB, graph, and memory evidence.

This module provides a read-only audit of whether files in ``data/imports``
are reflected across:

- imports metadata index (``imports_metadata.json``)
- ingestion log (``ingestion_log.json``)
- Qdrant knowledge store (source payload presence)
- Neo4j graph (global stats + optional source-property evidence)
- memory systems (explicitly reports current per-file verification limits)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError
from qdrant_client import AsyncQdrantClient, models

from brain_os.brain.embeddings import EmbeddingService
from brain_os.brain.imports_metadata_index import IMPORTS_DIR, load_index
from brain_os.brain.ingestion_gatekeeper import scan_for_undigested
from brain_os.brain.ingestion_log import load_log
from brain_os.brain.qdrant_manager import QdrantManager
from brain_os.config import get_settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SUPPORTED_IMPORT_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".xlsx",
        ".xls",
        ".docx",
        ".txt",
        ".csv",
        ".pptx",
        ".html",
        ".md",
        ".eml",
        ".json",
    }
)


def _collect_supported_import_files(imports_dir: Path) -> list[Path]:
    if not imports_dir.exists():
        return []
    files: list[Path] = []
    for p in imports_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in _SUPPORTED_IMPORT_EXTENSIONS:
            files.append(p.resolve())
    return sorted(files)


def _pct(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return round((num / den) * 100.0, 2)


def _normalise_qdrant_source(source: str) -> str:
    """Normalize Qdrant source payload values to absolute file paths when possible."""
    raw = source.strip()
    if not raw:
        return raw
    p = Path(raw)
    if p.is_absolute():
        return str(p.resolve())
    # Ingest sources may be stored as relative paths from project root.
    return str((_PROJECT_ROOT / p).resolve())


async def _fetch_qdrant_sources(collection: str) -> tuple[set[str], dict[str, Any]]:
    """Return all distinct source payloads currently present in Qdrant collection."""
    cfg = get_settings().qdrant
    client = AsyncQdrantClient(
        url=cfg.url,
        api_key=(cfg.api_key.get_secret_value() or None),
        check_compatibility=False,
        timeout=cfg.timeout,
    )
    raw_sources: set[str] = set()
    scanned_points = 0
    offset: models.PointId | None = None

    try:
        while True:
            points, offset = await client.scroll(
                collection_name=collection,
                limit=1000,
                with_payload=["source"],
                with_vectors=False,
                offset=offset,
            )
            if not points:
                break
            scanned_points += len(points)
            for point in points:
                payload = point.payload or {}
                source = payload.get("source")
                if isinstance(source, str) and source.strip():
                    raw_sources.add(source)
            if offset is None:
                break

        count_info = await client.count(collection_name=collection, exact=False)
        normalized = {_normalise_qdrant_source(s) for s in raw_sources}
        return normalized, {
            "status": "ok",
            "collection": collection,
            "count_estimate": count_info.count,
            "points_scanned": scanned_points,
            "distinct_sources": len(normalized),
        }
    except (httpx.HTTPError, OSError, ValueError, TypeError) as exc:
        return set(), {"status": "error", "error": str(exc), "collection": collection}
    finally:
        await client.close()


async def _fetch_neo4j_evidence() -> dict[str, Any]:
    """Fetch global graph evidence and optional source-property linkage evidence."""
    cfg = get_settings().neo4j

    # Only use configured credentials; report clear failure if unavailable.
    user, password = cfg.resolved_auth()
    if not user or not password:
        return {
            "status": "error",
            "error": "Neo4j credentials not configured (set NEO4J_PASSWORD or NEO4J_AUTH)",
            "per_file_source_supported": False,
        }

    app_cfg = get_settings().app
    driver = AsyncGraphDatabase.driver(
        cfg.uri,
        auth=(user, password),
        max_connection_pool_size=app_cfg.neo4j_max_pool_size,
        connection_acquisition_timeout=60.0,
    )
    try:
        async with driver.session() as session:
            nodes_row = await (await session.run("MATCH (n) RETURN count(n) AS c")).single()
            rels_row = await (await session.run("MATCH ()-[r]->() RETURN count(r) AS c")).single()
            src_nodes_row = await (
                await session.run("MATCH (n) WHERE n.source IS NOT NULL RETURN count(n) AS c")
            ).single()
            src_rows = await (
                await session.run(
                    "MATCH (n) WHERE n.source IS NOT NULL RETURN DISTINCT n.source AS source LIMIT 10000"
                )
            ).data()

        sources = {
            _normalise_qdrant_source(str(r.get("source", ""))) for r in src_rows if r.get("source")
        }
        return {
            "status": "ok",
            "nodes": int(nodes_row["c"]) if nodes_row else 0,
            "relationships": int(rels_row["c"]) if rels_row else 0,
            "nodes_with_source": int(src_nodes_row["c"]) if src_nodes_row else 0,
            "distinct_source_values": len(sources),
            "source_values": sources,
            "per_file_source_supported": True,
        }
    except (Neo4jError, OSError, ValueError, TypeError) as exc:
        return {"status": "error", "error": str(exc), "per_file_source_supported": False}
    finally:
        await driver.close()


async def run_ingestion_audit() -> dict[str, Any]:
    """Run read-only cross-store ingestion coverage audit."""
    imports_root = IMPORTS_DIR.resolve()
    imports_files = _collect_supported_import_files(imports_root)
    imports_abs = {str(p) for p in imports_files}
    imports_rel = {str(p.relative_to(imports_root)) for p in imports_files}

    index = await load_index()
    index_files = set(index.get("files", {}).keys())
    index_matched = imports_rel & index_files

    ingestion_log = await load_log()
    log_files = set(ingestion_log.get("files", {}).keys())
    log_matched = imports_rel & log_files

    needs_ingestion = await scan_for_undigested(force=False)
    by_reason: dict[str, int] = {}
    for row in needs_ingestion:
        reason = str(row.get("reason", "unknown"))
        by_reason[reason] = by_reason.get(reason, 0) + 1

    qdrant_collection = get_settings().qdrant.collection
    embedding = EmbeddingService()
    qdrant = QdrantManager(embedding_service=embedding)
    try:
        await qdrant.ensure_collection()
    finally:
        await qdrant.close()
    qdrant_sources, qdrant_info = await _fetch_qdrant_sources(qdrant_collection)
    qdrant_matched = imports_abs & qdrant_sources

    neo4j_info = await _fetch_neo4j_evidence()
    neo4j_matched = 0
    if neo4j_info.get("status") == "ok" and neo4j_info.get("per_file_source_supported"):
        source_values = neo4j_info.get("source_values", set())
        if isinstance(source_values, set):
            neo4j_matched = len(imports_abs & source_values)

    log_entries = ingestion_log.get("files", {})
    memory_rows = [
        v
        for v in log_entries.values()
        if isinstance(v, dict) and isinstance(v.get("memory_write"), dict)
    ]
    memory_attempted = sum(1 for row in memory_rows if row["memory_write"].get("attempted"))
    memory_stored = sum(1 for row in memory_rows if row["memory_write"].get("status") == "stored")
    memory_source_id_rows = sum(
        1 for row in log_entries.values() if isinstance(row, dict) and row.get("source_id")
    )

    memory_info = {
        "status": "ok" if memory_rows else "limited",
        "per_file_supported": bool(memory_rows and memory_source_id_rows),
        "detail": (
            "Coverage based on ingestion log memory_write evidence."
            if memory_rows
            else "No memory_write evidence present in ingestion log yet."
        ),
        "attempted": memory_attempted,
        "stored": memory_stored,
        "source_id_rows": memory_source_id_rows,
        "coverage_pct": _pct(memory_stored, len(imports_rel)),
    }

    summary = {
        "imports": {
            "supported_file_count": len(imports_files),
            "imports_dir": str(imports_root),
        },
        "metadata_index": {
            "indexed_files": len(index_files),
            "matched_import_files": len(index_matched),
            "coverage_pct": _pct(len(index_matched), len(imports_rel)),
            "built_at": index.get("built_at"),
        },
        "ingestion_log": {
            "logged_files": len(log_files),
            "matched_import_files": len(log_matched),
            "coverage_pct": _pct(len(log_matched), len(imports_rel)),
            "last_full_scan": ingestion_log.get("last_full_scan"),
            "pipeline_versions": sorted(
                {
                    str(v.get("pipeline", "unknown"))
                    for v in ingestion_log.get("files", {}).values()
                    if isinstance(v, dict)
                }
            ),
        },
        "gatekeeper": {
            "needs_ingestion_count": len(needs_ingestion),
            "by_reason": by_reason,
        },
        "qdrant": {
            **qdrant_info,
            "matched_import_files": len(qdrant_matched),
            "coverage_pct": _pct(len(qdrant_matched), len(imports_abs)),
        },
        "neo4j": {
            **{k: v for k, v in neo4j_info.items() if k != "source_values"},
            "matched_import_files": neo4j_matched,
            "coverage_pct": _pct(neo4j_matched, len(imports_abs)),
        },
        "memory": memory_info,
    }

    gaps: list[str] = []
    if summary["metadata_index"]["coverage_pct"] < 100:
        gaps.append("metadata_index_incomplete")
    if summary["ingestion_log"]["coverage_pct"] < 100:
        gaps.append("ingestion_log_incomplete")
    if summary["qdrant"].get("status") != "ok":
        gaps.append("qdrant_unavailable")
    elif summary["qdrant"]["coverage_pct"] < 100:
        gaps.append("qdrant_incomplete")
    if summary["neo4j"].get("status") != "ok":
        gaps.append("neo4j_unavailable")
    if not summary["memory"]["per_file_supported"]:
        gaps.append("memory_per_file_audit_not_supported")
    elif summary["memory"]["coverage_pct"] < 100:
        gaps.append("memory_incomplete")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": summary,
        "gaps": gaps,
        "healthy": len(gaps) == 0,
    }


async def write_audit_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(report, default=str))
    await asyncio.to_thread(output_path.write_text, json.dumps(serializable, indent=2))
