"""Ingestion gatekeeper — Alexandros's brain for deciding what to ingest.

Compares the imports metadata index against the ingestion log to find
files that are new, changed, or were ingested by an older pipeline.
Processes them through the DigestiveSystem and updates the log.

Supports parallel processing: multiple files are digested concurrently
(each file's GPT calls run independently), giving near-linear speedup
since all work is I/O-bound (API calls to OpenAI / Qdrant / Neo4j).

  Concurrency 1 (serial):   ~45s/file  — 712 files = ~9 hours
  Concurrency 5 (default):  ~9s/file   — 712 files = ~1.8 hours
  Concurrency 10:           ~5s/file   — 712 files = ~1 hour

Can be called from the CLI (``brain ingest``), the respiratory system's
inhale cycle, or directly by the Alexandros agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain_os.brain.imports_metadata_index import load_index
from brain_os.brain.ingestion_log import (
    CURRENT_PIPELINE,
    file_fingerprint,
    load_log,
    needs_ingestion,
    record_ingestion,
    record_ingestion_attempt,
    save_log,
)
from brain_os.brain.source_identity import make_source_id
from brain_os.contracts.memory_protocols import LongTermMemoryStoreProtocol
from brain_os.exceptions import IngestionError, BrainOSError
from brain_os.knowledge.teacher_provenance import (
    SOURCE_CATEGORY_TEACHER_CANON,
    teacher_metadata_for_corpus,
    teacher_metadata_from_rel_path,
)

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 5
_MIN_TEXT_FOR_DIRECT_INGEST = 80
_INGESTION_METRICS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "brain" / "ingestion_metrics.jsonl"
)

_ASANA_IMPORT_MARKERS = ("23_asana", "asana_grounded_gold_sets")
_ASANA_DOC_TYPE_TO_CATEGORY: dict[str, str] = {
    "order": "orders_and_pos",
    "invoice": "orders_and_pos",
    "quote": "orders_and_pos",
    "contract": "contracts_and_legal",
    "technical_spec": "production",
    "manual": "production",
    "report": "production",
    "spreadsheet": "production",
    "presentation": "project_case_studies",
}

# Acme Corp finance imports — Qdrant ``source_category`` (slugged at upsert).
_FINANCE_PATH_TO_CATEGORY: tuple[tuple[str, str], ...] = (
    ("19_business_plans/", "business_plans"),
    ("18_tally_exports/", "tally_exports"),
    ("10_company_internal/demo_finance/", "demo_finance"),
)


def _resolve_source_category(rel_path: str, meta: dict[str, Any]) -> str:
    """Map imports metadata to an ingestion category.

    For the Asana gold-set package we prefer Atlas-friendly operational
    categories so category-filtered retrieval can find shop-floor context.
    """
    rel_lower = rel_path.lower().replace("\\", "/")
    if "teacher_canon/" in rel_lower:
        return SOURCE_CATEGORY_TEACHER_CANON
    doc_type = str(meta.get("doc_type", "other")).strip().lower() or "other"
    for prefix, category in _FINANCE_PATH_TO_CATEGORY:
        if rel_lower.startswith(prefix):
            return category
    if any(marker in rel_lower for marker in _ASANA_IMPORT_MARKERS):
        return _ASANA_DOC_TYPE_TO_CATEGORY.get(doc_type, "production")
    return doc_type


async def scan_for_undigested(
    *,
    force: bool = False,
    exclude_prefixes: tuple[str, ...] = (),
    include_prefixes: tuple[str, ...] = (),
    teacher: str | None = None,
) -> list[dict[str, Any]]:
    """Compare the metadata index against the ingestion log.

    Returns a list of file dicts needing ingestion, each with keys
    ``rel_path``, ``path``, ``hash``, ``reason``, ``category``,
    ``extension``, ``name``.

    If *include_prefixes* is non-empty, only files whose relative path
    starts with one of those prefixes (e.g. ``01_Quotes_and_Proposals/``)
    are considered. Otherwise all files (subject to excludes) are considered.
    """
    index = await load_index()
    log = await load_log()
    queue: list[dict[str, Any]] = []

    normalized_excludes = tuple(
        p.strip().lower().rstrip("/") + "/" for p in exclude_prefixes if p and p.strip()
    )
    normalized_includes = tuple(
        p.strip().lower().rstrip("/") + "/" for p in include_prefixes if p and p.strip()
    )

    for rel_path, meta in index.get("files", {}).items():
        rel_path_lower = rel_path.lower()
        if any(rel_path_lower.startswith(prefix) for prefix in normalized_excludes):
            continue
        if normalized_includes and not any(
            rel_path_lower.startswith(prefix) for prefix in normalized_includes
        ):
            continue

        filepath = Path(meta.get("path", ""))
        if not filepath.exists():
            continue

        current_hash = meta.get("hash", "")
        if not current_hash:
            try:
                current_hash = file_fingerprint(filepath)
            except OSError:
                continue

        reason = needs_ingestion(log, rel_path, current_hash, force=force)
        if reason:
            doc_type = meta.get("doc_type", "other")
            teacher_meta = teacher_metadata_from_rel_path(rel_path)
            if teacher and not teacher_meta:
                teacher_meta = teacher_metadata_for_corpus(teacher.strip().lower())
            entry: dict[str, Any] = {
                "rel_path": rel_path,
                "path": str(filepath),
                "hash": current_hash,
                "reason": reason,
                "category": _resolve_source_category(rel_path, meta),
                "doc_type": doc_type,
                "extension": meta.get("extension", filepath.suffix.lower()),
                "name": meta.get("name", filepath.name),
                "size_kb": meta.get("size_kb", 0),
            }
            if teacher_meta:
                entry["teacher_metadata"] = teacher_meta
                entry["category"] = SOURCE_CATEGORY_TEACHER_CANON
            queue.append(entry)

    queue.sort(
        key=lambda f: (
            0
            if f["reason"] == "new"
            else 1
            if f["reason"] == "changed"
            else 2
            if f["reason"] == "forced"
            else 3,
            f["size_kb"],  # smaller files first within each reason group
        )
    )

    logger.info(
        "Gatekeeper scan: %d files need ingestion (%s)",
        len(queue),
        ", ".join(
            f"{r}: {sum(1 for f in queue if f['reason'] == r)}"
            for r in sorted({f["reason"] for f in queue})
        )
        if queue
        else "none",
    )
    return queue


ProgressCallback = Any  # Callable[[int, int, str, dict], None]


async def run_ingestion_cycle(
    *,
    force: bool = False,
    batch_size: int = 712,
    concurrency: int = DEFAULT_CONCURRENCY,
    exclude_prefixes: tuple[str, ...] = (),
    include_prefixes: tuple[str, ...] = (),
    teacher: str | None = None,
    progress_callback: ProgressCallback | None = None,
    long_term_memory: LongTermMemoryStoreProtocol | None = None,
) -> dict[str, Any]:
    """Run a gated ingestion cycle through the DigestiveSystem.

    Processes up to *batch_size* files with *concurrency* files in
    parallel.  The log is updated as each file completes.

    *progress_callback(done, total, filename, file_result)* is called
    after each file finishes so the CLI can update a progress bar.
    """
    from brain_os.brain.document_ingestor import DocumentIngestor, _get_reader
    from brain_os.brain.embeddings import EmbeddingService
    from brain_os.brain.knowledge_graph import KnowledgeGraph
    from brain_os.brain.qdrant_manager import QdrantManager
    from brain_os.config import get_settings
    from brain_os.systems.digestive import DigestiveSystem

    queue = await scan_for_undigested(
        force=force,
        exclude_prefixes=exclude_prefixes,
        include_prefixes=include_prefixes,
        teacher=teacher,
    )
    if not queue:
        logger.info("Gatekeeper: nothing to ingest")
        return {"files_processed": 0, "files_skipped": 0, "reason": "up_to_date"}

    batch = queue[:batch_size]
    batch_total = len(batch)
    logger.info(
        "Gatekeeper: ingesting %d / %d files (concurrency=%d)",
        batch_total,
        len(queue),
        concurrency,
    )

    # Shared services — thread-safe for concurrent use
    embedding = EmbeddingService()
    qdrant = QdrantManager(embedding_service=embedding)
    await qdrant.ensure_collection()
    graph = KnowledgeGraph()
    ingestor = DocumentIngestor(qdrant=qdrant, knowledge_graph=graph)
    digestive = DigestiveSystem(
        ingestor=ingestor,
        knowledge_graph=graph,
        embedding_service=embedding,
        qdrant=qdrant,
    )

    settings = get_settings()
    collection = settings.qdrant.collection
    log = await load_log()

    # Shared counters protected by a lock (concurrent writes from parallel tasks)
    lock = asyncio.Lock()
    processed = 0
    skipped = 0
    failed = 0
    total_chunks = 0
    total_entities: dict[str, int] = {"companies": 0, "people": 0, "machines": 0}
    memory_written = 0
    memory_attempted = 0
    errors: list[str] = []
    done_count = 0

    if long_term_memory is None:
        from brain_os.memory.long_term import LongTermMemory

        long_term_memory = LongTermMemory()

    async def _store_ingestion_memory(
        *,
        file_info: dict[str, Any],
        source_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Write a compact learning fact to long-term memory for this source."""
        chunks = int(result.get("chunks_created", 0))
        entities = result.get("entities_found", {})
        content = (
            f"Ingested source {file_info.get('name', 'unknown')} "
            f"(category={file_info.get('category', 'other')}, source_id={source_id}) "
            f"with {chunks} chunks and entities={entities}."
        )
        metadata = {
            "type": "ingested_source",
            "source_id": source_id,
            "source_path": file_info.get("path", ""),
            "source_name": file_info.get("name", ""),
            "source_category": file_info.get("category", "other"),
            "doc_type": file_info.get("doc_type", "other"),
            "chunk_count": chunks,
            "entities": entities,
        }
        memories = await long_term_memory.store(content, user_id="global", metadata=metadata)
        return {
            "attempted": True,
            "status": "stored" if memories else "not_stored",
            "count": len(memories),
        }

    async def _process_one(file_info: dict[str, Any]) -> None:
        nonlocal \
            processed, \
            skipped, \
            failed, \
            total_chunks, \
            done_count, \
            memory_written, \
            memory_attempted

        rel_path = file_info["rel_path"]
        filepath = Path(file_info["path"])
        source_id = make_source_id(str(filepath), file_info["hash"])

        try:
            ext = file_info["extension"]
            reader = _get_reader(ext)
            if reader is None:
                async with lock:
                    record_ingestion_attempt(
                        log,
                        rel_path,
                        file_info["hash"],
                        collection=collection,
                        source_id=source_id,
                        status="unsupported_extension",
                        detail=ext,
                    )
                    await save_log(log)
                    skipped += 1
                    done_count += 1
                    if progress_callback is not None:
                        progress_callback(done_count, batch_total, file_info["name"], {})
                return

            text = reader(filepath)
            if not text or not text.strip():
                async with lock:
                    record_ingestion_attempt(
                        log,
                        rel_path,
                        file_info["hash"],
                        collection=collection,
                        source_id=source_id,
                        status="empty_content",
                    )
                    await save_log(log)
                    skipped += 1
                    done_count += 1
                    if progress_callback is not None:
                        progress_callback(done_count, batch_total, file_info["name"], {})
                return

            teacher_extra = file_info.get("teacher_metadata")
            extra_meta = teacher_extra if isinstance(teacher_extra, dict) else None
            result = await digestive.ingest(
                raw_data=text,
                source=str(filepath),
                source_category=file_info["category"],
                source_id=source_id,
                doc_type=file_info.get("doc_type", "other"),
                extra_metadata=extra_meta,
            )

            chunks = int(result.get("chunks_created", 0))
            if chunks == 0 and len(text.strip()) >= _MIN_TEXT_FOR_DIRECT_INGEST:
                direct_chunks = await ingestor.ingest_file(file_info, force=True)
                if direct_chunks > 0:
                    chunks = direct_chunks
                    result = {
                        "chunks_created": direct_chunks,
                        "nutrients_extracted": {"protein": 0, "carbs": 0, "waste": 0},
                        "entities_found": {},
                        "ingest_status": "direct_chunk_fallback",
                    }

            async with lock:
                done_count += 1
                if chunks > 0:
                    memory_result: dict[str, Any]
                    try:
                        memory_result = await _store_ingestion_memory(
                            file_info=file_info,
                            source_id=source_id,
                            result=result,
                        )
                    except BrainOSError as exc:
                        logger.warning("Memory write failed for %s", rel_path, exc_info=True)
                        memory_result = {
                            "attempted": True,
                            "status": "error",
                            "count": 0,
                        }
                    result["memory_write"] = memory_result
                    memory_attempted += 1 if memory_result.get("attempted") else 0
                    memory_written += 1 if memory_result.get("status") == "stored" else 0

                    record_ingestion(
                        log,
                        rel_path,
                        file_info["hash"],
                        result,
                        collection,
                        source_id=source_id,
                    )
                    await save_log(log)
                    processed += 1
                    total_chunks += chunks
                    for k in total_entities:
                        total_entities[k] += result.get("entities_found", {}).get(k, 0)
                else:
                    record_ingestion_attempt(
                        log,
                        rel_path,
                        file_info["hash"],
                        collection=collection,
                        source_id=source_id,
                        status="no_chunks",
                    )
                    await save_log(log)
                    skipped += 1
                if progress_callback is not None:
                    progress_callback(done_count, batch_total, file_info["name"], result)

        except (IngestionError, OSError, ValueError, TypeError) as exc:
            logger.exception("Gatekeeper: failed to ingest %s", rel_path)
            async with lock:
                errors.append(f"{rel_path}: {exc}")
                failed += 1
                done_count += 1
                if progress_callback is not None:
                    progress_callback(done_count, batch_total, file_info["name"], {})

    # Run with bounded concurrency using a semaphore
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(file_info: dict[str, Any]) -> None:
        async with sem:
            await _process_one(file_info)

    try:
        await asyncio.gather(*[_guarded(f) for f in batch])

        log["last_full_scan"] = datetime.now(UTC).isoformat()
        await save_log(log)
    finally:
        for closeable in [qdrant, graph]:
            try:
                await closeable.close()
            except (BrainOSError, OSError, RuntimeError):
                logger.debug("Failed to close %s", type(closeable).__name__, exc_info=True)
        try:
            ingestor.close()
        except (BrainOSError, OSError, RuntimeError):
            logger.debug("Failed to close ingestor", exc_info=True)

    summary = {
        "files_processed": processed,
        "files_skipped": skipped,
        "files_failed": failed,
        "files_remaining": len(queue) - batch_total,
        "total_chunks": total_chunks,
        "total_entities": total_entities,
        "memory_attempted": memory_attempted,
        "memory_written": memory_written,
        "errors": errors,
        "pipeline": CURRENT_PIPELINE,
        "batch_size": batch_total,
        "total_queued": len(queue),
        "concurrency": concurrency,
        "exclude_prefixes": list(exclude_prefixes),
    }
    logger.info("Gatekeeper ingestion cycle complete: %s", summary)

    # Append metrics for trend tracking (one JSON object per line).
    try:
        _INGESTION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        metrics_line = (
            json.dumps(
                {"ts": datetime.now(UTC).isoformat(), **summary},
                default=str,
            )
            + "\n"
        )

        def _write() -> None:
            with _INGESTION_METRICS_PATH.open("a", encoding="utf-8") as f:
                f.write(metrics_line)

        await asyncio.to_thread(_write)
    except (OSError, TypeError) as e:
        logger.debug("Could not write ingestion metrics: %s", e)

    return summary
