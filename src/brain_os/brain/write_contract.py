"""Write-contract receipts for Mem0, Qdrant, and Neo4j operations.

Provides a lightweight JSONL receipt log plus a small retry queue for failed
write attempts. Used by pipeline/system write paths and surfaced by CLI/API.

Core receipt types and JSONL I/O live in ``brain_os.contracts.write_receipt`` so
``brain_os.memory`` never imports this module. Mem0 replay is registered from
``brain_os.memory.write_replay``.

Replay handlers (async) are registered per ``(store, operation)`` key. Failed
replays increment ``attempt_count``; after ``max_attempts`` the row is appended
to ``write_contract_retry_dead_letter.jsonl`` and removed from the queue.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import httpx

from brain_os.contracts import write_receipt as _wr
from brain_os.exceptions import DatabaseError, IraError
from brain_os.memory.write_replay import install_mem0_write_replay_handler

logger = logging.getLogger(__name__)

_BUILTIN_HANDLERS_INSTALLED = False

# Re-export contract API (backward compatible imports from ``brain_os.brain.write_contract``).
WriteReceipt = _wr.WriteReceipt
ReplayHandler = _wr.ReplayHandler
build_receipt = _wr.build_receipt
record_receipt = _wr.record_receipt
enqueue_retry = _wr.enqueue_retry
register_write_replay_handler = _wr.register_write_replay_handler
unregister_write_replay_handler = _wr.unregister_write_replay_handler
recent_receipts = _wr.recent_receipts
summarize_recent = _wr.summarize_recent
append_dead_letter = _wr.append_dead_letter

# Tests patch ``brain_os.brain.write_contract.get_data_dir``.


async def _replay_qdrant_pipeline_turn_summary(row: dict[str, Any]) -> bool:
    meta = dict(row.get("metadata") or {})
    if not meta.get("replay_turn_summary"):
        return False
    email = str(meta.get("contact_email") or "").strip()
    rid = str(row.get("run_id") or "").strip()
    user_prev = meta.get("turn_user_preview") or ""
    asst_prev = meta.get("turn_assistant_preview") or ""
    if not email or not rid:
        return False
    text = meta.get("turn_content_full")
    if not text:
        text = f"User: {str(user_prev)[:1200]}\nAssistant: {str(asst_prev)[:1200]}"
    try:
        point_id = uuid.uuid5(uuid.NAMESPACE_URL, f"conversation_turn::{email}::{rid}")
    except ValueError:
        return False

    from brain_os.brain.embeddings import EmbeddingService
    from brain_os.brain.qdrant_manager import QdrantManager
    from brain_os.data.models import KnowledgeItem

    embedding = EmbeddingService()
    qm = QdrantManager(embedding_service=embedding)
    item = KnowledgeItem(
        id=point_id,
        source=f"conversation:turn:{email}",
        source_category="conversation_turn",
        content=str(text),
        metadata={
            "contact_email": email,
            "run_id": rid,
            "replay": True,
        },
    )
    try:
        n = await qm.upsert_items([item])
        return int(n or 0) > 0
    except (DatabaseError, IraError, httpx.HTTPError, OSError, ValueError, TypeError) as exc:
        logger.warning("Replay Qdrant pipeline.turn_summary failed", exc_info=True)
        return False
    finally:
        try:
            await qm.close()
        except (OSError, RuntimeError) as exc:
            logger.debug("QdrantManager close failed after replay", exc_info=True)


async def _replay_neo_pipeline_turn_relationship(row: dict[str, Any]) -> bool:
    meta = dict(row.get("metadata") or {})
    if not meta.get("replay_turn_relationship"):
        return False
    email = str(meta.get("contact_email") or "").strip()
    rid = str(row.get("run_id") or "").strip()
    route = str(meta.get("route_method") or "")
    if not email:
        return False
    from brain_os.brain.knowledge_graph import KnowledgeGraph

    graph = KnowledgeGraph()
    try:
        await graph.add_person(
            name=email.split("@")[0] if "@" in email else email,
            email=email,
            company_name="",
            role="contact",
            source_id=f"write_contract::{rid}",
        )
        await graph.add_application(
            name="ConversationTurn",
            description="General conversation turns for write contract",
        )
        ok = await graph.add_relationship(
            from_type="Person",
            from_key=email,
            rel_type="REFERS_TO",
            to_type="Application",
            to_key="ConversationTurn",
            properties={"run_id": rid, "route_method": route},
            source_id=f"write_contract::{rid}",
        )
        return bool(ok)
    except DatabaseError as exc:
        logger.warning("Replay Neo4j pipeline.turn_relationship failed", exc_info=True)
        return False
    finally:
        try:
            await graph.close()
        except (OSError, RuntimeError) as exc:
            logger.debug("KnowledgeGraph close failed after replay", exc_info=True)


def _install_builtin_handlers() -> None:
    global _BUILTIN_HANDLERS_INSTALLED
    if _BUILTIN_HANDLERS_INSTALLED:
        return
    install_mem0_write_replay_handler()
    register_write_replay_handler(
        "qdrant", "pipeline.turn_summary", _replay_qdrant_pipeline_turn_summary
    )
    register_write_replay_handler(
        "neo4j", "pipeline.turn_relationship", _replay_neo_pipeline_turn_relationship
    )
    _BUILTIN_HANDLERS_INSTALLED = True


# Exposed for tests that restore the default mem0 handler after patching.
async def _replay_mem0_store(row: dict[str, Any]) -> bool:
    from brain_os.memory.write_replay import replay_mem0_store

    return await replay_mem0_store(row)


_install_builtin_handlers()


async def replay_write_contract_retry_queue(
    *,
    max_to_process: int = 50,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run replay handlers against queued rows until ``max_to_process`` attempts are consumed."""

    rows = _wr.read_retry_queue()
    if not rows:
        return {
            "processed": 0,
            "remaining": 0,
            "replay_succeeded": 0,
            "replay_failed": 0,
            "dead_lettered": 0,
            "skipped_budget": 0,
            "dry_run": dry_run,
        }

    attempt_budget = max(0, int(max_to_process))
    out_queue: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "processed": 0,
        "remaining": 0,
        "replay_succeeded": 0,
        "replay_failed": 0,
        "dead_lettered": 0,
        "skipped_budget": 0,
        "dry_run": dry_run,
    }

    _install_builtin_handlers()

    for row in rows:
        store = str(row.get("store") or "")
        op = str(row.get("operation") or "")
        attempts = int(row.get("attempt_count") or 0)
        max_attempts = max(1, int(row.get("max_attempts") or 3))

        if attempts >= max_attempts:
            row.setdefault("terminal_reason", "already_exhausted_in_queue")
            append_dead_letter(row)
            stats["dead_lettered"] += 1
            continue

        if attempt_budget <= 0:
            stats["skipped_budget"] += 1
            out_queue.append(row)
            continue

        stats["processed"] += 1
        attempt_budget -= 1

        if dry_run:
            out_queue.append(row)
            continue

        handler = _wr.get_write_replay_handler(store, op)
        ok = False
        if handler is not None:
            try:
                ok = bool(await handler(row))
            except (
                Exception
            ):  # intentional — replay handlers are plugin-defined; must not abort queue
                logger.exception("Replay handler crashed for store=%s op=%s", store, op)
                ok = False
        else:
            logger.debug(
                "No replay handler for store=%s op=%s — counting as failed replay", store, op
            )

        if ok:
            stats["replay_succeeded"] += 1
            continue

        attempts += 1
        row["attempt_count"] = attempts
        row["last_attempt_at"] = _wr._utcnow_iso()

        stats["replay_failed"] += 1

        if attempts >= max_attempts:
            row["terminal_reason"] = (
                row.get("terminal_reason") or "max_attempts_replay_failed_or_no_handler"
            )
            append_dead_letter(row)
            stats["dead_lettered"] += 1
        else:
            out_queue.append(row)

    _wr.write_retry_queue(out_queue)

    stats["remaining"] = len(out_queue)
    return stats


def retry_failed_receipts(*, max_to_process: int = 50) -> dict[str, Any]:
    """Synchronous helper: replay queued rows using :func:`replay_write_contract_retry_queue`."""

    return asyncio.run(
        replay_write_contract_retry_queue(max_to_process=max_to_process, dry_run=False),
    )
