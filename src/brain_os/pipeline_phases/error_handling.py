"""Phase: error handling helpers for RequestPipeline entry/finalization paths.

These helpers are intentionally side-effect focused and dependency-light so
``brain_os.pipeline`` can delegate timeout/finalization boilerplate without changing
behavior.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from brain_os.brain.run_record_assembler import RunRecordAssemblyContext
from brain_os.pipeline_phases.compile import schedule_finalize_run_record


def build_pipeline_timeout_response(
    *,
    timeout_seconds: int,
    sender_id: str,
    metadata: dict[str, Any] | None,
    logger: logging.Logger,
) -> tuple[str, list[str], str]:
    """Build the standard timeout triple and preserve/inject ``pipeline_run_id``."""
    timeout_run_id = str(uuid.uuid4())
    if isinstance(metadata, dict):
        metadata.setdefault("pipeline_run_id", timeout_run_id)
        timeout_run_id = str(metadata.get("pipeline_run_id") or timeout_run_id)
    logger.error(
        "Pipeline timed out after %ds for sender=%s",
        timeout_seconds,
        sender_id,
    )
    msg = (
        f"I'm sorry, the request timed out after {timeout_seconds} seconds. "
        "Please try a simpler question or ask about one topic at a time."
    )
    if isinstance(metadata, dict):
        trace = {
            "run_id": timeout_run_id,
            "early_exit": "timeout",
            "agents": ["timeout"],
        }
        schedule_finalize_run_record(
            meta=metadata,
            ctx=RunRecordAssemblyContext(
                run_id=timeout_run_id,
                channel=str(metadata.get("channel") or "API"),
                sender_id=sender_id,
                ts_start=float(metadata.get("_run_record_ts_start") or 0.0) or time.time(),
                trace=trace,
                meta=metadata,
                agents_used=["timeout"],
                raw_input=str(metadata.get("_run_record_raw_input") or "")[:2000],
                response_text=msg,
            ),
        )
    return msg, ["timeout"], timeout_run_id


def finalize_pipeline_scope(
    *,
    metadata: dict[str, Any],
    trace: dict[str, Any],
    attach_pipeline_trace: Callable[[dict[str, Any], dict[str, Any]], None],
    mem0_scope_token: Any,
    reset_mem0_scope: Callable[[Any], None],
    logger: logging.Logger,
    run_record_fallback: RunRecordAssemblyContext | None = None,
) -> None:
    """Attach trace defensively and always reset Mem0 scope token."""
    try:
        attach_pipeline_trace(metadata, trace)
    except Exception:
        logger.debug("pipeline_trace finalize failed", exc_info=True)
    if run_record_fallback is not None and not metadata.get("_run_record_persist_scheduled"):
        schedule_finalize_run_record(meta=metadata, ctx=run_record_fallback)
    reset_mem0_scope(mem0_scope_token)


def start_background_task_with_lifecycle(
    *,
    run_id: str,
    task_coro_factory: Callable[[], Any],
    record_started: Callable[[str], None],
    record_finished: Callable[[str, BaseException | None], None],
    logger: logging.Logger,
) -> asyncio.Task[None]:
    """Create a background task and wire started/finished lifecycle callbacks."""
    record_started(run_id)
    task: asyncio.Task[None] = asyncio.create_task(task_coro_factory())

    def _done_cb(done_task: asyncio.Task[None], _rid: str = run_id) -> None:
        err: BaseException | None = None
        try:
            if done_task.cancelled():
                err = None
            else:
                err = done_task.exception()
        except Exception:  # pragma: no cover — defensive
            err = None
        if err is not None:
            logger.warning("LEARN background task failed: %s", err)
        record_finished(_rid, err)

    task.add_done_callback(_done_cb)
    return task


async def finalize_response_with_dedup_cache(
    *,
    recent_messages: dict[str, tuple[Any, ...]],
    fingerprint: str,
    now_epoch_s: float,
    shaped: str,
    agents_used: list[str],
    run_id: str,
    redis_cache: Any | None,
    redis_key: str,
    encode_dedup_payload: Callable[[str, list[str], str], str],
) -> tuple[str, list[str], str]:
    """Persist in-memory/Redis dedup artifacts and return pipeline response triple."""
    recent_messages[fingerprint] = (shaped, now_epoch_s, list(agents_used), run_id)
    if redis_cache is not None and redis_cache.available:
        await redis_cache.dedup_store(
            redis_key,
            encode_dedup_payload(shaped, agents_used, run_id),
            ttl_seconds=300,
        )
    return shaped, agents_used, run_id


def unpack_inproc_dedup_entry(entry: tuple[Any, ...]) -> tuple[str, list[str], str]:
    """Decode in-proc dedup tuple shape (legacy-safe) into response/agents/run_id."""
    cached_resp = str(entry[0]) if len(entry) > 0 else ""
    cached_agents: list[str] = (
        list(entry[2]) if len(entry) > 2 and isinstance(entry[2], list) else []
    )
    cached_run_id = str(entry[3]) if len(entry) > 3 and entry[3] else ""
    return cached_resp, cached_agents, cached_run_id


def record_dedup_hit_metadata(
    *,
    meta: dict[str, Any],
    trace: dict[str, Any],
    hit: str,
    original_run_id: str,
    original_agents_used: list[str],
) -> None:
    """Record standardized dedup-hit metadata into trace and envelope metadata."""
    trace["early_exit"] = f"{hit}_dedup_hit"
    meta["dedup"] = {
        "hit": hit,
        "original_run_id": original_run_id,
        "original_agents_used": list(original_agents_used),
    }


async def complete_short_circuit_response(
    *,
    route_label: str,
    contact_email: str,
    started_at: float,
    logger: logging.Logger,
    learn_fn: Callable[[], Awaitable[None]],
    push_timings_fn: Callable[[], None],
    finalize_response_fn: Callable[[], Awaitable[tuple[str, list[str], str]]],
    record_stage_fn: Callable[[str], None],
    graphe_log: dict[str, Any] | None = None,
) -> tuple[str, list[str], str]:
    """Finalize early-exit route with learn timing and dedup persistence."""
    if graphe_log:
        from brain_os.brain.graphe_instrumentation import log_graphe_pipeline_turn

        await log_graphe_pipeline_turn(**graphe_log)
    record_stage_fn("route")
    record_stage_fn("shape")
    await learn_fn()
    record_stage_fn("learn")
    push_timings_fn()
    elapsed_ms = (time.monotonic() - started_at) * 1000
    logger.info("RETURN (%s) | %s | %.0fms", route_label, contact_email, elapsed_ms)
    return await finalize_response_fn()


__all__ = [
    "build_pipeline_timeout_response",
    "complete_short_circuit_response",
    "finalize_pipeline_scope",
    "finalize_response_with_dedup_cache",
    "record_dedup_hit_metadata",
    "start_background_task_with_lifecycle",
    "unpack_inproc_dedup_entry",
]
