"""Phase: compile. Extracted from pipeline.py 2026-05-15.

Phase modules MUST NOT import from brain_os.pipeline (circular). Shared helpers
live in ira.pipeline_runtime; import from there when this slice needs them.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from brain_os.brain.run_record_assembler import RunRecordAssemblyContext, assemble_from_pipeline
from brain_os.brain.run_record_store import RunRecordStore
from brain_os.data.models import Contact

logger = logging.getLogger(__name__)

_RUN_RECORD_STORE: RunRecordStore | None = None


def _get_run_record_store() -> RunRecordStore:
    global _RUN_RECORD_STORE
    if _RUN_RECORD_STORE is None:
        _RUN_RECORD_STORE = RunRecordStore()
    return _RUN_RECORD_STORE


def _reset_run_record_store_cache() -> None:
    """Clear lazy store singleton (tests only)."""
    global _RUN_RECORD_STORE
    _RUN_RECORD_STORE = None


async def finalize_run_record(ctx: RunRecordAssemblyContext) -> None:
    """Assemble and persist one run record (fail-open)."""
    try:
        from brain_os.config import get_settings

        if not get_settings().app.run_record_enabled:
            return
        record = await assemble_from_pipeline(ctx)
        store = _get_run_record_store()
        await store.save(record)
        from brain_os.brain.context_graph_sync import sync_pipeline_run

        await sync_pipeline_run(record)
    except Exception:
        logger.debug("finalize_run_record failed", exc_info=True)


def schedule_exit_run_record(
    *,
    meta: dict[str, Any],
    run_id: str,
    channel: str,
    sender_id: str,
    ts_start: float,
    trace: dict[str, Any],
    agents_used: list[str],
    raw_input: str,
    response_text: str,
    tool_stats_tracker: Any | None = None,
) -> None:
    """Schedule a minimal run record for early-exit paths."""
    schedule_finalize_run_record(
        meta=meta,
        ctx=RunRecordAssemblyContext(
            run_id=run_id,
            channel=channel,
            sender_id=sender_id,
            ts_start=ts_start,
            trace=trace,
            meta=meta,
            agents_used=list(agents_used),
            raw_input=raw_input,
            response_text=response_text,
            tool_stats_tracker=tool_stats_tracker,
        ),
    )


def schedule_finalize_run_record(
    *,
    meta: dict[str, Any],
    ctx: RunRecordAssemblyContext,
) -> None:
    """Schedule background persist; idempotent per request metadata."""
    try:
        from brain_os.config import get_settings as _gs_schedule_rr

        if not _gs_schedule_rr().app.run_record_enabled:
            return
        if meta.get("_run_record_persist_scheduled"):
            return
        meta["_run_record_persist_scheduled"] = True

        task = asyncio.create_task(finalize_run_record(ctx))

        def _done(done_task: asyncio.Task[None]) -> None:
            try:
                if done_task.cancelled():
                    return
                err = done_task.exception()
                if err is not None:
                    logger.warning("run_record background persist failed: %s", err)
            except Exception:
                logger.debug("run_record done callback failed", exc_info=True)

        task.add_done_callback(_done)
    except Exception:
        logger.debug("schedule_finalize_run_record failed", exc_info=True)


def format_source_limitation_note(
    agents_used: list[str],
    raw_response: str,
) -> tuple[str, list[str]]:
    """Return ``(markdown_suffix, limitation_reasons)`` for partial-answer disclosure."""
    limitations: list[str] = []
    for au in agents_used:
        if f"Agent '{au}' timed out" in raw_response:
            limitations.append(f"{au} timed out")
        if f"Agent '{au}' encountered an error" in raw_response:
            limitations.append(f"{au} failed")
    if "timeout" in agents_used:
        limitations.append("pipeline timed out")
    if not limitations:
        return "", []
    note = (
        "\n\n> **Note:** " + "; ".join(limitations) + ". This answer may be based on partial data."
    )
    return note, limitations


def format_metis_stability_note(metis_result: dict[str, Any] | None) -> str:
    """Appendix text from Metis stability scoring, or empty string."""
    if not metis_result:
        return ""
    if metis_result.get("should_announce_stable"):
        return (
            f"\n\n---\n**Metis (Stability Monitor):** I think we are stable now "
            f"at max_rounds={metis_result['max_rounds']} "
            f"(rolling avg: {metis_result['rolling_avg']}/100). Confirm?"
        )
    if metis_result.get("should_report"):
        return (
            f"\n\n---\n*Stability score: {metis_result['score']}/100 "
            f"(avg: {metis_result['rolling_avg']}/100, "
            f"max_rounds: {metis_result['max_rounds']})*"
        )
    return ""


async def build_execution_summary_block(
    tool_stats_tracker: Any,
    *,
    run_id: str,
    route_method: str,
    agents_used: list[str],
    email_scope: str,
    shared_kb_evidence: list[dict[str, Any]],
    tool_audit: list[dict[str, Any]],
) -> str:
    """Build a compact Markdown block summarizing how the answer was produced."""
    delegations: list[dict[str, Any]] = []
    tracker = tool_stats_tracker
    if tracker is not None and hasattr(tracker, "get_recent_delegations"):
        try:
            delegations = await tracker.get_recent_delegations(run_id=run_id)
        except Exception:
            logger.debug("execution summary: delegation lookup failed", exc_info=True)

    delegation_ok = sum(1 for d in delegations if bool(d.get("ok")))
    delegation_failed = max(0, len(delegations) - delegation_ok)
    kb_docs = len([r for r in shared_kb_evidence if r.get("content")])
    email_tools_used = sum(
        1
        for row in (tool_audit or [])
        if str(row.get("tool") or "") in {"search_emails", "read_email_thread"}
    )
    lines = [
        "\n\n---\n**Agentic execution summary**",
        f"- Route: `{route_method}`",
        f"- Agents consulted: `{', '.join(agents_used) if agents_used else 'none'}`",
        f"- Delegations: `{len(delegations)}` (ok: `{delegation_ok}`, failed: `{delegation_failed}`)",
        f"- Evidence signals: kb_docs=`{kb_docs}`, email_scope=`{email_scope}`, email_tools_used=`{email_tools_used}`",
    ]
    return "\n".join(lines)


async def run_reflection_step(
    *,
    inner_voice: Any,
    contact_email: str,
    channel: str,
    raw_input: str,
    raw_response: str,
    logger: logging.Logger,
) -> str:
    """Run InnerVoice reflection; return surfaced markdown suffix or empty string."""
    if inner_voice is None:
        return ""
    try:
        reflection = await inner_voice.reflect(
            context=f"User ({contact_email}) on {channel}: {raw_input}",
            trigger=raw_response[:500],
        )
        if reflection.get("should_surface") and reflection.get("content"):
            return f"\n\n_{reflection['content']}_"
    except Exception:
        logger.exception("InnerVoice reflection failed")
    return ""


def apply_source_limitation_note(
    *,
    raw_response: str,
    agents_used: list[str],
    logger: logging.Logger,
) -> str:
    """Append source limitation disclosure (if any) and return updated response."""
    from brain_os.knowledge.teacher_provenance import format_maestro_pipeline_footer

    lim_note, limitations = format_source_limitation_note(agents_used, raw_response)
    if lim_note:
        raw_response += lim_note
        logger.info("SOURCE LIMITATION | %s", limitations)
    maestro_footer = format_maestro_pipeline_footer(agents_used)
    if maestro_footer and maestro_footer.strip() not in raw_response:
        raw_response += maestro_footer
    return raw_response


async def shape_pipeline_response(
    *,
    email_gold_eval_json: str | None,
    confidence_prefix: str,
    raw_response: str,
    reflection_text: str,
    contact_info: dict[str, Any],
    contact_email: str,
    channel: str,
    voice: Any,
    endocrine: Any | None,
    logger: logging.Logger,
    record_shape_stage: Callable[[], None],
) -> str:
    """Build final shaped response for step 9, preserving gold-eval bypass behavior."""
    if email_gold_eval_json is not None:
        shaped = email_gold_eval_json
        record_shape_stage()
        logger.info("SHAPE | skipped voice (IRA_EMAIL_GOLD_EVAL_v1) len=%d", len(shaped))
        return shaped

    full_response = confidence_prefix + raw_response + reflection_text
    recipient = Contact(
        name=contact_info.get("name", ""),
        email=contact_email,
        company=contact_info.get("company"),
        region=contact_info.get("region"),
        source="pipeline",
    )

    modifiers = {}
    if endocrine is not None:
        try:
            modifiers = endocrine.get_behavioral_modifiers()
        except Exception:
            logger.exception("Endocrine modifiers failed")

    shaped = await voice.shape_response(
        full_response,
        channel,
        recipient=recipient,
        behavioral_modifiers=modifiers,
    )
    record_shape_stage()
    logger.info("SHAPE | channel=%s len=%d", channel, len(shaped))
    return shaped


async def apply_metis_stability_note(
    *,
    shaped: str,
    pantheon: Any,
    agents_used: list[str],
    raw_response: str,
    logger: logging.Logger,
) -> tuple[str, dict[str, Any] | None]:
    """Run Metis stability scoring and append any user-facing note."""
    metis = pantheon.get_agent("metis")
    metis_result: dict[str, Any] | None = None
    if metis is not None:
        try:
            metis_result = await metis.score_and_track(agents_used, raw_response)
        except Exception:
            logger.debug("Metis scoring failed", exc_info=True)
    metis_note = format_metis_stability_note(metis_result)
    if metis_note:
        shaped += metis_note
    return shaped, metis_result


async def append_post_shape_artifacts(
    *,
    shaped: str,
    pantheon: Any,
    raw_input: str,
    raw_response: str,
    agents_used: list[str],
    run_id: str,
    channel: str,
    route_method: str,
    email_scope: str,
    contact_email: str,
    goal_lineage_payload: dict[str, Any] | None,
    learning_meta: dict[str, Any],
    tool_audit: list[dict[str, Any]],
    email_gold_eval_json: str | None,
    tool_stats_tracker: Any,
    shared_kb_evidence: list[dict[str, Any]],
    logger: logging.Logger,
    meta: dict[str, Any] | None = None,
    trace: dict[str, Any] | None = None,
    ts_start: float | None = None,
    stage_timings: dict[str, float] | None = None,
    sender_id: str = "",
) -> str:
    """Append Graphe logging + optional execution summary to shaped response."""
    graphe = pantheon.get_agent("graphe")
    if graphe is not None:
        try:
            await graphe.log_turn(
                query=raw_input,
                agents_used=agents_used,
                response_summary=raw_response[:300],
                tool_calls=tool_audit,
                run_id=run_id,
                work_context={
                    "channel": channel,
                    "route_method": route_method,
                    "email_scope": email_scope,
                    "contact_email": (contact_email or "")[:120],
                    "goal_lineage": goal_lineage_payload,
                },
                learning_meta=learning_meta,
            )
        except Exception:
            logger.debug("Graphe session logging failed", exc_info=True)

    try:
        from brain_os.config import get_settings as _gs_exec_summary

        if (
            email_gold_eval_json is None
            and _gs_exec_summary().app.response_include_execution_summary
        ):
            shaped += await build_execution_summary_block(
                tool_stats_tracker,
                run_id=run_id,
                route_method=route_method,
                agents_used=agents_used,
                email_scope=email_scope,
                shared_kb_evidence=shared_kb_evidence,
                tool_audit=tool_audit,
            )
    except Exception:
        logger.debug("Execution summary append failed", exc_info=True)

    if meta is not None and trace is not None and ts_start is not None:
        schedule_finalize_run_record(
            meta=meta,
            ctx=RunRecordAssemblyContext(
                run_id=run_id,
                channel=channel,
                sender_id=sender_id or contact_email,
                ts_start=ts_start,
                trace=trace,
                meta=meta,
                agents_used=list(agents_used),
                raw_input=raw_input,
                response_text=shaped,
                tool_audit=tool_audit,
                shared_kb_evidence=shared_kb_evidence,
                learning_meta=learning_meta,
                tool_stats_tracker=tool_stats_tracker,
                stage_timings=stage_timings,
                email_scope=email_scope or None,
            ),
        )
    return shaped


__all__ = [
    "_reset_run_record_store_cache",
    "append_post_shape_artifacts",
    "apply_metis_stability_note",
    "apply_source_limitation_note",
    "build_execution_summary_block",
    "finalize_run_record",
    "format_metis_stability_note",
    "format_source_limitation_note",
    "run_reflection_step",
    "schedule_exit_run_record",
    "schedule_finalize_run_record",
    "shape_pipeline_response",
]
