"""Assemble :class:`~brain_os.schemas.run_record.RunRecord` from pipeline context."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from brain_os.config import get_settings
from brain_os.schemas.run_record import (
    _SUMMARY_INPUT_MAX,
    _SUMMARY_RESPONSE_MAX,
    CostEstimate,
    DelegationHop,
    EvidenceRef,
    QualityInfo,
    RouteInfo,
    RunOutcome,
    RunRecord,
    RunRecordArtifacts,
    SafetyFlags,
    ToolInvocationRef,
    truncate_summary,
)
from brain_os.systems.llm_budget import estimate_turn_tokens

logger = logging.getLogger(__name__)

_EMAIL_TOOLS = frozenset({"search_emails", "read_email_thread"})


@dataclass
class RunRecordAssemblyContext:
    """Inputs collected at pipeline exit for run-record assembly."""

    run_id: str
    channel: str
    sender_id: str
    ts_start: float
    trace: dict[str, Any]
    meta: dict[str, Any]
    agents_used: list[str]
    raw_input: str
    response_text: str
    tool_audit: list[dict[str, Any]] | None = None
    shared_kb_evidence: list[dict[str, Any]] | None = None
    learning_meta: dict[str, Any] | None = None
    tool_stats_tracker: Any | None = None
    stage_timings: dict[str, float] | None = None
    email_scope: str | None = None
    faithfulness_tier: str | None = None
    caveat_appended: bool = False


def _redact_identifier(value: str, *, prefix: str = "") -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if not get_settings().app.run_record_redact_pii:
        return raw[:256]
    digest = hashlib.sha256(f"{prefix}:{raw}".encode()).hexdigest()[:12]
    return f"redacted:{digest}"


def _resolve_outcome(
    *,
    trace: dict[str, Any],
    agents_used: list[str],
) -> tuple[RunOutcome, str | None]:
    early = trace.get("early_exit")
    early_str = str(early).strip() if early is not None else None
    agents = [str(a).strip().lower() for a in (agents_used or []) if a]
    if agents == ["timeout"]:
        return "timeout", early_str
    if agents == ["budget"]:
        return "budget_gate", early_str
    if early_str:
        return "early_exit", early_str
    return "ok", None


def _normalize_tools(tool_audit: list[dict[str, Any]] | None) -> list[ToolInvocationRef]:
    out: list[ToolInvocationRef] = []
    for row in tool_audit or []:
        if not isinstance(row, dict):
            continue
        tool = str(row.get("tool") or "").strip()
        if not tool:
            continue
        ok_val = row.get("ok")
        if ok_val is None:
            ok_val = row.get("success", True)
        success = bool(ok_val) if ok_val is not None else True
        dur = row.get("duration_ms")
        duration_ms = int(dur) if dur is not None else None
        err = row.get("error_code")
        out.append(
            ToolInvocationRef(
                agent=str(row.get("agent") or row.get("agent_name") or "").strip()[:128],
                tool=tool[:256],
                success=success,
                duration_ms=duration_ms,
                error_code=str(err)[:128] if err else None,
            )
        )
    return out


def _collect_evidence(
    *,
    shared_kb_evidence: list[dict[str, Any]] | None,
    tool_audit: list[dict[str, Any]] | None,
) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    seen: set[tuple[str, str]] = set()
    for row in shared_kb_evidence or []:
        if not isinstance(row, dict):
            continue
        ref = str(
            row.get("id") or row.get("doc_id") or row.get("source") or row.get("document_id") or ""
        ).strip()
        if not ref:
            continue
        key = ("kb", ref)
        if key in seen:
            continue
        seen.add(key)
        score_raw = row.get("score")
        score = float(score_raw) if score_raw is not None else None
        refs.append(EvidenceRef(kind="kb", ref=ref[:256], score=score))
        if len(refs) >= 25:
            break
    for row in tool_audit or []:
        if not isinstance(row, dict):
            continue
        tool = str(row.get("tool") or "")
        if tool not in _EMAIL_TOOLS:
            continue
        tid = str(row.get("thread_id") or row.get("thread") or "").strip()
        if not tid:
            result = row.get("result")
            if isinstance(result, dict):
                tid = str(result.get("thread_id") or "").strip()
        if not tid:
            continue
        key = ("email_thread", tid)
        if key in seen:
            continue
        seen.add(key)
        refs.append(EvidenceRef(kind="email_thread", ref=tid[:128]))
    return refs


async def _load_delegations(
    tracker: Any | None,
    run_id: str,
) -> list[DelegationHop]:
    if tracker is None or not hasattr(tracker, "get_recent_delegations"):
        return []
    try:
        rows = await tracker.get_recent_delegations(run_id=run_id)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        logger.debug("run_record: delegation lookup failed", exc_info=True)
        return []
    out: list[DelegationHop] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            DelegationHop.model_validate(
                {
                    "from": row.get("from") or row.get("from_agent") or "",
                    "to": row.get("to") or row.get("to_agent") or "",
                    "duration_ms": int(row.get("duration_ms") or 0),
                    "ok": bool(row.get("ok", True)),
                }
            )
        )
    return out


async def assemble_from_pipeline(ctx: RunRecordAssemblyContext) -> RunRecord:
    """Build a :class:`RunRecord` from pipeline exit context."""
    ts_end = time.time()
    duration_ms = max(0, int((ts_end - float(ctx.ts_start)) * 1000))
    trace = ctx.trace if isinstance(ctx.trace, dict) else {}
    meta = ctx.meta if isinstance(ctx.meta, dict) else {}
    agents_used = list(ctx.agents_used or [])
    outcome, early_exit = _resolve_outcome(trace=trace, agents_used=agents_used)

    route_method = str(trace.get("route") or "")
    rte = trace.get("routing_telemetry")
    telemetry = dict(rte) if isinstance(rte, dict) else {}

    learning = ctx.learning_meta if isinstance(ctx.learning_meta, dict) else {}
    confidence_raw = trace.get("confidence")
    confidence = float(confidence_raw) if confidence_raw is not None else None
    metis_score = learning.get("metis_score")
    metis_int = int(metis_score) if metis_score is not None else None

    delegations = await _load_delegations(ctx.tool_stats_tracker, ctx.run_id)
    tools = _normalize_tools(ctx.tool_audit)
    evidence = _collect_evidence(
        shared_kb_evidence=ctx.shared_kb_evidence,
        tool_audit=ctx.tool_audit,
    )

    tokens = estimate_turn_tokens(
        ctx.raw_input,
        ctx.response_text,
        agents_used,
    )

    dedup = meta.get("dedup")
    artifacts = RunRecordArtifacts(
        dedup=dict(dedup) if isinstance(dedup, dict) else None,
    )

    transitions_raw = trace.get("transitions")
    transitions = dict(transitions_raw) if isinstance(transitions_raw, dict) else {}

    sender = _redact_identifier(ctx.sender_id, prefix="sender")

    return RunRecord(
        run_id=ctx.run_id,
        ts_start=float(ctx.ts_start),
        ts_end=ts_end,
        duration_ms=duration_ms,
        channel=(ctx.channel or "")[:64],
        sender_id=sender,
        outcome=outcome,
        early_exit=early_exit,
        input_summary=truncate_summary(ctx.raw_input, max_len=_SUMMARY_INPUT_MAX),
        response_summary=truncate_summary(ctx.response_text, max_len=_SUMMARY_RESPONSE_MAX),
        route=RouteInfo(method=route_method, telemetry=telemetry),
        email_scope=(ctx.email_scope or None),
        agents=agents_used,
        delegations=delegations,
        tools=tools,
        evidence=evidence,
        safety=SafetyFlags(
            had_provenance_warning=bool(learning.get("had_provenance_warning")),
            had_dlp_flag=bool(learning.get("had_dlp_flag")),
            faithfulness_tier=ctx.faithfulness_tier,
            caveat_appended=ctx.caveat_appended,
        ),
        quality=QualityInfo(
            confidence=confidence,
            metis_score=metis_int,
            stage_timings=dict(ctx.stage_timings or {}),
        ),
        cost_estimate=CostEstimate(tokens_heuristic=tokens),
        artifacts=artifacts,
        transitions=transitions,
    )
