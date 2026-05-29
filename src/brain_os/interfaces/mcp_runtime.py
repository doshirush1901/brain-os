"""Shared runtime for the MCP server surface.

Holds the FastMCP instance, lazy-init globals, bootstrap helpers, and
cross-tool utilities. Tool modules under ``brain_os.interfaces.mcp_tools.*``
import from here; ``brain_os.interfaces.mcp_server`` re-exports a stable subset
for test monkeypatches and CLI callers.

Lifecycle: ``_ensure_initialized()`` builds the global services on first
tool invocation. Subsequent calls are no-ops behind ``_init_lock``.

This module mirrors ``brain_os.interfaces.server_runtime`` — the W2 closure pattern.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from brain_os.brain.tool_discovery import rank_mcp_tools
from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool
from brain_os.pipeline_loop import AgentLoop
from brain_os.services.resilience import RetryPolicy

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "ira",
    instructions=(
        "Ira is Machinecraft's multi-agent assistant for thermoforming machinery sales & operations "
        "(industrial OEM). Tools query Ira's pipeline, knowledge base, CRM, Gmail (when configured), "
        "memory, graph, web search, and specialist agents. "
        "**Never expose this MCP server to untrusted networks** — Gmail/CRM/memory-write tools can leak or mutate data."
    ),
)

# ── Service globals (populated by _ensure_initialized) ──────────────────

_pantheon: Any = None
_shared_services: dict[str, Any] = {}
_pipeline: Any = None
_retriever: Any = None
_crm: Any = None
_ingestor: Any = None
_long_term_memory: Any = None
_conversation_memory: Any = None
_relationship_memory: Any = None
_goal_manager: Any = None
_knowledge_graph: Any = None
_email_processor: Any = None
_tinder_email_mode: Any = None
_task_orchestrator: Any = None
_agent_loop: AgentLoop | None = None
_initialized = False
_init_lock: asyncio.Lock | None = None

_SHORT_FACTUAL_QUERY_PATTERN = re.compile(
    r"\b(lead\s*time|specs?|what\s+is|machine|model|delivery|turnaround|timeline)\b",
    re.IGNORECASE,
)
_VALIDATOR_MODES = {"strict", "crm_only", "relaxed_evidence"}
_AGENT_LOOP_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay_seconds=0.8,
    max_delay_seconds=8.0,
)
_MCP_DOCAI_MAX_BYTES = 40 * 1024 * 1024


def _mcp_query_sender_id(user_id: str | None) -> str:
    """Resolve pipeline ``sender_id`` for MCP ``query_ira``."""
    raw = (user_id or "").strip()
    if raw:
        return raw
    from brain_os.config import get_settings

    fb = (get_settings().app.default_user_id or "").strip()
    return fb or "mcp_user"


def _ensure_init_lock() -> asyncio.Lock:
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


def _all_mcp_tool_names() -> list[str]:
    """Return registered MCP tool names in stable sorted order."""
    return sorted(mcp._tool_manager._tools.keys())


def _rank_tools_for_query(query: str) -> list[str]:
    """Rank MCP tools using shared discovery heuristics."""
    return rank_mcp_tools(query, _all_mcp_tool_names())


def _normalize_validator_mode(raw: str, *, default: str = "strict") -> str:
    mode = (raw or "").strip().lower()
    if mode in _VALIDATOR_MODES:
        return mode
    return default


def _is_transient_agent_loop_error(exc: BaseException) -> bool:
    """True for bounded retries on plan/execute/report MCP tools."""
    if isinstance(
        exc, (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout, httpx.WriteError)
    ):
        return True
    if isinstance(exc, asyncio.TimeoutError):
        return True
    msg = str(exc).lower()
    needles = (
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "econnreset",
        "rate limit",
        "503",
        "502",
        "429",
    )
    return any(n in msg for n in needles)


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _loop_telemetry(*, tool: str, started_wall: str, t0: float) -> dict[str, Any]:
    return {
        "tool": tool,
        "started_at": started_wall,
        "finished_at": _utc_now_iso(),
        "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def _is_short_factual_query(query: str) -> bool:
    cleaned = (query or "").strip()
    if not cleaned:
        return False
    if len(cleaned) > 180:
        return False
    return bool(_SHORT_FACTUAL_QUERY_PATTERN.search(cleaned))


async def _quick_answer_core(
    question: str,
    *,
    max_snippets: int,
    min_score: float,
) -> tuple[str, bool, dict[str, Any]]:
    """Fast retrieval-only answer path for short factual MCP prompts."""
    from brain_os.interfaces import mcp_server as srv

    await _ensure_initialized()
    retriever = srv._retriever
    if retriever is None:
        return "Retriever not available.", False, {"reason": "retriever_unavailable"}

    try:
        limit = max(3, min(max_snippets * 2, 20))
        hits = await retriever.search(question, limit=limit)
        if not hits:
            return (
                "No strong knowledge hits found for this question. Try `query_ira` for a full synthesis.",
                False,
                {"reason": "no_hits", "hits": 0},
            )
        ranked = sorted(hits, key=lambda r: float(r.get("score", 0) or 0), reverse=True)
        shortlisted: list[dict[str, Any]] = []
        for hit in ranked:
            score = float(hit.get("score", 0) or 0)
            if score >= min_score:
                shortlisted.append(hit)
            if len(shortlisted) >= max_snippets:
                break
        top = ranked[: max(1, min(max_snippets, len(ranked)))]
        selected = shortlisted or top
        top_score = float(ranked[0].get("score", 0) or 0)
        confident = bool(shortlisted)
        lines = []
        if confident:
            lines.append(f"Quick answer (retrieval-first) for: {question}")
        else:
            lines.append("Quick answer (low confidence; escalated synthesis recommended):")
        for idx, hit in enumerate(selected, start=1):
            content = " ".join(str(hit.get("content", "")).split())[:220]
            source = str(hit.get("source", "") or "unknown")
            score = float(hit.get("score", 0) or 0)
            lines.append(f"{idx}. {content} [source={source}, score={score:.3f}]")
        if not confident:
            lines.append("Use `query_ira` if you need a fully verified, synthesized response.")
        return (
            "\n".join(lines),
            confident,
            {
                "reason": "ok",
                "hits": len(ranked),
                "top_score": round(top_score, 3),
                "confident": confident,
            },
        )
    except Exception as exc:
        logger.exception("MCP quick_answer failed")
        return f"Error: {exc}", False, {"reason": "error", "error": str(exc)}


def _hydrate_from_server_facade() -> None:
    """Pull handles from ``mcp_server`` when runtime globals were skipped (early-return path)."""
    global _pantheon, _shared_services, _pipeline, _retriever, _crm, _ingestor
    global _long_term_memory, _conversation_memory, _relationship_memory
    global _goal_manager, _knowledge_graph, _email_processor, _tinder_email_mode
    global _task_orchestrator, _agent_loop

    from brain_os.interfaces import mcp_server as srv

    if getattr(srv, "_pantheon", None) is not None:
        _pantheon = srv._pantheon
    if getattr(srv, "_shared_services", None):
        _shared_services = srv._shared_services
    if getattr(srv, "_pipeline", None) is not None:
        _pipeline = srv._pipeline
    if getattr(srv, "_retriever", None) is not None:
        _retriever = srv._retriever
    if getattr(srv, "_crm", None) is not None:
        _crm = srv._crm
    if getattr(srv, "_ingestor", None) is not None:
        _ingestor = srv._ingestor
    if getattr(srv, "_long_term_memory", None) is not None:
        _long_term_memory = srv._long_term_memory
    elif _pipeline is not None:
        _long_term_memory = getattr(_pipeline, "_long_term", None)
    if getattr(srv, "_conversation_memory", None) is not None:
        _conversation_memory = srv._conversation_memory
    if getattr(srv, "_relationship_memory", None) is not None:
        _relationship_memory = srv._relationship_memory
    if getattr(srv, "_goal_manager", None) is not None:
        _goal_manager = srv._goal_manager
    if getattr(srv, "_knowledge_graph", None) is not None:
        _knowledge_graph = srv._knowledge_graph
    if getattr(srv, "_email_processor", None) is not None:
        _email_processor = srv._email_processor
    if getattr(srv, "_tinder_email_mode", None) is not None:
        _tinder_email_mode = srv._tinder_email_mode
    if getattr(srv, "_task_orchestrator", None) is not None:
        _task_orchestrator = srv._task_orchestrator
    if getattr(srv, "_agent_loop", None) is not None:
        _agent_loop = srv._agent_loop


def _sync_mcp_server_facade() -> None:
    """Mirror runtime globals onto ``mcp_server`` (test patch surface + import stability)."""
    from brain_os.interfaces import mcp_server as srv

    srv._pantheon = _pantheon
    srv._shared_services = _shared_services
    srv._pipeline = _pipeline
    srv._retriever = _retriever
    srv._crm = _crm
    srv._ingestor = _ingestor
    srv._long_term_memory = _long_term_memory
    srv._conversation_memory = _conversation_memory
    srv._relationship_memory = _relationship_memory
    srv._goal_manager = _goal_manager
    srv._knowledge_graph = _knowledge_graph
    srv._email_processor = _email_processor
    srv._tinder_email_mode = _tinder_email_mode
    srv._task_orchestrator = _task_orchestrator
    srv._agent_loop = _agent_loop
    srv._initialized = _initialized


async def _ensure_initialized() -> None:
    """Lazy-init the Ira subsystems on first tool call."""
    global _pantheon, _shared_services, _pipeline, _retriever, _crm, _ingestor
    global _long_term_memory, _conversation_memory, _relationship_memory
    global _goal_manager, _knowledge_graph, _email_processor, _tinder_email_mode
    global _task_orchestrator, _agent_loop, _initialized
    if _initialized:
        return

    try:
        from brain_os.interfaces import mcp_server as srv

        if getattr(srv, "_initialized", False):
            _hydrate_from_server_facade()
            pipe = _pipeline or getattr(srv, "_pipeline", None)
            ltm = _long_term_memory or getattr(srv, "_long_term_memory", None)
            if ltm is None and pipe is not None:
                _long_term_memory = getattr(pipe, "_long_term", None)
            # Honor server facade shim (tests monkeypatch ``mcp_server._initialized``).
            _initialized = True
            _sync_mcp_server_facade()
            return
    except ImportError:
        pass

    async with _ensure_init_lock():
        if _initialized:
            return

        from brain_os.interfaces.cli_runtime import _build_pantheon, _build_pipeline
        from brain_os.service_keys import ServiceKey as SK

        _pantheon, _shared_services = _build_pantheon()
        _retriever = _shared_services.get(SK.RETRIEVER)
        _crm = _shared_services.get(SK.CRM)

        _pipeline, _, _, _ = await _build_pipeline(_pantheon, _shared_services)

        _conversation_memory = getattr(_pipeline, "_conversation", None)
        _relationship_memory = getattr(_pipeline, "_relationship", None)
        _goal_manager = getattr(_pipeline, "_goals", None)

        _long_term_memory = getattr(_pipeline, "_long_term", None)
        if _long_term_memory is None and _pantheon is not None:
            mnemosyne = _pantheon.get_agent("mnemosyne")
            if mnemosyne and hasattr(mnemosyne, "_services"):
                _long_term_memory = mnemosyne._services.get(SK.LONG_TERM_MEMORY)

        from brain_os.brain.knowledge_graph import KnowledgeGraph

        _knowledge_graph = KnowledgeGraph()

        digestive = _shared_services.get(SK.DIGESTIVE)
        if digestive and hasattr(digestive, "_ingestor"):
            _ingestor = digestive._ingestor

        try:
            from brain_os.interfaces.email_processor import EmailProcessor
            from brain_os.systems.sensory import SensorySystem

            graph = KnowledgeGraph()
            sensory = SensorySystem(knowledge_graph=graph)
            delphi = _pantheon.get_agent("delphi") if _pantheon else None
            _email_processor = EmailProcessor(
                delphi=delphi,
                digestive=digestive,
                sensory=sensory,
                crm=_crm,
                pantheon=_pantheon,
            )
        except Exception:
            logger.info("Email processor not available for MCP — continuing without it")

        try:
            from brain_os.systems.tinder_email_mode import TinderEmailModeService

            _tinder_email_mode = TinderEmailModeService()
        except Exception:
            logger.info("Tinder email mode service not initialized — continuing without it")

        try:
            from brain_os.systems.task_orchestrator import TaskOrchestrator

            redis_cache = _shared_services.get(SK.REDIS)
            voice = _shared_services.get(SK.VOICE)
            _task_orchestrator = TaskOrchestrator(
                pantheon=_pantheon,
                redis_cache=redis_cache,
                voice=voice,
                pdfco=_shared_services.get(SK.PDFCO),
            )
        except Exception:
            logger.info("Task orchestrator not available for MCP — continuing without it")

        if _pantheon is not None:
            _agent_loop = AgentLoop(_pantheon)

        _initialized = True
        _sync_mcp_server_facade()
        logger.info(
            "Ira MCP server initialized (%d tools registered)", len(mcp._tool_manager._tools)
        )


def _model_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a SQLAlchemy model or Pydantic model to a JSON-safe dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {"value": str(obj)}


__all__ = [
    "_AGENT_LOOP_RETRY_POLICY",
    "_MCP_DOCAI_MAX_BYTES",
    "_VALIDATOR_MODES",
    "_agent_loop",
    "_all_mcp_tool_names",
    "_conversation_memory",
    "_crm",
    "_email_processor",
    "_ensure_init_lock",
    "_ensure_initialized",
    "_goal_manager",
    "_ingestor",
    "_init_lock",
    "_initialized",
    "_is_short_factual_query",
    "_is_transient_agent_loop_error",
    "_knowledge_graph",
    "_long_term_memory",
    "_loop_telemetry",
    "_mcp_query_sender_id",
    "_model_to_dict",
    "_normalize_validator_mode",
    "_pantheon",
    "_pipeline",
    "_quick_answer_core",
    "_rank_tools_for_query",
    "_relationship_memory",
    "_retriever",
    "_shared_services",
    "_sync_mcp_server_facade",
    "_task_orchestrator",
    "_tinder_email_mode",
    "_utc_now_iso",
    "hardened_mcp_tool",
    "mcp",
]
