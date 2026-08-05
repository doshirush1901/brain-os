"""Master request pipeline from raw input to shaped response.

The canonical stage contract lives in ``AGENTS.md``; this module contains the
runtime orchestration and timing instrumentation for those stages.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from typing import Any

from langfuse.decorators import observe

from brain_os.brain.outreach_intents import is_outreach_prioritization_query
from brain_os.brain.write_contract import (
    recent_receipts,
    summarize_recent,
)
from brain_os.schemas.llm_outputs import (
    ClarificationPayload,
)

logger = logging.getLogger(__name__)


class RequestPipeline:
    """Stateful pipeline that holds references to every subsystem.

    Construct once at application startup (e.g. in the FastAPI lifespan)
    and reuse for every request.
    """

    def __init__(
        self,
        *,
        sensory: Any,
        conversation_memory: Any,
        relationship_memory: Any | None = None,
        goal_manager: Any | None = None,
        procedural_memory: Any | None = None,
        metacognition: Any | None = None,
        inner_voice: Any | None = None,
        pantheon: Any,
        voice: Any,
        endocrine: Any | None = None,
        crm: Any | None = None,
        musculoskeletal: Any | None = None,
        unified_context: Any | None = None,
        adaptive_style: Any | None = None,
        realtime_observer: Any | None = None,
        power_level_tracker: Any | None = None,
        redis_cache: Any | None = None,
        episodic_memory: Any | None = None,
        long_term_memory: Any | None = None,
        memory_block_store: Any | None = None,
        tool_stats_tracker: Any | None = None,
        agent_journal: Any | None = None,
        qdrant_manager: Any | None = None,
        knowledge_graph: Any | None = None,
    ) -> None:
        self._sensory = sensory
        self._conversation = conversation_memory
        self._relationship = relationship_memory
        self._goals = goal_manager
        self._procedural = procedural_memory
        self._metacognition = metacognition
        self._inner_voice = inner_voice
        self._pantheon = pantheon
        self._voice = voice
        self._endocrine = endocrine
        self._crm = crm
        self._musculoskeletal = musculoskeletal
        self._unified_ctx = unified_context
        self._adaptive_style = adaptive_style
        self._realtime_observer = realtime_observer
        self._power_level_tracker = power_level_tracker
        self._redis = redis_cache
        self._episodic = episodic_memory
        self._long_term = long_term_memory
        self._memory_blocks = memory_block_store
        self._tool_stats_tracker = tool_stats_tracker
        self._agent_journal = agent_journal
        self._qdrant = qdrant_manager
        self._graph = knowledge_graph

        self._router = pantheon.router
        self._pending_clarifications: dict[str, dict[str, Any]] = {}
        self._recent_messages: dict[str, tuple[str, float]] = {}
        self._state_lock = asyncio.Lock()
        self._request_semaphore = asyncio.Semaphore(5)
        self._stage_timings: deque[dict[str, Any]] = deque(maxlen=100)
        self._last_learn_task: asyncio.Task[None] | None = None
        self._learn_metrics: dict[str, Any] = {
            "runs": 0,
            "success": 0,
            "failed": 0,
            "last_duration_ms": 0.0,
            "last_error_count": 0,
            "last_contract_pass": False,
            "contract_pass": 0,
            "contract_fail": 0,
            # Phase-2 hardening: per-task lifecycle so operators can see if
            # the last LEARN background task actually finished or failed.
            "last_run_id": None,
            "last_started_at": None,
            "last_finished_at": None,
            "last_error": None,
            "last_failed_at": None,
            "failures_recent": [],  # bounded list of {run_id, error, at}
        }
        self._learn_failures_cap: int = 25
        # Soft-fail observability: last request's structured degradation rows
        # (from trace["degradation"]), independent of LEARN success.
        self._last_degradation: list[dict[str, str]] = []
        self._last_degradation_run_id: str | None = None

        self._load_pending_clarifications()

    _CLARIFICATION_REDIS_KEY = "ira:pending_clarifications"

    async def wait_for_background_tasks(self, timeout: float = 30.0) -> None:
        """Wait for the background learn task to finish (with timeout).

        Call this before shutting down the event loop (e.g. in CLI mode)
        to avoid asyncio.CancelledError on in-flight DB transactions.
        """
        if self._last_learn_task is not None and not self._last_learn_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._last_learn_task),
                    timeout=timeout,
                )
            except TimeoutError:
                logger.warning(
                    "Background learn task did not finish in %.0fs; leaving it running",
                    timeout,
                )
            except Exception:
                logger.debug("Background learn task raised", exc_info=True)

    def get_recent_write_receipts(self, limit: int = 100) -> list[dict[str, Any]]:
        return recent_receipts(limit=limit)

    def get_write_contract_summary(self, limit: int = 500) -> dict[str, Any]:
        return summarize_recent(limit=limit)

    def _load_pending_clarifications(self) -> None:
        """Restore pending clarifications from Redis on startup."""
        if self._redis is None:
            return
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                return
        except RuntimeError:
            return

    async def _persist_clarification(self, sender_id: str, data: dict[str, Any]) -> None:
        """Store a pending clarification in Redis for cross-invocation persistence."""
        if self._redis is None:
            return
        try:
            import json as _json

            await self._redis.hset(
                self._CLARIFICATION_REDIS_KEY,
                sender_id,
                _json.dumps(data),
            )
        except Exception:
            logger.warning("Failed to persist clarification to Redis", exc_info=True)

    def _attach_pipeline_trace(self, metadata: dict[str, Any], trace: dict[str, Any]) -> None:
        """Copy structured observability trace for API/MCP/eval callers."""
        try:
            # Always stash last degradation for GET /api/learn/last (ops one-stop),
            # even when full pipeline_trace attach is disabled.
            deg = trace.get("degradation")
            if isinstance(deg, list):
                self._last_degradation = [
                    {str(k): str(v) for k, v in row.items()} for row in deg if isinstance(row, dict)
                ]
            else:
                self._last_degradation = []
            rid = trace.get("run_id")
            self._last_degradation_run_id = str(rid) if rid else None

            from brain_os.config import get_settings as _gst

            if not _gst().app.pipeline_attach_observability_trace:
                return
            metadata["pipeline_trace"] = dict(trace)
        except Exception:
            logger.debug("pipeline_trace attach skipped", exc_info=True)

    def get_last_degradation(self) -> dict[str, Any]:
        """Return soft-fail rows from the most recent request that attached a trace."""
        return {
            "run_id": self._last_degradation_run_id,
            "events": list(self._last_degradation),
        }

    def _rank_optional_for_query(self, query: str, optional_agents: list[str]) -> list[str]:
        from brain_os.config import get_settings as _gro

        if not _gro().app.router_optional_keyword_ranking:
            return list(optional_agents)
        words = {w.lower() for w in (query or "").split() if len(w) >= 3}

        def sort_key(name: str) -> tuple[int, str]:
            n = name.lower().replace("_", " ")
            hit = sum(1 for part in n.split() if part in words)
            return (-hit, name)

        return sorted(optional_agents, key=sort_key)

    async def _maybe_router_embedding_tiebreak(
        self,
        query: str,
        routing: dict[str, Any] | None,
        scoreboard: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        return await maybe_router_embedding_tiebreak(
            self._router,
            self._qdrant,
            query,
            routing,
            scoreboard,
        )

    async def _pop_clarification(
        self,
        sender_id: str,
        *,
        gate_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Pop a pending clarification from both memory and Redis."""
        from brain_os.pipeline_phases.clarification_state import pop_clarification

        return await pop_clarification(
            redis=self._redis,
            pending=self._pending_clarifications,
            lock=self._state_lock,
            sender_id=sender_id,
            redis_key=self._CLARIFICATION_REDIS_KEY,
            gate_id=gate_id,
        )

    async def _get_clarification(
        self,
        *,
        sender_id: str | None = None,
        gate_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Peek pending clarification without popping."""
        from brain_os.pipeline_phases.clarification_state import get_clarification

        return await get_clarification(
            redis=self._redis,
            pending=self._pending_clarifications,
            lock=self._state_lock,
            redis_key=self._CLARIFICATION_REDIS_KEY,
            sender_id=sender_id,
            gate_id=gate_id,
        )

    async def _store_clarification(
        self,
        *,
        sender_id: str,
        agent_name: str,
        original_query: str,
        payload: ClarificationPayload,
        checkpoint: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Persist normalized clarification state and return rendered question text."""
        from brain_os.pipeline_phases.clarification_state import store_clarification

        return await store_clarification(
            redis=self._redis,
            pending=self._pending_clarifications,
            lock=self._state_lock,
            sender_id=sender_id,
            agent_name=agent_name,
            original_query=original_query,
            payload=payload,
            redis_key=self._CLARIFICATION_REDIS_KEY,
            checkpoint=checkpoint,
            extra=extra,
        )

    def get_recent_stage_timings(self, n: int = 24) -> list[dict[str, Any]]:
        """Return the last n pipeline runs' per-stage durations (seconds) and timestamps."""
        recent = list(self._stage_timings)
        return recent[-n:] if len(recent) > n else recent

    def get_learn_metrics(self) -> dict[str, Any]:
        """Return aggregate health signals for Step 10 (learn).

        Phase-2 hardening adds per-task lifecycle keys so operators can see
        whether the most recent background ``_learn`` task actually finished
        or failed silently:

        - ``last_run_id``: ``run_id`` of the most recently scheduled LEARN task.
        - ``last_started_at`` / ``last_finished_at``: epoch seconds, or ``None``.
        - ``last_error`` / ``last_failed_at``: populated when the task raised.
        - ``failures_recent``: bounded list of ``{run_id, error, at}`` entries
          (last :pyattr:`_learn_failures_cap` failures).
        """
        snap = dict(self._learn_metrics)
        # Defensive copy — callers must not mutate the rolling failure list.
        failures = snap.get("failures_recent")
        if isinstance(failures, list):
            snap["failures_recent"] = list(failures)
        return snap

    def _record_learn_started(self, run_id: str) -> None:
        """Mark a LEARN background task as started; called inline before the await."""
        try:
            self._learn_metrics["last_run_id"] = run_id
            self._learn_metrics["last_started_at"] = time.time()
            self._learn_metrics["last_finished_at"] = None
            # Do not clear last_error/last_failed_at — they persist as the
            # most recent failure until a *new* failure overwrites them.
        except Exception:  # pragma: no cover — defensive
            logger.debug("learn_metrics: started bookkeeping failed", exc_info=True)

    def _record_learn_finished(self, run_id: str, error: BaseException | None) -> None:
        """Mark a LEARN background task as finished (success or failure).

        Must never raise — it is invoked from an asyncio done-callback.
        """
        try:
            now = time.time()
            self._learn_metrics["last_finished_at"] = now
            if error is None:
                return
            # Truncate the error string so a giant traceback cannot bloat
            # observability snapshots / API responses.
            err_str = f"{type(error).__name__}: {error!s}"
            if len(err_str) > 400:
                err_str = err_str[:400] + "…"
            self._learn_metrics["last_error"] = err_str
            self._learn_metrics["last_failed_at"] = now
            failures = self._learn_metrics.get("failures_recent")
            if not isinstance(failures, list):
                failures = []
                self._learn_metrics["failures_recent"] = failures
            failures.append({"run_id": run_id, "error": err_str, "at": now})
            cap = max(1, int(self._learn_failures_cap or 1))
            if len(failures) > cap:
                # Drop oldest; keep the most recent ``cap`` entries.
                del failures[: len(failures) - cap]
        except Exception:  # pragma: no cover — defensive
            logger.debug("learn_metrics: finished bookkeeping failed", exc_info=True)

    async def _maybe_record_llm_budget_turn(
        self,
        sender_id: str,
        raw_input: str,
        response: str,
        agents_used: list[str],
    ) -> None:
        """Heuristic monthly token accrual (Redis). Skips empty agent lists and gate failures."""
        from brain_os.config import get_settings
        from brain_os.systems.llm_budget import (
            add_monthly_usage,
            estimate_turn_tokens,
            resolve_budget_bucket,
        )

        app = get_settings().app
        if app.llm_monthly_token_budget <= 0:
            return
        if self._redis is None or not self._redis.available:
            return
        if not agents_used or agents_used == ["timeout"] or agents_used == ["budget"]:
            return
        scope, bucket = resolve_budget_bucket(sender_id, scope_mode=app.llm_budget_scope)
        est = estimate_turn_tokens(raw_input, response, agents_used)
        if est <= 0:
            return
        await add_monthly_usage(self._redis, scope=scope, bucket=bucket, tokens=est)

    # ── Public entry point ────────────────────────────────────────────────

    @observe(name="pipeline.process_request")
    async def process_request(
        self,
        raw_input: str,
        channel: str,
        sender_id: str,
        metadata: dict[str, Any] | None = None,
        on_progress: Any | None = None,
    ) -> tuple[str, list[str], str]:
        """Run the full 17-step pipeline with concurrency and timeout guards.

        Returns ``(response, agents_used, run_id)``. ``run_id`` is also written to
        ``metadata["pipeline_run_id"]`` when *metadata* is a mutable dict shared by the caller.

        Stage breakdown (including substeps) is documented in AGENTS.md § Request Pipeline.
        """
        from brain_os.services.llm_caller_context import InvocationSource, llm_caller_scope

        meta = metadata if isinstance(metadata, dict) else {}
        source: InvocationSource = (
            "background"
            if str(meta.get("llm_source") or "").lower() == "background"
            else "operator"
        )
        with llm_caller_scope(
            job="pipeline",
            pipeline_step="process_request",
            source=source,
            call_site=f"pipeline:{channel}",
        ):
            async with self._request_semaphore:
                try:
                    from brain_os.config import get_settings

                    _timeout = get_settings().app.pipeline_timeout
                    out = await asyncio.wait_for(
                        self._process_request_inner(
                            raw_input,
                            channel,
                            sender_id,
                            metadata,
                            on_progress,
                        ),
                        timeout=_timeout,
                    )
                    await self._maybe_record_llm_budget_turn(
                        sender_id,
                        raw_input,
                        out[0],
                        out[1],
                    )
                    return out
                except TimeoutError:
                    from brain_os.config import get_settings

                    _timeout = get_settings().app.pipeline_timeout
                    return build_pipeline_timeout_response(
                        timeout_seconds=_timeout,
                        sender_id=sender_id,
                        metadata=metadata,
                        logger=logger,
                    )

    async def _process_request_inner(
        self,
        raw_input: str,
        channel: str,
        sender_id: str,
        metadata: dict[str, Any] | None = None,
        on_progress: Any | None = None,
    ) -> tuple[str, list[str], str]:
        """Run the full 17-step pipeline and return ``(shaped_response, agents_used, run_id)``."""
        from brain_os.pipeline_phases.request_inner import run_process_request_inner

        return await run_process_request_inner(
            self,
            raw_input,
            channel,
            sender_id,
            metadata,
            on_progress,
        )

    # ── Email scope resolver ─────────────────────────────────────────────

    _LIVE_EMAIL_PATTERNS = re.compile(
        r"(latest|recent|today|yesterday|new|unread|inbox|just sent|this morning|"
        r"last\s+(hour|day|week)|what did .+ (say|send|write|reply))",
        re.IGNORECASE,
    )
    _EMAIL_PATTERNS = re.compile(
        r"(email|mail|thread|inbox|gmail|message from|correspondence|"
        r"draft|reply|follow.?up email|send .+ email)",
        re.IGNORECASE,
    )
    _THREAD_ID_PATTERN = re.compile(
        r"(?:gmail\s+thread\s*:?\s*|thread[_\s-]*id\s*:?\s*)([a-zA-Z0-9]{8,})",
        re.IGNORECASE,
    )
    _HIGH_STAKES_LLM_ROUTE_PATTERN = re.compile(
        r"("
        r"\bquote\b|\bpricing\b|\bprice\b|\bcost\b|\bcontract\b|\bmsa\b|\bsow\b|"
        r"\blegal\b|\bcompliance\b|\bconfidential\b|\bnda\b|"
        r"\binvoice\b|\bpayment\b|\brefund\b|\bpo\b|\bpurchase\s+order\b|"
        r"\bsend\b.*\bemail\b|\bemail\b.*\bsend\b"
        r")",
        re.IGNORECASE,
    )
    _OUTREACH_WORKFLOW_AGENTS = ["prometheus", "chiron", "clio"]

    def _is_outreach_prioritization_query(self, query: str) -> bool:
        return is_outreach_prioritization_query(query)

    @staticmethod
    def _uncensored_local_llm_active(metadata: dict[str, Any] | None) -> bool:
        """True when this turn opts into local uncensored relaxations (env or request metadata)."""
        meta = metadata if isinstance(metadata, dict) else {}
        raw = meta.get("uncensored_local_llm_mode")
        if raw is True:
            return True
        if isinstance(raw, str) and raw.strip().lower() in ("1", "true", "yes", "on"):
            return True
        try:
            from brain_os.config import get_settings

            return bool(get_settings().app.uncensored_local_llm_mode)
        except Exception:
            return False

    @staticmethod
    def _ollama_openai_compat_ready() -> bool:
        """True when Brain OS's OpenAI-compatible local client (``ollama`` slot) is configured."""
        try:
            from brain_os.config import get_settings
            from brain_os.services.llm_client import get_llm_client

            llm_cfg = get_settings().llm
            if not (llm_cfg.ollama_base_url or "").strip():
                return False
            return get_llm_client().is_provider_configured("ollama")
        except Exception:
            return False

    def _resolve_email_scope(self, query: str) -> str:
        """Classify the query's email data scope: live_email, imported_email, both, or no_email."""
        has_email_mention = bool(self._EMAIL_PATTERNS.search(query))
        needs_live = bool(self._LIVE_EMAIL_PATTERNS.search(query))
        outreach_prioritization = self._is_outreach_prioritization_query(query)

        # Outreach sequencing requires the latest inbox intent signals; default
        # to blended scope so agents can read live threads + indexed history.
        if outreach_prioritization:
            has_email_mention = True
            needs_live = True

        if needs_live and has_email_mention:
            return "both"
        if needs_live:
            return "live_email"
        if not has_email_mention:
            return "no_email"
        return "imported_email"

    def _is_high_stakes_llm_route_query(self, query: str) -> bool:
        """True when query should escalate routed LLM calls to a cloud model."""
        return bool(self._HIGH_STAKES_LLM_ROUTE_PATTERN.search(query or ""))

    @staticmethod
    def _resolve_llm_route_provider_override(
        query: str,
        *,
        route_method: str,
        channel: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Pick primary provider override for this turn's LLM calls (or None)."""
        from brain_os.config import get_settings
        from brain_os.services.llm_client import get_llm_client

        if route_method != "llm":
            return None
        s = get_settings()
        llm_cfg = s.llm
        app_cfg = s.app
        llm_client = get_llm_client()
        md = metadata if isinstance(metadata, dict) else {}
        hints = md.get("mcp_runtime_hints", {}) if isinstance(md, dict) else {}

        ollama_configured = bool(
            (llm_cfg.ollama_base_url or "").strip()
        ) and llm_client.is_provider_configured("ollama")
        if RequestPipeline._uncensored_local_llm_active(md) and ollama_configured:
            return "ollama"
        if str(channel).lower() == "mcp":
            prefer_cloud = bool(
                hints.get("route_prefer_cloud_fast", app_cfg.mcp_route_prefer_cloud_fast)
            )
            fast_fail_to_openai = bool(
                hints.get("fast_fail_to_openai", app_cfg.mcp_fast_fail_to_openai)
            )
            if fast_fail_to_openai and llm_client.is_provider_configured("openai"):
                return "openai"
            if prefer_cloud:
                if llm_client.is_provider_configured("openai"):
                    return "openai"
                if llm_client.is_provider_configured("anthropic"):
                    return "anthropic"
                return "ollama" if ollama_configured else None
        if (
            app_cfg.llm_route_high_stakes_cloud_escalation
            and RequestPipeline._HIGH_STAKES_LLM_ROUTE_PATTERN.search(query or "")
        ):
            if llm_client.is_provider_configured("openai"):
                return "openai"
            if llm_client.is_provider_configured("anthropic"):
                return "anthropic"
            return "ollama" if ollama_configured else None
        if app_cfg.llm_route_prefer_ollama and ollama_configured:
            return "ollama"
        return None

    # ── Execution helpers ─────────────────────────────────────────────────

    async def _execute_routed(
        self,
        agent_names: list[str],
        query: str,
        context: dict[str, Any],
        on_progress: Any | None = None,
    ) -> tuple[str, list[str]]:
        """Execute one or more routed agents with bounded parallelism."""
        from brain_os.pipeline_phases.pipeline_execute import execute_routed_agents

        return await execute_routed_agents(
            pantheon=self._pantheon,
            agent_names=agent_names,
            query=query,
            context=context,
            on_progress=on_progress,
        )

    # ── Learning step ─────────────────────────────────────────────────────

    def _pipeline_learn_deps(self) -> Any:
        from brain_os.pipeline_phases.learn import PipelineLearnDeps

        return PipelineLearnDeps(
            conversation=self._conversation,
            relationship=self._relationship,
            episodic=self._episodic,
            memory_blocks=self._memory_blocks,
            power_level_tracker=self._power_level_tracker,
            long_term=self._long_term,
            qdrant=self._qdrant,
            graph=self._graph,
            crm=self._crm,
            musculoskeletal=self._musculoskeletal,
            goals=self._goals,
            procedural=self._procedural,
            pantheon=self._pantheon,
            realtime_observer=self._realtime_observer,
            endocrine=self._endocrine,
            unified_ctx=self._unified_ctx,
            learn_metrics=self._learn_metrics,
        )

    async def _learn(
        self,
        *,
        contact_email: str,
        channel: str,
        raw_input: str,
        raw_response: str,
        route_method: str,
        agents_used: list[str],
        active_goal: Any | None,
        resolved_input: str,
        run_id: str,
    ) -> None:
        """Step 10: record the interaction across all memory and tracking systems."""
        from brain_os.pipeline_phases.learn import learn_pipeline_turn

        await learn_pipeline_turn(
            self._pipeline_learn_deps(),
            contact_email=contact_email,
            channel=channel,
            raw_input=raw_input,
            raw_response=raw_response,
            route_method=route_method,
            agents_used=agents_used,
            active_goal=active_goal,
            resolved_input=resolved_input,
            run_id=run_id,
        )


# Back-compat re-exports for callers that still import from pipeline.
from brain_os.pipeline_runtime import (  # noqa: I001
    _AMBIGUITY_SENSITIVE_PATTERNS,
    _DEFAULT_CHEAP_EXIT_BYPASS_KEYWORDS,
    _DEDUP_ENVELOPE_VERSION,
    _QUICK_PIPELINE_PATTERN,
    _csv_tokens,
    _decode_dedup_payload,
    _encode_dedup_payload,
    _extract_teaching_facts,
    _faithfulness_strict_intent,
    _format_pipeline_summary_md,
    _is_ambiguity_sensitive_query,
    _is_quick_pipeline_query,
    _should_bypass_cheap_exits,
)

from brain_os.pipeline_phases.error_handling import (
    build_pipeline_timeout_response,
)
from brain_os.pipeline_phases.outreach import prefetch_outreach_thread_evidence
from brain_os.pipeline_phases.plan import (
    maybe_router_embedding_tiebreak,
)
