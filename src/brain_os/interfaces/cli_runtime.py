"""Shared CLI bootstrap helpers (pantheon, pipeline, revenue bundle).

Extracted from ``cli.py`` so Typer sub-apps under ``interfaces/commands/`` can
import runtime wiring **without** importing ``cli`` (avoids circular imports
once ``cli`` mounts those command modules).

Command modules must depend on this module — not on ``brain_os.interfaces.cli``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import json
import logging
import re
import signal
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import httpx

from brain_os.exceptions import ConfigurationError, BrainOSError
from brain_os.service_keys import ServiceKey as SK
from brain_os.services.resilience import RetryPolicy, run_with_retry

logger = logging.getLogger(__name__)

T_co = TypeVar("T_co")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _coerce_strings_for_json(obj: Any) -> Any:
    """Walk dict/list and replace raw newlines in strings so ``json.dumps`` is valid."""
    if isinstance(obj, dict):
        return {k: _coerce_strings_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_strings_for_json(x) for x in obj]
    if isinstance(obj, str):
        return obj.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return obj


def _extract_first_json_block(text: str) -> Any:
    """Robust JSON extractor for LLM responses with wrappers/noise."""
    """Robust JSON extractor for LLM responses with wrappers/noise."""
    if isinstance(text, (dict, list)):
        return text
    raw = (text or "").strip()
    lowered = raw.lower()
    if not raw or lowered in {"(no response)", "no response", "none", "null"}:
        raise ValueError("No valid JSON found in agent response.")

    candidates: list[str] = []
    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())
    candidates.append(raw)

    decoder = json.JSONDecoder()
    parsed_candidates: list[Any] = []
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
            parsed_candidates.append(value)
            continue
        except json.JSONDecodeError:
            pass

        for idx, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            try:
                value, _end = decoder.raw_decode(candidate[idx:])
                parsed_candidates.append(value)
                break
            except json.JSONDecodeError:
                continue

    for value in parsed_candidates:
        if isinstance(value, (dict, list)):
            return value
    raise ValueError("No valid JSON found in agent response.")


def _is_connection_error(exc: Exception) -> bool:
    """True if exc is a Qdrant/HTTP connection failure (e.g. Docker down)."""
    if isinstance(exc, httpx.ConnectError):
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and isinstance(cause, httpx.ConnectError):
        return True
    if type(exc).__name__ == "ResponseHandlingException" and cause is not None:
        return "connection" in str(cause).lower() or isinstance(cause, OSError)
    return False


async def run_with_connection_retry(
    operation: Callable[[], Awaitable[T_co]],
    *,
    policy: RetryPolicy | None = None,
) -> T_co:
    """Run *operation* with bounded retries on transient transport failures (CLI/MCP helpers)."""
    pol = policy or RetryPolicy(max_attempts=3, base_delay_seconds=1.0, max_delay_seconds=10.0)
    return await run_with_retry(operation, policy=pol, is_retryable=_is_connection_error)


def _confidence_score_to_label(score: float) -> str:
    if score >= 2.5:
        return "high"
    if score >= 1.5:
        return "medium"
    if score > 0:
        return "low"
    return "unknown"


def _run(coro: Any) -> Any:
    """Run an async coroutine from synchronous CLI context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _configure_logging(verbose: bool = False, *, dream_progress: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if dream_progress and not verbose:
        for name in (
            "brain_os.memory.dream_mode",
            "brain_os.systems.respiratory",
        ):
            logging.getLogger(name).setLevel(logging.INFO)


def _build_pantheon() -> tuple[Any, dict[str, Any]]:
    """Construct a Pantheon with full service wiring for CLI use.

    Returns a ``(pantheon, shared_services)`` tuple so that callers can
    forward the shared CRM / quotes / pricing / retriever instances to
    ``_build_pipeline`` instead of creating duplicate connections.
    """
    from brain_os.brain.embeddings import EmbeddingService
    from brain_os.brain.knowledge_graph import KnowledgeGraph
    from brain_os.brain.pricing_engine import PricingEngine
    from brain_os.brain.qdrant_manager import QdrantManager
    from brain_os.brain.retriever import UnifiedRetriever
    from brain_os.config import get_settings
    from brain_os.data.crm import CRMDatabase
    from brain_os.data.quotes import QuoteManager
    from brain_os.message_bus import MessageBus
    from brain_os.pantheon import Pantheon
    from brain_os.skills.handlers import bind_services as bind_skill_services

    settings = get_settings()

    embedding = EmbeddingService()
    qdrant = QdrantManager(embedding_service=embedding)
    graph = KnowledgeGraph()

    mem0_client = None
    mem0_key = settings.memory.api_key.get_secret_value()
    if mem0_key:
        try:
            from mem0 import MemoryClient  # type: ignore[import-untyped]

            mem0_client = MemoryClient(api_key=mem0_key)
        except (ConfigurationError, Exception):
            logger.warning("Mem0 init failed — continuing without conversational memory")

    retriever = UnifiedRetriever(qdrant=qdrant, graph=graph, mem0_client=mem0_client)
    bus = MessageBus()

    crm = CRMDatabase()
    quotes = QuoteManager(session_factory=crm.session_factory)
    pricing_engine = PricingEngine(retriever=retriever, crm=crm)

    from brain_os.systems.data_event_bus import DataEventBus

    data_event_bus = DataEventBus()
    crm.set_event_bus(data_event_bus)
    graph.set_event_bus(data_event_bus)
    qdrant.set_event_bus(data_event_bus)

    from brain_os.systems.circulatory import CirculatorySystem

    CirculatorySystem(
        data_event_bus,
        crm=crm,
        graph=graph,
        qdrant=qdrant,
        embedding=embedding,
    )

    pantheon = Pantheon(retriever=retriever, bus=bus)

    # Some runtime contexts (notably stale MCP subprocesses) may load an older
    # ServiceKey class that does not define QDRANT. Fall back to canonical key.
    qdrant_key = getattr(SK, "QDRANT", "qdrant")

    shared_services = {
        SK.CRM: crm,
        SK.QUOTES: quotes,
        SK.PRICING_ENGINE: pricing_engine,
        SK.RETRIEVER: retriever,
        SK.DATA_EVENT_BUS: data_event_bus,
        qdrant_key: qdrant,
        SK.GRAPH: graph,
        SK.MEM0_CLIENT: mem0_client,
    }
    try:
        from brain_os.systems.pdfco import PdfCoService

        _pdfco = PdfCoService()
        if _pdfco.available:
            shared_services[SK.PDFCO] = _pdfco
            logger.debug("CLI pantheon: PDF.co service injected")
    except Exception as e:
        logger.debug("CLI pantheon: PDF.co not available: %s", e)

    pantheon.inject_services(shared_services)
    bind_skill_services(shared_services)

    return pantheon, shared_services


async def _build_pipeline(
    pantheon: Any,
    shared_services: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    """Construct a full RequestPipeline and FeedbackHandler for CLI use.

    *shared_services* is the dict returned by ``_build_pantheon()`` and
    contains at least ``crm`` and ``quotes`` so we reuse the same
    connection pool instead of opening a duplicate.

    Returns ``(pipeline, feedback_handler, redis_cache, voice)``.
    """
    data_event_bus = shared_services.get(SK.DATA_EVENT_BUS)
    if data_event_bus is not None:
        await data_event_bus.start()

    from brain_os.brain.correction_store import CorrectionStore
    from brain_os.brain.feedback_handler import FeedbackHandler
    from brain_os.brain.power_levels import PowerLevelTracker
    from brain_os.brain.tool_stats import ToolStatsTracker
    from brain_os.config import get_settings as _cli_gepa_settings
    from brain_os.context import UnifiedContextManager
    from brain_os.memory.conversation import ConversationMemory
    from brain_os.memory.emotional_intelligence import EmotionalIntelligence
    from brain_os.memory.episodic import EpisodicMemory
    from brain_os.memory.goal_manager import GoalManager
    from brain_os.memory.inner_voice import InnerVoice
    from brain_os.memory.long_term import LongTermMemory
    from brain_os.memory.metacognition import Metacognition
    from brain_os.memory.procedural import ProceduralMemory
    from brain_os.memory.relationship import RelationshipMemory
    from brain_os.pipeline import RequestPipeline
    from brain_os.systems.endocrine import EndocrineSystem
    from brain_os.systems.immune import ImmuneSystem
    from brain_os.systems.learning_hub import LearningHub
    from brain_os.systems.musculoskeletal import MusculoskeletalSystem
    from brain_os.systems.sensory import SensorySystem
    from brain_os.systems.voice import VoiceSystem

    _INIT_TIMEOUT = 30

    async def _safe_init(name: str, coro: Any) -> None:
        try:
            await asyncio.wait_for(coro, timeout=_INIT_TIMEOUT)
            logger.info("CLI: initialised %s", name)
        except TimeoutError:
            logger.warning(
                "CLI: timed out initialising %s after %ds — skipping",
                name,
                _INIT_TIMEOUT,
            )
        except (ConfigurationError, Exception):
            logger.warning("CLI: failed to initialise %s — skipping", name, exc_info=True)

    retriever = shared_services[SK.RETRIEVER]
    qdrant = retriever._qdrant
    graph = retriever._graph
    embedding_service = qdrant._embeddings

    sensory = SensorySystem(knowledge_graph=graph)
    try:
        await asyncio.wait_for(sensory.create_tables(), timeout=_INIT_TIMEOUT)
    except (TimeoutError, Exception):
        logger.warning("SensorySystem table creation slow/failed — continuing", exc_info=True)

    long_term = LongTermMemory()
    episodic = EpisodicMemory(long_term=long_term)
    await _safe_init("episodic", episodic.initialize())

    conversation = ConversationMemory()
    await _safe_init("conversation", conversation.initialize())

    musculoskeletal = MusculoskeletalSystem()
    await _safe_init("musculoskeletal", musculoskeletal.create_tables())

    relationship_memory = RelationshipMemory()
    await relationship_memory.initialize()
    goal_manager = GoalManager()
    await goal_manager.initialize()
    procedural_memory = ProceduralMemory()
    await procedural_memory.initialize()
    metacognition = Metacognition()
    await metacognition.initialize()
    inner_voice = InnerVoice()
    await inner_voice.initialize()

    emotional_intelligence = EmotionalIntelligence()
    await emotional_intelligence.initialize()

    voice = VoiceSystem()
    endocrine = EndocrineSystem()
    immune = ImmuneSystem(
        qdrant=qdrant,
        knowledge_graph=graph,
        embedding_service=embedding_service,
    )
    immune.set_endocrine(endocrine)
    crm = shared_services[SK.CRM]
    await crm.create_tables()
    unified_context = UnifiedContextManager()

    learning_hub = LearningHub(crm=crm, procedural_memory=procedural_memory)

    correction_store = CorrectionStore()
    await correction_store.initialize()

    power_level_tracker = PowerLevelTracker()
    await _safe_init("power_level_tracker", power_level_tracker._load())

    from brain_os.memory.blocks import MemoryBlockStore

    memory_block_store = MemoryBlockStore()
    await memory_block_store.initialize()
    from brain_os.memory.block_store_access import register_live_block_store_lookup

    register_live_block_store_lookup(lambda: memory_block_store)

    mem0_client = shared_services.get(SK.MEM0_CLIENT)

    feedback_handler = FeedbackHandler(
        learning_hub=learning_hub,
        correction_store=correction_store,
        mem0_client=mem0_client,
        procedural_memory=procedural_memory,
        data_event_bus=data_event_bus,
        power_level_tracker=power_level_tracker,
        memory_block_store=memory_block_store,
    )
    await feedback_handler.load_scores()

    sensory.configure_memory(
        emotional_intelligence=emotional_intelligence,
        conversation_memory=conversation,
        relationship_memory=relationship_memory,
        memory_block_store=memory_block_store,
    )

    agent_journal = None
    try:
        from brain_os.memory.agent_journal import AgentJournal

        agent_journal = AgentJournal()
        await agent_journal.initialize()
    except Exception:
        logger.debug("AgentJournal not available for CLI", exc_info=True)

    agent_services: dict[str, Any] = {
        SK.LONG_TERM_MEMORY: long_term,
        SK.EPISODIC_MEMORY: episodic,
        SK.CONVERSATION_MEMORY: conversation,
        SK.RELATIONSHIP_MEMORY: relationship_memory,
        SK.MEMORY_BLOCK_STORE: memory_block_store,
        SK.GOAL_MANAGER: goal_manager,
        SK.PROCEDURAL_MEMORY: procedural_memory,
        SK.EMOTIONAL_INTELLIGENCE: emotional_intelligence,
        SK.LEARNING_HUB: learning_hub,
        SK.PANTHEON: pantheon,
        SK.DATA_EVENT_BUS: shared_services.get(SK.DATA_EVENT_BUS),
        SK.AGENT_JOURNAL: agent_journal,
        SK.IMMUNE: immune,
        SK.POWER_LEVEL_TRACKER: power_level_tracker,
    }
    try:
        digestive, _, _ = _build_digestive()
        email_processor = _build_email_processor(pantheon, digestive, shared_services)
        agent_services[SK.EMAIL_PROCESSOR] = email_processor
        logger.debug("CLI: email processor injected — mailbox access enabled")
    except Exception as e:
        logger.debug("CLI: email processor not available (mailbox access disabled): %s", e)

    try:
        from brain_os.services.scheduling import SchedulingService
        from brain_os.systems.google_calendar import GoogleCalendarService

        google_calendar = GoogleCalendarService()
        await google_calendar.connect()
        scheduling = SchedulingService(crm=crm, calendar=google_calendar)
        agent_services[SK.GOOGLE_CALENDAR] = google_calendar
        agent_services[SK.SCHEDULING] = scheduling
        shared_services[SK.GOOGLE_CALENDAR] = google_calendar
        shared_services[SK.SCHEDULING] = scheduling
        logger.debug("CLI: scheduling service injected")
    except Exception as e:
        logger.debug("CLI: scheduling service not available: %s", e)

    try:
        from brain_os.systems.pdfco import PdfCoService

        _pdfco = PdfCoService()
        if _pdfco.available:
            agent_services[SK.PDFCO] = _pdfco
            shared_services[SK.PDFCO] = _pdfco
            logger.debug("CLI: PDF.co service injected")
    except Exception as e:
        logger.debug("CLI: PDF.co not available: %s", e)

    pantheon.inject_services(agent_services)

    nemesis = pantheon.get_agent("nemesis")
    if nemesis is not None and hasattr(nemesis, "configure"):
        nemesis.configure(learning_hub=learning_hub, peer_agents=pantheon.agents)

    try:
        health_report = await immune.run_startup_validation()
        healthy = all(v.get("status") == "healthy" for v in health_report.values())
        status = "ALL HEALTHY" if healthy else "DEGRADED"
        logger.info("CLI startup validation: %s — %s", status, list(health_report))
    except (BrainOSError, Exception):
        logger.exception("CLI startup validation failed — continuing in degraded mode")

    redis_cache = None
    try:
        from brain_os.systems.redis_cache import RedisCache

        redis_cache = RedisCache()
        await redis_cache.connect()
    except (ConfigurationError, Exception):
        logger.info("Redis not available for CLI — pipeline state will be in-memory only")

    logger.info("CLI pipeline ready with feedback handler")

    tool_invocation_store = None
    if _cli_gepa_settings().app.gepa_persist_tool_invocations:
        from brain_os.brain.tool_invocation_store import ToolInvocationStore

        tool_invocation_store = ToolInvocationStore()
        await tool_invocation_store.initialize()
    tool_stats_tracker = ToolStatsTracker(invocation_store=tool_invocation_store)

    adaptive_style: Any | None = None
    realtime_observer: Any | None = None
    try:
        from brain_os.brain.adaptive_style import AdaptiveStyleTracker

        adaptive_style = AdaptiveStyleTracker()
        await adaptive_style._load()
    except Exception:
        logger.debug("CLI: AdaptiveStyleTracker unavailable", exc_info=True)
    try:
        from brain_os.brain.realtime_observer import RealTimeObserver

        realtime_observer = RealTimeObserver()
        await realtime_observer._load()
    except Exception:
        logger.debug("CLI: RealTimeObserver unavailable", exc_info=True)

    pipeline = RequestPipeline(
        sensory=sensory,
        conversation_memory=conversation,
        relationship_memory=relationship_memory,
        goal_manager=goal_manager,
        procedural_memory=procedural_memory,
        metacognition=metacognition,
        inner_voice=inner_voice,
        pantheon=pantheon,
        voice=voice,
        endocrine=endocrine,
        crm=crm,
        musculoskeletal=musculoskeletal,
        unified_context=unified_context,
        adaptive_style=adaptive_style,
        realtime_observer=realtime_observer,
        redis_cache=redis_cache,
        episodic_memory=episodic,
        long_term_memory=long_term,
        memory_block_store=memory_block_store,
        tool_stats_tracker=tool_stats_tracker,
        agent_journal=agent_journal,
        power_level_tracker=power_level_tracker,
    )

    return pipeline, feedback_handler, redis_cache, voice


def _build_digestive() -> tuple[Any, Any, Any]:
    """Return (digestive_system, document_ingestor, qdrant_manager)."""
    from brain_os.brain.document_ingestor import DocumentIngestor
    from brain_os.brain.embeddings import EmbeddingService
    from brain_os.brain.knowledge_graph import KnowledgeGraph
    from brain_os.brain.qdrant_manager import QdrantManager
    from brain_os.systems.digestive import DigestiveSystem

    embedding = EmbeddingService()
    qdrant = QdrantManager(embedding_service=embedding)
    graph = KnowledgeGraph()
    ingestor = DocumentIngestor(qdrant=qdrant, knowledge_graph=graph)
    digestive = DigestiveSystem(
        ingestor=ingestor,
        knowledge_graph=graph,
        embedding_service=embedding,
        qdrant=qdrant,
    )
    return digestive, ingestor, qdrant


def _build_email_processor(
    pantheon: Any,
    digestive: Any,
    shared_services: dict[str, Any] | None = None,
) -> Any:
    """Construct an EmailProcessor wired to the Pantheon's Delphi agent.

    Reuses the CRM instance from *shared_services* when available to
    avoid opening a duplicate database connection pool.
    """
    from brain_os.brain.knowledge_graph import KnowledgeGraph
    from brain_os.interfaces.email_processor import EmailProcessor
    from brain_os.systems.sensory import SensorySystem

    graph = KnowledgeGraph()
    sensory = SensorySystem(knowledge_graph=graph)

    if shared_services and SK.CRM in shared_services:
        crm = shared_services[SK.CRM]
    else:
        from brain_os.data.crm import CRMDatabase

        crm = CRMDatabase()

    delphi = pantheon.get_agent("delphi")
    return EmailProcessor(
        delphi=delphi,
        digestive=digestive,
        sensory=sensory,
        crm=crm,
    )


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = line.strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _looks_us_lead(lead: dict[str, Any]) -> bool:
    geography = str(lead.get("geography") or "").lower()
    text_blob = " ".join(
        str(lead.get(field) or "")
        for field in (
            "geography",
            "why_relevant",
            "trigger_event",
            "entry_angle",
            "decision_maker_hint",
        )
    ).lower()
    markers = (
        "usa",
        "u.s.",
        "united states",
        "north america",
        "us market",
        "us-based",
    )
    return any(m in geography for m in markers) or any(m in text_blob for m in markers)


def _revenue_deal_bundle(
    company: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    from brain_os.services import revenue_mode

    c = company.strip()
    lead = revenue_mode.find_latest_lead(c)
    pilot = revenue_mode.find_pilot_review(c)
    pipe = revenue_mode.get_pipeline_row(c)
    deal = dict((pipe or {}).get("deal") or {})
    return lead, pilot, pipe, deal


def _reconfigure_stdio_line_buffering() -> None:
    """So hosted terminals (e.g. IDE) see log lines as they are emitted during long runs."""
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(line_buffering=True)
            except (OSError, ValueError, io.UnsupportedOperation):
                pass


_LONG_JOB_STATUS_PATH = Path("data/brain/long_job_status.json")
_long_job_signal_name: str | None = None
_prev_sigint: Any = None
_prev_sigterm: Any = None


def _write_long_job_status(payload: dict[str, Any]) -> None:
    try:
        _LONG_JOB_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        merged = {**payload, "updated_at": datetime.now(UTC).isoformat()}
        _LONG_JOB_STATUS_PATH.write_text(
            json.dumps(merged, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError:
        logger.debug("long_job_status write failed", exc_info=True)


def _long_job_signal_handler(signum: int, frame: Any) -> None:
    job = _long_job_signal_name or "unknown"
    _write_long_job_status({"job": job, "phase": "interrupted", "signal": int(signum)})
    if int(signum) == signal.SIGINT:
        raise KeyboardInterrupt
    sys.exit(128 + int(signum))


def _install_long_job_signal_handlers(job: str) -> None:
    global _long_job_signal_name, _prev_sigint, _prev_sigterm
    _long_job_signal_name = job
    _prev_sigint = signal.signal(signal.SIGINT, _long_job_signal_handler)
    if hasattr(signal, "SIGTERM"):
        _prev_sigterm = signal.signal(signal.SIGTERM, _long_job_signal_handler)
    else:
        _prev_sigterm = None


def _reset_long_job_signal_handlers() -> None:
    global _long_job_signal_name, _prev_sigint, _prev_sigterm
    if _prev_sigint is not None:
        signal.signal(signal.SIGINT, _prev_sigint)
        _prev_sigint = None
    if _prev_sigterm is not None:
        signal.signal(signal.SIGTERM, _prev_sigterm)
        _prev_sigterm = None
    _long_job_signal_name = None
