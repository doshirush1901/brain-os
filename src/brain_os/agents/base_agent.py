"""Abstract base class for all Pantheon agents.

Every specialist agent inherits from :class:`BaseAgent`, which provides
LLM access (OpenAI, Anthropic, optional Ollama) via the centralised
:class:`~brain_os.services.llm_client.LLMClient`, knowledge-base search via
the :class:`~brain_os.brain.retriever.UnifiedRetriever`, a reference to the
:class:`~brain_os.message_bus.MessageBus` for inter-agent communication,
and an opt-in ReAct (Reason-Act-Observe) loop for agentic tool use.

Implementation is composed from mixins under :mod:`brain_os.agents.base`
(MOVE split; public import surface stays this module).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC
from datetime import UTC, datetime
from typing import Any

from brain_os.agents.base.delegation_mixin import DelegationMixin
from brain_os.agents.base.memory_mixin import MemoryMixin
from brain_os.agents.base.prompting_mixin import PromptingMixin
from brain_os.agents.base.react_mixin import ReactMixin
from brain_os.agents.base.retrieval_mixin import RetrievalMixin, _normalize_searchapi_results
from brain_os.agents.base.shared import (
    _AGENT_ENRICHMENT_ERRORS,
    _CRAWL_SCRAPE_ERRORS,
    _DELEGATION_ERRORS,
    _LLM_REACT_ERRORS,
    _RELATIONSHIP_EVENT_ERRORS,
    _coerce_optional_str,
    _coerce_tool_limit,
    _coerce_tool_query,
    _is_delegation_tool_name,
)
from brain_os.agents.base.telemetry_mixin import TelemetryMixin
from brain_os.agents.base.tools_mixin import AgentTool, ToolsMixin
from brain_os.agents.invocation_context import (
    AgentInvocation,
    AgentState,
    get_invocation,
)
from brain_os.brain.retriever import UnifiedRetriever
from brain_os.config import get_settings
from brain_os.message_bus import MessageBus
from brain_os.prompt_loader import load_prompt, load_soul_preamble
from brain_os.services.llm_client import get_llm_client

# Patch surface for tests (``patch("brain_os.agents.base_agent.get_llm_client")`` etc.).
# Mixins resolve these via :mod:`brain_os.agents.base.patchable`.

logger = logging.getLogger(__name__)

# Re-export historical public/private surface (callers import from base_agent).
__all__ = [
    "_AGENT_ENRICHMENT_ERRORS",
    "_CRAWL_SCRAPE_ERRORS",
    "_DELEGATION_ERRORS",
    "_LLM_REACT_ERRORS",
    "_RELATIONSHIP_EVENT_ERRORS",
    "AgentInvocation",
    "AgentState",
    "AgentTool",
    "BaseAgent",
    "_coerce_optional_str",
    "_coerce_tool_limit",
    "_coerce_tool_query",
    "_is_delegation_tool_name",
    "_normalize_searchapi_results",
    "get_llm_client",
    "get_settings",
    "load_prompt",
    "load_soul_preamble",
]


class BaseAgent(
    ReactMixin,
    ToolsMixin,
    RetrievalMixin,
    DelegationMixin,
    MemoryMixin,
    TelemetryMixin,
    PromptingMixin,
    ABC,
):
    """Abstract base for every agent in the Pantheon."""

    name: str = "base"
    role: str = ""
    description: str = ""
    model_provider: str = (
        "openai"  # "openai" or "anthropic" (ignored when BRAIN_DEFAULT_LLM_PROVIDER=ollama)
    )
    #: When set, ``_reason`` / ``call_llm`` pass this ``model_profile`` (e.g. ``\"writing\"`` for Calliope).
    preferred_model_profile: str | None = None
    knowledge_categories: list[str] = []
    timeout: int | None = None
    created_at: datetime = datetime(2024, 1, 1, tzinfo=UTC)

    _LLM_CACHE_TTL = 3600

    @property
    def age_in_days(self) -> int:
        """Days since this agent's creation (identity continuity)."""
        delta = datetime.now(UTC) - self.created_at
        return max(0, delta.days)

    @property
    def _context(self) -> dict[str, Any]:
        """Request-scoped context (ContextVar when inside ``run`` / ``handle``)."""
        inv = get_invocation()
        if inv is not None:
            return inv.context
        return self._idle_context

    @_context.setter
    def _context(self, value: dict[str, Any] | None) -> None:
        ctx = value if isinstance(value, dict) else {}
        inv = get_invocation()
        if inv is not None:
            inv.context = ctx
        else:
            self._idle_context = ctx

    @property
    def state(self) -> AgentState:
        inv = get_invocation()
        if inv is not None:
            return inv.state
        return self._idle_state

    @state.setter
    def state(self, value: AgentState) -> None:
        inv = get_invocation()
        if inv is not None:
            inv.state = value
        else:
            self._idle_state = value

    @property
    def _last_grounding_score(self) -> float:
        inv = get_invocation()
        if inv is not None:
            return inv.grounding_score
        return self._idle_grounding_score

    @_last_grounding_score.setter
    def _last_grounding_score(self, value: float) -> None:
        inv = get_invocation()
        if inv is not None:
            inv.grounding_score = float(value)
        else:
            self._idle_grounding_score = float(value)

    def _svc(self, key: str, default: Any = None) -> Any:
        """Resolve a service: invocation overlay first, then process-stable ``_services``."""
        inv = get_invocation()
        if inv is not None and key in inv.services_overlay:
            return inv.services_overlay[key]
        return self._services.get(key, default)

    def __init__(
        self,
        retriever: UnifiedRetriever,
        bus: MessageBus,
        *,
        services: dict[str, Any] | None = None,
    ) -> None:
        self._retriever = retriever
        self._bus = bus
        self._services: dict[str, Any] = services or {}

        # Lazy: tests construct agents under ``mock_settings`` without patching
        # ``get_llm_client``; real LLM is resolved on first ``call_llm`` / ``_reason``.
        self._llm: Any | None = None

        self.tools: list[AgentTool] = []
        self.max_iterations: int = get_settings().app.react_max_iterations
        self._idle_state: AgentState = AgentState.THINKING
        self._idle_context: dict[str, Any] = {}
        self._idle_grounding_score: float = 0.0
        self._default_tools_registered: bool = False
        self._tools_lock = asyncio.Lock()
        self._llm_lock = asyncio.Lock()

    def inject_services(self, services: dict[str, Any]) -> None:
        """Late-bind shared services after construction.

        Resets the default-tool flag so the next ``run()`` call will
        re-register tools with the newly available services.
        """
        from brain_os.service_keys import ALL_SERVICE_KEYS

        for key in services:
            if key not in ALL_SERVICE_KEYS:
                logger.warning("Unknown service key injected: %r", key)
        self._services.update(services)
        self._default_tools_registered = False
