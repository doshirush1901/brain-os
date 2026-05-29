"""Abstract base class for all Pantheon agents.

Every specialist agent inherits from :class:`BaseAgent`, which provides
LLM access (OpenAI, Anthropic, optional Ollama) via the centralised
:class:`~ira.services.llm_client.LLMClient`, knowledge-base search via
the :class:`~ira.brain.retriever.UnifiedRetriever`, a reference to the
:class:`~ira.message_bus.MessageBus` for inter-agent communication,
and an opt-in ReAct (Reason-Act-Observe) loop for agentic tool use.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from langfuse.decorators import observe
from pydantic import ValidationError

from brain_os.brain.retriever import UnifiedRetriever
from brain_os.config import get_settings
from brain_os.exceptions import IraError, LLMError, ToolExecutionError
from brain_os.message_bus import MessageBus
from brain_os.prompt_loader import load_prompt, load_soul_preamble
from brain_os.schemas.agent_handoff import AgentHandoffBrief
from brain_os.service_keys import ServiceKey as SK
from brain_os.services.llm_client import get_llm_client
from brain_os.services.tool_runner import is_retryable_transient, run_tool
from brain_os.skills import SKILL_MATRIX
from brain_os.skills.handlers import use_skill as _use_skill

logger = logging.getLogger(__name__)

_AGENT_ENRICHMENT_ERRORS = (OSError, KeyError, ValueError, TypeError, AttributeError, RuntimeError)
# TODO: define AgentDelegationError
_DELEGATION_ERRORS = (
    ToolExecutionError,
    IraError,
    ValueError,
    KeyError,
    RuntimeError,
    asyncio.TimeoutError,
    RecursionError,
)
_HANDOFF_JSON_ERRORS = (ValidationError, json.JSONDecodeError, ValueError, TypeError, KeyError)
_EMAIL_TOOL_ERRORS = (
    ToolExecutionError,
    httpx.HTTPError,
    asyncio.TimeoutError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
)
_LLM_REACT_ERRORS = (LLMError, httpx.HTTPError, asyncio.TimeoutError, ToolExecutionError)
_CRAWL_SCRAPE_ERRORS = (
    httpx.HTTPError,
    OSError,
    ToolExecutionError,
    RuntimeError,
    ImportError,
    ValueError,
    TypeError,
)
_RELATIONSHIP_EVENT_ERRORS = (
    IraError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
)

_DELEGATION_TOOL_NAMES = frozenset({"ask_agent", "delegate_to_agent"})
_EMPTY_RESULT_TOOL_NAMES = frozenset(
    {
        "scrape_url",
        "web_search",
        "searchapi_search",
        "search_emails",
        "read_email_thread",
        "fetch_news",
        "search_knowledge_base",
        "search_market_research",
        "search_linkedin_data",
    }
)


def _delegation_observation_meta(
    tool_name: str,
    safe_inputs: dict[str, Any],
    delegation_ms: int | None,
) -> dict[str, Any]:
    """Extra IRA_TOOL_META fields for delegation tools."""
    if tool_name not in _DELEGATION_TOOL_NAMES:
        return {}
    to_a = str(safe_inputs.get("agent_name") or "").lower().strip()
    meta: dict[str, Any] = {
        "delegation_target": to_a,
        "handoff_present": bool(str(safe_inputs.get("handoff_json") or "").strip()),
    }
    if delegation_ms is not None:
        meta["delegation_ms"] = delegation_ms
    return meta


def _normalize_searchapi_results(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Map SearchAPI.io JSON to uniform title/url/snippet rows (engine-dependent keys)."""
    for key in ("organic_results", "news_results"):
        raw = payload.get(key)
        if not isinstance(raw, list):
            continue
        rows: list[dict[str, str]] = []
        for r in raw:
            if not isinstance(r, dict):
                continue
            title = str(r.get("title") or r.get("name") or "")
            link = r.get("link") or r.get("url")
            url = str(link) if link is not None else ""
            snippet = str(
                r.get("snippet")
                or r.get("description")
                or r.get("summary")
                or r.get("content")
                or ""
            )
            if title or url or snippet:
                rows.append({"title": title, "url": url, "snippet": snippet})
        if rows:
            return rows
    return []


# ── ReAct infrastructure ─────────────────────────────────────────────────


class AgentState(Enum):
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    RESPONDING = "responding"


@dataclass
class AgentTool:
    """A tool that an agent can invoke during its ReAct loop."""

    name: str
    description: str
    parameters: dict[str, str]  # param_name -> description
    handler: Callable[..., Awaitable[str]]


class BaseAgent(ABC):
    """Abstract base for every agent in the Pantheon."""

    name: str = "base"
    role: str = ""
    description: str = ""
    model_provider: str = (
        "openai"  # "openai" or "anthropic" (ignored when IRA_DEFAULT_LLM_PROVIDER=ollama)
    )
    #: When set, ``_reason`` / ``call_llm`` pass this ``model_profile`` (e.g. ``\"writing\"`` for Calliope).
    preferred_model_profile: str | None = None
    knowledge_categories: list[str] = []
    timeout: int | None = None
    created_at: datetime = datetime(2024, 1, 1, tzinfo=UTC)

    @property
    def age_in_days(self) -> int:
        """Days since this agent's creation (identity continuity)."""
        delta = datetime.now(UTC) - self.created_at
        return max(0, delta.days)

    def _primary_llm_provider(self) -> str:
        """ReAct / ``call_llm`` primary provider before cloud fallback.

        When :envvar:`IRA_DEFAULT_LLM_PROVIDER` is ``ollama`` and ``OLLAMA_BASE_URL`` is set,
        all agents use local Ollama first regardless of per-agent ``model_provider``.
        """
        ctx = getattr(self, "_context", None)
        if isinstance(ctx, dict):
            if self.name == "athena":
                synth = str(ctx.get("llm_synthesis_provider_override") or "").strip().lower()
                if synth in {"openai", "anthropic", "ollama"}:
                    return synth
            override = str(ctx.get("llm_provider_override") or "").strip().lower()
            if override in {"openai", "anthropic", "ollama"}:
                return override
        llm_cfg = get_settings().llm
        if llm_cfg.default_llm_provider == "ollama" and (llm_cfg.ollama_base_url or "").strip():
            return "ollama"
        return "anthropic" if self.model_provider == "anthropic" else "openai"

    async def _compose_system_prompt(
        self,
        system_prompt: str = "",
        *,
        allow_default: bool = True,
    ) -> str:
        """Build a system prompt with SOUL preamble, identity (age), and latest journal."""
        prompt = system_prompt.strip()
        if not prompt and allow_default:
            prompt = (
                f"You are {self.name}, the {self.role} of the Machinecraft AI Pantheon. "
                f"{self.description}"
            )

        identity = f"You are {self.name}, the {self.role}. You are {self.age_in_days} days old."
        prompt = f"{identity}\n\n{prompt}"

        journal = None
        journal_svc = self._services.get(SK.AGENT_JOURNAL)
        if journal_svc is not None:
            try:
                journal = await journal_svc.get_latest_journal_entry(self.name)
            except _AGENT_ENRICHMENT_ERRORS:
                logger.debug("Failed to get latest journal for %s", self.name, exc_info=True)
        if journal and journal.strip():
            prompt = f"{prompt}\n\nYour most recent journal entry:\n{journal.strip()}"

        sense_lost = {}
        immune = self._services.get(SK.IMMUNE)
        if immune is not None and hasattr(immune, "get_sense_lost"):
            try:
                sense_lost = immune.get_sense_lost()
            except (AttributeError, TypeError, RuntimeError):
                pass
        if sense_lost:
            lost = [k for k, v in sense_lost.items() if v]
            if lost:
                warnings = []
                if "qdrant" in lost:
                    warnings.append(
                        "WARNING: Your semantic memory (Qdrant) is currently severed. "
                        "You cannot search past documents. Rely on immediate context and ask the user for clarification or documents if needed."
                    )
                if "neo4j" in lost:
                    warnings.append(
                        "WARNING: Your knowledge graph (Neo4j) is currently unavailable. "
                        "You cannot traverse relationships. Rely on other sources."
                    )
                if "openai" in lost or "voyage" in lost:
                    warnings.append(
                        "WARNING: Some model/embedding services are degraded. "
                        "Be explicit about limitations if you cannot fulfill a request."
                    )
                if warnings:
                    prompt = f"{prompt}\n\n" + "\n\n".join(warnings)

        tracker = self._services.get(SK.POWER_LEVEL_TRACKER)
        if tracker is not None and hasattr(tracker, "get_trust_matrix"):
            try:
                trust_matrix = tracker.get_trust_matrix(self.name)
                if trust_matrix:
                    from brain_os.brain.power_levels import DEFAULT_TRUST

                    informative: list[tuple[str, float]] = []
                    for other, score in trust_matrix.items():
                        try:
                            score_f = float(score)
                        except (TypeError, ValueError):
                            continue
                        if abs(score_f - DEFAULT_TRUST) >= 0.1:
                            informative.append((other, score_f))

                    informative.sort(key=lambda item: abs(item[1] - DEFAULT_TRUST), reverse=True)
                    lines = []
                    for other, score in informative[:8]:
                        if score >= 0.6:
                            lines.append(f"You trust {other}'s outputs (Trust: {score:.1f}).")
                        else:
                            lines.append(
                                f"You do not trust {other}'s outputs (Trust: {score:.1f}); verify before relying."
                            )
                    if lines:
                        prompt = f"{prompt}\n\n" + " ".join(lines)
            except _AGENT_ENRICHMENT_ERRORS:
                logger.debug("Trust matrix injection failed for %s", self.name, exc_info=True)

        soul = ""
        _ctx = getattr(self, "_context", None)
        if isinstance(_ctx, dict):
            _snap = _ctx.get("_soul_preamble_snapshot")
            if isinstance(_snap, str) and _snap.strip():
                soul = _snap.strip()
        if not soul:
            soul = load_soul_preamble()
        if soul and not prompt.startswith(soul):
            prompt = f"{soul}\n\n{prompt}" if prompt else soul
        try:
            guard = load_prompt("injection_guard")
            if guard.strip():
                prompt = f"{prompt}\n\n---\n{guard.strip()}"
        except FileNotFoundError:
            logger.debug("injection_guard prompt missing — skipping")
        return prompt

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
        self.state: AgentState = AgentState.THINKING
        self._default_tools_registered: bool = False
        self._last_grounding_score: float = 0.0
        self._tools_lock = asyncio.Lock()
        self._llm_lock = asyncio.Lock()

    async def _ensure_llm(self) -> None:
        if self._llm is not None:
            return
        async with self._llm_lock:
            if self._llm is None:
                self._llm = get_llm_client()

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

    # ── tool registration ────────────────────────────────────────────────

    def register_tool(self, tool: AgentTool) -> None:
        """Register a tool for use in the ReAct loop.

        Silently replaces an existing tool with the same name.
        """
        self.tools = [t for t in self.tools if t.name != tool.name]
        self.tools.append(tool)

    def _register_default_tools(self) -> None:
        """Register the standard tools available to every agent.

        Only registers tools whose backing service is present in
        ``self._services``.  Called lazily at the start of ``run()``.
        """
        if self._default_tools_registered:
            return
        self._default_tools_registered = True

        self.register_tool(
            AgentTool(
                name="search_knowledge",
                description="Search the internal knowledge base (Qdrant + Neo4j + Mem0).",
                parameters={"query": "Search query string", "limit": "Max results (default 10)"},
                handler=self._tool_search_knowledge,
            )
        )

        if self._services.get(SK.LONG_TERM_MEMORY):
            self.register_tool(
                AgentTool(
                    name="recall_memory",
                    description="Search long-term semantic memory (Mem0) for past facts and context.",
                    parameters={"query": "What to recall", "user_id": "User ID (default 'global')"},
                    handler=self._tool_recall_memory,
                )
            )
            self.register_tool(
                AgentTool(
                    name="store_memory",
                    description="Store an important fact or insight in long-term memory.",
                    parameters={
                        "content": "Fact to remember",
                        "user_id": "User ID (default 'global')",
                    },
                    handler=self._tool_store_memory,
                )
            )

        if self._services.get(SK.CONVERSATION_MEMORY):
            self.register_tool(
                AgentTool(
                    name="get_conversation_history",
                    description="Retrieve recent conversation history for a user. Returns the last 5 messages by default. Use recall_episodes for older context.",
                    parameters={
                        "user_id": "User ID",
                        "channel": "Channel (default 'CLI')",
                        "limit": "Max messages (default 5)",
                    },
                    handler=self._tool_get_conversation_history,
                )
            )

        if self._services.get(SK.RELATIONSHIP_MEMORY):
            self.register_tool(
                AgentTool(
                    name="check_relationship",
                    description="Look up the relationship profile for a contact (warmth, history, preferences).",
                    parameters={"contact_id": "Contact identifier"},
                    handler=self._tool_check_relationship,
                )
            )

        if self._services.get(SK.GOAL_MANAGER):
            self.register_tool(
                AgentTool(
                    name="check_goals",
                    description="Get the active goal for a contact (slot-filling progress, type).",
                    parameters={"contact_id": "Contact identifier"},
                    handler=self._tool_check_goals,
                )
            )

        if self._services.get(SK.MEMORY_BLOCK_STORE):
            self.register_tool(
                AgentTool(
                    name="read_memory_block",
                    description=(
                        "Read pinned core-memory blocks for a contact (human, relationship, "
                        "timeline, commitments). Omit label to return all blocks."
                    ),
                    parameters={
                        "contact_id": "Contact email scope",
                        "label": "Block label (optional; one of human, relationship, timeline, commitments)",
                    },
                    handler=self._tool_read_memory_block,
                )
            )
            self.register_tool(
                AgentTool(
                    name="update_memory_block",
                    description=(
                        "Update a pinned core-memory block for a contact. "
                        "Labels: human, relationship, timeline, commitments."
                    ),
                    parameters={
                        "contact_id": "Contact email scope",
                        "label": "Block label",
                        "value": "New block content",
                    },
                    handler=self._tool_update_memory_block,
                )
            )

        if self._services.get(SK.EPISODIC_MEMORY):
            self.register_tool(
                AgentTool(
                    name="recall_episodes",
                    description="Search episodic memory for past interaction narratives and key events.",
                    parameters={
                        "query": "What to search for",
                        "user_id": "User/contact ID",
                        "limit": "Max results (default 5)",
                    },
                    handler=self._tool_recall_episodes,
                )
            )

        if self._services.get(SK.PANTHEON):
            self.register_tool(
                AgentTool(
                    name="ask_agent",
                    description="Delegate a question to another specialist agent in the Pantheon.",
                    parameters={
                        "agent_name": "Name of the agent (e.g. 'clio', 'prometheus')",
                        "question": "The question to ask",
                        "handoff_json": (
                            "Optional JSON for AgentHandoffBrief: goal, bullets[], constraints[], "
                            "domain?, source_agent? — omit or leave empty if not needed"
                        ),
                    },
                    handler=self._tool_ask_agent,
                )
            )

        if self._services.get(SK.AGENT_JOURNAL):
            self.register_tool(
                AgentTool(
                    name="read_my_journal",
                    description="Look up your own past journal entries (nightly reflections written during Dream Mode).",
                    parameters={
                        "query": "Optional search query to filter entries; leave empty for recent entries"
                    },
                    handler=self._tool_read_journal,
                )
            )

        self.register_tool(
            AgentTool(
                name="web_search",
                description=(
                    "Search the web for real-time information. Returns titles, URLs, and snippets. "
                    "Requires at least one search API key (Tavily, Serper, or SearchAPI)."
                ),
                parameters={"query": "Web search query"},
                handler=self._tool_web_search_default,
            )
        )

        self.register_tool(
            AgentTool(
                name="scrape_url",
                description=(
                    "Fetch a web page and return its content as clean markdown. "
                    "Use after web_search to read the full content of a result URL."
                ),
                parameters={"url": "The full URL to scrape"},
                handler=self._tool_scrape_url,
            )
        )

        self.register_tool(
            AgentTool(
                name="check_known_entities",
                description=(
                    "Check if a company or contact is a known entity (existing customer, "
                    "vendor, sales agent, defunct company, or sister company). Use this "
                    "BEFORE classifying any contact as a lead to avoid misclassification."
                ),
                parameters={"query": "Company name, contact name, or email to look up"},
                handler=self._tool_check_known_entities,
            )
        )

        if self._services.get(SK.EMAIL_PROCESSOR):
            self.register_tool(
                AgentTool(
                    name="search_emails",
                    description=(
                        "Search Gmail for emails matching filters. Use this when the user asks "
                        "to find, show, or pull up emails from a person, company, or about a topic."
                    ),
                    parameters={
                        "from_address": "Sender email or partial match (e.g. 'contact@acme-corp.com' or 'acme-corp')",
                        "to_address": "Recipient email (optional)",
                        "subject": "Subject keyword (optional)",
                        "label": "Gmail label/folder (optional, e.g. 'HR' or 'Recruitment CVs')",
                        "query": "Free-form Gmail search query (optional, e.g. 'has:attachment')",
                        "after": "Date filter YYYY/MM/DD (optional)",
                        "before": "Date filter YYYY/MM/DD (optional)",
                        "max_results": "Max emails to return (default 10)",
                    },
                    handler=self._tool_search_emails,
                )
            )
            self.register_tool(
                AgentTool(
                    name="read_email_thread",
                    description=(
                        "Fetch the full email thread by thread ID. Use this after search_emails "
                        "to read the complete conversation history. If the search result had source_mailbox "
                        "(e.g. secondary@example.com for vendor/procurement mailbox), pass that as mailbox."
                    ),
                    parameters={
                        "thread_id": "Gmail thread ID from a search result",
                        "mailbox": "Optional: email of the mailbox that owns the thread (e.g. secondary@example.com for secondary/vendor mailbox)",
                    },
                    handler=self._tool_read_email_thread,
                )
            )

        if get_settings().apollo.api_key.get_secret_value():
            self.register_tool(
                AgentTool(
                    name="enrich_contact_apollo",
                    description=(
                        "Enrich a contact using Apollo.io (title, company, LinkedIn). "
                        "Pass email and optionally name or company/domain. Uses Apollo credits."
                    ),
                    parameters={
                        "email": "Contact email (best for matching)",
                        "name": "Full name (optional)",
                        "company_or_domain": "Company name or domain e.g. acme.com (optional)",
                    },
                    handler=self._tool_enrich_contact_apollo,
                )
            )
            self.register_tool(
                AgentTool(
                    name="sync_crm_apollo",
                    description=(
                        "Sync CRM with Apollo.io: enrich contacts (role, LinkedIn) and companies "
                        "(industry, website, employee count, region). Use when the user asks to sync or "
                        "refresh CRM data with Apollo, or to enrich all contacts/companies from Apollo."
                    ),
                    parameters={
                        "dry_run": "If true, only report what would be updated (default false)",
                        "limit": "Max contacts to process (default 20 for agent; use 0 or omit for no limit)",
                        "contacts_only": "If true, only enrich contacts, not companies (default false)",
                    },
                    handler=self._tool_sync_crm_apollo,
                )
            )
            self.register_tool(
                AgentTool(
                    name="search_people_apollo",
                    description=(
                        "Discover people at companies via Apollo.io search (mixed_people/api_search) "
                        "and optionally reveal work emails (uses credits; needs api_search-capable key). "
                        "Pass job titles and optionally company name and/or domains."
                    ),
                    parameters={
                        "person_titles": "Comma-separated job titles, e.g. 'CEO,Head of Procurement'",
                        "organization_name": "Company name keyword (optional)",
                        "organization_domains": "Comma-separated domains to narrow, e.g. 'acme.com' (optional)",
                        "page": "Results page (default 1)",
                        "per_page": "Page size 1–25 (default 10)",
                        "max_email_reveals": "Max reveal attempts 0–5 (default 3; each may consume credits)",
                    },
                    handler=self._tool_search_people_apollo,
                )
            )

        gdocs = self._services.get(SK.GOOGLE_DOCS)
        if gdocs is not None and getattr(gdocs, "calendar_available", False):
            self.register_tool(
                AgentTool(
                    name="check_calendar",
                    description=(
                        "List upcoming calendar events (primary Google Calendar). "
                        "Use when the user asks about schedule, availability, meetings, or what's coming up."
                    ),
                    parameters={
                        "days": "Number of days ahead to show (default 7)",
                        "max_results": "Max events to return (default 20)",
                    },
                    handler=self._tool_check_calendar,
                )
            )

        scheduling = self._services.get(SK.SCHEDULING)
        if scheduling is not None:
            self.register_tool(
                AgentTool(
                    name="propose_meeting_slots",
                    description=(
                        "Generate customer-safe meeting slots from explicit availability windows, "
                        "using Google Calendar free/busy and signed booking links."
                    ),
                    parameters={
                        "contact_email": "Customer email address",
                        "contact_name": "Customer name (optional)",
                        "purpose": "Meeting purpose",
                        "windows_json": (
                            "JSON list of windows, each with date YYYY-MM-DD, start_time HH:MM, "
                            "end_time HH:MM, and optional timezone (e.g. Asia/Kolkata)"
                        ),
                        "duration_minutes": "Meeting duration in minutes (default 30)",
                        "max_slots": "Maximum slots to offer (default 5)",
                        "thread_id": "Gmail thread id if replying in a thread (optional)",
                        "owner_mailbox": "Mailbox sending the request (optional)",
                    },
                    handler=self._tool_propose_meeting_slots,
                )
            )
            self.register_tool(
                AgentTool(
                    name="draft_scheduling_email",
                    description=(
                        "Create a polished scheduling email after slots have been proposed. "
                        "Pass the JSON returned by propose_meeting_slots."
                    ),
                    parameters={
                        "proposal_json": "JSON output from propose_meeting_slots",
                    },
                    handler=self._tool_draft_scheduling_email,
                )
            )

    # ── default tool handlers ─────────────────────────────────────────────

    async def _tool_search_knowledge(self, query: str, limit: str = "10") -> str:
        results = await self._retriever.search(query, limit=int(limit))
        if not results:
            return "No results found."
        lines = []
        for r in results:
            chunk_id = r.get("id", "?")
            source = r.get("source", "?")
            lines.append(f"- [Source: {source}, Chunk: {chunk_id}] {r.get('content', '')[:400]}")
        return "\n".join(lines)

    async def _tool_recall_memory(self, query: str, user_id: str | None = None) -> str:
        mem = self._services[SK.LONG_TERM_MEMORY]
        ctx = self._context or {}
        uid = user_id or ctx.get("mem0_user_id") or "global"
        results = await mem.search(query, user_id=uid)
        if not results:
            return "No memories found."
        lines = []
        for m in results:
            lines.append(f"- {m.get('memory', m.get('content', ''))}")
        return "\n".join(lines)

    async def _tool_store_memory(self, content: str, user_id: str | None = None) -> str:
        from brain_os.memory.store_policy import should_skip_mem_store

        skip, reason = should_skip_mem_store(content)
        if skip:
            return f"Not stored ({reason})."
        mem = self._services[SK.LONG_TERM_MEMORY]
        ctx = self._context or {}
        uid = user_id or ctx.get("mem0_user_id") or "global"
        result = await mem.store(content, user_id=uid)
        return f"Stored. ({len(result)} memory entries affected)"

    async def _tool_get_conversation_history(
        self,
        user_id: str,
        channel: str = "CLI",
        limit: str = "5",
    ) -> str:
        conv = self._services[SK.CONVERSATION_MEMORY]
        history = await conv.get_history(user_id, channel, limit=int(limit))
        if not history:
            return "No conversation history found."
        lines = []
        for msg in history:
            lines.append(f"[{msg.get('role', '?')}] {msg.get('content', '')[:300]}")
        return "\n".join(lines)

    async def _tool_check_relationship(self, contact_id: str) -> str:
        rel_mem = self._services[SK.RELATIONSHIP_MEMORY]
        rel = await rel_mem.get_relationship(contact_id)
        return json.dumps(
            {
                "contact_id": rel.contact_id,
                "warmth_level": rel.warmth_level.value
                if hasattr(rel.warmth_level, "value")
                else str(rel.warmth_level),
                "interaction_count": rel.interaction_count,
                "memorable_moments": rel.memorable_moments[:5],
                "learned_preferences": rel.learned_preferences,
            },
            default=str,
        )

    async def _tool_check_goals(self, contact_id: str) -> str:
        gm = self._services[SK.GOAL_MANAGER]
        goal = await gm.get_active_goal(contact_id)
        if goal is None:
            return f"No active goal for contact '{contact_id}'."
        return json.dumps(
            {
                "id": str(goal.id),
                "type": goal.goal_type.value,
                "status": goal.status.value,
                "progress": goal.progress,
                "slots": goal.required_slots,
            },
            default=str,
        )

    async def _tool_read_memory_block(self, contact_id: str, label: str = "") -> str:
        from brain_os.memory.blocks import BLOCK_SPECS, DEFAULT_BLOCK_LABELS, render_blocks_xml

        store = self._services[SK.MEMORY_BLOCK_STORE]
        scope = contact_id.strip().lower()
        wanted = (label.strip(),) if label.strip() else DEFAULT_BLOCK_LABELS
        if label.strip() and label.strip() not in BLOCK_SPECS:
            return f"Unknown label '{label}'. Valid: {', '.join(DEFAULT_BLOCK_LABELS)}"
        blocks = await store.get_blocks_for_scope(scope, wanted)
        xml = render_blocks_xml(blocks)
        if not xml:
            return f"No memory blocks for '{scope}'."
        return xml

    async def _tool_update_memory_block(self, contact_id: str, label: str, value: str) -> str:
        from brain_os.memory.blocks import BLOCK_SPECS

        store = self._services[SK.MEMORY_BLOCK_STORE]
        scope = contact_id.strip().lower()
        key = label.strip()
        if key not in BLOCK_SPECS:
            return f"Unknown label '{label}'. Valid: {', '.join(BLOCK_SPECS)}"
        block = await store.set_block(scope, key, value)
        return f"Updated {key} for {scope} ({len(block.value)} chars)."

    async def _tool_recall_episodes(
        self, query: str, user_id: str = "global", limit: str = "5"
    ) -> str:
        ep = self._services[SK.EPISODIC_MEMORY]
        results = await ep.surface_relevant_episodes(query, user_id)
        if not results:
            return "No episodic memories found."
        lines = []
        for e in results[: int(limit)]:
            ts = e.get("created_at", "?")
            narrative = e.get("narrative", e.get("content", ""))[:400]
            lines.append(f"- [{ts}] {narrative}")
        return "\n".join(lines)

    @property
    def _max_delegation_depth(self) -> int:
        try:
            from brain_os.config import get_settings

            return get_settings().app.max_delegation_depth
        except (ImportError, AttributeError, TypeError):
            return 5

    async def _tool_read_journal(self, query: str = "") -> str:
        """Return this agent's past journal entries (from Dream Mode reflections)."""
        journal = self._services.get(SK.AGENT_JOURNAL)
        if not journal:
            return "Journal service not available."
        try:
            entries = await journal.search_past_journals(
                agent_name=self.name,
                query=query.strip(),
                limit=10,
            )
        except _AGENT_ENRICHMENT_ERRORS as exc:
            logger.warning("read_my_journal failed in %s: %s", self.name, exc)
            return f"Journal lookup error: {exc}"
        if not entries:
            return "No journal entries found."
        lines = []
        for e in entries:
            lines.append(f"[{e.get('date', '?')}] {e.get('reflection_text', '')[:500]}")
        return "\n\n".join(lines)

    async def _tool_ask_agent(self, agent_name: str, question: str, handoff_json: str = "") -> str:
        depth = self._services.get("_delegation_depth", 0)
        limit = self._max_delegation_depth
        if depth >= limit:
            raise IraError(
                "Delegation depth limit reached. "
                "Please synthesize your answer from the information already gathered."
            )
        pantheon = self._services.get(SK.PANTHEON)
        if not pantheon:
            raise IraError("Pantheon service unavailable.")
        agent = pantheon.get_agent(agent_name.lower())
        if agent is None:
            raise IraError(f"Agent '{agent_name}' not found.")
        composed = question
        raw_hj = (handoff_json or "").strip()
        if raw_hj:
            try:
                brief = AgentHandoffBrief.model_validate_json(raw_hj)
                if not brief.source_agent:
                    brief = brief.model_copy(update={"source_agent": self.name})
                composed = AgentHandoffBrief.compose_query(brief, question)
            except _HANDOFF_JSON_ERRORS as exc:
                raise IraError(f"Invalid handoff_json: {exc}") from exc
        child_ctx: dict[str, Any] = {"_delegation_depth": depth + 1}
        if isinstance(getattr(self, "_context", None), dict):
            for key in ("email_scope", "run_id", "contact_id", "channel", "_tool_audit"):
                val = self._context.get(key)
                if val is not None:
                    child_ctx[key] = val
        try:
            return await agent.handle(
                composed,
                {
                    "services": child_ctx,
                    **child_ctx,
                },
            )
        except _DELEGATION_ERRORS as exc:
            logger.warning("Delegation to '%s' failed in %s: %s", agent_name, self.name, exc)
            raise IraError(f"Agent '{agent_name}' error: {exc}") from exc

    async def _tool_search_emails(
        self,
        from_address: str = "",
        to_address: str = "",
        subject: str = "",
        label: str = "",
        query: str = "",
        after: str = "",
        before: str = "",
        max_results: str = "10",
    ) -> str:
        _email_scope = self._context.get("email_scope", "both") if self._context else "both"
        if _email_scope == "no_email":
            return "Email search skipped (query does not require email data). Use search_knowledge for KB results."
        if _email_scope == "imported_email":
            return "Live email search skipped (using imported/indexed email via KB). Use search_knowledge for faster results from Qdrant."
        ep = self._services[SK.EMAIL_PROCESSOR]
        try:
            emails = await ep.search_emails(
                from_address=from_address,
                to_address=to_address,
                subject=subject,
                label=label,
                query=query,
                after=after,
                before=before,
                max_results=int(max_results),
            )
        except _EMAIL_TOOL_ERRORS as exc:
            logger.warning("search_emails failed in %s: %s", self.name, exc)
            return f"Email search error: {exc}"
        if not emails:
            return "No emails found matching the search criteria."
        lines = []
        for e in emails:
            mb = f" | Mailbox: {e.source_mailbox}" if getattr(e, "source_mailbox", None) else ""
            lines.append(
                f"- [{e.received_at.strftime('%Y-%m-%d %H:%M')}] "
                f"From: {e.from_address} | To: {e.to_address} | "
                f"Subject: {e.subject} | Thread: {e.thread_id}{mb}\n"
                f"  Body preview: {e.body[:500]}"
            )
        return "\n".join(lines)

    async def _tool_read_email_thread(self, thread_id: str, mailbox: str = "") -> str:
        ep = self._services[SK.EMAIL_PROCESSOR]
        try:
            emails = await ep.get_thread(thread_id, mailbox=mailbox.strip() or None)
        except _EMAIL_TOOL_ERRORS as exc:
            logger.warning("read_email_thread failed in %s: %s", self.name, exc)
            return f"Thread read error: {exc}"
        if not emails:
            return "Thread is empty or not found."
        lines = []
        for e in emails:
            mb = f" [Mailbox: {e.source_mailbox}]" if getattr(e, "source_mailbox", None) else ""
            lines.append(
                f"--- [{e.received_at.strftime('%Y-%m-%d %H:%M')}]{mb} "
                f"From: {e.from_address} → To: {e.to_address} ---\n"
                f"Subject: {e.subject}\n{e.body}\n"
            )
        return "\n".join(lines)

    async def _tool_enrich_contact_apollo(
        self,
        email: str = "",
        name: str = "",
        company_or_domain: str = "",
    ) -> str:
        from brain_os.systems.apollo_client import enrich_person_async

        if not email and not name and not company_or_domain:
            return "Provide at least one of: email, name, or company_or_domain."
        domain = None
        org = None
        if company_or_domain:
            s = company_or_domain.strip()
            if "." in s and " " not in s:
                domain = s.replace("www.", "")
            else:
                org = s
        try:
            result = await enrich_person_async(
                email=email.strip() or None,
                name=name.strip() or None,
                domain=domain,
                organization_name=org,
            )
        except (
            ToolExecutionError,
            httpx.TransportError,
            httpx.TimeoutException,
            KeyError,
            ValueError,
        ) as exc:
            logger.warning("enrich_contact_apollo failed in %s: %s", self.name, exc)
            return f"Apollo enrichment error: {exc}"
        if not result:
            return "No Apollo match found for that contact."
        parts = [f"**{result.get('name') or 'Unknown'}**"]
        if result.get("title"):
            parts.append(f"Title: {result['title']}")
        if result.get("organization_name"):
            parts.append(f"Company: {result['organization_name']}")
        if result.get("email"):
            parts.append(f"Email: {result['email']}")
        if result.get("linkedin_url"):
            parts.append(f"LinkedIn: {result['linkedin_url']}")
        return "\n".join(parts)

    async def _tool_sync_crm_apollo(
        self,
        dry_run: str = "false",
        limit: str = "20",
        contacts_only: str = "false",
    ) -> str:
        """Run Apollo sync for CRM; returns a short summary of contacts/companies updated."""
        crm = self._services.get(SK.CRM)
        if not crm:
            return "CRM is not available; cannot run Apollo sync."
        from brain_os.systems.apollo_crm_sync import sync_crm_with_apollo

        try:
            limit_val: int | None = int(limit) if limit and limit.strip() else 20
            if limit_val <= 0:
                limit_val = None
            result = await sync_crm_with_apollo(
                crm,
                dry_run=dry_run.strip().lower() in ("true", "1", "yes"),
                limit=limit_val,
                contact_type=None,
                contacts_only=contacts_only.strip().lower() in ("true", "1", "yes"),
            )
        except (
            ToolExecutionError,
            httpx.TransportError,
            httpx.TimeoutException,
            KeyError,
            ValueError,
        ) as exc:
            logger.warning("sync_crm_apollo failed in %s: %s", self.name, exc)
            return f"Apollo sync error: {exc}"

        if result.get("contact_type_error"):
            return result["contact_type_error"]
        return (
            f"Apollo sync done. Contacts updated: {result['contacts_updated']}, "
            f"companies updated: {result['companies_updated']}, "
            f"skipped (no email): {result['skipped_no_email']}, no match: {result['no_match']}, errors: {result['errors']}."
        )

    async def _tool_search_people_apollo(
        self,
        person_titles: str,
        organization_name: str = "",
        organization_domains: str = "",
        page: str = "1",
        per_page: str = "10",
        max_email_reveals: str = "3",
    ) -> str:
        from brain_os.systems.apollo_client import search_people_async

        def _csv_parts(raw: str) -> list[str]:
            return [p.strip() for p in (raw or "").split(",") if p.strip()]

        titles = _csv_parts(person_titles)
        if not titles:
            return "Provide person_titles (comma-separated), e.g. 'CEO,VP Engineering'."
        org_names = [organization_name.strip()] if (organization_name or "").strip() else []
        domains_raw = _csv_parts(organization_domains)
        domains = domains_raw or None
        try:
            page_i = max(1, int(page)) if (page or "").strip() else 1
            per_i = min(25, max(1, int(per_page))) if (per_page or "").strip() else 10
            reveals = (
                min(5, max(0, int(max_email_reveals))) if (max_email_reveals or "").strip() else 3
            )
        except ValueError:
            return "Invalid page, per_page, or max_email_reveals (integers expected)."

        try:
            rows = await search_people_async(
                organization_names=org_names,
                person_titles=titles,
                organization_domains=domains,
                page=page_i,
                per_page=per_i,
                max_email_reveals=reveals,
            )
        except (
            ToolExecutionError,
            httpx.TransportError,
            httpx.TimeoutException,
            KeyError,
            ValueError,
        ) as exc:
            logger.warning("search_people_apollo failed in %s: %s", self.name, exc)
            return f"Apollo people search error: {exc}"
        if not rows:
            return (
                "No people returned (empty search, no matches, or Apollo api_search blocked for this key). "
                "Try different titles/domains or check Apollo key permissions."
            )
        return json.dumps(rows, indent=2, default=str)

    async def _tool_check_calendar(
        self,
        days: str = "7",
        max_results: str = "20",
    ) -> str:
        gdocs = self._services.get(SK.GOOGLE_DOCS)
        if not gdocs or not getattr(gdocs, "calendar_available", False):
            return "Calendar is not available."
        try:
            d = int(days) if days else 7
            m = int(max_results) if max_results else 20
            d = max(1, min(d, 31))
            m = max(1, min(m, 50))
        except ValueError:
            d, m = 7, 20
        try:
            events = await gdocs.list_upcoming_events(days=d, max_results=m)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("check_calendar failed in %s: %s", self.name, exc)
            return f"Calendar error: {exc}"
        if not events:
            return f"No events in the next {d} days."
        lines = [f"Upcoming events (next {d} days):"]
        for ev in events:
            start = ev.get("start", "") or ""
            summary = ev.get("summary", "(No title)")
            loc = ev.get("location", "")
            line = f"- {start}: {summary}"
            if loc:
                line += f" @ {loc}"
            lines.append(line)
        return "\n".join(lines)

    async def _tool_propose_meeting_slots(
        self,
        contact_email: str,
        contact_name: str = "",
        purpose: str = "a quick meeting",
        windows_json: str = "[]",
        duration_minutes: str = "30",
        max_slots: str = "5",
        thread_id: str = "",
        owner_mailbox: str = "",
    ) -> str:
        from brain_os.services.scheduling import SchedulingWindow

        scheduling = self._services.get(SK.SCHEDULING)
        if scheduling is None:
            return "Scheduling service is not available."
        try:
            raw_windows = json.loads(windows_json or "[]")
            if not isinstance(raw_windows, list):
                return "windows_json must be a JSON list."
            default_tz = get_settings().app.scheduling_default_timezone
            windows = [
                SchedulingWindow(
                    date=str(item["date"]),
                    start_time=str(item["start_time"]),
                    end_time=str(item["end_time"]),
                    timezone_name=str(item.get("timezone") or default_tz),
                )
                for item in raw_windows
                if isinstance(item, dict)
            ]
            result = await scheduling.propose_slots(
                contact_email=contact_email,
                contact_name=contact_name or None,
                purpose=purpose,
                windows=windows,
                duration_minutes=int(duration_minutes or 30),
                max_slots=int(max_slots or 5),
                thread_id=thread_id or None,
                owner_mailbox=owner_mailbox or None,
            )
            return json.dumps(result, indent=2, default=str)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("propose_meeting_slots failed in %s: %s", self.name, exc)
            return f"Scheduling error: {exc}"

    async def _tool_draft_scheduling_email(self, proposal_json: str) -> str:
        try:
            proposal = json.loads(proposal_json or "{}")
        except json.JSONDecodeError as exc:
            return f"Invalid proposal_json: {exc}"
        body = proposal.get("email_body")
        if body:
            return str(body)
        slots = proposal.get("slots") or []
        if not slots:
            return "No slots found in proposal_json."
        lines = [
            "Hi,",
            "",
            f"Would any of these slots work for {proposal.get('purpose') or 'a quick meeting'}?",
            "",
        ]
        for slot in slots:
            lines.append(f"- {slot.get('label')}: {slot.get('booking_url')}")
        lines.extend(["", "Best regards,", "Brain OS Operator"])
        return "\n".join(lines)

    async def _tool_web_search_default(self, query: str) -> str:
        results = await self.web_search(query, max_results=5)
        if not results:
            return "No web search results found."
        lines = []
        for r in results:
            lines.append(f"- [{r['title']}]({r['url']}): {r['snippet'][:200]}")
        return "\n".join(lines)

    async def _tool_scrape_url(self, url: str) -> str:
        return await self.scrape_url(url)

    _KNOWN_ENTITIES_PATH = Path("data/brain/known_entities.json")

    async def _tool_check_known_entities(self, query: str) -> str:
        """Check if a company/contact matches a known entity."""
        try:
            if not self._KNOWN_ENTITIES_PATH.exists():
                return "Known entities registry not found."
            raw = await asyncio.to_thread(self._KNOWN_ENTITIES_PATH.read_text)
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            return f"Error reading known entities: {exc}"

        query_lower = query.lower()
        matches: list[str] = []

        for agent in data.get("agents", []):
            if (
                query_lower in agent.get("name", "").lower()
                or query_lower in agent.get("email", "").lower()
            ):
                matches.append(
                    f"AGENT: {agent['name']} ({agent.get('email', '')}) — "
                    f"Region: {agent.get('region', '?')}. {agent.get('note', '')}"
                )

        for cust in data.get("existing_customers", []):
            if (
                query_lower in cust.get("name", "").lower()
                or query_lower in cust.get("email", "").lower()
            ):
                matches.append(
                    f"EXISTING CUSTOMER: {cust['name']} — "
                    f"Machine: {cust.get('machine', '?')}. {cust.get('note', '')}"
                )

        for vendor in data.get("vendors", []):
            if (
                query_lower in vendor.get("name", "").lower()
                or query_lower in vendor.get("email", "").lower()
            ):
                matches.append(
                    f"VENDOR: {vendor['name']} — "
                    f"Service: {vendor.get('service', '?')}. {vendor.get('note', '')}"
                )

        for defunct in data.get("defunct_companies", []):
            if query_lower in defunct.get("name", "").lower():
                matches.append(f"DEFUNCT COMPANY: {defunct['name']} — {defunct.get('note', '')}")

        for sister in data.get("sister_companies", []):
            if (
                query_lower in sister.get("name", "").lower()
                or query_lower in sister.get("domain", "").lower()
            ):
                matches.append(f"SISTER COMPANY: {sister['name']} — {sister.get('note', '')}")

        for excluded in data.get("not_machinecraft", []):
            if query_lower in excluded.get("name", "").lower():
                matches.append(f"NOT MACHINECRAFT: {excluded['name']} — {excluded.get('note', '')}")

        if not matches:
            return f"No known entity match for '{query}'. Proceed with normal classification."
        return "\n".join(matches)

    async def _log_journal_action(self, query: str, outcome: str) -> None:
        """Log this agent's action to the journal when the service is available."""
        journal = self._services.get(SK.AGENT_JOURNAL)
        if journal is None:
            return
        try:
            await journal.log_action(
                agent_name=self.name,
                action_text=f"Handled query: {query[:100]}",
                outcome=outcome,
            )
        except _AGENT_ENRICHMENT_ERRORS:
            logger.debug("AgentJournal.log_action failed for %s", self.name, exc_info=True)

    # ── grounding score ────────────────────────────────────────────────────

    _RETRIEVAL_TOOLS: frozenset[str] = frozenset(
        {
            "search_knowledge",
            "recall_memory",
            "recall_episodes",
            "search_emails",
            "read_email_thread",
            "ask_agent",
            "web_search",
            "scrape_url",
        }
    )
    # Agents that may answer without retrieval: gatekeeper (clarify only), gap resolver (synthesis).
    _GROUNDING_EXEMPT_AGENTS: frozenset[str] = frozenset({"sphinx", "gapper"})

    @staticmethod
    def _compute_grounding_score(scratchpad: list[dict[str, str]]) -> float:
        """Score how well the answer is grounded in tool outputs.

        Returns 1.0 if retrieval tools were used, 0.5 if only
        non-retrieval tools, 0.0 if no tools were called.
        """
        if not scratchpad:
            return 0.0
        tool_names = set()
        for entry in scratchpad:
            action = entry.get("action", "")
            paren = action.find("(")
            if paren > 0:
                tool_names.add(action[:paren])
        if not tool_names:
            return 0.0
        if tool_names & BaseAgent._RETRIEVAL_TOOLS:
            return 1.0
        return 0.5

    # ── ReAct loop ────────────────────────────────────────────────────────

    def _build_tool_descriptions(self, context: dict[str, Any] | None = None) -> str:
        """Format registered tools into a description block for the LLM.

        When pipeline ``tool_discovery`` metadata is present, preferred tools
        are listed first and annotated so ReAct tool selection can follow the
        progressive-discovery shortlist while still keeping full fallback.
        """
        if not self.tools:
            return "(No tools available)"
        preferred_tools: list[str] = []
        if isinstance(context, dict):
            td = context.get("tool_discovery")
            if isinstance(td, dict):
                raw = td.get("shortlisted_tools", [])
                if isinstance(raw, list):
                    preferred_tools = [str(t) for t in raw if isinstance(t, str)]

        tools = list(self.tools)
        if preferred_tools:
            preferred_set = set(preferred_tools)
            tools.sort(key=lambda t: (0 if t.name in preferred_set else 1, t.name))

        lines = []
        if preferred_tools:
            lines.append(
                "Preferred tools for this request (try first): " + ", ".join(preferred_tools)
            )
        for t in tools:
            params = ", ".join(f"{k}: {v}" for k, v in t.parameters.items())
            preferred_tag = " [preferred]" if t.name in preferred_tools else ""
            lines.append(f"  - {t.name}({params}): {t.description}{preferred_tag}")
        return "\n".join(lines)

    def _session_user_for_llm(self) -> tuple[str | None, str | None]:
        """``run_id`` and ``mem0_user_id`` from pipeline context for tracing (Langfuse + Helicone)."""
        ctx = getattr(self, "_context", None)
        if not isinstance(ctx, dict):
            return (None, None)
        rid = str(ctx.get("run_id") or "").strip()
        uid = str(ctx.get("mem0_user_id") or "").strip()
        return (rid or None, uid or None)

    async def _reason(
        self,
        agent_system_prompt: str,
        query: str,
        context: dict[str, Any] | None,
        scratchpad: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Ask the LLM to decide the next action (tool call or final answer)."""
        react_prompt = load_prompt("react_system")

        tool_block = self._build_tool_descriptions(context)
        system = (
            f"{agent_system_prompt}\n\n"
            f"--- TOOLS ---\n{tool_block}\n\n"
            f"--- REASONING PROTOCOL ---\n{react_prompt}"
        )

        scratchpad_text = ""
        if scratchpad:
            parts = []
            for entry in scratchpad:
                parts.append(
                    f"Thought: {entry.get('thought', '')}\n"
                    f"Action: {entry.get('action', '')}\n"
                    f"Observation: {entry.get('observation', '')}"
                )
            scratchpad_text = "\n---\n".join(parts)

        ctx_text = ""
        if context:
            ctx_text = f"\n\nAdditional context: {json.dumps(context, default=str)[:2000]}"

        delimited_query = f"<<<USER INPUT>>>\n{query}\n<<<END INPUT>>>"
        user_msg = f"Query: {delimited_query}{ctx_text}"
        if scratchpad_text:
            user_msg += f"\n\nPrevious reasoning steps:\n{scratchpad_text}\n\nContinue reasoning."

        llm_cfg = get_settings().llm
        primary = llm_cfg.resolve_primary_provider_for_profile(
            baseline_primary=self._primary_llm_provider(),
            model_profile=self.preferred_model_profile,
        )
        _sess, _user = self._session_user_for_llm()
        await self._ensure_llm()
        try:
            raw = await self._llm.generate_text_with_fallback(
                system,
                user_msg,
                primary=primary,
                temperature=0.2,
                name=f"{self.name}.reason",
                model_profile=self.preferred_model_profile,
                session_id=_sess,
                user_id=_user,
            )
        except _LLM_REACT_ERRORS as exc:
            logger.warning("ReAct LLM call failed in %s: %s", self.name, exc)
            return {"thought": "LLM unavailable.", "final_answer": "(LLM call failed)"}

        try:
            parsed = self._parse_json_response(raw)
            # `_parse_json_response` returns {} when no JSON is found; do not treat
            # that as a valid empty ReAct decision (would yield "(No response)").
            if isinstance(parsed, dict) and parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse ReAct JSON in %s", self.name, exc_info=True)

        return {"thought": "Could not parse structured response.", "final_answer": raw}

    @staticmethod
    def _ira_tool_observation(tool_name: str, ok: bool, body: str, **meta: Any) -> str:
        payload: dict[str, Any] = {"ok": ok, "tool": tool_name}
        for k, v in meta.items():
            if v is not None:
                payload[k] = v
        return "IRA_TOOL_META:" + json.dumps(payload, default=str) + "\n" + body

    async def _execute_tool(self, name: str, inputs: dict[str, Any]) -> str:
        """Find and execute a registered tool by name.

        Strips unexpected keyword arguments that the LLM may hallucinate
        (e.g. ``limit``, ``max_results``) so tool handlers don't crash
        with ``got an unexpected keyword argument``.

        Successful and failed executions prefix the observation with a single
        ``IRA_TOOL_META:{...}`` JSON line (machine-readable audit) before the
        human-readable tool body.
        """

        def _audit(
            success: bool,
            tool_input: dict[str, Any] | None = None,
            *,
            delegation_ms: int | None = None,
        ) -> None:
            ctx = getattr(self, "_context", None)
            bucket = ctx.get("_tool_audit") if isinstance(ctx, dict) else None
            if isinstance(bucket, list):
                entry: dict[str, Any] = {
                    "agent": self.name,
                    "tool": name,
                    "success": success,
                }
                if name == "read_email_thread":
                    tid = str((tool_input or {}).get("thread_id") or "").strip()
                    if tid:
                        entry["thread_id"] = tid
                if name in _DELEGATION_TOOL_NAMES and tool_input:
                    to_a = str(tool_input.get("agent_name") or "").lower().strip()
                    if to_a:
                        entry["to_agent"] = to_a
                    if delegation_ms is not None:
                        entry["duration_ms"] = delegation_ms
                    hj = str(tool_input.get("handoff_json") or "").strip()
                    entry["handoff_present"] = bool(hj)
                    if hj:
                        try:
                            fp = AgentHandoffBrief.model_validate_json(hj).domain_fingerprint()
                            if fp:
                                entry["handoff_domain_fp"] = fp
                        except (ValueError, TypeError, json.JSONDecodeError):
                            pass
                bucket.append(entry)

        tracker = self._services.get("tool_stats_tracker")
        _ctx_run = getattr(self, "_context", None)
        _run_id = str(_ctx_run.get("run_id") or "").strip() if isinstance(_ctx_run, dict) else ""
        for tool in self.tools:
            if tool.name == name:
                declared_params = set(tool.parameters.keys())
                safe_inputs = {k: v for k, v in inputs.items() if k in declared_params}
                if len(safe_inputs) < len(inputs):
                    _dropped = set(inputs) - declared_params
                    logger.debug("Tool '%s': dropped undeclared params %s", name, _dropped)
                app_cfg = get_settings().app
                max_retries = max(0, int(app_cfg.react_tool_transient_retries))
                tool_timeout = float(app_cfg.react_tool_timeout_seconds)

                async def _record_delegation(ok: bool, ms: int | None) -> None:
                    if tracker is None or name not in _DELEGATION_TOOL_NAMES or ms is None:
                        return
                    if not hasattr(tracker, "record_delegation"):
                        return
                    to_a = str(safe_inputs.get("agent_name") or "").lower().strip()
                    await tracker.record_delegation(self.name, to_a, ms, ok=ok, run_id=_run_id)

                async def _invoke() -> Any:
                    return await tool.handler(**safe_inputs)

                t_start = time.monotonic()
                tool_result = await run_tool(
                    f"{self.name}.{name}",
                    _invoke,
                    retries=max_retries,
                    backoff_seconds=0.3,
                    timeout_seconds=tool_timeout,
                    is_retryable=is_retryable_transient,
                )
                elapsed_ms = int((time.monotonic() - t_start) * 1000)
                deleg_ms = elapsed_ms if name in _DELEGATION_TOOL_NAMES else None
                obs_meta = {
                    **_delegation_observation_meta(name, safe_inputs, deleg_ms),
                    "attempts": tool_result.attempts,
                }

                if tool_result.ok:
                    raw = tool_result.value
                    if name in _EMPTY_RESULT_TOOL_NAMES:
                        empty = (
                            raw is None
                            or (isinstance(raw, str) and not raw.strip())
                            or (isinstance(raw, (list, dict)) and not raw)
                        )
                        if empty:
                            if tracker is not None:
                                await tracker.record_tool_call(
                                    self.name,
                                    name,
                                    False,
                                    error_code="empty_result",
                                    run_id=_run_id or None,
                                    duration_ms=elapsed_ms,
                                )
                                await _record_delegation(False, deleg_ms)
                            _audit(False, safe_inputs, delegation_ms=deleg_ms)
                            return self._ira_tool_observation(
                                name,
                                False,
                                "Tool returned no usable data. Try different parameters or another tool.",
                                error_code="empty_result",
                                **obs_meta,
                            )
                    if tracker is not None:
                        await tracker.record_tool_call(
                            self.name,
                            name,
                            True,
                            run_id=_run_id or None,
                            duration_ms=elapsed_ms,
                        )
                        await _record_delegation(True, deleg_ms)
                    _audit(True, safe_inputs, delegation_ms=deleg_ms)
                    return self._ira_tool_observation(
                        name,
                        True,
                        str(raw)[:4000],
                        **obs_meta,
                    )

                exc = tool_result.last_exception
                if isinstance(exc, ToolExecutionError):
                    if tracker is not None:
                        await tracker.record_tool_call(
                            self.name,
                            name,
                            False,
                            error_code="ToolExecutionError",
                            run_id=_run_id or None,
                            duration_ms=elapsed_ms,
                        )
                        await _record_delegation(False, deleg_ms)
                    _audit(False, safe_inputs, delegation_ms=deleg_ms)
                    raise exc

                if isinstance(exc, httpx.HTTPStatusError):
                    code = f"http_{exc.response.status_code}" if exc.response else "http_error"
                    if tracker is not None:
                        await tracker.record_tool_call(
                            self.name,
                            name,
                            False,
                            error_code=code,
                            run_id=_run_id or None,
                            duration_ms=elapsed_ms,
                        )
                        await _record_delegation(False, deleg_ms)
                    _audit(False, safe_inputs, delegation_ms=deleg_ms)
                    return self._ira_tool_observation(
                        name,
                        False,
                        "",
                        error_code=code,
                        **obs_meta,
                    )

                logger.warning(
                    "Tool '%s' failed in %s after %d attempts: %s",
                    name,
                    self.name,
                    tool_result.attempts,
                    tool_result.error,
                )
                err_code = type(exc).__name__ if exc is not None else "tool_failed"
                if tracker is not None:
                    await tracker.record_tool_call(
                        self.name,
                        name,
                        False,
                        error_code=err_code,
                        run_id=_run_id or None,
                        duration_ms=elapsed_ms,
                    )
                    await _record_delegation(False, deleg_ms)
                _audit(False, safe_inputs, delegation_ms=deleg_ms)
                detail = tool_result.error or (str(exc) if exc else "unknown")
                msg = (
                    f"Tool execution failed after {tool_result.attempts} attempt(s): {detail}. "
                    "Please check your parameters and try again, or use a different tool."
                )
                return self._ira_tool_observation(
                    name,
                    False,
                    msg,
                    error_code=err_code,
                    **obs_meta,
                )
        if tracker is not None:
            await tracker.record_tool_call(
                self.name,
                name,
                False,
                error_code="unknown_tool",
                run_id=_run_id or None,
            )
        _audit(False)
        return self._ira_tool_observation(
            name,
            False,
            f"Unknown tool: {name}",
            error_code="unknown_tool",
        )

    async def _force_final_answer(
        self,
        agent_system_prompt: str,
        query: str,
        scratchpad: list[dict[str, str]],
    ) -> str:
        """Synthesise a final answer when max iterations are reached."""
        observations = "\n".join(
            f"- {e.get('thought', '')}: {e.get('observation', '')[:300]}"
            for e in scratchpad
            if e.get("observation")
        )
        user_msg = (
            f"Original query: {query}\n\n"
            f"Research gathered so far:\n{observations}\n\n"
            "You have reached the maximum number of reasoning steps. "
            "Synthesise the best possible answer from the information above."
        )
        return await self.call_llm(agent_system_prompt, user_msg, system_already_composed=True)

    @observe(name="agent.run")
    async def run(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        *,
        system_prompt: str = "",
        on_progress: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> str:
        """Execute the ReAct (Reason-Act-Observe) loop.

        Subclasses that want agentic behaviour should call this method
        (typically from their ``handle()`` override) instead of doing a
        single-pass LLM call.

        Parameters
        ----------
        query:
            The user/caller query.
        context:
            Optional context dict forwarded from the caller.  May contain
            a ``"services"`` key with live service references from the
            pipeline, which are merged into ``self._services`` so that
            ReAct tools can query memory dynamically.
        system_prompt:
            The agent-specific system prompt to prepend to the ReAct
            protocol.  If empty, a minimal default is used.
        on_progress:
            Optional async callback for streaming progress events.
        """
        _progress = on_progress or (context or {}).get("_on_progress")

        self._context = context or {}
        self.state = AgentState.THINKING
        self._last_grounding_score = 0.0
        self._services.pop("_delegation_depth", None)

        if context and "services" in context:
            for key, svc in context["services"].items():
                if svc is not None and key not in self._services:
                    self._services[key] = svc
            self._default_tools_registered = False

        async with self._tools_lock:
            self._register_default_tools()

        agent_prompt = await self._compose_system_prompt(system_prompt)
        if get_settings().app.gepa_strategy_overlay_runtime:
            from brain_os.brain.strategy_overlays import append_strategy_overlay_fragments

            agent_prompt = append_strategy_overlay_fragments(
                agent=self.name,
                user_query=query,
                prompt=agent_prompt,
            )

        scratchpad: list[dict[str, str]] = []
        self.state = AgentState.THINKING

        for iteration in range(self.max_iterations):
            self.state = AgentState.THINKING
            if _progress:
                await _progress(
                    {"type": "agent_thinking", "agent": self.name, "iteration": iteration + 1}
                )

            decision = await self._reason(agent_prompt, query, context, scratchpad)

            thought = decision.get("thought", "")

            if "final_answer" in decision:
                self.state = AgentState.RESPONDING
                self._last_grounding_score = self._compute_grounding_score(scratchpad)
                if (
                    self._last_grounding_score == 0.0
                    and len(decision["final_answer"]) > 100
                    and self.name not in BaseAgent._GROUNDING_EXEMPT_AGENTS
                ):
                    logger.warning(
                        "GROUNDING | %s answered without retrieval tools — hallucination risk",
                        self.name,
                    )
                logger.info(
                    "%s reached final answer after %d iterations (grounding=%.1f)",
                    self.name,
                    iteration + 1,
                    self._last_grounding_score,
                )
                await self._log_journal_action(query, "success")
                return decision["final_answer"]

            tool_call = decision.get("tool_to_use")
            if not isinstance(tool_call, dict) or "name" not in tool_call:
                self.state = AgentState.RESPONDING
                self._last_grounding_score = self._compute_grounding_score(scratchpad)
                await self._log_journal_action(query, "success")
                return decision.get("final_answer", thought or "(No response)")

            tool_name = tool_call["name"]
            tool_input = tool_call.get("input", {})

            self.state = AgentState.ACTING
            logger.info(
                "%s [iter %d] calling tool '%s'",
                self.name,
                iteration + 1,
                tool_name,
            )
            if _progress:
                await _progress(
                    {
                        "type": "tool_called",
                        "agent": self.name,
                        "tool": tool_name,
                        "iteration": iteration + 1,
                    }
                )
            observation = await self._execute_tool(tool_name, tool_input)

            self.state = AgentState.OBSERVING
            scratchpad.append(
                {
                    "thought": thought,
                    "action": f"{tool_name}({json.dumps(tool_input, default=str)})",
                    "observation": observation,
                }
            )

        logger.warning(
            "%s hit max iterations (%d) — forcing final answer",
            self.name,
            self.max_iterations,
        )
        self._last_grounding_score = self._compute_grounding_score(scratchpad)
        await self._log_journal_action(query, "success")
        return await self._force_final_answer(agent_prompt, query, scratchpad)

    # ── abstract interface ───────────────────────────────────────────────

    @abstractmethod
    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        """Process a query and return a response string.

        Existing agents implement this as a single-pass function.
        To opt into the ReAct loop, an agent's ``handle()`` can call
        ``await self.run(query, context, system_prompt=...)`` instead.
        """

    # ── LLM access ───────────────────────────────────────────────────────

    _LLM_CACHE_TTL = 3600

    def _llm_cache_key(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        model_profile: str | None = None,
    ) -> str:
        """Deterministic hash for caching LLM responses."""
        raw = (
            f"{self.name}:{temperature:.2f}:{model_profile or ''}:"
            f"{system_prompt[:500]}:{user_message[:2000]}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    async def call_llm(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.3,
        model_profile: str | None = None,
        system_already_composed: bool = False,
    ) -> str:
        """Call the primary LLM provider; fall back to the other on failure.

        When ``system_already_composed`` is True, ``system_prompt`` is sent
        unchanged (e.g. the string produced once at the start of
        :meth:`run`). Skips :meth:`_compose_system_prompt` so ReAct
        follow-up calls do not duplicate identity, journal, or injection
        guard blocks — and keeps the system prefix stable for prefix-cache
        friendly providers.
        """
        redis = self._services.get(SK.REDIS)
        if system_already_composed:
            resolved_system_prompt = system_prompt
        else:
            resolved_system_prompt = await self._compose_system_prompt(system_prompt)
        resolved_profile = (
            model_profile if model_profile is not None else self.preferred_model_profile
        )
        cache_key = self._llm_cache_key(
            resolved_system_prompt,
            user_message,
            temperature,
            resolved_profile,
        )

        if redis is not None and temperature <= 0.3:
            try:
                cached = await redis.get_llm_cache(cache_key)
                if cached is not None:
                    logger.debug("LLM cache hit for %s (key=%s)", self.name, cache_key[:8])
                    return cached
            except (AttributeError, TypeError, RuntimeError):
                pass

        llm_cfg = get_settings().llm
        primary = llm_cfg.resolve_primary_provider_for_profile(
            baseline_primary=self._primary_llm_provider(),
            model_profile=resolved_profile,
        )
        _sess, _user = self._session_user_for_llm()
        await self._ensure_llm()
        result = await self._llm.generate_text_with_fallback(
            resolved_system_prompt,
            user_message,
            primary=primary,
            temperature=temperature,
            name=f"{self.name}.call_llm",
            model_profile=resolved_profile,
            session_id=_sess,
            user_id=_user,
        )

        if redis is not None and temperature <= 0.3 and not result.startswith("("):
            try:
                await redis.set_llm_cache(cache_key, result, self._LLM_CACHE_TTL)
            except (AttributeError, TypeError, RuntimeError):
                pass

        return result

    # ── knowledge retrieval ──────────────────────────────────────────────

    async def search_knowledge(
        self,
        query: str,
        limit: int = 10,
        sources: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the unified knowledge base."""
        return await self._retriever.search(query, sources=sources, limit=limit)

    async def search_category(
        self,
        query: str,
        category: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search within a specific knowledge category."""
        return await self._retriever.search_by_category(query, category, limit=limit)

    async def search_domain_knowledge(
        self,
        query: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Search across all source categories declared in *knowledge_categories*.

        Fires one ``search_category`` call per category in parallel, then
        deduplicates and sorts by score.  Falls back to the generic
        ``search_knowledge`` when no categories are configured.
        """
        if not self.knowledge_categories:
            return await self.search_knowledge(query, limit=limit)

        per_cat = max(3, limit // len(self.knowledge_categories))
        results_lists = await asyncio.gather(
            *(self.search_category(query, cat, limit=per_cat) for cat in self.knowledge_categories)
        )

        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for results in results_lists:
            for r in results:
                key = r.get("source", "") + r.get("content", "")[:100]
                if key not in seen:
                    seen.add(key)
                    merged.append(r)

        merged.sort(key=lambda r: r.get("score", 0), reverse=True)
        return merged[:limit]

    # ── inter-agent communication ────────────────────────────────────────

    async def send_to(
        self, to_agent: str, query: str, context: dict[str, Any] | None = None
    ) -> None:
        """Send a message to another agent via the message bus."""
        await self._bus.send(self.name, to_agent, query, context)

    # ── web scraping (Firecrawl → Crawl4AI fallback) ────────────────────

    async def scrape_url(self, url: str, *, max_chars: int = 8000) -> str:
        """Fetch a URL and return its content as clean markdown.

        Order: Firecrawl (if FIRECRAWL_API_KEY) → Jina Reader (no key) →
        Crawl4AI (Playwright) → web_search fallback.
        """
        firecrawl_key = get_settings().firecrawl.api_key.get_secret_value()
        if firecrawl_key:

            async def _firecrawl_markdown() -> str:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        "https://api.firecrawl.dev/v1/scrape",
                        headers={
                            "Authorization": f"Bearer {firecrawl_key}",
                            "Content-Type": "application/json",
                        },
                        json={"url": url, "formats": ["markdown"]},
                    )
                    resp.raise_for_status()
                    data = resp.json().get("data", {})
                    text = data.get("markdown", "")
                    return text[:max_chars] if text else ""

            fc_result = await run_tool(
                f"{self.name}.scrape_url.firecrawl",
                _firecrawl_markdown,
                retries=1,
                backoff_seconds=0.3,
            )
            if fc_result.ok and fc_result.value:
                return str(fc_result.value)
            if not fc_result.ok:
                logger.warning(
                    "Firecrawl failed for %s in %s after retries, trying next scraper: %s",
                    url,
                    self.name,
                    fc_result.error,
                )

        # Jina Reader: no API key for basic use (https://jina.ai/reader/)
        text = await self._scrape_url_jina(url, max_chars=max_chars)
        if text:
            return text
        return await self._scrape_url_crawl4ai(url, max_chars=max_chars)

    async def _scrape_url_jina(self, url: str, *, max_chars: int = 8000) -> str:
        """Jina Reader: GET r.jina.ai/<url> returns markdown. Uses JINA_API_KEY when set for better rate limits."""
        try:
            headers: dict[str, str] = {}
            jina_key = get_settings().jina.api_key.get_secret_value()
            if jina_key:
                headers["Authorization"] = f"Bearer {jina_key}"
            async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
                resp = await client.get(f"https://r.jina.ai/{url}", headers=headers or None)
                resp.raise_for_status()
                text = (resp.text or "").strip()
                if text and len(text) > 100:
                    return text[:max_chars]
        except (httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            logger.debug("Jina Reader failed for %s: %s", url, exc)
        return ""

    async def _scrape_url_crawl4ai(self, url: str, *, max_chars: int = 8000) -> str:
        """Crawl4AI fallback for scrape_url."""
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

            config = CrawlerRunConfig(
                word_count_threshold=50,
                excluded_tags=["nav", "footer", "header", "aside"],
                exclude_external_links=True,
            )
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=config)
                if not result.success:
                    return f"Failed to scrape {url}: {result.error_message or 'unknown error'}"
                # Crawl4AI 0.8+: markdown is MarkdownGenerationResult with .raw_markdown; markdown_v2 removed
                md = getattr(result, "markdown", None)
                if (
                    md is not None
                    and hasattr(md, "raw_markdown")
                    and getattr(md, "raw_markdown", None)
                ):
                    text = md.raw_markdown
                elif isinstance(md, str):
                    text = md
                else:
                    text = ""
                if not text:
                    return f"Failed to scrape {url}: no markdown content"
                return text[:max_chars]
        except ImportError:
            return await self._scrape_url_web_search_fallback(url, max_chars)
        except _CRAWL_SCRAPE_ERRORS as exc:
            logger.warning("scrape_url (Crawl4AI) failed for %s in %s: %s", url, self.name, exc)
            return await self._scrape_url_web_search_fallback(url, max_chars)

    async def _scrape_url_web_search_fallback(self, url: str, max_chars: int = 8000) -> str:
        """When Firecrawl and Crawl4AI fail, use web_search to get snippets about the URL."""
        try:
            # Search for the URL or the domain so we return something useful
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc or url
            query = f"site:{domain}" if domain else url
            results = await self.web_search(query, max_results=5)
            if not results:
                return f"Scrape and search fallback had no results for {url}. Install Playwright (playwright install) or set FIRECRAWL_API_KEY for better scraping."
            parts = []
            for r in results:
                title = r.get("title") or ""
                snippet = r.get("snippet") or ""
                link = r.get("url") or ""
                if snippet or title:
                    parts.append(f"**{title}**\n{snippet}\nSource: {link}")
            out = "\n\n".join(parts)[:max_chars]
            return out or f"No content from search for {url}."
        except _CRAWL_SCRAPE_ERRORS as exc:
            logger.warning("scrape_url web_search fallback failed for %s: %s", url, exc)
            return f"Scrape error and fallback failed: {exc}. Install Playwright (playwright install) or set FIRECRAWL_API_KEY for URL scraping."

    # ── web search ────────────────────────────────────────────────────────

    async def web_search(self, query: str, *, max_results: int = 5) -> list[dict[str, str]]:
        """Search the web using the best available provider.

        Returns a list of dicts with 'title', 'url', and 'snippet' keys.
        Tries Tavily first, then Serper, then SearchAPI.
        """
        settings = get_settings().search
        tavily = settings.tavily_api_key.get_secret_value()
        serper = settings.serper_api_key.get_secret_value()
        searchapi = settings.searchapi_api_key.get_secret_value()

        if tavily:
            return await self._search_tavily(query, tavily, max_results)
        if serper:
            return await self._search_serper(query, serper, max_results)
        if searchapi:
            return await self._search_searchapi(query, searchapi, max_results, engine=None)
        logger.warning("Agent '%s' tried web_search but no search API key is configured", self.name)
        return []

    async def searchapi_search(
        self, query: str, *, engine: str | None = None, max_results: int = 5
    ) -> list[dict[str, str]]:
        """Call SearchAPI.io directly, ignoring Tavily/Serper priority.

        Uses ``SEARCHAPI_API_KEY``. When ``engine`` is omitted, uses ``SEARCHAPI_ENGINE``
        (default ``google``). See https://www.searchapi.io/docs for supported engines.
        """
        key = get_settings().search.searchapi_api_key.get_secret_value().strip()
        if not key:
            logger.warning(
                "Agent '%s' called searchapi_search but SEARCHAPI_API_KEY is empty", self.name
            )
            return []
        override: str | None = None
        if engine is not None and str(engine).strip():
            override = str(engine).strip()
        return await self._search_searchapi(query, key, max_results, engine=override)

    async def _search_tavily(
        self, query: str, api_key: str, max_results: int
    ) -> list[dict[str, str]]:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": api_key, "query": query, "max_results": max_results},
                )
                resp.raise_for_status()
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", ""),
                    }
                    for r in resp.json().get("results", [])
                ]
        except (httpx.HTTPError, KeyError):
            logger.exception("Tavily search failed in %s", self.name)
            return []

    async def _search_serper(
        self, query: str, api_key: str, max_results: int
    ) -> list[dict[str, str]]:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    "https://google.serper.dev/search",
                    json={"q": query, "num": max_results},
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                )
                resp.raise_for_status()
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("link", ""),
                        "snippet": r.get("snippet", ""),
                    }
                    for r in resp.json().get("organic", [])
                ]
        except (httpx.HTTPError, KeyError):
            logger.exception("Serper search failed in %s", self.name)
            return []

    async def _search_searchapi(
        self,
        query: str,
        api_key: str,
        max_results: int,
        *,
        engine: str | None,
    ) -> list[dict[str, str]]:
        sch = get_settings().search
        eng = (engine if engine is not None else sch.searchapi_engine).strip() or "google"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://www.searchapi.io/api/v1/search",
                    params={
                        "engine": eng,
                        "q": query,
                        "num": max_results,
                        "api_key": api_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and data.get("error"):
                    logger.warning(
                        "SearchAPI error in %s (engine=%s): %s",
                        self.name,
                        eng,
                        data.get("error"),
                    )
                    return []
                if isinstance(data, dict):
                    return _normalize_searchapi_results(data)
                return []
        except (httpx.HTTPError, KeyError, TypeError):
            logger.exception("SearchAPI search failed in %s", self.name)
            return []

    # ── relationship reporting ──────────────────────────────────────────

    async def report_relationship(
        self,
        from_type: str,
        from_key: str,
        rel: str,
        to_type: str,
        to_key: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Emit a discovered relationship via the DataEventBus.

        Any agent can call this to contribute graph edges without needing
        direct access to the KnowledgeGraph.
        """
        event_bus = self._services.get(SK.DATA_EVENT_BUS)
        if event_bus is None:
            return
        from brain_os.systems.data_event_bus import DataEvent, EventType, SourceStore

        try:
            await event_bus.emit(
                DataEvent(
                    event_type=EventType.RELATIONSHIP_DISCOVERED,
                    entity_type="relationship",
                    entity_id=f"{from_key}-{rel}-{to_key}",
                    payload={
                        "from_type": from_type,
                        "from_key": from_key,
                        "rel": rel,
                        "to_type": to_type,
                        "to_key": to_key,
                        "properties": properties or {},
                    },
                    source_store=SourceStore.NEO4J,
                )
            )
        except _RELATIONSHIP_EVENT_ERRORS:
            logger.debug("Relationship event emission failed in %s", self.name, exc_info=True)

    # ── utility ──────────────────────────────────────────────────────────

    def _format_context(self, kb_results: list[dict[str, Any]]) -> str:
        """Format knowledge-base results into a context string for LLM prompts."""
        if not kb_results:
            return "(No relevant context found)"
        lines = []
        for r in kb_results:
            chunk_id = r.get("id", "?")
            source = r.get("source", "unknown")
            lines.append(f"- [Source: {source}, Chunk: {chunk_id}] {r.get('content', '')[:500]}")
        return "\n".join(lines)

    def _parse_json_response(self, raw: str) -> dict[str, Any] | list[Any]:
        """Attempt to parse an LLM response as JSON, stripping markdown fences. Returns {} on failure."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Try to extract first complete {...} or [...]
        for open_b, close_b in (("{", "}"), ("[", "]")):
            i = cleaned.find(open_b)
            if i < 0:
                continue
            depth = 0
            for j in range(i, len(cleaned)):
                if cleaned[j] == open_b:
                    depth += 1
                elif cleaned[j] == close_b:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[i : j + 1])
                        except json.JSONDecodeError:
                            break
                        break
        logger.debug("_parse_json_response could not parse JSON in %s", self.name)
        return {}

    # ── skill execution ──────────────────────────────────────────────────

    async def use_skill(self, skill_name: str, **kwargs: Any) -> str:
        """Execute a skill from the SKILL_MATRIX by name.

        Every agent inherits this method, giving the entire Pantheon
        uniform access to the shared skill library.

        Raises :class:`ValueError` for unrecognised skill names.
        """
        logger.info("Agent '%s' invoking skill '%s'", self.name, skill_name)
        return await _use_skill(skill_name, **kwargs)

    @staticmethod
    def available_skills() -> dict[str, str]:
        """Return the full skill matrix for introspection."""
        return dict(SKILL_MATRIX)
