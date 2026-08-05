"""Per-user, per-channel conversation history backed by SQLite."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import aiosqlite
from langfuse.decorators import observe

from brain_os.memory.conversation_backend import (
    ConversationMemoryBackend,
    build_conversation_backend,
)
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import ConversationEntities
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(
        self,
        db_path: str = "data/conversations.db",
    ) -> None:
        self._db_path = db_path
        self._backend: ConversationMemoryBackend = build_conversation_backend(db_path)
        self._llm = get_llm_client()

    @property
    def _db(self):
        return self._backend.sqlite_db_connection

    async def initialize(self) -> None:
        await self._backend.initialize()

    async def should_start_new_conversation(self, user_id: str, channel: str) -> bool:
        return await self._backend.should_start_new_conversation(user_id, channel)

    async def add_message(
        self,
        user_id: str,
        channel: str,
        role: str,
        content: str,
    ) -> None:
        await self._backend.add_message(user_id, channel, role, content)

    async def get_history(
        self,
        user_id: str,
        channel: str,
        limit: int = 20,
    ) -> list[dict]:
        return await self._backend.get_history(user_id, channel, limit)

    @observe()
    async def get_summarized_history(
        self,
        user_id: str,
        channel: str,
        recent_limit: int = 5,
        full_limit: int = 20,
    ) -> tuple[list[dict], str]:
        """Return recent messages verbatim plus a summary of older messages.

        Returns ``(recent_messages, older_summary)`` where *older_summary*
        is an LLM-generated 2-3 sentence summary of messages beyond the
        recent window, or an empty string if there are none.
        """
        full_history = await self.get_history(user_id, channel, limit=full_limit)
        if len(full_history) <= recent_limit:
            return full_history, ""

        recent = full_history[-recent_limit:]
        older = full_history[:-recent_limit]

        older_text = "\n".join(f"[{m['role']}] {m['content'][:300]}" for m in older)
        try:
            summary = await self._llm.generate_text(
                "Summarize this conversation history in 2-3 concise sentences. "
                "Focus on key facts, decisions, and open questions. "
                "Do not include greetings or filler.",
                older_text,
                name="conversation.summarize_history",
                model_tier="cheap",
            )
            return recent, (summary or "").strip()
        except Exception:
            logger.warning("History summarization failed; returning recent only", exc_info=True)
            return recent, ""

    @observe()
    async def extract_entities(self, message: str) -> dict[str, list]:
        """Extract entities from a message. Returns dict with keys: companies, people, emails, machines, quote_ids, dates, amounts."""
        system = load_prompt("conversation_extract_entities")
        empty: dict[str, list] = {
            "companies": [],
            "people": [],
            "emails": [],
            "machines": [],
            "quote_ids": [],
            "dates": [],
            "amounts": [],
        }
        try:
            result = await self._llm.generate_structured(
                system,
                message,
                ConversationEntities,
                name="conversation.extract_entities",
                model_tier="cheap",
            )
            return result.model_dump()
        except Exception:
            logger.exception("Entity extraction LLM call failed")
        return empty

    @observe()
    async def resolve_coreferences(self, message: str, history: list[dict]) -> str:
        context = "\n".join(f"[{h['role']}] {h['content']}" for h in history[-10:])
        system = load_prompt("conversation_resolve_coreferences")
        user_text = f"Context:\n{context}\n\nMessage to rewrite:\n{message}"
        try:
            result = await self._llm.generate_text(
                system,
                user_text,
                name="conversation.resolve_coreferences",
                model_tier="cheap",
            )
        except Exception:
            logger.exception("Coreference resolution LLM call failed")
            return message
        if not result or not result.strip():
            return message
        return result.strip()

    async def close(self) -> None:
        await self._backend.close()

    async def __aenter__(self) -> ConversationMemory:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
