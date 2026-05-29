"""Real-time observer — extracts learnings from every conversation turn.

After each response, the observer fires a lightweight LLM call to extract
corrections, facts, and preferences from the exchange.  These are persisted
to ``data/brain/realtime_learnings.jsonl`` and injected into the next turn's
system prompt so Ira immediately benefits from what it just learned.

Designed to run fire-and-forget so it never blocks the response path.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langfuse.decorators import observe

from brain_os.brain.untrusted_input import sanitize_for_llm_embedding
from brain_os.exceptions import ConfigurationError, LLMError
from brain_os.schemas.llm_outputs import ObservedTurn
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_LEARNINGS_PATH = Path("data/brain/realtime_learnings.jsonl")

_EXTRACT_SYSTEM = (
    "You are analyzing a conversation turn between a user and an AI assistant. "
    "Extract any new facts, corrections, or preferences the user revealed. "
    'Return JSON: {"facts": ["..."], "corrections": ["..."], "preferences": ["..."]}\n'
    'If nothing new was learned, return {"facts": [], "corrections": [], "preferences": []}'
)

_MAX_LEARNINGS_PER_CONTACT = 50


class RealTimeObserver:
    """Extracts and persists learnings from each conversation turn."""

    def __init__(self) -> None:
        self._llm = get_llm_client()
        self._learnings: dict[str, list[dict[str, Any]]] = {}

    async def _load(self) -> None:
        if not _LEARNINGS_PATH.exists():
            return
        try:
            raw = await asyncio.to_thread(_LEARNINGS_PATH.read_text)
            for line in raw.splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                cid = entry.get("contact_id", "unknown")
                self._learnings.setdefault(cid, []).append(entry)
        except ConfigurationError:
            logger.debug("Failed to load realtime learnings")

    async def _persist(self, entry: dict[str, Any]) -> None:
        _LEARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry) + "\n"

        def _append() -> None:
            with open(_LEARNINGS_PATH, "a") as f:
                f.write(line)

        await asyncio.to_thread(_append)

    @observe()
    async def observe_turn(
        self,
        query: str,
        response: str,
        contact_id: str,
    ) -> dict[str, Any]:
        """Extract learnings from a conversation turn (fire-and-forget safe)."""
        u = sanitize_for_llm_embedding(query, max_chars=2000)
        a = sanitize_for_llm_embedding(response, max_chars=2000)
        prompt = f"USER: {u}\n\nASSISTANT: {a}"

        try:
            result = await self._llm.generate_structured(
                _EXTRACT_SYSTEM,
                prompt,
                ObservedTurn,
                model_profile="fast",
                name="realtime_observer.extract",
            )
            extracted = result.model_dump()
        except LLMError:
            logger.debug("RealTimeObserver extraction failed")
            return {"facts": [], "corrections": [], "preferences": []}

        has_content = any(extracted.get(k) for k in ("facts", "corrections", "preferences"))
        if not has_content:
            return extracted

        entry = {
            "contact_id": contact_id,
            "timestamp": datetime.now(UTC).isoformat(),
            **extracted,
        }
        self._learnings.setdefault(contact_id, []).append(entry)

        if len(self._learnings[contact_id]) > _MAX_LEARNINGS_PER_CONTACT:
            self._learnings[contact_id] = self._learnings[contact_id][-_MAX_LEARNINGS_PER_CONTACT:]

        await self._persist(entry)
        logger.info(
            "RealTimeObserver: %d facts, %d corrections, %d preferences for %s",
            len(extracted.get("facts", [])),
            len(extracted.get("corrections", [])),
            len(extracted.get("preferences", [])),
            contact_id,
        )
        return extracted

    def format_for_prompt(self, contact_id: str, limit: int = 10) -> str:
        """Return recent learnings formatted for system prompt injection."""
        entries = self._learnings.get(contact_id, [])
        if not entries:
            return ""

        recent = entries[-limit:]
        lines: list[str] = []
        for e in recent:
            for fact in e.get("facts", []):
                lines.append(f"- Learned fact: {fact}")
            for corr in e.get("corrections", []):
                lines.append(f"- Correction: {corr}")
            for pref in e.get("preferences", []):
                lines.append(f"- Preference: {pref}")

        if not lines:
            return ""
        return "Recent learnings about this contact:\n" + "\n".join(lines[-15:])
