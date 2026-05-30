"""Active gap resolution — researches and fills knowledge gaps.

Used by Dream Mode Stage 3e to automatically research high-priority
knowledge gaps identified by Metacognition, store the discovered facts
in long-term memory, and mark the gaps as resolved.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from brain_os.exceptions import BrainOSError, LLMError
from brain_os.prompt_loader import load_prompt
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_RESOLUTION_SYSTEM = load_prompt("gap_resolution")


class GapResolver:
    """Researches knowledge gaps using web search and stores results."""

    def __init__(
        self,
        long_term_memory: Any,
        metacognition: Any | None = None,
    ) -> None:
        self._long_term = long_term_memory
        self._metacognition = metacognition
        self._llm = get_llm_client()

    def prioritize_gaps(self, gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score and sort gaps by priority (frequency, recency, severity).

        Each gap dict should have ``query``, ``state``, ``created_at``,
        and optionally ``gaps`` (list of gap descriptions).
        """
        now = datetime.now(UTC)
        scored: list[tuple[float, dict[str, Any]]] = []

        query_counts: dict[str, int] = {}
        for g in gaps:
            q = g.get("query", "").lower().strip()
            query_counts[q] = query_counts.get(q, 0) + 1

        for g in gaps:
            q = g.get("query", "").lower().strip()
            frequency = query_counts.get(q, 1)

            try:
                created = datetime.fromisoformat(g.get("created_at", "").replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                days_old = max(0, (now - created).days)
                recency_score = max(0.1, 1.0 / (1 + days_old))
            except (ValueError, TypeError):
                recency_score = 0.5

            severity = 1.0 if g.get("state") == "UNKNOWN" else 0.5

            score = (frequency * 3) + (recency_score * 2) + severity
            scored.append((score, g))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [g for _, g in scored]

    async def resolve_gap(
        self,
        gap: dict[str, Any],
        web_search_fn: Any | None = None,
        scrape_fn: Any | None = None,
    ) -> str | None:
        """Attempt to resolve a single knowledge gap.

        Uses web search (if provided) to gather information, then
        synthesizes a factual answer via the LLM. Stores the result
        in long-term memory and marks the gap as resolved.

        Returns the resolution text, or ``None`` if resolution failed.
        """
        query = gap.get("query", "")
        gap_descriptions = gap.get("gaps", [])
        gap_id = gap.get("id")

        search_context = ""
        if web_search_fn is not None:
            try:
                search_query = f"Acme Corp {query}"
                results = await web_search_fn(search_query, max_results=3)
                if results:
                    snippets = []
                    for r in results:
                        snippets.append(f"[{r.get('title', '')}] {r.get('snippet', '')}")
                    search_context = "\n".join(snippets)

                    if scrape_fn is not None and results:
                        try:
                            url = results[0].get("url", "")
                            if url:
                                page_content = await scrape_fn(url)
                                search_context += f"\n\nFull page:\n{page_content[:2000]}"
                        except BrainOSError:
                            logger.debug("Scrape failed during gap resolution", exc_info=True)
            except BrainOSError:
                logger.warning("Web search failed during gap resolution", exc_info=True)

        user_msg = (
            f"Knowledge gap query: {query}\n"
            f"Gap descriptions: {gap_descriptions}\n\n"
            f"Web search results:\n{search_context or '(no results)'}"
        )

        try:
            resolution = await self._llm.generate_text(
                _RESOLUTION_SYSTEM,
                user_msg,
                name="gap_resolver.resolve",
            )
        except LLMError:
            logger.warning("LLM gap resolution failed for: %s", query, exc_info=True)
            return None

        if not resolution or resolution.strip().upper() in ("NONE", "UNKNOWN", "N/A"):
            return None

        resolution = resolution.strip()

        try:
            from brain_os.brain.retrieval_context import mem0_user_id_var

            _muid = mem0_user_id_var.get()
            await self._long_term.store_fact(
                resolution,
                source=f"gap_resolution:{query[:100]}",
                confidence=0.6,
                mem0_user_id=_muid,
            )
        except BrainOSError:
            logger.warning("Failed to store resolved gap fact", exc_info=True)

        if self._metacognition is not None and gap_id is not None:
            try:
                await self._metacognition.mark_gap_resolved(gap_id, resolution)
            except BrainOSError:
                logger.warning("Failed to mark gap #%s as resolved", gap_id, exc_info=True)

        logger.info("Resolved gap: %s -> %s", query[:60], resolution[:80])
        return resolution
