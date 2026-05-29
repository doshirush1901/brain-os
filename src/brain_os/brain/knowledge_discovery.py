"""On-the-fly knowledge gap detection and filling.

When retrieval results are thin or miss the mark, :class:`KnowledgeDiscovery`
detects the gap via an LLM call, scans the raw imports archive for candidate
files, deep-extracts relevant facts, and stores them in Qdrant so future
queries hit.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from langfuse.decorators import observe

from brain_os.brain.embeddings import EmbeddingService
from brain_os.brain.imports_metadata_index import (
    _extract_preview,
    search_index,
)
from brain_os.brain.qdrant_manager import QdrantManager
from brain_os.brain.retriever import UnifiedRetriever
from brain_os.brain.untrusted_input import sanitize_for_llm_embedding
from brain_os.data.models import KnowledgeItem
from brain_os.exceptions import DatabaseError, IraError, PathTraversalError
from brain_os.schemas.llm_outputs import DeepFacts, KnowledgeGap
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_GAP_DETECT_SYSTEM = (
    "You are a knowledge-gap analyst. The user asked a question and received "
    "search results. Determine whether the results adequately answer the query. "
    "If there is a gap, return JSON: "
    '{"gap_type": "missing|partial|outdated", '
    '"description": "what is missing", '
    '"suggested_search": "keywords to find the missing info"}. '
    "If the results are sufficient, return null."
)

_DEEP_EXTRACT_SYSTEM = (
    "You are a fact-extraction engine. Given a document excerpt and a user "
    "query, extract every fact from the document that is relevant to the query. "
    "Return JSON: "
    '{"facts": [{"content": "...", "entity": "...", "category": "..."}]}'
)


class KnowledgeDiscovery:
    """Detect retrieval gaps and fill them from the raw imports archive."""

    def __init__(
        self,
        retriever: UnifiedRetriever,
        qdrant_manager: QdrantManager,
        embedding_service: EmbeddingService,
        imports_dir: Path = Path("data/imports"),
    ) -> None:
        self._retriever = retriever
        self._qdrant = qdrant_manager
        self._embeddings = embedding_service
        self._imports_dir = imports_dir
        self._llm = get_llm_client()

    # ── public API ────────────────────────────────────────────────────────

    @observe()
    async def detect_gap(
        self,
        query: str,
        search_results: list[dict],
    ) -> dict | None:
        """LLM analyzes query vs results and returns a gap descriptor or None."""
        results_summary = json.dumps(
            [
                {
                    "content": r.get("content", "")[:300],
                    "score": r.get("score", 0),
                    "source": r.get("source", ""),
                }
                for r in search_results[:10]
            ],
            default=str,
        )
        q = sanitize_for_llm_embedding(query, max_chars=16_000)
        user_msg = f"QUERY: {q}\n\nSEARCH RESULTS:\n{results_summary}"
        result = await self._llm.generate_structured(
            _GAP_DETECT_SYSTEM,
            user_msg,
            KnowledgeGap,
            name="discovery.detect_gap",
        )

        if not result.gap_type:
            return None
        parsed = result.model_dump()
        logger.info("Gap detected for '%s': %s", query[:60], parsed.get("gap_type"))
        return parsed

    async def find_candidate_files(self, gap: dict) -> list[dict]:
        """Score files in the imports directory by relevance to the gap.

        Uses the imports metadata index for keyword/entity/doc_type scoring.
        Returns the top 5 candidates.
        """
        search_query = gap.get("suggested_search", gap.get("description", ""))
        if not search_query:
            return []

        candidates = await search_index(search_query, limit=5)
        logger.info(
            "Found %d candidate files for gap '%s'",
            len(candidates),
            gap.get("description", "")[:60],
        )
        return candidates

    @observe()
    async def deep_scan_and_extract(
        self,
        filepath: str,
        query: str,
    ) -> list[dict]:
        """Extract text from a file and use LLM to pull relevant facts."""
        path = Path(filepath)
        project_root = Path(__file__).resolve().parents[3]
        data_root = project_root / "data"
        if not path.resolve().is_relative_to(data_root):
            raise PathTraversalError(f"Path {filepath} is outside the data directory")

        if not path.exists():
            logger.warning("Candidate file not found: %s", filepath)
            return []

        text = await asyncio.to_thread(_extract_preview, path)
        if not text:
            return []

        if path.suffix in (".txt", ".csv", ".md", ".json"):
            full_text = await asyncio.to_thread(path.read_text, errors="ignore")
        else:
            full_text = text

        q = sanitize_for_llm_embedding(query, max_chars=4_000)
        doc_snip = sanitize_for_llm_embedding(full_text[:8000], max_chars=8_000)
        user_msg = f"QUERY: {q}\n\nDOCUMENT ({path.name}):\n{doc_snip}"
        result = await self._llm.generate_structured(
            _DEEP_EXTRACT_SYSTEM,
            user_msg,
            DeepFacts,
            name="discovery.deep_extract",
        )
        facts = [f.model_dump() for f in result.facts]
        logger.info("Extracted %d facts from %s", len(facts), path.name)
        return facts

    async def discover_and_store(
        self,
        query: str,
        search_results: list[dict],
    ) -> list[dict]:
        """Full pipeline: detect gap -> find candidates -> deep scan -> store.

        Returns the list of newly discovered knowledge items.
        """
        gap = await self.detect_gap(query, search_results)
        if gap is None:
            return []

        candidates = await self.find_candidate_files(gap)
        if not candidates:
            logger.info("No candidate files found for gap: %s", gap.get("description", ""))
            return []

        discovered: list[dict] = []
        for candidate in candidates:
            filepath = candidate.get("path", "")
            if not filepath:
                continue

            facts = await self.deep_scan_and_extract(filepath, query)
            for fact in facts:
                content = fact.get("content", "")
                if not content:
                    continue

                entity = fact.get("entity", "")
                if entity and await self._contradicts_existing(entity, content):
                    logger.warning(
                        "Discovery skipped contradicting fact for '%s': %s",
                        entity,
                        content[:80],
                    )
                    continue

                item = KnowledgeItem(
                    source=f"discovery:{filepath}",
                    source_category=fact.get("category", "discovered").lower(),
                    content=content,
                    metadata={
                        "discovery_source": filepath,
                        "entity": entity,
                        "gap_type": gap.get("gap_type", ""),
                        "original_query": query[:200],
                    },
                )
                try:
                    await self._qdrant.upsert_items([item])
                    discovered.append(fact)
                except DatabaseError:
                    logger.exception("Failed to store discovered fact from %s", filepath)

        logger.info(
            "Discovery complete: %d new facts stored for query '%s'",
            len(discovered),
            query[:60],
        )
        return discovered

    async def _contradicts_existing(self, entity: str, new_content: str) -> bool:
        """Check if new content contradicts existing KB entries for an entity.

        Returns True if a contradiction is detected, False otherwise.
        Errs on the side of allowing the upsert (returns False on failure).
        """
        try:
            existing = await self._qdrant.search(entity, limit=3)
            if not existing:
                return False

            existing_text = " | ".join(
                r.get("content", "")[:200] for r in existing if r.get("score", 0) > 0.6
            )
            if not existing_text:
                return False

            verdict = await self._llm.generate_text(
                "You are a fact-checker. Given EXISTING facts and a NEW fact about "
                "the same entity, determine if the NEW fact contradicts the EXISTING "
                "facts. Reply with only YES or NO.",
                f"Entity: {sanitize_for_llm_embedding(entity, max_chars=500)}\n"
                f"EXISTING: {sanitize_for_llm_embedding(existing_text, max_chars=2_000)}\n"
                f"NEW: {sanitize_for_llm_embedding(new_content, max_chars=300)}",
                name="discovery.contradiction_check",
            )
            return verdict.strip().upper().startswith("YES")
        except IraError:
            logger.debug("Contradiction check failed; allowing upsert", exc_info=True)
            return False
