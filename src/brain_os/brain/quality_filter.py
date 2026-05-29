"""Filters low-quality data before it enters the knowledge base.

Applies heuristic quality checks (length, uniqueness, numeric density)
and optional semantic deduplication via Qdrant to prevent noisy or
redundant content from polluting the vector store.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from brain_os.exceptions import DatabaseError

logger = logging.getLogger(__name__)

_MIN_WORDS = 20
_MIN_UNIQUE_RATIO = 0.3
_MAX_NUMERIC_RATIO = 0.7
_MAX_WHITESPACE_RATIO = 0.6
_DEFAULT_DUPLICATE_THRESHOLD = 0.95

_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"\bconfidential\b", re.IGNORECASE),
    re.compile(r"all\s+rights\s+reserved", re.IGNORECASE),
    re.compile(r"table\s+of\s+contents", re.IGNORECASE),
    re.compile(r"©\s*\d{4}", re.IGNORECASE),
    re.compile(r"^\s*[-=]{5,}\s*$", re.MULTILINE),
]

_REPEATED_HEADER_PATTERN = re.compile(r"^(.{10,})\n(?:.*\n){0,3}\1", re.MULTILINE)


class QualityFilter:
    """Rejects low-quality text chunks before they enter the knowledge base."""

    def __init__(
        self,
        qdrant_manager: Any | None = None,
        embedding_service: Any | None = None,
    ) -> None:
        self._qdrant = qdrant_manager
        self._embeddings = embedding_service

    # ── single-chunk evaluation ───────────────────────────────────────────

    def filter_chunk(self, text: str) -> dict[str, Any]:
        """Evaluate *text* and return pass/fail with a quality score."""
        stripped = text.strip()
        if not stripped:
            return {"pass": False, "reason": "empty text", "quality_score": 0.0}

        words = stripped.split()
        word_count = len(words)

        if word_count < _MIN_WORDS:
            return {
                "pass": False,
                "reason": f"too short ({word_count} words, minimum {_MIN_WORDS})",
                "quality_score": round(word_count / _MIN_WORDS, 2),
            }

        unique_ratio = len(set(w.lower() for w in words)) / word_count
        if unique_ratio < _MIN_UNIQUE_RATIO:
            return {
                "pass": False,
                "reason": f"low vocabulary diversity ({unique_ratio:.2f})",
                "quality_score": round(unique_ratio, 2),
            }

        numeric_chars = sum(1 for c in stripped if c.isdigit())
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        total_meaningful = numeric_chars + alpha_chars
        numeric_ratio = numeric_chars / total_meaningful if total_meaningful else 0
        if numeric_ratio > _MAX_NUMERIC_RATIO:
            return {
                "pass": False,
                "reason": f"mostly numeric content ({numeric_ratio:.2f})",
                "quality_score": round(1 - numeric_ratio, 2),
            }

        whitespace_ratio = sum(1 for c in text if c.isspace()) / len(text)
        if whitespace_ratio > _MAX_WHITESPACE_RATIO:
            return {
                "pass": False,
                "reason": f"mostly whitespace ({whitespace_ratio:.2f})",
                "quality_score": round(1 - whitespace_ratio, 2),
            }

        if self.detect_boilerplate(stripped):
            return {
                "pass": False,
                "reason": "boilerplate content detected",
                "quality_score": 0.2,
            }

        score = min(
            1.0,
            round(
                0.4 * min(word_count / 100, 1.0) + 0.3 * unique_ratio + 0.3 * (1 - numeric_ratio),
                2,
            ),
        )
        return {"pass": True, "reason": "ok", "quality_score": score}

    def detect_boilerplate(self, text: str) -> bool:
        """Return True if *text* matches common boilerplate patterns."""
        boilerplate_hits = sum(1 for p in _BOILERPLATE_PATTERNS if p.search(text))
        if boilerplate_hits >= 2:
            return True

        if _REPEATED_HEADER_PATTERN.search(text):
            return True

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 4:
            unique_lines = set(lines)
            if len(unique_lines) / len(lines) < 0.5:
                return True

        return False

    # ── semantic deduplication ────────────────────────────────────────────

    async def check_semantic_duplicate(
        self,
        text: str,
        threshold: float = _DEFAULT_DUPLICATE_THRESHOLD,
    ) -> bool:
        """Embed *text* and check Qdrant for near-duplicates above *threshold*."""
        if self._qdrant is None or self._embeddings is None:
            return False

        try:
            results = await self._qdrant.search(text, limit=3)
            for hit in results:
                if hit.get("score", 0) >= threshold:
                    logger.debug(
                        "Semantic duplicate found (score=%.3f): %s",
                        hit["score"],
                        hit.get("content", "")[:80],
                    )
                    return True
        except DatabaseError:
            logger.exception("Semantic duplicate check failed")

        return False

    # ── batch filtering ───────────────────────────────────────────────────

    def filter_chunks(self, chunks: list[str]) -> tuple[list[str], list[str]]:
        """Evaluate a batch of chunks; return (passed, rejected)."""
        passed: list[str] = []
        rejected: list[str] = []
        for chunk in chunks:
            result = self.filter_chunk(chunk)
            if result["pass"]:
                passed.append(chunk)
            else:
                rejected.append(chunk)
                logger.debug("Rejected chunk: %s", result["reason"])
        logger.info(
            "Quality filter: %d passed, %d rejected out of %d",
            len(passed),
            len(rejected),
            len(chunks),
        )
        return passed, rejected

    # ── cleanup ───────────────────────────────────────────────────────────

    async def cleanup_waste(self, collection: str | None = None) -> dict[str, Any]:
        """Remove low-quality points from Qdrant."""
        if self._qdrant is None:
            return {"status": "skipped", "reason": "no qdrant manager"}

        removed = 0
        scanned = 0
        try:
            probe_queries = [
                "page of confidential",
                "table of contents",
                "all rights reserved",
            ]
            seen_ids: set[str] = set()

            for query in probe_queries:
                results = await self._qdrant.search(
                    query,
                    collection=collection,
                    limit=50,
                )
                for hit in results:
                    point_id = hit.get("metadata", {}).get("id", "")
                    if point_id in seen_ids:
                        continue
                    seen_ids.add(point_id)
                    scanned += 1

                    content = hit.get("content", "")
                    result = self.filter_chunk(content)
                    if not result["pass"]:
                        removed += 1
                        logger.debug(
                            "Waste candidate: score=%.2f reason=%s content=%s",
                            hit.get("score", 0),
                            result["reason"],
                            content[:60],
                        )

            logger.info("Cleanup scan: %d scanned, %d waste candidates", scanned, removed)
            return {"status": "ok", "scanned": scanned, "waste_candidates": removed}
        except DatabaseError:
            logger.exception("Cleanup waste scan failed")
            return {"status": "error"}
