"""Usage-based knowledge graph tuning during dream mode.

Analyzes retrieval logs to find which knowledge chunks are frequently
accessed together, then strengthens the Neo4j relationships between
co-accessed entities and decays stale nodes that haven't been touched
in a configurable window.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles

from brain_os.brain.knowledge_graph import KnowledgeGraph
from brain_os.exceptions import DatabaseError, IraError

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path("data/brain/retrieval_log.jsonl")
_DEFAULT_COMPANY_CANDIDATES = Path("data/brain/company_name_normalization_candidates.json")


class GraphConsolidation:
    """Tune the knowledge graph based on real retrieval usage patterns."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        retrieval_log_path: Path = _DEFAULT_LOG_PATH,
        company_candidates_path: Path = _DEFAULT_COMPANY_CANDIDATES,
    ) -> None:
        self._graph = knowledge_graph
        self._log_path = retrieval_log_path
        self._company_candidates_path = company_candidates_path

    # ── logging ───────────────────────────────────────────────────────────

    async def log_retrieval(
        self,
        query: str,
        chunks_retrieved: list[str],
        source_types: list[str],
    ) -> None:
        """Append a retrieval event to the JSONL log."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "query": query,
            "chunks": chunks_retrieved,
            "source_types": source_types,
        }
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(entry, default=str) + "\n"
            async with aiofiles.open(self._log_path, mode="a", encoding="utf-8") as f:
                await f.write(line)
        except OSError:
            logger.exception("Failed to write retrieval log entry")

    # ── analysis ──────────────────────────────────────────────────────────

    async def build_co_access_matrix(self) -> dict:
        """Analyze the retrieval log to find chunks frequently retrieved together.

        Returns a dict mapping ``(chunk_a, chunk_b)`` tuple-keys (serialized
        as ``"chunk_a|||chunk_b"``) to co-occurrence counts.
        """
        if not self._log_path.exists():
            return {}

        try:
            co: dict[str, int] = defaultdict(int)
            async with aiofiles.open(self._log_path, encoding="utf-8") as f:
                raw = await f.read()
            for raw_line in raw.splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                chunks = entry.get("chunks", [])
                for i, a in enumerate(chunks):
                    for b in chunks[i + 1 :]:
                        key = "|||".join(sorted([a, b]))
                        co[key] += 1
            co_access = dict(co)
        except OSError:
            logger.exception("Failed to read retrieval log")
            co_access = {}

        logger.info("Co-access matrix: %d pairs analyzed", len(co_access))
        return co_access

    async def tune_relationships(self, co_access: dict) -> None:
        """Strengthen relationships between co-accessed entities in Neo4j.

        Pairs accessed together >= 3 times get a ``CO_RELEVANT`` edge with
        a ``strength`` property.  Only links *existing* labeled nodes — never
        creates label-less orphans.
        """
        strengthened = 0
        for pair_key, count in co_access.items():
            if count < 3:
                continue
            parts = pair_key.split("|||")
            if len(parts) != 2:
                continue
            entity_a, entity_b = parts

            try:
                result = await self._graph._run_cypher_write(
                    """
                    MATCH (a) WHERE (a.name = $a OR a.email = $a OR a.model = $a
                                     OR a.source = $a)
                                    AND size(labels(a)) > 0
                    MATCH (b) WHERE (b.name = $b OR b.email = $b OR b.model = $b
                                     OR b.source = $b)
                                    AND size(labels(b)) > 0
                    WITH a, b LIMIT 1
                    MERGE (a)-[r:CO_RELEVANT]-(b)
                    SET r.strength = COALESCE(r.strength, 0) + $boost,
                        r.updated_at = $now
                    RETURN count(r) AS created
                    """,
                    params={
                        "a": entity_a,
                        "b": entity_b,
                        "boost": min(count, 10),
                        "now": datetime.now(UTC).isoformat(),
                    },
                )
                if result and result[0].get("created", 0) > 0:
                    strengthened += 1
            except DatabaseError:
                logger.debug("Failed to strengthen edge %s <-> %s", entity_a, entity_b)

        logger.info("Tuned %d co-access relationships", strengthened)

    async def decay_stale_nodes(self, days_threshold: int = 30) -> None:
        """Mark nodes not accessed in *days_threshold* days as stale.

        Sets a ``stale`` property to ``true`` and records the decay timestamp.
        """
        cutoff = datetime.now(UTC)
        accessed_entities: set[str] = set()
        if not self._log_path.exists():
            pass
        else:
            try:
                async with aiofiles.open(self._log_path, encoding="utf-8") as f:
                    raw = await f.read()
                for raw_line in raw.splitlines():
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        entry = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    ts_str = entry.get("timestamp", "")
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=UTC)
                    except (ValueError, TypeError):
                        continue
                    age_days = (cutoff - ts).days
                    if age_days <= days_threshold:
                        for chunk in entry.get("chunks", []):
                            accessed_entities.add(chunk)
            except OSError:
                logger.exception("Failed to read retrieval log for decay analysis")

        try:
            result = await self._graph._run_cypher_write(
                """
                MATCH (n)
                WHERE n.name IS NOT NULL AND size(labels(n)) > 0
                      AND NOT n.name IN $active
                      AND NOT COALESCE(n.source, '') IN $active
                SET n.stale = true, n.stale_since = $now
                RETURN count(n) AS decayed
                """,
                params={
                    "active": list(accessed_entities),
                    "now": datetime.now(UTC).isoformat(),
                },
            )
            decayed = result[0].get("decayed", 0) if result else 0
            logger.info("Marked %d nodes as stale (threshold=%d days)", decayed, days_threshold)
        except DatabaseError:
            logger.exception("Failed to decay stale nodes")

    async def reconcile_company_duplicates(self) -> int:
        """Merge safe duplicate company variants from candidate report."""
        if not self._company_candidates_path.exists():
            return 0
        try:
            raw = json.loads(self._company_candidates_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Could not read company normalization candidates from %s",
                self._company_candidates_path,
                exc_info=True,
            )
            return 0
        candidates = raw.get("candidates", []) if isinstance(raw, dict) else []
        merged = 0
        for row in candidates:
            if not isinstance(row, dict) or not row.get("safe_exact_key_merge"):
                continue
            canonical = str(row.get("canonical_name") or "").strip()
            variants = [str(v).strip() for v in row.get("variants", []) if str(v).strip()]
            if not canonical or len(variants) < 2:
                continue
            for variant in variants:
                if variant == canonical:
                    continue
                try:
                    await self._graph._run_cypher_write(
                        """
                        MATCH (src:Company {name: $variant})
                        MATCH (dst:Company {name: $canonical})
                        WHERE src <> dst
                        CALL apoc.refactor.mergeNodes([dst, src], {
                            properties: 'combine',
                            mergeRels: true
                        })
                        YIELD node
                        SET node.updated_at = $now
                        RETURN 1 AS merged
                        """,
                        params={
                            "variant": variant,
                            "canonical": canonical,
                            "now": datetime.now(UTC).isoformat(),
                        },
                    )
                    merged += 1
                except DatabaseError:
                    logger.debug(
                        "Failed merge candidate %s -> %s",
                        variant,
                        canonical,
                        exc_info=True,
                    )
        logger.info("Safe duplicate reconciliation merged %d company variants", merged)
        return merged

    async def merge_duplicate_company_ids(self) -> int:
        """Merge Company nodes that share the same ``company_id`` (Aura-safe via APOC)."""
        now = datetime.now(UTC).isoformat()
        try:
            rows = await self._graph._run_cypher_write(
                """
                MATCH (c:Company)
                WHERE c.company_id IS NOT NULL AND c.company_id <> ''
                WITH c.company_id AS cid, collect(c) AS nodes
                WHERE size(nodes) > 1
                CALL apoc.refactor.mergeNodes(nodes, {
                    properties: 'combine',
                    mergeRels: true
                })
                YIELD node
                SET node.updated_at = $now
                RETURN count(*) AS merged
                """,
                params={"now": now},
            )
            merged = int(rows[0].get("merged", 0)) if rows else 0
            logger.info("Merged %d duplicate company_id groups", merged)
            return merged
        except DatabaseError:
            logger.exception("company_id duplicate merge failed")
            return 0

    async def cleanup_relationship_noise(self) -> int:
        """Prune stale weak CO_RELEVANT edges and standardize weighted metadata."""
        try:
            rows = await self._graph._run_cypher_write(
                """
                MATCH ()-[r:CO_RELEVANT]-()
                WHERE COALESCE(r.strength, 0) <= 1
                  AND (
                        r.updated_at IS NULL
                        OR datetime(r.updated_at) < datetime() - duration({days: 45})
                      )
                DELETE r
                RETURN count(r) AS removed
                """
            )
            removed = int(rows[0].get("removed", 0)) if rows else 0
            now = datetime.now(UTC).isoformat()
            await self._graph._run_cypher_write(
                """
                MATCH ()-[r:CO_RELEVANT]-()
                SET r.strength = COALESCE(r.strength, 1),
                    r.updated_at = COALESCE(r.updated_at, $now)
                RETURN count(r) AS touched
                """,
                params={"now": now},
            )
            logger.info("Relationship cleanup removed %d weak stale CO_RELEVANT edges", removed)
            return removed
        except DatabaseError as exc:
            logger.exception("Relationship cleanup failed")
            return 0

    # ── full pipeline ─────────────────────────────────────────────────────

    async def run_consolidation(self) -> dict:
        """Execute the full consolidation pipeline and return stats."""
        stats: dict[str, Any] = {}

        try:
            co_access = await self.build_co_access_matrix()
            stats["co_access_pairs"] = len(co_access)

            await self.tune_relationships(co_access)
            stats["tuning"] = "completed"

            await self.decay_stale_nodes()
            stats["decay"] = "completed"

            stats["company_id_merges"] = await self.merge_duplicate_company_ids()
            stats["safe_company_merges"] = await self.reconcile_company_duplicates()
            stats["relationship_noise_removed"] = await self.cleanup_relationship_noise()

            stats["status"] = "success"
        except IraError:
            logger.exception("Graph consolidation failed")
            stats["status"] = "error"

        logger.info("Graph consolidation complete: %s", stats)
        return stats
