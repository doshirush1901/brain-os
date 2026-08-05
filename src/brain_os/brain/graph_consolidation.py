"""Usage-based knowledge graph tuning during dream mode.

Analyzes retrieval logs to find which knowledge chunks are frequently
accessed together, then strengthens the Neo4j relationships between
co-accessed entities and decays stale nodes that haven't been touched
in a configurable window.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles

from brain_os.brain.knowledge_graph import KnowledgeGraph
from brain_os.config import data_root
from brain_os.exceptions import DatabaseError, BrainOSError

logger = logging.getLogger(__name__)


def _default_retrieval_log_path() -> Path:
    return data_root() / "brain" / "retrieval_log.jsonl"


def _default_company_candidates_path() -> Path:
    return data_root() / "brain" / "company_name_normalization_candidates.json"


_DEFAULT_LOG_PATH = Path("brain/retrieval_log.jsonl")  # resolved via data_root in __init__
_DEFAULT_COMPANY_CANDIDATES = Path("brain/company_name_normalization_candidates.json")
_CO_ACCESS_MIN_COUNT = 3
_TUNE_RELATIONSHIP_BATCH_SIZE = 75

_BATCH_CO_RELEVANT_CYPHER = """
UNWIND $pairs AS pair
CALL {
    WITH pair
    MATCH (a)
    WHERE (a.name = pair.a OR a.email = pair.a OR a.model = pair.a OR a.source = pair.a)
          AND size(labels(a)) > 0
    MATCH (b)
    WHERE (b.name = pair.b OR b.email = pair.b OR b.model = pair.b OR b.source = pair.b)
          AND size(labels(b)) > 0
    WITH a, b, pair.boost AS boost
    LIMIT 1
    MERGE (a)-[r:CO_RELEVANT]-(b)
    SET r.strength = COALESCE(r.strength, 0) + boost,
        r.updated_at = $now
    RETURN 1 AS created
}
RETURN coalesce(sum(created), 0) AS strengthened
"""


@dataclass(frozen=True)
class RetrievalLogSnapshot:
    """Single-pass parse of ``retrieval_log.jsonl`` for dream graph stages."""

    co_access: dict[str, int] = field(default_factory=dict)
    active_entities: set[str] = field(default_factory=set)


class GraphConsolidation:
    """Tune the knowledge graph based on real retrieval usage patterns."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        retrieval_log_path: Path | None = None,
        company_candidates_path: Path | None = None,
    ) -> None:
        self._graph = knowledge_graph
        if retrieval_log_path is None:
            self._log_path = _default_retrieval_log_path()
        elif not retrieval_log_path.is_absolute():
            self._log_path = data_root() / retrieval_log_path
        else:
            self._log_path = retrieval_log_path
        if company_candidates_path is None:
            self._company_candidates_path = _default_company_candidates_path()
        elif not company_candidates_path.is_absolute():
            self._company_candidates_path = data_root() / company_candidates_path
        else:
            self._company_candidates_path = company_candidates_path

    # ── logging ───────────────────────────────────────────────────────────

    async def log_retrieval(
        self,
        query: str,
        chunks_retrieved: list[str],
        source_types: list[str],
        stitch_stats: dict[str, Any] | None = None,
    ) -> None:
        """Append a retrieval event to the JSONL log (size-rotated)."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "query": query,
            "chunks": chunks_retrieved,
            "source_types": source_types,
        }
        if stitch_stats:
            entry.update(stitch_stats)
        try:
            from brain_os.systems.jsonl_log import append_jsonl

            # Sync append + rotate; retrieval logging must stay cheap and durable.
            await asyncio.to_thread(append_jsonl, self._log_path, entry, rotate_mb=50)
        except OSError:
            logger.exception("Failed to write retrieval log entry")
        except Exception:
            # Fallback without rotation if thread/import fails.
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                line = json.dumps(entry, default=str) + "\n"
                async with aiofiles.open(self._log_path, mode="a", encoding="utf-8") as f:
                    await f.write(line)
            except OSError:
                logger.exception("Failed to write retrieval log entry")

    # ── analysis ──────────────────────────────────────────────────────────

    async def load_retrieval_log_snapshot(
        self, *, days_threshold: int = 30
    ) -> RetrievalLogSnapshot:
        """Read the retrieval log once; build co-access matrix and active-entity set."""
        if not self._log_path.exists():
            return RetrievalLogSnapshot()

        co: dict[str, int] = defaultdict(int)
        active_entities: set[str] = set()
        cutoff = datetime.now(UTC)

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
                chunks = entry.get("chunks", [])
                if isinstance(chunks, list):
                    for i, a in enumerate(chunks):
                        if not isinstance(a, str):
                            continue
                        for b in chunks[i + 1 :]:
                            if not isinstance(b, str):
                                continue
                            key = "|||".join(sorted([a, b]))
                            co[key] += 1
                ts_str = entry.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    age_days = (cutoff - ts).days
                    if age_days <= days_threshold and isinstance(chunks, list):
                        for chunk in chunks:
                            if isinstance(chunk, str):
                                active_entities.add(chunk)
                except (ValueError, TypeError):
                    pass
        except OSError:
            logger.exception("Failed to read retrieval log")
            return RetrievalLogSnapshot()

        co_access = dict(co)
        logger.info(
            "Retrieval log snapshot: %d co-access pairs, %d active entities",
            len(co_access),
            len(active_entities),
        )
        return RetrievalLogSnapshot(co_access=co_access, active_entities=active_entities)

    async def build_co_access_matrix(self) -> dict:
        """Analyze the retrieval log to find chunks frequently retrieved together.

        Returns a dict mapping ``(chunk_a, chunk_b)`` tuple-keys (serialized
        as ``"chunk_a|||chunk_b"``) to co-occurrence counts.
        """
        snapshot = await self.load_retrieval_log_snapshot()
        return snapshot.co_access

    async def tune_relationships(self, co_access: dict) -> int:
        """Strengthen relationships between co-accessed entities in Neo4j.

        Pairs accessed together >= 3 times get a ``CO_RELEVANT`` edge with
        a ``strength`` property.  Only links *existing* labeled nodes — never
        creates label-less orphans.  Writes are batched via ``UNWIND``.
        """
        now = datetime.now(UTC).isoformat()
        pair_rows: list[dict[str, Any]] = []
        for pair_key, count in co_access.items():
            if count < _CO_ACCESS_MIN_COUNT:
                continue
            parts = pair_key.split("|||")
            if len(parts) != 2:
                continue
            pair_rows.append(
                {
                    "a": parts[0],
                    "b": parts[1],
                    "boost": min(count, 10),
                }
            )

        if not pair_rows:
            return 0

        strengthened = 0
        for offset in range(0, len(pair_rows), _TUNE_RELATIONSHIP_BATCH_SIZE):
            batch = pair_rows[offset : offset + _TUNE_RELATIONSHIP_BATCH_SIZE]
            try:
                result = await self._graph._run_cypher_write(
                    _BATCH_CO_RELEVANT_CYPHER,
                    params={"pairs": batch, "now": now},
                )
                if result:
                    strengthened += int(result[0].get("strengthened", 0) or 0)
            except DatabaseError:
                logger.debug(
                    "Failed batched co-access tune (offset=%d, size=%d)",
                    offset,
                    len(batch),
                    exc_info=True,
                )

        logger.info(
            "Tuned %d co-access relationships (%d pairs in %d batch(es))",
            strengthened,
            len(pair_rows),
            (len(pair_rows) + _TUNE_RELATIONSHIP_BATCH_SIZE - 1) // _TUNE_RELATIONSHIP_BATCH_SIZE,
        )
        return strengthened

    async def decay_stale_nodes(
        self,
        days_threshold: int = 30,
        *,
        active_entities: set[str] | None = None,
    ) -> None:
        """Mark nodes not accessed in *days_threshold* days as stale.

        Sets a ``stale`` property to ``true`` and records the decay timestamp.
        When *active_entities* is provided, the retrieval log is not read again.
        """
        if active_entities is None:
            snapshot = await self.load_retrieval_log_snapshot(days_threshold=days_threshold)
            accessed_entities = snapshot.active_entities
        else:
            accessed_entities = active_entities

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
        except DatabaseError:
            logger.exception("Relationship cleanup failed")
            return 0

    # ── full pipeline ─────────────────────────────────────────────────────

    async def run_consolidation(
        self,
        snapshot: RetrievalLogSnapshot | None = None,
    ) -> dict:
        """Execute the full consolidation pipeline and return stats."""
        stats: dict[str, Any] = {}

        try:
            if snapshot is None:
                snapshot = await self.load_retrieval_log_snapshot()
            co_access = snapshot.co_access
            stats["co_access_pairs"] = len(co_access)

            stats["relationships_strengthened"] = await self.tune_relationships(co_access)
            stats["tuning"] = "completed"

            await self.decay_stale_nodes(active_entities=snapshot.active_entities)
            stats["decay"] = "completed"

            stats["company_id_merges"] = await self.merge_duplicate_company_ids()
            stats["safe_company_merges"] = await self.reconcile_company_duplicates()
            stats["relationship_noise_removed"] = await self.cleanup_relationship_noise()

            stats["status"] = "success"
        except BrainOSError:
            logger.exception("Graph consolidation failed")
            stats["status"] = "error"

        logger.info("Graph consolidation complete: %s", stats)
        return stats
