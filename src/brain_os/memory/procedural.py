"""Procedural memory — learned response patterns for recurring request types."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any, Literal

from langfuse.decorators import observe
from pydantic import BaseModel

from brain_os.memory.procedural_backend import ProceduralMemoryBackend, build_procedural_backend
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import PatternExtraction
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_PATTERN_SYSTEM_PROMPT = load_prompt("pattern_extraction")


class Procedure(BaseModel):
    id: int | None = None
    trigger_pattern: str
    steps: list[str]
    success_rate: float = 1.0
    times_used: int = 1
    last_used: datetime | None = None


_ROUTE_SUCCESS_RATE_MIN = 0.7
_ROUTE_TIMES_USED_MIN = 3
_ROUTE_SIMILARITY_MIN = 0.5


def procedure_route_eligible(p: Procedure) -> bool:
    return p.success_rate >= _ROUTE_SUCCESS_RATE_MIN and p.times_used >= _ROUTE_TIMES_USED_MIN


class ProcedureMatchExplanation(BaseModel):
    query: str
    would_route: bool
    similarity: float
    matched: Procedure | None = None
    best_candidate: Procedure | None = None
    best_candidate_similarity: float = 0.0
    blockers: list[str] = []


def procedure_to_dict(p: Procedure) -> dict[str, Any]:
    return {
        "id": p.id,
        "trigger_pattern": p.trigger_pattern,
        "steps": p.steps,
        "success_rate": p.success_rate,
        "times_used": p.times_used,
        "last_used": p.last_used.isoformat() if p.last_used else None,
        "eligible": procedure_route_eligible(p),
    }


def procedure_match_to_dict(explanation: ProcedureMatchExplanation) -> dict[str, Any]:
    matched = explanation.matched
    best = explanation.best_candidate
    return {
        "query": explanation.query,
        "would_route": explanation.would_route,
        "similarity": round(explanation.similarity, 4),
        "matched": procedure_to_dict(matched) if matched else None,
        "best_candidate": procedure_to_dict(best) if best else None,
        "best_candidate_similarity": round(explanation.best_candidate_similarity, 4),
        "blockers": explanation.blockers,
    }


def _procedure_from_row(row: tuple[Any, ...]) -> Procedure:
    return Procedure(
        id=row[0],
        trigger_pattern=row[1],
        steps=json.loads(row[2]),
        success_rate=row[3],
        times_used=row[4],
        last_used=(datetime.fromisoformat(row[5].replace("Z", "+00:00")) if row[5] else None),
    )


class ProceduralMemory:
    def __init__(
        self,
        db_path: str = "data/conversations.db",
    ) -> None:
        self._db_path = db_path
        self._llm = get_llm_client()
        self._backend: ProceduralMemoryBackend = build_procedural_backend(db_path)
        self._cache: list[Procedure] = []
        self._cache_time: float = 0.0

    @property
    def _db(self):
        return self._backend.sqlite_db_connection

    async def initialize(self) -> None:
        await self._backend.initialize()

    def _jaccard_similarity(self, a: str, b: str) -> float:
        def tokenize(s: str) -> set[str]:
            tokens = set(re.findall(r"\w+", s.lower()))
            return {t for t in tokens if not re.match(r"^\{.*\}$", t)}

        set_a = tokenize(a)
        set_b = tokenize(b)
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)

    async def _load_cache(self) -> list[Procedure]:
        if time.monotonic() - self._cache_time < 60:
            return self._cache
        rows = await self._backend.load_all_rows()
        self._cache = []
        for row in rows:
            self._cache.append(
                Procedure(
                    id=row[0],
                    trigger_pattern=row[1],
                    steps=json.loads(row[2]),
                    success_rate=row[3],
                    times_used=row[4],
                    last_used=(
                        datetime.fromisoformat(row[5].replace("Z", "+00:00")) if row[5] else None
                    ),
                )
            )
        self._cache_time = time.monotonic()
        return self._cache

    @observe()
    async def learn_procedure(
        self,
        query: str,
        successful_response_path: list[str],
    ) -> Procedure:
        user_msg = f"Query: {query}\n\nSuccessful agent path: {', '.join(successful_response_path)}"
        result = await self._llm.generate_structured(
            _PATTERN_SYSTEM_PROMPT,
            user_msg,
            PatternExtraction,
            name="procedural.learn",
        )
        if result.trigger.strip():
            trigger_pattern = result.trigger.strip()
        else:
            trigger_pattern = query.lower()

        procedures = await self._load_cache()
        best_match: Procedure | None = None
        best_sim = 0.0
        for p in procedures:
            sim = self._jaccard_similarity(trigger_pattern, p.trigger_pattern)
            if sim > 0.6 and sim > best_sim:
                best_sim = sim
                best_match = p

        now = datetime.now(UTC).isoformat()

        if best_match is not None and best_match.id is not None:
            new_rate = 0.9 * best_match.success_rate + 0.1 * 1.0
            steps_json = json.dumps(successful_response_path) if best_match.times_used < 3 else None
            await self._backend.update_procedure_merge(best_match.id, now, new_rate, steps_json)
            self._cache_time = 0.0
            return Procedure(
                id=best_match.id,
                trigger_pattern=best_match.trigger_pattern,
                steps=successful_response_path if best_match.times_used < 3 else best_match.steps,
                success_rate=new_rate,
                times_used=best_match.times_used + 1,
                last_used=datetime.now(UTC),
            )

        rid = await self._backend.insert_procedure(
            trigger_pattern, json.dumps(successful_response_path), now
        )
        self._cache_time = 0.0
        return Procedure(
            id=rid,
            trigger_pattern=trigger_pattern,
            steps=successful_response_path,
            success_rate=1.0,
            times_used=1,
            last_used=datetime.now(UTC),
        )

    async def merge_procedure_from_learning(
        self,
        trigger_pattern: str,
        steps: list[str],
    ) -> Procedure:
        """Insert or merge a procedure without an LLM call (learning compiler / promotion)."""
        tp = (trigger_pattern or "").strip()
        if not tp:
            raise ValueError("trigger_pattern must be non-empty")
        if not steps:
            raise ValueError("steps must be non-empty")

        procedures = await self._load_cache()
        best_match: Procedure | None = None
        best_sim = 0.0
        for p in procedures:
            sim = self._jaccard_similarity(tp, p.trigger_pattern)
            if sim > 0.6 and sim > best_sim:
                best_sim = sim
                best_match = p

        now = datetime.now(UTC).isoformat()
        steps_json = json.dumps(steps)

        if best_match is not None and best_match.id is not None:
            new_rate = 0.9 * best_match.success_rate + 0.1 * 1.0
            merge_steps = steps_json if best_match.times_used < 3 else None
            await self._backend.update_procedure_merge(best_match.id, now, new_rate, merge_steps)
            self._cache_time = 0.0
            return Procedure(
                id=best_match.id,
                trigger_pattern=best_match.trigger_pattern,
                steps=steps if best_match.times_used < 3 else best_match.steps,
                success_rate=new_rate,
                times_used=best_match.times_used + 1,
                last_used=datetime.now(UTC),
            )

        rid = await self._backend.insert_procedure(tp, steps_json, now)
        self._cache_time = 0.0
        return Procedure(
            id=rid,
            trigger_pattern=tp,
            steps=list(steps),
            success_rate=1.0,
            times_used=1,
            last_used=datetime.now(UTC),
        )

    async def explain_procedure_match(self, query: str) -> ProcedureMatchExplanation:
        procedures = await self._load_cache()
        best_routable: Procedure | None = None
        best_routable_sim = 0.0
        best_candidate: Procedure | None = None
        best_candidate_sim = 0.0
        for p in procedures:
            sim = self._jaccard_similarity(query, p.trigger_pattern)
            if sim > best_candidate_sim:
                best_candidate_sim = sim
                best_candidate = p
            if not procedure_route_eligible(p):
                continue
            if sim >= _ROUTE_SIMILARITY_MIN and sim > best_routable_sim:
                best_routable_sim = sim
                best_routable = p
        blockers: list[str] = []
        if not procedures:
            blockers.append("no procedures in store")
        elif best_routable is None and best_candidate is not None:
            if best_candidate_sim < _ROUTE_SIMILARITY_MIN:
                blockers.append(
                    f"similarity {best_candidate_sim:.3f} < {_ROUTE_SIMILARITY_MIN} threshold"
                )
            if not procedure_route_eligible(best_candidate):
                if best_candidate.success_rate < _ROUTE_SUCCESS_RATE_MIN:
                    blockers.append(
                        f"success_rate {best_candidate.success_rate:.2f} < {_ROUTE_SUCCESS_RATE_MIN}"
                    )
                if best_candidate.times_used < _ROUTE_TIMES_USED_MIN:
                    blockers.append(
                        f"times_used {best_candidate.times_used} < {_ROUTE_TIMES_USED_MIN}"
                    )
        return ProcedureMatchExplanation(
            query=query,
            would_route=best_routable is not None,
            similarity=best_routable_sim if best_routable else best_candidate_sim,
            matched=best_routable,
            best_candidate=best_candidate,
            best_candidate_similarity=best_candidate_sim,
            blockers=blockers,
        )

    async def find_procedure(self, query: str) -> Procedure | None:
        return (await self.explain_procedure_match(query)).matched

    async def get_top_procedures(self, limit: int = 10) -> list[Procedure]:
        rows = await self._backend.get_top_rows(limit)
        return [_procedure_from_row(row) for row in rows]

    async def list_procedures(
        self,
        *,
        limit: int = 50,
        agent: str | None = None,
        sort: Literal["last_used", "score"] = "last_used",
    ) -> list[Procedure]:
        sql_limit = limit if agent is None else 500
        rows = await self._backend.list_rows(limit=sql_limit, sort=sort)
        out = [_procedure_from_row(row) for row in rows]
        if agent:
            key = agent.strip().lower()
            out = [p for p in out if any(s.lower() == key for s in p.steps)]
        return out[:limit]

    async def count_procedures(self) -> int:
        """Return the total number of learned procedures."""
        return await self._backend.count_procedures()

    async def record_failure(self, query: str) -> None:
        procedures = await self._load_cache()
        best_match: Procedure | None = None
        best_sim = 0.0
        for p in procedures:
            sim = self._jaccard_similarity(query, p.trigger_pattern)
            if sim > best_sim:
                best_sim = sim
                best_match = p

        if best_match is None or best_match.id is None:
            return

        new_rate = 0.9 * best_match.success_rate + 0.1 * 0.0
        if new_rate < 0.3:
            await self._backend.delete_procedure(best_match.id)
        else:
            await self._backend.update_success_rate(best_match.id, new_rate)
        self._cache_time = 0.0

    async def close(self) -> None:
        await self._backend.close()

    async def __aenter__(self) -> ProceduralMemory:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
