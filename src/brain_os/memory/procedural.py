"""Procedural memory — learned response patterns for recurring request types."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from langfuse.decorators import observe
from pydantic import BaseModel, Field

from brain_os.memory.instinct_lifecycle import (
    DEFAULT_INSTINCT_TTL_DAYS,
    instinct_age_days,
    instinct_confidence,
    instinct_expired,
)
from brain_os.memory.procedural_backend import ProceduralMemoryBackend, build_procedural_backend
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import PatternExtraction
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_PATTERN_SYSTEM_PROMPT = load_prompt("pattern_extraction")

#: Jaccard route threshold (degrade path / no embedder).
_ROUTE_JACCARD_MIN = 0.5
#: Learn/merge Jaccard threshold when embeddings unavailable.
_MERGE_JACCARD_MIN = 0.6
#: Default cosine route threshold (tuned on live Voyage paraphrase fixture).
#: See ``tests/test_procedural_embedding_match.py`` — highest thr with TP=TN=10.
_DEFAULT_COSINE_MIN = 0.63
#: Within this cosine of the top score, Jaccard breaks ties.
_DEFAULT_COSINE_TIEBREAK = 0.02

_embed_degrade_warned = False


class ProcedureEmbedder(Protocol):
    """Minimal Voyage/EmbeddingService surface used by procedural matching."""

    async def embed_texts(
        self, texts: Sequence[str], *, input_type: str = "document"
    ) -> list[list[float]]: ...

    async def embed_query(self, query: str) -> list[float]: ...


class Procedure(BaseModel):
    id: int | None = None
    trigger_pattern: str
    steps: list[str]
    success_rate: float = 1.0
    times_used: int = 1
    last_used: datetime | None = None
    trigger_embedding: list[float] | None = None


_ROUTE_SUCCESS_RATE_MIN = 0.7
_ROUTE_TIMES_USED_MIN = 3


def _instinct_ttl_days() -> int:
    try:
        from brain_os.config import get_settings

        return int(get_settings().app.instinct_ttl_days)
    except Exception:
        return DEFAULT_INSTINCT_TTL_DAYS


def _instinct_soft_expire_enabled() -> bool:
    try:
        from brain_os.config import get_settings

        return bool(get_settings().app.instinct_ttl_soft_expire_enabled)
    except Exception:
        return False


def _cosine_min() -> float:
    try:
        from brain_os.config import get_settings

        return float(get_settings().app.procedural_match_cosine_min)
    except Exception:
        return _DEFAULT_COSINE_MIN


def _cosine_tiebreak() -> float:
    try:
        from brain_os.config import get_settings

        return float(get_settings().app.procedural_match_cosine_tiebreak)
    except Exception:
        return _DEFAULT_COSINE_TIEBREAK


def _embed_match_enabled() -> bool:
    try:
        from brain_os.config import get_settings

        return bool(get_settings().app.procedural_embed_match_enabled)
    except Exception:
        return True


def procedure_route_eligible(p: Procedure) -> bool:
    return p.success_rate >= _ROUTE_SUCCESS_RATE_MIN and p.times_used >= _ROUTE_TIMES_USED_MIN


class ProcedureMatchExplanation(BaseModel):
    query: str
    would_route: bool
    similarity: float
    matched: Procedure | None = None
    best_candidate: Procedure | None = None
    best_candidate_similarity: float = 0.0
    blockers: list[str] = Field(default_factory=list)
    match_method: str = "jaccard"  # cosine | jaccard


def procedure_to_dict(p: Procedure) -> dict[str, Any]:
    ttl_days = _instinct_ttl_days()
    age_days = instinct_age_days(p)
    return {
        "id": p.id,
        "trigger_pattern": p.trigger_pattern,
        "steps": p.steps,
        "success_rate": p.success_rate,
        "times_used": p.times_used,
        "last_used": p.last_used.isoformat() if p.last_used else None,
        "eligible": procedure_route_eligible(p),
        "confidence": instinct_confidence(p, ttl_days=ttl_days),
        "age_days": round(age_days, 2) if age_days is not None else None,
        "expired": instinct_expired(p, ttl_days=ttl_days),
        "has_embedding": bool(p.trigger_embedding),
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
        "match_method": explanation.match_method,
    }


def normalize_trigger_for_embedding(text: str) -> str:
    """Strip ``{placeholder}`` tokens so Voyage sees natural language."""
    cleaned = re.sub(r"\{[^}]+\}", " ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def jaccard_similarity(a: str, b: str) -> float:
    """Token Jaccard (legacy matcher + cosine tiebreaker)."""

    def tokenize(s: str) -> set[str]:
        tokens = set(re.findall(r"\w+", s.lower()))
        return {t for t in tokens if not re.match(r"^\{.*\}$", t)}

    set_a = tokenize(a)
    set_b = tokenize(b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _encode_embedding(vec: Sequence[float]) -> str:
    return json.dumps([float(x) for x in vec], separators=(",", ":"))


def _decode_embedding(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        try:
            return [float(x) for x in raw]
        except (TypeError, ValueError):
            return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    try:
        return [float(x) for x in payload]
    except (TypeError, ValueError):
        return None


def _procedure_from_row(row: tuple[Any, ...]) -> Procedure:
    emb = _decode_embedding(row[6]) if len(row) > 6 else None
    return Procedure(
        id=row[0],
        trigger_pattern=row[1],
        steps=json.loads(row[2]),
        success_rate=row[3],
        times_used=row[4],
        last_used=(datetime.fromisoformat(row[5].replace("Z", "+00:00")) if row[5] else None),
        trigger_embedding=emb,
    )


def pick_best_by_cosine_then_jaccard(
    scored: list[tuple[float, float, Procedure]],
    *,
    tiebreak: float = _DEFAULT_COSINE_TIEBREAK,
) -> tuple[Procedure | None, float]:
    """``scored`` items are ``(cosine, jaccard, procedure)``.

    Prefer highest cosine; among candidates within ``tiebreak`` of the top
    cosine, break ties with Jaccard.
    """
    if not scored:
        return None, 0.0
    best_cos = max(s[0] for s in scored)
    near = [s for s in scored if s[0] >= best_cos - tiebreak]
    near.sort(key=lambda s: (s[1], s[0]), reverse=True)
    winner = near[0]
    return winner[2], winner[0]


class ProceduralMemory:
    def __init__(
        self,
        db_path: str = "data/conversations.db",
        *,
        embedder: ProcedureEmbedder | None = None,
    ) -> None:
        self._db_path = db_path
        self._llm = get_llm_client()
        self._backend: ProceduralMemoryBackend = build_procedural_backend(db_path)
        self._embedder = embedder
        self._cache: list[Procedure] = []
        self._cache_time: float = 0.0

    @property
    def _db(self):
        return self._backend.sqlite_db_connection

    async def initialize(self) -> None:
        await self._backend.initialize()

    def _jaccard_similarity(self, a: str, b: str) -> float:
        return jaccard_similarity(a, b)

    async def _load_cache(self) -> list[Procedure]:
        if time.monotonic() - self._cache_time < 60:
            return self._cache
        rows = await self._backend.load_all_rows()
        self._cache = [_procedure_from_row(row) for row in rows]
        self._cache_time = time.monotonic()
        return self._cache

    async def _embed_trigger(self, text: str) -> list[float] | None:
        if self._embedder is None or not _embed_match_enabled():
            return None
        try:
            cleaned = normalize_trigger_for_embedding(text)
            if not cleaned:
                return None
            vecs = await self._embedder.embed_texts([cleaned], input_type="document")
            if not vecs or not vecs[0]:
                return None
            return list(vecs[0])
        except Exception:
            global _embed_degrade_warned
            if not _embed_degrade_warned:
                logger.warning(
                    "procedural embedding failed; degrading to Jaccard",
                    exc_info=True,
                )
                _embed_degrade_warned = True
            return None

    async def backfill_embeddings(self, *, force: bool = False) -> dict[str, int]:
        """Embed trigger_patterns missing a cached vector (batch via EmbeddingService)."""
        if self._embedder is None:
            return {"attempted": 0, "written": 0, "failed": 0, "skipped": 0}
        procedures = await self._load_cache()
        missing = [p for p in procedures if p.id is not None and (force or not p.trigger_embedding)]
        if not missing:
            return {"attempted": 0, "written": 0, "failed": 0, "skipped": len(procedures)}
        attempted = len(missing)
        written = 0
        failed = 0
        batch_size = 32
        for i in range(0, len(missing), batch_size):
            chunk = missing[i : i + batch_size]
            texts = [normalize_trigger_for_embedding(p.trigger_pattern) for p in chunk]
            try:
                vectors = await self._embedder.embed_texts(texts, input_type="document")
            except Exception:
                global _embed_degrade_warned
                if not _embed_degrade_warned:
                    logger.warning(
                        "procedural embedding backfill failed; degrading to Jaccard",
                        exc_info=True,
                    )
                    _embed_degrade_warned = True
                failed += len(chunk)
                continue
            for proc, vec, text in zip(chunk, vectors, texts, strict=False):
                if not text or not vec or proc.id is None:
                    failed += 1
                    continue
                try:
                    await self._backend.update_trigger_embedding(proc.id, _encode_embedding(vec))
                    proc.trigger_embedding = list(vec)
                    written += 1
                except Exception:
                    failed += 1
                    logger.debug(
                        "procedural embedding persist failed id=%s", proc.id, exc_info=True
                    )
        self._cache_time = 0.0
        return {
            "attempted": attempted,
            "written": written,
            "failed": failed,
            "skipped": len(procedures) - attempted,
        }

    async def _score_candidates(
        self, query: str, procedures: list[Procedure]
    ) -> tuple[list[tuple[float, float, Procedure]], str]:
        """Return ``(cosine_or_jaccard, jaccard, procedure)`` and method name."""
        if not procedures:
            return [], "jaccard"
        jaccards = [jaccard_similarity(query, p.trigger_pattern) for p in procedures]
        use_embed = self._embedder is not None and _embed_match_enabled()
        if not use_embed:
            return [(j, j, p) for j, p in zip(jaccards, procedures, strict=True)], "jaccard"

        # Lazy-fill missing vectors so match works before an explicit backfill.
        missing = [p for p in procedures if not p.trigger_embedding and p.id is not None]
        if missing:
            await self.backfill_embeddings(force=False)
            procedures = await self._load_cache()
            jaccards = [jaccard_similarity(query, p.trigger_pattern) for p in procedures]

        try:
            q_text = normalize_trigger_for_embedding(query) or query
            q_vec = await self._embedder.embed_query(q_text)  # type: ignore[union-attr]
        except Exception:
            global _embed_degrade_warned
            if not _embed_degrade_warned:
                logger.warning(
                    "procedural embedding match failed; degrading to Jaccard",
                    exc_info=True,
                )
                _embed_degrade_warned = True
            return [(j, j, p) for j, p in zip(jaccards, procedures, strict=True)], "jaccard"

        if not q_vec:
            return [(j, j, p) for j, p in zip(jaccards, procedures, strict=True)], "jaccard"

        scored: list[tuple[float, float, Procedure]] = []
        for p, jac in zip(procedures, jaccards, strict=True):
            if p.trigger_embedding:
                cos = cosine_similarity(q_vec, p.trigger_embedding)
            else:
                cos = jac  # no vector → fall back to Jaccard for this row
            scored.append((cos, jac, p))
        return scored, "cosine"

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
        scored, method = await self._score_candidates(trigger_pattern, procedures)
        merge_min = _cosine_min() if method == "cosine" else _MERGE_JACCARD_MIN
        best_match: Procedure | None = None
        best_sim = 0.0
        for sim, _jac, p in scored:
            if sim > merge_min and sim > best_sim:
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
                trigger_embedding=best_match.trigger_embedding,
            )

        emb = await self._embed_trigger(trigger_pattern)
        emb_json = _encode_embedding(emb) if emb else None
        rid = await self._backend.insert_procedure(
            trigger_pattern,
            json.dumps(successful_response_path),
            now,
            trigger_embedding=emb_json,
        )
        self._cache_time = 0.0
        return Procedure(
            id=rid,
            trigger_pattern=trigger_pattern,
            steps=successful_response_path,
            success_rate=1.0,
            times_used=1,
            last_used=datetime.now(UTC),
            trigger_embedding=emb,
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
        scored, method = await self._score_candidates(tp, procedures)
        merge_min = _cosine_min() if method == "cosine" else _MERGE_JACCARD_MIN
        best_match: Procedure | None = None
        best_sim = 0.0
        for sim, _jac, p in scored:
            if sim > merge_min and sim > best_sim:
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
                trigger_embedding=best_match.trigger_embedding,
            )

        emb = await self._embed_trigger(tp)
        emb_json = _encode_embedding(emb) if emb else None
        rid = await self._backend.insert_procedure(tp, steps_json, now, trigger_embedding=emb_json)
        self._cache_time = 0.0
        return Procedure(
            id=rid,
            trigger_pattern=tp,
            steps=list(steps),
            success_rate=1.0,
            times_used=1,
            last_used=datetime.now(UTC),
            trigger_embedding=emb,
        )

    async def explain_procedure_match(self, query: str) -> ProcedureMatchExplanation:
        procedures = await self._load_cache()
        now = datetime.now(UTC)
        ttl_days = _instinct_ttl_days()
        soft_expire = _instinct_soft_expire_enabled()
        scored, method = await self._score_candidates(query, procedures)
        route_min = _cosine_min() if method == "cosine" else _ROUTE_JACCARD_MIN
        tiebreak = _cosine_tiebreak() if method == "cosine" else 0.0

        def _route_eligible(p: Procedure) -> bool:
            if not procedure_route_eligible(p):
                return False
            if soft_expire and instinct_expired(p, now=now, ttl_days=ttl_days):
                return False
            return True

        best_candidate, best_candidate_sim = pick_best_by_cosine_then_jaccard(
            scored, tiebreak=tiebreak
        )
        routable = [(c, j, p) for c, j, p in scored if _route_eligible(p)]
        best_routable, best_routable_sim = pick_best_by_cosine_then_jaccard(
            routable, tiebreak=tiebreak
        )
        if best_routable is not None and best_routable_sim < route_min:
            best_routable = None
            best_routable_sim = 0.0

        blockers: list[str] = []
        if not procedures:
            blockers.append("no procedures in store")
        elif best_routable is None and best_candidate is not None:
            if best_candidate_sim < route_min:
                blockers.append(
                    f"similarity {best_candidate_sim:.3f} < {route_min} threshold ({method})"
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
            if soft_expire and instinct_expired(best_candidate, now=now, ttl_days=ttl_days):
                age = instinct_age_days(best_candidate, now=now)
                blockers.append(f"expired: idle {age:.0f}d > {ttl_days}d TTL")
        return ProcedureMatchExplanation(
            query=query,
            would_route=best_routable is not None,
            similarity=best_routable_sim if best_routable else best_candidate_sim,
            matched=best_routable,
            best_candidate=best_candidate,
            best_candidate_similarity=best_candidate_sim,
            blockers=blockers,
            match_method=method,
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

    async def list_decay_audit_candidates(
        self,
        *,
        success_rate_below: float = 0.35,
        times_used_below: int = 3,
        limit: int = 50,
    ) -> list[Procedure]:
        """Dry-run prune candidates (low rate + low use); does not delete."""
        procedures = await self._load_cache()
        out: list[Procedure] = []
        for p in procedures:
            if p.success_rate >= success_rate_below:
                continue
            if p.times_used >= times_used_below:
                continue
            out.append(p)
        out.sort(key=lambda x: (x.success_rate, x.times_used))
        return out[:limit]

    async def prune_decay_audit_candidates(
        self,
        *,
        success_rate_below: float = 0.35,
        times_used_below: int = 3,
        limit: int = 20,
    ) -> int:
        """Delete low-value procedures (dream stage 5); returns count removed."""
        candidates = await self.list_decay_audit_candidates(
            success_rate_below=success_rate_below,
            times_used_below=times_used_below,
            limit=limit,
        )
        pruned = 0
        for proc in candidates:
            if proc.id is None:
                continue
            old_rate = proc.success_rate
            await self._backend.delete_procedure(proc.id)
            from brain_os.memory.procedural_decay_metrics import append_decay_event

            append_decay_event(
                action="pruned",
                procedure_id=proc.id,
                trigger_pattern=proc.trigger_pattern,
                success_rate_before=old_rate,
                query_snippet="dream_stage5_prune",
            )
            pruned += 1
        if pruned:
            self._cache_time = 0.0
        return pruned

    async def touch(self, procedure_id: int) -> None:
        """Bump ``last_used`` to now so an actively-routed instinct does not expire.

        Procedural route hits (pipeline step 4) do not re-learn, so without this
        a heavily-used instinct would look idle and hit the TTL. Updates the
        timestamp only — never ``times_used`` or ``success_rate``.
        """
        now = datetime.now(UTC).isoformat()
        await self._backend.touch_procedure(procedure_id, now)
        self._cache_time = 0.0

    async def list_expired_instincts(
        self,
        *,
        ttl_days: int,
        limit: int = 50,
        now: datetime | None = None,
    ) -> list[Procedure]:
        """Dry-run list of instincts idle longer than ``ttl_days`` (oldest first)."""
        procedures = await self._load_cache()
        ref = now or datetime.now(UTC)
        out = [p for p in procedures if instinct_expired(p, now=ref, ttl_days=ttl_days)]
        out.sort(key=lambda x: instinct_age_days(x, now=ref) or 0.0, reverse=True)
        return out[:limit]

    async def prune_expired_instincts(
        self,
        *,
        ttl_days: int,
        limit: int = 20,
        now: datetime | None = None,
    ) -> int:
        """Delete TTL-expired instincts (dream stage 5); returns count removed."""
        ref = now or datetime.now(UTC)
        candidates = await self.list_expired_instincts(ttl_days=ttl_days, limit=limit, now=ref)
        pruned = 0
        for proc in candidates:
            if proc.id is None:
                continue
            age = instinct_age_days(proc, now=ref)
            await self._backend.delete_procedure(proc.id)
            from brain_os.memory.procedural_decay_metrics import append_decay_event

            append_decay_event(
                action="expired",
                procedure_id=proc.id,
                trigger_pattern=proc.trigger_pattern,
                success_rate_before=proc.success_rate,
                query_snippet=(
                    f"ttl_{ttl_days}d_idle_{age:.0f}d" if age is not None else f"ttl_{ttl_days}d"
                ),
            )
            pruned += 1
        if pruned:
            self._cache_time = 0.0
        return pruned

    async def record_failure(self, query: str) -> None:
        procedures = await self._load_cache()
        scored, _method = await self._score_candidates(query, procedures)
        best_match, best_sim = pick_best_by_cosine_then_jaccard(scored, tiebreak=_cosine_tiebreak())
        if best_match is None or best_match.id is None or best_sim <= 0:
            return

        old_rate = best_match.success_rate
        new_rate = 0.9 * best_match.success_rate + 0.1 * 0.0
        if new_rate < 0.3:
            await self._backend.delete_procedure(best_match.id)
            from brain_os.memory.procedural_decay_metrics import append_decay_event

            append_decay_event(
                action="pruned",
                procedure_id=best_match.id,
                trigger_pattern=best_match.trigger_pattern,
                success_rate_before=old_rate,
                query_snippet=query,
            )
        else:
            await self._backend.update_success_rate(best_match.id, new_rate)
            from brain_os.memory.procedural_decay_metrics import append_decay_event

            append_decay_event(
                action="demoted",
                procedure_id=best_match.id,
                trigger_pattern=best_match.trigger_pattern,
                success_rate_before=old_rate,
                success_rate_after=new_rate,
                query_snippet=query,
            )
        self._cache_time = 0.0

    async def close(self) -> None:
        await self._backend.close()

    async def __aenter__(self) -> ProceduralMemory:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
