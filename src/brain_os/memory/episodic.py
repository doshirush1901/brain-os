"""Episodic memory — narrative summaries of significant interactions."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from langfuse.decorators import observe

from brain_os.memory.episode_hygiene import should_skip_episode_write
from brain_os.memory.episodic_backend import EpisodicMemoryBackend, build_episodic_backend
from brain_os.memory.long_term import LongTermMemory
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import EpisodeConsolidation
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_CONSOLIDATE_SYSTEM_PROMPT = load_prompt("consolidate_episode")

_WEAVE_SYSTEM_PROMPT = load_prompt("weave_episodes")


class EpisodicMemory:
    def __init__(
        self,
        long_term: LongTermMemory,
        db_path: str = "data/conversations.db",
    ) -> None:
        self._long_term = long_term
        self._db_path = db_path
        self._llm = get_llm_client()
        self._backend: EpisodicMemoryBackend = build_episodic_backend(db_path)

    @property
    def _db(self):
        return self._backend.sqlite_db_connection

    async def initialize(self) -> None:
        await self._backend.initialize()

    @observe()
    async def consolidate_episode(self, conversation: list[dict], user_id: str) -> dict:
        transcript = "".join(
            f"[{m.get('role', 'unknown')}] {m.get('content', '')}\n" for m in conversation
        )
        fallback = {
            "narrative": "(Consolidation failed)",
            "key_topics": [],
            "decisions_made": [],
            "commitments": [],
            "emotional_tone": "unknown",
            "relationship_impact": "unknown",
        }
        try:
            result = await self._llm.generate_structured(
                _CONSOLIDATE_SYSTEM_PROMPT,
                transcript,
                EpisodeConsolidation,
                name="episodic.consolidate",
                model_tier="cheap",
            )
            episode = {
                "narrative": result.narrative or fallback["narrative"],
                "key_topics": result.key_topics,
                "decisions_made": result.decisions_made,
                "commitments": result.commitments,
                "emotional_tone": result.emotional_tone or fallback["emotional_tone"],
                "relationship_impact": result.relationship_impact
                or fallback["relationship_impact"],
            }
        except Exception:
            logger.exception("Structured LLM call failed in EpisodicMemory.consolidate_episode")
            episode = fallback

        skip_reason = should_skip_episode_write(
            narrative=episode["narrative"],
            transcript=conversation,
            user_id=user_id,
        )
        if skip_reason:
            return {
                "skipped": True,
                "skip_reason": skip_reason,
                "user_id": user_id,
                **episode,
            }

        now = datetime.now(UTC).isoformat()

        async def _write_episode() -> int:
            return await self._backend.insert_episode(
                user_id,
                episode["narrative"],
                json.dumps(episode["key_topics"]),
                json.dumps(episode["decisions_made"]),
                json.dumps(episode["commitments"]),
                episode["emotional_tone"],
                episode["relationship_impact"],
                now,
            )

        if episode != fallback:
            mem0_task = self._long_term.store_gated(
                episode["narrative"],
                user_id,
                metadata={
                    "type": "episode",
                    "key_topics": json.dumps(episode["key_topics"]),
                    "emotional_tone": episode["emotional_tone"],
                    "memory_category": "episode",
                },
                source="episodic:consolidate",
                category="episode",
            )
            ep_id, _ = await asyncio.gather(_write_episode(), mem0_task)
        else:
            ep_id = await _write_episode()

        return {
            "id": ep_id,
            "user_id": user_id,
            **episode,
            "created_at": now,
        }

    async def weave_episodes(self, user_id: str, topic: str | None = None) -> str:
        rows = await self._backend.weave_rows(user_id, topic, 10)
        if not rows:
            return "No episodes found for this user."

        formatted = "\n".join(f"{i + 1}. [{r[2]}] {r[1]}" for i, r in enumerate(rows))
        try:
            raw = await self._llm.generate_text(
                _WEAVE_SYSTEM_PROMPT,
                formatted,
                name="episodic.weave",
                model_tier="cheap",
            )
        except Exception:
            logger.exception("Text LLM call failed in EpisodicMemory.weave_episodes")
            return "(Narrative weaving failed)"
        if not raw or not raw.strip():
            return "(Narrative weaving failed)"
        return raw.strip()

    async def surface_relevant_episodes(self, query: str, user_id: str) -> list[dict]:
        # Server-side type filter (Mem0 v2) — only episode memories come back.
        mem0_results = await self._long_term.search(
            query, user_id, limit=5, metadata_filter={"type": "episode"}
        )
        mem0_episodes = [r for r in mem0_results if r.get("metadata", {}).get("type") == "episode"]

        keywords = query.lower().strip().split()[:3]
        sqlite_episodes: list[dict] = []
        if keywords:
            rows = await self._backend.surface_rows(user_id, keywords, 10)
            sqlite_episodes = [
                {
                    "id": r[0],
                    "narrative": r[1],
                    "key_topics": r[2],
                    "emotional_tone": r[3],
                    "relationship_impact": r[4],
                    "created_at": r[5],
                }
                for r in rows
            ]

        seen: set[str] = set()
        merged: list[dict] = []
        for r in mem0_episodes:
            narrative = r.get("memory", "")
            sig = narrative[:100] if narrative else ""
            if sig and sig not in seen:
                seen.add(sig)
                merged.append(
                    {
                        "id": r.get("id", ""),
                        "narrative": narrative,
                        "key_topics": r.get("metadata", {}).get("key_topics", "[]"),
                        "emotional_tone": r.get("metadata", {}).get("emotional_tone", ""),
                        "relationship_impact": "",
                        "created_at": r.get("created_at", ""),
                    }
                )
        for r in sqlite_episodes:
            narrative = r.get("narrative", "")
            sig = narrative[:100] if narrative else ""
            if sig and sig not in seen:
                seen.add(sig)
                merged.append(r)
        return merged[:5]

    async def close(self) -> None:
        await self._backend.close()

    async def __aenter__(self) -> EpisodicMemory:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
