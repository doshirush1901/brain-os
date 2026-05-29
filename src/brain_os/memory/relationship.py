"""Relationship memory — tracks depth and quality of contact relationships."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import aiosqlite  # noqa: F401 — characterization tests patch module connect
from langfuse.decorators import observe
from pydantic import BaseModel, Field

from brain_os.data.models import Interaction, WarmthLevel
from brain_os.memory.relationship_backend import (
    RelationshipMemoryBackend,
    build_relationship_backend,
)
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import MemorableMoment, MemorableMoments
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_MOMENTS_SYSTEM_PROMPT = load_prompt("memorable_moments")

_GREETINGS: dict[WarmthLevel, str] = {
    WarmthLevel.STRANGER: "Hello, thank you for reaching out to Machinecraft. How can I help you today?",
    WarmthLevel.ACQUAINTANCE: "Hello! Good to hear from you again. How can I assist you?",
    WarmthLevel.FAMILIAR: "Hi there! Nice to connect again. What can I do for you?",
    WarmthLevel.WARM: "Hey, great to hear from you! What's on your mind?",
    WarmthLevel.TRUSTED: "Hey! Always good to hear from you. What can I help with?",
}


class Relationship(BaseModel):
    contact_id: str
    warmth_level: WarmthLevel = WarmthLevel.STRANGER
    interaction_count: int = 0
    memorable_moments: list[str] = Field(default_factory=list)
    learned_preferences: dict[str, str] = Field(default_factory=dict)
    first_interaction: datetime | None = None
    last_interaction: datetime | None = None


def _relationship_from_row(row: tuple[Any, ...]) -> Relationship:
    first = None
    if row[5]:
        first = datetime.fromisoformat(str(row[5]).replace("Z", "+00:00"))
    last = None
    if row[6]:
        last = datetime.fromisoformat(str(row[6]).replace("Z", "+00:00"))
    try:
        warmth = WarmthLevel(row[1])
    except ValueError:
        warmth = WarmthLevel.STRANGER
    moments = json.loads(row[3]) if isinstance(row[3], str) else []
    prefs = json.loads(row[4]) if isinstance(row[4], str) else {}
    return Relationship(
        contact_id=row[0],
        warmth_level=warmth,
        interaction_count=row[2] or 0,
        memorable_moments=moments if isinstance(moments, list) else [],
        learned_preferences=prefs if isinstance(prefs, dict) else {},
        first_interaction=first,
        last_interaction=last,
    )


class RelationshipMemory:
    def __init__(
        self,
        db_path: str = "data/relationships.db",
    ) -> None:
        self._db_path = db_path
        self._llm = get_llm_client()
        self._backend: RelationshipMemoryBackend = build_relationship_backend(db_path)

    @property
    def _db(self):
        return self._backend.sqlite_db_connection

    async def initialize(self) -> None:
        await self._backend.initialize()

    async def _persist(self, contact_id: str, rel: Relationship) -> None:
        first_iso = rel.first_interaction.isoformat() if rel.first_interaction else None
        last_iso = rel.last_interaction.isoformat() if rel.last_interaction else None
        await self._backend.upsert_row(
            contact_id,
            rel.warmth_level.value,
            rel.interaction_count,
            json.dumps(rel.memorable_moments),
            json.dumps(rel.learned_preferences),
            first_iso,
            last_iso,
        )

    @observe()
    async def update_relationship(self, contact_id: str, interaction: Interaction) -> Relationship:
        rel = await self.get_relationship(contact_id)
        rel.interaction_count += 1
        rel.last_interaction = interaction.created_at
        if rel.first_interaction is None:
            rel.first_interaction = interaction.created_at

        if interaction.content and len(interaction.content) > 50:
            result = await self._llm.generate_structured(
                _MOMENTS_SYSTEM_PROMPT,
                interaction.content,
                MemorableMoments,
                name="relationship.moments",
            )
            for m in result.moments:
                if isinstance(m, MemorableMoment):
                    text = m.content.strip()
                else:
                    text = m.strip()
                if text:
                    rel.memorable_moments.append(text)
            rel.memorable_moments = rel.memorable_moments[-50:]

        rel.warmth_level = self._check_warmth_upgrade(rel)
        await self._persist(contact_id, rel)
        return rel

    def _check_warmth_upgrade(self, rel: Relationship) -> WarmthLevel:
        current = rel.warmth_level
        if current == WarmthLevel.STRANGER and rel.interaction_count >= 3:
            return WarmthLevel.ACQUAINTANCE
        if current == WarmthLevel.ACQUAINTANCE:
            if (
                rel.interaction_count >= 10
                and rel.first_interaction is not None
                and rel.last_interaction is not None
            ):
                days = (rel.last_interaction - rel.first_interaction).days
                if days >= 14:
                    return WarmthLevel.FAMILIAR
        if current == WarmthLevel.FAMILIAR:
            if rel.interaction_count >= 20 and len(rel.memorable_moments) >= 3:
                return WarmthLevel.WARM
        return current

    async def get_relationship(self, contact_id: str) -> Relationship:
        row = await self._backend.get_row(contact_id)
        if row is None:
            return Relationship(contact_id=contact_id)
        return _relationship_from_row(row)

    def get_greeting_style(self, relationship: Relationship) -> str:
        return _GREETINGS.get(relationship.warmth_level, _GREETINGS[WarmthLevel.STRANGER])

    async def promote_to_trusted(self, contact_id: str) -> Relationship:
        rel = await self.get_relationship(contact_id)
        if rel.warmth_level != WarmthLevel.WARM:
            raise ValueError("Can only promote WARM contacts to TRUSTED")
        rel.warmth_level = WarmthLevel.TRUSTED
        await self._persist(contact_id, rel)
        return rel

    async def get_all_relationships(
        self,
        min_warmth: WarmthLevel | None = None,
    ) -> list[Relationship]:
        rows = await self._backend.list_rows(min_warmth=min_warmth)
        return [_relationship_from_row(row) for row in rows]

    async def close(self) -> None:
        await self._backend.close()

    async def __aenter__(self) -> RelationshipMemory:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
