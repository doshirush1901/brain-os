"""Inner voice — evolving personality traits and internal reflections."""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import aiosqlite
from langfuse.decorators import observe
from pydantic import BaseModel, Field

from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import InnerReflection
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_DEFAULT_TRAITS: list[dict[str, Any]] = [
    {
        "name": "warmth",
        "value": 0.7,
        "description": "Tendency toward friendly, caring communication",
    },
    {
        "name": "directness",
        "value": 0.6,
        "description": "Preference for concise, straight-to-the-point responses",
    },
    {"name": "humor", "value": 0.3, "description": "Inclination to use light humor or wit"},
    {
        "name": "curiosity",
        "value": 0.8,
        "description": "Drive to ask follow-up questions and explore topics",
    },
    {"name": "formality", "value": 0.5, "description": "Level of formal vs. casual language"},
    {
        "name": "empathy",
        "value": 0.7,
        "description": "Sensitivity to emotional context and user feelings",
    },
]


class PersonalityTrait(BaseModel):
    name: str
    value: float = Field(ge=0.0, le=1.0)
    description: str


class ReflectionType(str, Enum):
    OBSERVATION = "OBSERVATION"
    OPINION = "OPINION"
    CELEBRATION = "CELEBRATION"
    CURIOSITY = "CURIOSITY"
    CONNECTION = "CONNECTION"
    CONCERN = "CONCERN"


_REFLECT_SYSTEM_TEMPLATE = load_prompt("inner_voice_reflect")


class InnerVoice:
    def __init__(
        self,
        db_path: str = "data/conversations.db",
        surface_probability: float = 0.1,
    ) -> None:
        self._db_path = db_path
        self._surface_probability = surface_probability
        self._llm = get_llm_client()
        self._db: aiosqlite.Connection | None = None
        self._traits: dict[str, PersonalityTrait] = {}

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path, timeout=30.0)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS personality_traits (
                name TEXT PRIMARY KEY,
                value REAL NOT NULL,
                description TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS trait_changelog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trait_name TEXT NOT NULL,
                old_value REAL NOT NULL,
                new_value REAL NOT NULL,
                delta REAL NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        await self._db.commit()

        cursor = await self._db.execute("SELECT COUNT(*) FROM personality_traits")
        row = await cursor.fetchone()
        await cursor.close()
        if row is not None and row[0] == 0:
            for t in _DEFAULT_TRAITS:
                await self._db.execute(
                    "INSERT INTO personality_traits (name, value, description) VALUES (?, ?, ?)",
                    (t["name"], t["value"], t["description"]),
                )
            await self._db.commit()

        cursor = await self._db.execute("SELECT name, value, description FROM personality_traits")
        rows = await cursor.fetchall()
        await cursor.close()
        self._traits = {
            r[0]: PersonalityTrait(name=r[0], value=r[1], description=r[2]) for r in rows
        }

    @observe()
    async def reflect(self, context: str, trigger: str) -> dict:
        trait_vals = {t.name: t.value for t in self._traits.values()}
        system = _REFLECT_SYSTEM_TEMPLATE.format(**trait_vals)
        user_content = f"CONTEXT: {context}\n\nTRIGGER: {trigger}"
        try:
            result = await self._llm.generate_structured(
                system,
                user_content,
                InnerReflection,
                name="inner_voice.reflect",
                model_tier="cheap",
            )
            reflection_type = ReflectionType(result.reflection_type)
            should_surface = result.should_surface
            if should_surface and reflection_type not in (
                ReflectionType.CELEBRATION,
                ReflectionType.CONCERN,
            ):
                if random.random() >= self._surface_probability:
                    should_surface = False
            return {
                "reflection_type": reflection_type,
                "content": result.content,
                "should_surface": should_surface,
            }
        except (ValueError, TypeError):
            logger.warning("InnerVoice reflect: LLM returned invalid data")
            return {
                "reflection_type": ReflectionType.OBSERVATION,
                "content": "",
                "should_surface": False,
            }

    async def update_trait(self, trait_name: str, delta: float, reason: str) -> None:
        if trait_name not in self._traits:
            raise ValueError(f"Unknown trait: {trait_name}")
        old_value = self._traits[trait_name].value
        new_value = max(0.0, min(1.0, old_value + delta))
        self._traits[trait_name] = PersonalityTrait(
            name=trait_name,
            value=new_value,
            description=self._traits[trait_name].description,
        )
        assert self._db is not None
        await self._db.execute(
            "UPDATE personality_traits SET value = ? WHERE name = ?",
            (new_value, trait_name),
        )
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            """
            INSERT INTO trait_changelog (trait_name, old_value, new_value, delta, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (trait_name, old_value, new_value, delta, reason, now),
        )
        await self._db.commit()

    def get_personality_summary(self) -> str:
        parts: list[str] = []
        for t in self._traits.values():
            if t.value >= 0.8:
                level = "very"
            elif t.value >= 0.6:
                level = "moderately"
            elif t.value >= 0.4:
                level = "balanced"
            else:
                level = "low"
            parts.append(f"{level} {t.name}")
        if not parts:
            return "Brain OS has no personality traits configured."
        return f"Brain OS is currently {', '.join(parts)}."

    def get_trait(self, trait_name: str) -> PersonalityTrait | None:
        return self._traits.get(trait_name)

    def get_all_traits(self) -> dict[str, PersonalityTrait]:
        return dict(self._traits)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> InnerVoice:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
