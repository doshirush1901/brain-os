"""Sensory system — unified perception and cross-channel identity resolution.

Every incoming message, regardless of channel, flows through ``perceive()``
to resolve the sender's identity, detect emotional state, and gather context
before reaching the agent pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
from neo4j.exceptions import Neo4jError
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, MetaData, String, Table, select
from sqlalchemy.ext.asyncio import create_async_engine

from brain_os.brain.knowledge_graph import KnowledgeGraph
from brain_os.config import get_settings
from brain_os.data.models import Channel, Contact, EmotionalState, WarmthLevel

logger = logging.getLogger(__name__)

_MEMORY_SUBSYSTEM_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    asyncio.TimeoutError,
    httpx.HTTPError,
    json.JSONDecodeError,
    UnicodeDecodeError,
)
_NEO4J_GRAPH_ERRORS = (
    Neo4jError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
)
_MEMORY_BLOCKS_ERRORS = _MEMORY_SUBSYSTEM_ERRORS + (ImportError,)

# ── Pydantic model ────────────────────────────────────────────────────────


class PerceptionEvent(BaseModel):
    """An incoming message from any channel."""

    channel: Channel
    raw_input: str
    sender_id: str
    sender_name: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = Field(default_factory=dict)


# ── SQLAlchemy identity mapping table ─────────────────────────────────────

identity_metadata = MetaData()

identity_mappings_table = Table(
    "identity_mappings",
    identity_metadata,
    Column("channel", String(20), primary_key=True),
    Column("sender_id", String(500), primary_key=True),
    Column("contact_email", String(500), nullable=False, index=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("last_seen_at", DateTime(timezone=True), nullable=False),
)


# ── system class ──────────────────────────────────────────────────────────


class SensorySystem:
    """Provides unified perception across all input channels."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        *,
        emotional_intelligence: Any | None = None,
        conversation_memory: Any | None = None,
        relationship_memory: Any | None = None,
        memory_block_store: Any | None = None,
        database_url: str | None = None,
    ) -> None:
        self._graph = knowledge_graph
        self._emotional_intelligence = emotional_intelligence
        self._conversation_memory = conversation_memory
        self._relationship_memory = relationship_memory
        self._memory_block_store = memory_block_store

        url = database_url or get_settings().database.url
        self._engine = create_async_engine(url)

        self._identity_cache: dict[tuple[str, str], str] = {}
        self._IDENTITY_CACHE_MAX = 1000

    def configure_memory(
        self,
        *,
        emotional_intelligence: Any | None = None,
        conversation_memory: Any | None = None,
        relationship_memory: Any | None = None,
        memory_block_store: Any | None = None,
    ) -> None:
        """Late-bind memory subsystems after construction."""
        if emotional_intelligence is not None:
            self._emotional_intelligence = emotional_intelligence
        if conversation_memory is not None:
            self._conversation_memory = conversation_memory
        if relationship_memory is not None:
            self._relationship_memory = relationship_memory
        if memory_block_store is not None:
            self._memory_block_store = memory_block_store

    async def create_tables(self) -> None:
        """Create the identity_mappings table if it doesn't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(identity_metadata.create_all)
        logger.info("SensorySystem tables ensured")

    # ── main perception entry point ───────────────────────────────────────

    async def perceive(self, event: PerceptionEvent) -> dict[str, Any]:
        """Build a unified perception dict from an incoming message."""
        contact = await self.resolve_identity(
            event.channel.value,
            event.sender_id,
            event.sender_name,
        )

        # Emotional state
        if self._emotional_intelligence is not None:
            try:
                emotional_result = await self._emotional_intelligence.detect_emotion(
                    event.raw_input
                )
            except _MEMORY_SUBSYSTEM_ERRORS:
                logger.exception("Emotion detection failed")
                emotional_result = {"state": EmotionalState.NEUTRAL.value, "confidence": 0.0}
        else:
            emotional_result = {"state": EmotionalState.NEUTRAL.value, "confidence": 0.0}

        # Conversation history
        if self._conversation_memory is not None:
            try:
                history = await self._conversation_memory.get_history(
                    contact.email,
                    event.channel.value,
                    limit=10,
                )
            except _MEMORY_SUBSYSTEM_ERRORS:
                logger.exception("Conversation history retrieval failed")
                history = []
        else:
            history = []

        # Relationship context
        if self._relationship_memory is not None:
            try:
                relationship = await self._relationship_memory.get_relationship(contact.email)
            except _MEMORY_SUBSYSTEM_ERRORS:
                logger.exception("Relationship retrieval failed")
                relationship = {"warmth": WarmthLevel.STRANGER.value}
        else:
            relationship = {"warmth": WarmthLevel.STRANGER.value}

        memory_blocks_xml = ""
        if self._memory_block_store is not None and contact.email and "@" in contact.email:
            try:
                from brain_os.memory.block_sync import ensure_contact_blocks_rendered

                memory_blocks_xml = await ensure_contact_blocks_rendered(
                    self._memory_block_store,
                    contact.email,
                    relationship=self._relationship_memory,
                )
            except _MEMORY_BLOCKS_ERRORS:
                logger.exception("Memory blocks perception failed")

        return {
            "resolved_contact": {
                "name": contact.name,
                "email": contact.email,
                "company": contact.company,
                "region": contact.region,
                "score": contact.score,
            },
            "emotional_state": emotional_result,
            "conversation_history": history,
            "relationship": relationship,
            "memory_blocks": memory_blocks_xml,
            "channel_context": {
                "channel": event.channel.value,
                "sender_id": event.sender_id,
                "metadata": event.metadata,
            },
            "timestamp": event.timestamp.isoformat(),
        }

    # ── identity resolution ───────────────────────────────────────────────

    async def resolve_identity(
        self,
        channel: str,
        sender_id: str,
        sender_name: str | None = None,
    ) -> Contact:
        """Look up or create a contact for the given channel + sender_id."""
        cache_key = (channel, sender_id)
        cached_email = self._identity_cache.get(cache_key)

        if cached_email is not None:
            return self._build_contact(cached_email, sender_name)

        # Database lookup
        stmt = select(identity_mappings_table).where(
            identity_mappings_table.c.channel == channel,
            identity_mappings_table.c.sender_id == sender_id,
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).first()

        now = datetime.now(UTC)

        if row is not None:
            contact_email = row.contact_email  # type: ignore[union-attr]
            # Update last_seen_at
            async with self._engine.begin() as conn:
                await conn.execute(
                    identity_mappings_table.update()
                    .where(
                        identity_mappings_table.c.channel == channel,
                        identity_mappings_table.c.sender_id == sender_id,
                    )
                    .values(last_seen_at=now)
                )
            self._identity_cache[cache_key] = contact_email
            if len(self._identity_cache) > self._IDENTITY_CACHE_MAX:
                del self._identity_cache[next(iter(self._identity_cache))]
            return self._build_contact(contact_email, sender_name)

        # New identity
        if channel == Channel.EMAIL.value:
            contact_email = sender_id
        elif "@" in sender_id and "." in sender_id.split("@", 1)[-1]:
            # API/CLI user_id is often a real work email; use it as the canonical key
            # so ConversationMemory and Mem0 align with one person.
            contact_email = sender_id.strip()
        else:
            contact_email = f"{channel.lower()}_{sender_id}@unknown"

        async with self._engine.begin() as conn:
            await conn.execute(
                identity_mappings_table.insert().values(
                    channel=channel,
                    sender_id=sender_id,
                    contact_email=contact_email,
                    created_at=now,
                    last_seen_at=now,
                )
            )

        if channel == Channel.EMAIL.value:
            try:
                await self._graph.add_person(
                    name=sender_name or sender_id,
                    email=contact_email,
                )
            except _NEO4J_GRAPH_ERRORS:
                logger.exception("Failed to add person to knowledge graph")

        self._identity_cache[cache_key] = contact_email
        if len(self._identity_cache) > self._IDENTITY_CACHE_MAX:
            del self._identity_cache[next(iter(self._identity_cache))]
        logger.info("New identity: %s/%s -> %s", channel, sender_id, contact_email)
        return self._build_contact(contact_email, sender_name)

    async def link_identity(
        self,
        channel: str,
        sender_id: str,
        contact_email: str,
    ) -> None:
        """Manually link a channel identity to an existing contact email."""
        now = datetime.now(UTC)

        # Check if mapping exists
        stmt = select(identity_mappings_table).where(
            identity_mappings_table.c.channel == channel,
            identity_mappings_table.c.sender_id == sender_id,
        )
        async with self._engine.connect() as conn:
            existing = (await conn.execute(stmt)).first()

        if existing:
            async with self._engine.begin() as conn:
                await conn.execute(
                    identity_mappings_table.update()
                    .where(
                        identity_mappings_table.c.channel == channel,
                        identity_mappings_table.c.sender_id == sender_id,
                    )
                    .values(contact_email=contact_email, last_seen_at=now)
                )
        else:
            async with self._engine.begin() as conn:
                await conn.execute(
                    identity_mappings_table.insert().values(
                        channel=channel,
                        sender_id=sender_id,
                        contact_email=contact_email,
                        created_at=now,
                        last_seen_at=now,
                    )
                )

        self._identity_cache[(channel, sender_id)] = contact_email
        if len(self._identity_cache) > self._IDENTITY_CACHE_MAX:
            del self._identity_cache[next(iter(self._identity_cache))]
        logger.info("Identity linked: %s/%s -> %s", channel, sender_id, contact_email)

    @staticmethod
    def _build_contact(email: str, name: str | None = None) -> Contact:
        return Contact(
            id=uuid4(),
            name=name or email.split("@")[0],
            email=email,
            source="sensory_system",
        )

    async def close(self) -> None:
        """Dispose the database engine."""
        await self._engine.dispose()
