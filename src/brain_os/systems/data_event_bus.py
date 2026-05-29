"""Data Event Bus -- typed event system for cross-store synchronisation.

Emits :class:`DataEvent` objects when any data store (CRM, Neo4j, Qdrant)
is written to.  Handlers subscribe by event type and propagate changes to
other stores.  The bus is fire-and-forget: emission never blocks the
writer, and handler failures are logged but do not propagate.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    CONTACT_CREATED = "contact_created"
    CONTACT_UPDATED = "contact_updated"
    CONTACT_CLASSIFIED = "contact_classified"
    COMPANY_CREATED = "company_created"
    DEAL_CREATED = "deal_created"
    DEAL_UPDATED = "deal_updated"
    INTERACTION_LOGGED = "interaction_logged"
    ENTITY_ADDED = "entity_added"
    RELATIONSHIP_DISCOVERED = "relationship_discovered"
    CHUNK_UPSERTED = "chunk_upserted"
    KNOWLEDGE_CORRECTED = "knowledge_corrected"
    CALENDAR_UPCOMING = "calendar_upcoming"
    HEARTBEAT_TICK = "heartbeat_tick"
    EMAIL_RECEIVED = "email_received"
    EMAIL_CLASSIFIED = "email_classified"
    SCHEDULING_REQUEST_CREATED = "scheduling_request_created"
    CALENDAR_BOOKING_CREATED = "calendar_booking_created"


class SourceStore(str, Enum):
    CRM = "crm"
    NEO4J = "neo4j"
    QDRANT = "qdrant"
    POPULATOR = "populator"
    EMAIL = "email"


@dataclass
class DataEvent:
    """A single data-mutation event emitted by a store writer."""

    event_type: EventType
    entity_type: str
    entity_id: str
    payload: dict[str, Any]
    source_store: SourceStore
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


EventHandler = Callable[[DataEvent], Awaitable[None]]


class DataEventBus:
    """Async event bus for data-store change propagation."""

    def __init__(self, maxsize: int = 2000) -> None:
        self._queue: asyncio.Queue[DataEvent | None] = asyncio.Queue(maxsize=maxsize)
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._running = False
        self._task: asyncio.Task[None] | None = None

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        self._handlers.setdefault(event_type, []).append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Register a handler that receives every event (e.g. the ledger)."""
        self._global_handlers.append(handler)

    async def emit(self, event: DataEvent) -> None:
        """Enqueue an event for async dispatch. Never blocks the caller."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "DataEventBus queue full — dropping %s for %s",
                event.event_type.value,
                event.entity_id,
            )

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop())
        logger.info("DataEventBus started")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._queue.put(None)
            await self._task
            self._task = None
        logger.info("DataEventBus stopped")

    async def _dispatch_loop(self) -> None:
        while self._running:
            event = await self._queue.get()
            if event is None:
                break
            await self._dispatch(event)
            self._queue.task_done()

    async def _dispatch(self, event: DataEvent) -> None:
        handlers = list(self._handlers.get(event.event_type, []))
        handlers.extend(self._global_handlers)

        for handler in handlers:
            try:
                await handler(event)
            except Exception:  # intentional — background loop, must not crash
                logger.exception(
                    "DataEventBus handler failed for %s (%s)",
                    event.event_type.value,
                    event.entity_id,
                )

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()
