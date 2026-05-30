"""Fire-and-forget helpers for :class:`~brain_os.systems.data_event_bus.DataEventBus`."""

from __future__ import annotations

import logging
from typing import Any

from brain_os.systems.data_event_bus import DataEvent, DataEventBus, EventType, SourceStore

logger = logging.getLogger(__name__)


async def emit_data_event(
    bus: DataEventBus | None,
    *,
    event_type: EventType,
    entity_type: str,
    entity_id: str,
    payload: dict[str, Any],
    source_store: SourceStore,
) -> None:
    """Enqueue a typed event; no-op when the bus is unset."""
    if bus is None:
        return
    try:
        await bus.emit(
            DataEvent(
                event_type=event_type,
                entity_type=entity_type,
                entity_id=entity_id,
                payload=payload,
                source_store=source_store,
            )
        )
    except Exception:  # intentional — background loop, must not crash
        logger.exception(
            "emit_data_event failed for %s/%s",
            event_type.value,
            entity_id,
        )
