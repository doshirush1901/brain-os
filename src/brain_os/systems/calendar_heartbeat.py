"""Emit ``CALENDAR_UPCOMING`` events for meetings in a lookahead window."""

from __future__ import annotations

import logging
from typing import Any

from brain_os.systems.data_event_bus import EventType, SourceStore
from brain_os.systems.event_emit import emit_data_event

logger = logging.getLogger(__name__)


async def emit_upcoming_calendar_events(
    event_bus: Any,
    google_calendar: Any,
    *,
    lookahead_hours: float = 24.0,
) -> int:
    """List upcoming events and emit one bus event per item. Returns count emitted."""
    if event_bus is None or google_calendar is None:
        return 0
    if not getattr(google_calendar, "available", False):
        try:
            await google_calendar.connect()
        except Exception:  # intentional — background loop, must not crash
            logger.exception("Calendar connect skipped for heartbeat")
    if not getattr(google_calendar, "available", False):
        return 0

    try:
        events = await google_calendar.list_upcoming_events(hours=lookahead_hours)
    except Exception:  # intentional — background loop, must not crash
        logger.exception("list_upcoming_events failed")
        return 0

    for ev in events:
        eid = str(ev.get("event_id") or ev.get("id") or "")
        if not eid:
            continue
        await emit_data_event(
            event_bus,
            event_type=EventType.CALENDAR_UPCOMING,
            entity_type="calendar_event",
            entity_id=eid,
            payload=ev,
            source_store=SourceStore.CRM,
        )
    return len(events)
