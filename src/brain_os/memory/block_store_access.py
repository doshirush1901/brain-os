"""Resolve the live memory block store without importing ``brain_os.interfaces``."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_live_store_lookup: Callable[[], Any | None] | None = None


def register_live_block_store_lookup(lookup: Callable[[], Any | None]) -> None:
    """Register a callable that returns the in-process block store, if any."""
    global _live_store_lookup
    _live_store_lookup = lookup


def clear_live_block_store_lookup() -> None:
    """Reset lookup (tests)."""
    global _live_store_lookup
    _live_store_lookup = None


async def resolve_block_store() -> tuple[Any, bool]:
    """Return ``(store, close_after)`` — open SQLite fallback when no live store."""
    if _live_store_lookup is not None:
        try:
            store = _live_store_lookup()
        except (OSError, RuntimeError, TypeError, ValueError):
            logger.debug("live block store lookup failed", exc_info=True)
            store = None
        if store is not None:
            return store, False

    from brain_os.memory.blocks import MemoryBlockStore

    store = MemoryBlockStore()
    await store.initialize()
    return store, True
