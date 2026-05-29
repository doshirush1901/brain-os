"""Apply corrections to contact memory blocks (shared by MCP, CLI, feedback)."""

from __future__ import annotations

import logging

from brain_os.memory.block_store_access import resolve_block_store
from brain_os.memory.block_sync import apply_corrections_to_blocks, scope_from_correction_text

logger = logging.getLogger(__name__)


async def apply_correction_to_memory_blocks(
    *,
    entity: str,
    correct_value: str,
    correction_id: int | str | None = None,
    category: str = "GENERAL",
    source: str = "correction",
) -> int:
    """Update pinned blocks when entity resolves to a contact email."""
    scope = scope_from_correction_text(entity, source)
    if not scope or not correct_value.strip():
        return 0

    store, close_after = await resolve_block_store()
    try:
        return await apply_corrections_to_blocks(
            store,
            [
                {
                    "id": correction_id,
                    "entity": entity,
                    "source": source,
                    "new_value": correct_value,
                    "category": category,
                }
            ],
        )
    except Exception:
        logger.debug("Memory block correction apply failed for %s", scope, exc_info=True)
        return 0
    finally:
        if close_after and store is not None:
            await store.close()
