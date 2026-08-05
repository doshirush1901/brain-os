"""Recall→store reconsolidation context (biological memory rewrite on retrieval).

When an agent recalls Mem0 rows then stores an update, the new write inherits
``recalled_from`` metadata and an audit row links old→new ids.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

_recall_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "mem0_recall_context",
    default=None,
)


def set_recall_context(ids: list[str], query: str, user_id: str) -> None:
    """Bind the latest recall session for the current async task."""
    clean = [str(i) for i in ids if i]
    if not clean:
        return
    _recall_context.set(
        {
            "ids": clean,
            "query": (query or "")[:500],
            "user_id": user_id or "global",
            "at": datetime.now(UTC).isoformat(),
        }
    )


def get_recall_context() -> dict[str, Any]:
    ctx = _recall_context.get()
    return dict(ctx) if ctx else {}


def clear_recall_context() -> None:
    _recall_context.set(None)


def reconsolidation_enabled() -> bool:
    from brain_os.config import get_settings

    return bool(getattr(get_settings().app, "mem0_reconsolidation_enabled", True))
