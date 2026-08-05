"""Mirror-upsert helpers for Qdrant cloud sync.

Extracted from ``qdrant_manager`` (L0.1 file-size peel). Re-exported from
``brain_os.brain.qdrant_manager`` so callers and tests keep the old names.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def vector_for_mirror_upsert(vec: Any) -> Any | None:
    """Preserve named dense/sparse vectors when copying points to a mirror.

    Hybrid collections store ``{"dense": [...], "sparse": SparseVector(...)}``.
    Older sync paths collapsed dicts to a single unnamed vector and dropped
    sparse — that breaks cloud hybrid search. Pass named dicts through.
    """
    if vec is None:
        return None
    if isinstance(vec, dict):
        if not vec:
            return None
        # Named hybrid (or any named vector map): keep as-is.
        if any(k != "" for k in vec):
            return vec
        # Legacy unnamed-in-dict form: ``{"": [...]}``.
        return vec.get("")
    return vec


async def upsert_points_with_backoff(
    client: Any,
    *,
    collection_name: str,
    points: list[Any],
    http_io_errors: tuple[type[BaseException], ...],
    max_attempts: int = 5,
) -> None:
    """Upsert *points* with exponential backoff on transient HTTP/IO errors.

    India→sa-east-1 often WriteTimeouts; upserts are idempotent so retry is safe.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            await client.upsert(collection_name=collection_name, points=points)
            return
        except http_io_errors as exc:
            last_exc = exc
            wait_s = min(2**attempt, 30)
            logger.warning(
                "Cloud upsert retry %d/%d after %s (sleep %ss, batch=%d)",
                attempt,
                max_attempts,
                type(exc).__name__,
                wait_s,
                len(points),
            )
            await asyncio.sleep(wait_s)
    if last_exc is not None:
        raise last_exc
