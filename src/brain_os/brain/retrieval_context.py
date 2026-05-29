"""Request-scoped retrieval identity (e.g. Mem0 user_id).

The unified retriever is a shared singleton; :class:`contextvars.ContextVar`
lets each concurrent pipeline run search Mem0 under the right resolved contact
without threading ``user_id`` through every :meth:`~ira.brain.retriever.UnifiedRetriever.search` call.
"""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

_T = TypeVar("_T")

# Resolved pipeline contact (synthetic e.g. api_user@unknown or a real email).
mem0_user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mem0_user_id", default="global"
)

# Optional ``pipeline_run_id`` for retrieval tracing / correlation (best-effort).
retrieval_run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "retrieval_run_id", default=None
)

# Named SLO profile for backend timeouts (see ``retrieval_slo``).
retrieval_profile_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "retrieval_profile", default="default"
)


async def with_retrieval_context(
    run_id: str | None,
    profile: str | None,
    fn: Callable[..., Awaitable[_T]],
    *args: Any,
    **kwargs: Any,
) -> _T:
    """Set retrieval context vars for the duration of ``await fn(*args, **kwargs)``."""
    tokens: list[tuple[contextvars.ContextVar[Any], contextvars.Token[Any]]] = []
    if run_id is not None:
        tokens.append((retrieval_run_id_var, retrieval_run_id_var.set(run_id)))
    if profile is not None:
        tokens.append((retrieval_profile_var, retrieval_profile_var.set(profile)))
    if not tokens:
        return await fn(*args, **kwargs)
    try:
        return await fn(*args, **kwargs)
    finally:
        for var, tok in reversed(tokens):
            var.reset(tok)
