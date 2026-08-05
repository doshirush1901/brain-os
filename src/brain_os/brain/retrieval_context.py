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

# Per-task last search health (replaces instance last-writer-wins field for isolation).
last_search_health_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "last_search_health", default=None
)


async def with_retrieval_context(
    run_id: str | None,
    profile: str | None,
    fn: Callable[..., Awaitable[_T]],
    *args: Any,
    mem0_user_id: str | None = None,
    **kwargs: Any,
) -> _T:
    """Set retrieval context vars for the duration of ``await fn(*args, **kwargs)``."""
    tokens: list[tuple[contextvars.ContextVar[Any], contextvars.Token[Any]]] = []
    if run_id is not None:
        tokens.append((retrieval_run_id_var, retrieval_run_id_var.set(run_id)))
    if profile is not None:
        tokens.append((retrieval_profile_var, retrieval_profile_var.set(profile)))
    if mem0_user_id is not None and str(mem0_user_id).strip():
        tokens.append((mem0_user_id_var, mem0_user_id_var.set(str(mem0_user_id).strip())))
    if not tokens:
        return await fn(*args, **kwargs)
    try:
        return await fn(*args, **kwargs)
    finally:
        for var, tok in reversed(tokens):
            var.reset(tok)


def get_last_search_health() -> dict[str, Any]:
    """Return a copy of the current-task retrieval health dict."""
    h = last_search_health_var.get()
    return dict(h) if isinstance(h, dict) else {}


def set_last_search_health(health: dict[str, Any]) -> None:
    """Publish search health for the current task (and optional process fallback)."""
    payload = dict(health) if isinstance(health, dict) else {}
    last_search_health_var.set(payload)
