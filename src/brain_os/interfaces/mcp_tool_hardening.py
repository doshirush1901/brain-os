"""MCP tool retry wrapper with Community/Pro license gate (Brain OS export)."""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar, overload

from brain_os.services.tool_runner import is_retryable_transient, run_tool

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


@overload
def hardened_mcp_tool(func: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R | str]]: ...


@overload
def hardened_mcp_tool() -> Callable[
    [Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R | str]]
]: ...


def hardened_mcp_tool(
    func: Callable[_P, Awaitable[_R]] | None = None,
) -> (
    Callable[_P, Awaitable[_R | str]]
    | Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R | str]]]
):
    """Wrap an MCP tool with license tier check, retries, and structured error strings."""

    def _decorate(fn: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R | str]]:
        tool_name = fn.__name__

        @functools.wraps(fn)
        async def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R | str:
            from brain_os.licensing.caps import mcp_tool_allowed, mcp_upgrade_hint

            if not mcp_tool_allowed(tool_name):
                return mcp_upgrade_hint(tool_name)

            from brain_os.config import get_settings

            app = get_settings().app
            retries = max(0, int(app.react_tool_transient_retries))
            timeout_s = float(getattr(app, "mcp_tool_timeout_seconds", 60.0))

            async def _op() -> _R:
                return await fn(*args, **kwargs)

            try:
                result = await run_tool(
                    f"mcp.{tool_name}",
                    _op,
                    retries=retries,
                    backoff_seconds=0.3,
                    timeout_seconds=timeout_s,
                    is_retryable=is_retryable_transient,
                )
            except asyncio.CancelledError:
                logger.warning("MCP tool %s cancelled by client", tool_name)
                return f"Error ({tool_name}): cancelled by client"
            except BaseException as exc:
                logger.exception("MCP tool %s hit fatal base exception: %s", tool_name, exc)
                return f"Error ({tool_name}): fatal {type(exc).__name__}: {exc}"
            if result.ok:
                return result.value
            detail = result.error or "unknown"
            return f"Error ({tool_name}): {detail} [attempts={result.attempts}]"

        return _wrapper

    if func is not None:
        return _decorate(func)
    return _decorate


__all__ = ["hardened_mcp_tool"]
