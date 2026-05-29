"""Phase-2 hardening — opt-in retry / fallback helper for agent tools."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx

from brain_os.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})

T = TypeVar("T")


class ToolFailure(Exception):
    def __init__(self, message: str, *, result: ToolResult | None = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass(slots=True)
class ToolResult:
    ok: bool
    value: Any = None
    attempts: int = 0
    duration_ms: float = 0.0
    fallback_used: bool = False
    error: str | None = None
    name: str = ""
    attempt_errors: list[str] = field(default_factory=list)
    last_exception: Exception | None = None


def _classify_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc!s}"


def is_retryable_transient(exc: Exception) -> bool:
    if isinstance(exc, ToolExecutionError):
        return False
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return False
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response is not None and exc.response.status_code in _TRANSIENT_HTTP_STATUS
    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, RuntimeError) and "circuit breaker" in str(exc).lower():
        return False
    msg = str(exc).lower()
    return any(
        n in msg
        for n in (
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "econnreset",
            "rate limit",
            "503",
            "502",
            "429",
        )
    )


def _is_retryable_default(exc: Exception) -> bool:
    return is_retryable_transient(exc)


async def _await_operation(
    operation: Callable[[], Awaitable[T]],
    *,
    timeout_seconds: float | None,
) -> T:
    if timeout_seconds is not None and timeout_seconds > 0:
        return await asyncio.wait_for(operation(), timeout=timeout_seconds)
    return await operation()


async def run_tool(
    name: str,
    operation: Callable[[], Awaitable[T]],
    *,
    retries: int = 2,
    backoff_seconds: float = 0.5,
    backoff_multiplier: float = 2.0,
    backoff_cap_seconds: float = 5.0,
    timeout_seconds: float | None = None,
    fallbacks: tuple[Callable[[], Awaitable[T]], ...] = (),
    is_retryable: Callable[[Exception], bool] | None = None,
    raise_on_failure: bool = False,
) -> ToolResult:
    if retries < 0:
        retries = 0
    if backoff_seconds < 0:
        backoff_seconds = 0
    if backoff_multiplier < 1.0:
        backoff_multiplier = 1.0
    predicate = is_retryable or _is_retryable_default

    started_at = time.monotonic()
    result = ToolResult(ok=False, name=name)

    async def _try(callable_: Callable[[], Awaitable[T]], *, is_fallback: bool) -> bool:
        last_exc: Exception | None = None
        delay = float(backoff_seconds)
        max_attempts = 1 + retries
        for attempt in range(1, max_attempts + 1):
            result.attempts += 1
            try:
                value = await _await_operation(callable_, timeout_seconds=timeout_seconds)
            except Exception as exc:
                last_exc = exc
                err_str = _classify_error(exc)
                result.attempt_errors.append(err_str)
                if attempt >= max_attempts or not predicate(exc):
                    break
                if delay > 0:
                    await asyncio.sleep(min(delay, backoff_cap_seconds))
                delay *= backoff_multiplier
                continue
            result.ok = True
            result.value = value
            result.fallback_used = is_fallback
            result.error = None
            result.last_exception = None
            return True
        if last_exc is not None:
            result.error = _classify_error(last_exc)
            result.last_exception = last_exc
        return False

    succeeded = await _try(operation, is_fallback=False)
    if not succeeded:
        for fallback in fallbacks:
            if await _try(fallback, is_fallback=True):
                succeeded = True
                break

    result.duration_ms = round((time.monotonic() - started_at) * 1000.0, 2)
    if result.ok:
        logger.info(
            "IRA_TOOL_RUN | name=%s ok=true attempts=%d duration_ms=%.0f fallback=%s",
            name,
            result.attempts,
            result.duration_ms,
            result.fallback_used,
        )
    else:
        logger.warning(
            "IRA_TOOL_RUN | name=%s ok=false attempts=%d duration_ms=%.0f error=%s",
            name,
            result.attempts,
            result.duration_ms,
            result.error,
        )
        if raise_on_failure:
            raise ToolFailure(
                f"tool '{name}' failed after {result.attempts} attempts: {result.error}",
                result=result,
            )
    return result


async def mcp_run_tool(
    name: str,
    operation: Callable[[], Awaitable[Any]],
    *,
    retries: int = 1,
    backoff_seconds: float = 0.3,
    timeout_seconds: float | None = None,
) -> str:
    result = await run_tool(
        f"mcp.{name}",
        operation,
        retries=retries,
        backoff_seconds=backoff_seconds,
        timeout_seconds=timeout_seconds,
        is_retryable=is_retryable_transient,
    )
    if result.ok:
        return str(result.value)
    return f"Error ({name}): {result.error}  [attempts={result.attempts}]"


__all__ = [
    "ToolFailure",
    "ToolResult",
    "is_retryable_transient",
    "mcp_run_tool",
    "run_tool",
]
