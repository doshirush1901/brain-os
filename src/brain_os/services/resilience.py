"""Shared resilience primitives: retry, backoff, and circuit breaker."""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 20.0
    jitter_ratio: float = 0.2


class CircuitBreaker:
    """Simple half-open circuit breaker for unstable upstream calls."""

    def __init__(self, *, threshold: int = 8, window_seconds: int = 180) -> None:
        self._threshold = threshold
        self._window_seconds = window_seconds
        self._failures: deque[float] = deque()
        self._opened_at: float | None = None

    def record_success(self) -> None:
        """Perform the record success operation."""
        self._failures.clear()
        self._opened_at = None

    def record_failure(self) -> None:
        """Perform the record failure operation."""
        now = time.time()
        self._failures.append(now)
        cutoff = now - self._window_seconds
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()
        if len(self._failures) >= self._threshold:
            self._opened_at = now

    def can_attempt(self) -> bool:
        """Perform the can attempt operation.

        Returns:
            bool: Result produced by this operation.
        """
        if self._opened_at is None:
            return True
        return (time.time() - self._opened_at) >= self._window_seconds

    def reset(self) -> None:
        """Clear failure state so the next attempt is allowed. Use after wallet reload or 'start Ira'."""
        self._failures.clear()
        self._opened_at = None

    def public_snapshot(self) -> dict[str, float | bool | int | str | None]:
        """Non-secret breaker state for deep-health / ops.

        Phase-2 hardening adds ``state`` (``closed`` / ``open`` / ``half_open``),
        ``threshold``, ``window_seconds``, and ``last_failure_age_s`` so the
        ``/api/deep-health`` payload tells operators not just *if* a breaker
        tripped but *why* and *how long ago*.
        """
        now = time.time()
        last_failure_age: float | None = None
        if self._failures:
            last_failure_age = max(0.0, now - float(self._failures[-1]))

        if self._opened_at is None:
            state = "closed"
        elif (now - float(self._opened_at)) >= float(self._window_seconds):
            # Past the cool-down window — next attempt would be allowed.
            state = "half_open"
        else:
            state = "open"

        return {
            "state": state,
            "can_attempt": bool(self.can_attempt()),
            "recent_failures": len(self._failures),
            "threshold": int(self._threshold),
            "window_seconds": int(self._window_seconds),
            "opened_at_unix": float(self._opened_at) if self._opened_at is not None else None,
            "last_failure_age_s": (
                round(float(last_failure_age), 2) if last_failure_age is not None else None
            ),
        }


def _backoff_delay(attempt: int, policy: RetryPolicy) -> float:
    base = min(policy.base_delay_seconds * (2 ** max(0, attempt - 1)), policy.max_delay_seconds)
    jitter = base * policy.jitter_ratio * random.random()
    return base + jitter


async def run_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[Exception], bool],
    circuit_breaker: CircuitBreaker | None = None,
) -> T:
    """Run async operation with retry and optional circuit breaker checks."""
    if circuit_breaker is not None and not circuit_breaker.can_attempt():
        raise RuntimeError("Circuit breaker open for operation")

    last_exc: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            result = await operation()
            if circuit_breaker is not None:
                circuit_breaker.record_success()
            return result
        except (
            Exception
        ) as exc:  # intentional broad catch — retry loop; re-raises on terminal failure
            last_exc = exc
            if circuit_breaker is not None:
                circuit_breaker.record_failure()
            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise
            await asyncio.sleep(_backoff_delay(attempt, policy))
    raise last_exc if last_exc else RuntimeError("retry failed without error")
