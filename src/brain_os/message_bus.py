"""Async pub/sub message bus for inter-agent communication.

Every agent publishes and receives :class:`~brain_os.data.models.AgentMessage`
objects through the bus.  Messages can be directed (to a specific agent)
or broadcast (to all subscribers).  A bounded in-process log of recent
messages is kept for debugging (``AppConfig.message_bus_log_maxlen``).

When a :class:`~brain_os.systems.redis_cache.RedisCache` instance is attached
via :meth:`set_redis`, every published message is also persisted to a
Redis Stream (``ira:bus:messages``) for durability and cross-process replay.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from brain_os.config import get_settings
from brain_os.data.models import AgentMessage
from brain_os.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

MessageHandler = Callable[[AgentMessage], Awaitable[None]]

_BROADCAST = "__broadcast__"
_REDIS_STREAM = "ira:bus:messages"
_STREAM_MAXLEN = 5000


class MessageBus:
    """Async message bus using :class:`asyncio.Queue`.

    Optionally backed by a Redis Stream for message persistence.
    """

    def __init__(self, maxsize: int = 1000, *, log_maxlen: int | None = None) -> None:
        self._queue: asyncio.Queue[AgentMessage] = asyncio.Queue(maxsize=maxsize)
        self._handlers: dict[str, list[MessageHandler]] = defaultdict(list)
        cap = log_maxlen if log_maxlen is not None else get_settings().app.message_bus_log_maxlen
        self._log: deque[AgentMessage] = deque(maxlen=int(cap))
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._redis: Any | None = None

    def set_redis(self, redis_cache: Any) -> None:
        """Attach a RedisCache for message persistence."""
        self._redis = redis_cache
        logger.info("MessageBus: Redis persistence enabled")

    # ── subscription ─────────────────────────────────────────────────────

    def subscribe(self, agent_name: str, handler: MessageHandler) -> None:
        """Register *handler* to receive messages addressed to *agent_name*."""
        self._handlers[agent_name].append(handler)
        logger.debug("Subscribed handler for '%s'", agent_name)

    def subscribe_broadcast(self, handler: MessageHandler) -> None:
        """Register *handler* to receive all broadcast messages."""
        self._handlers[_BROADCAST].append(handler)

    # ── publishing ───────────────────────────────────────────────────────

    async def publish(self, message: AgentMessage) -> None:
        """Enqueue a message for delivery and persist to Redis if available."""
        await self._queue.put(message)
        logger.debug(
            "Published message from '%s' to '%s'",
            message.from_agent,
            message.to_agent,
        )
        await self._persist_to_redis(message)

    async def _persist_to_redis(self, message: AgentMessage) -> None:
        if self._redis is None or not self._redis.available:
            return
        try:
            entry = {
                "from": message.from_agent,
                "to": message.to_agent,
                "query": message.query[:2000],
                "ts": datetime.now(UTC).isoformat(),
            }
            await self._redis._client.xadd(
                _REDIS_STREAM,
                entry,
                maxlen=_STREAM_MAXLEN,
                approximate=True,
            )
        except Exception:
            logger.debug("Redis stream persist failed", exc_info=True)

    async def send(
        self,
        from_agent: str,
        to_agent: str,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Convenience: build and publish an :class:`AgentMessage`."""
        msg = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            query=query,
            context=context or {},
        )
        await self.publish(msg)

    async def broadcast(
        self,
        from_agent: str,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Publish a message to all broadcast subscribers."""
        await self.send(from_agent, _BROADCAST, query, context)

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the dispatch loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop())
        logger.info("MessageBus started")

    async def stop(self) -> None:
        """Drain remaining messages and stop the dispatch loop."""
        self._running = False
        if self._task is not None:
            await self._queue.put(None)  # type: ignore[arg-type]
            await self._task
            self._task = None
        logger.info("MessageBus stopped")

    # ── dispatch ─────────────────────────────────────────────────────────

    async def _dispatch_loop(self) -> None:
        while self._running:
            message = await self._queue.get()
            if message is None:
                break
            self._log.append(message)
            await self._dispatch(message)
            self._queue.task_done()

    async def _dispatch(self, message: AgentMessage) -> None:
        target = message.to_agent
        handlers = list(self._handlers.get(target, []))
        if target != _BROADCAST:
            handlers.extend(self._handlers.get(_BROADCAST, []))

        for handler in handlers:
            try:
                await handler(message)
            except (ToolExecutionError, Exception):
                logger.exception(
                    "Handler failed for message from '%s' to '%s'",
                    message.from_agent,
                    target,
                )

    # ── introspection ────────────────────────────────────────────────────

    @property
    def message_log(self) -> list[AgentMessage]:
        return list(self._log)

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()
