"""Rules-driven :class:`~brain_os.systems.task_orchestrator.TaskOrchestrator` on bus events."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from brain_os.config import get_settings
from brain_os.systems.data_event_bus import DataEvent, DataEventBus
from brain_os.systems.llm_budget import check_budget_allows, resolve_budget_bucket

logger = logging.getLogger(__name__)

_DEDUP_PREFIX = "event_task:dedup:"
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_.]+)\}")


def load_event_task_rules(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        logger.warning("Event task rules file missing: %s", p)
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Invalid event task rules (%s): %s", p, exc)
        return []
    rules = raw.get("rules") if isinstance(raw, dict) else raw
    if not isinstance(rules, list):
        return []
    return [r for r in rules if isinstance(r, dict)]


def _get_path(obj: dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _rule_matches(rule: dict[str, Any], event: DataEvent) -> bool:
    if not rule.get("enabled"):
        return False
    types = rule.get("event_types") or []
    if types and event.event_type.value not in types:
        return False
    match = rule.get("match")
    if not isinstance(match, dict) or not match:
        return True
    ctx = {
        "payload": event.payload,
        "entity_id": event.entity_id,
        "entity_type": event.entity_type,
        "event_type": event.event_type.value,
    }
    for key, expected in match.items():
        actual = _get_path(ctx, key) if "." in key else ctx.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _format_goal(template: str, event: DataEvent) -> str:
    ctx = {
        "payload": event.payload,
        "entity_id": event.entity_id,
        "entity_type": event.entity_type,
        "event_type": event.event_type.value,
    }

    def _repl(m: re.Match[str]) -> str:
        val = _get_path(ctx, m.group(1))
        return "" if val is None else str(val)

    return _PLACEHOLDER_RE.sub(_repl, template)


class EventTaskReactor:
    """Subscribe to :class:`DataEventBus` and spawn bounded tasks from JSON rules."""

    def __init__(
        self,
        event_bus: DataEventBus,
        task_orchestrator: Any,
        redis: Any,
        *,
        rules_path: str | Path,
        max_inflight: int = 2,
    ) -> None:
        self._bus = event_bus
        self._orchestrator = task_orchestrator
        self._redis = redis
        self._rules_path = Path(rules_path)
        self._rules = load_event_task_rules(self._rules_path)
        self._semaphore = asyncio.Semaphore(max(1, max_inflight))
        self._started = False
        self._settings = get_settings()

    async def start(self) -> None:
        if self._started:
            return
        self._bus.subscribe_all(self._on_event)
        self._started = True
        logger.info(
            "EventTaskReactor started (%d rules from %s)",
            len(self._rules),
            self._rules_path,
        )

    async def stop(self) -> None:
        self._started = False

    async def _on_event(self, event: DataEvent) -> None:
        if not self._started:
            return
        for rule in self._rules:
            if not _rule_matches(rule, event):
                continue
            asyncio.create_task(self._handle_rule(rule, event))

    async def _handle_rule(self, rule: dict[str, Any], event: DataEvent) -> None:
        rule_id = str(rule.get("id") or "unnamed")
        async with self._semaphore:
            if await self._in_cooldown(rule_id, event):
                return
            if not await self._budget_allows():
                logger.info("EventTaskReactor skipped %s — LLM budget", rule_id)
                return
            template = str(rule.get("goal_template") or "").strip()
            if not template:
                return
            goal = _format_goal(template, event)
            try:
                task_id = await self._orchestrator.create_task(
                    goal,
                    user_id="event_reactor",
                    trigger=f"event:{rule_id}",
                    event_meta={
                        "rule_id": rule_id,
                        "event_type": event.event_type.value,
                        "entity_id": event.entity_id,
                    },
                )
            except Exception:  # intentional — background loop, must not crash
                logger.exception("EventTaskReactor create_task failed for %s", rule_id)
                return
            await self._set_cooldown(rule_id, event, int(rule.get("cooldown_seconds") or 3600))
            logger.info(
                "EventTaskReactor spawned task %s for rule %s (%s)",
                task_id,
                rule_id,
                event.event_type.value,
            )

            async def _run() -> None:
                try:
                    await self._orchestrator.run_task(task_id)
                except Exception:  # intentional — background loop, must not crash
                    logger.exception("Event task %s failed (rule %s)", task_id, rule_id)

            asyncio.create_task(_run())

    async def _in_cooldown(self, rule_id: str, event: DataEvent) -> bool:
        if self._redis is None or not getattr(self._redis, "available", False):
            return False
        key = f"{_DEDUP_PREFIX}{rule_id}:{event.entity_id}"
        val = await self._redis.get(key)
        return bool(val)

    async def _set_cooldown(self, rule_id: str, event: DataEvent, seconds: int) -> None:
        if self._redis is None or not getattr(self._redis, "available", False):
            return
        key = f"{_DEDUP_PREFIX}{rule_id}:{event.entity_id}"
        await self._redis.set(key, "1", ttl_seconds=max(60, seconds))

    async def _budget_allows(self) -> bool:
        limit = int(self._settings.app.llm_monthly_token_budget or 0)
        if limit <= 0:
            return True
        if self._redis is None or not getattr(self._redis, "available", False):
            return True
        scope, bucket = resolve_budget_bucket("event_reactor", scope_mode="global")
        block = await check_budget_allows(
            self._redis,
            limit=limit,
            scope=scope,
            bucket=bucket,
        )
        return not block


def maybe_start_event_task_reactor(
    event_bus: DataEventBus,
    task_orchestrator: Any,
    redis: Any,
) -> EventTaskReactor | None:
    """Construct reactor when ``APP__EVENT_TASK_REACTOR_ENABLED`` is true."""
    settings = get_settings()
    if not settings.app.event_task_reactor_enabled:
        return None
    return EventTaskReactor(
        event_bus,
        task_orchestrator,
        redis,
        rules_path=settings.app.event_task_rules_path,
        max_inflight=settings.app.event_task_max_inflight,
    )
