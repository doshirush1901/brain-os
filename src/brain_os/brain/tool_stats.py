"""In-memory tool call success/failure tracking for observability.

Used by the dashboard to show per-agent (and per-tool) success rates.
No persistence; resets on process restart.

Optional :class:`~brain_os.brain.tool_invocation_store.ToolInvocationStore` persists
each outcome when ``APP__GEPA_PERSIST_TOOL_INVOCATIONS`` is enabled.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from typing import Any

_DELEGATION_RING_MAX = 200

logger = logging.getLogger(__name__)


class ToolStatsTracker:
    """Tracks tool call outcomes per agent and per tool for success-rate reporting."""

    def __init__(self, *, invocation_store: Any | None = None) -> None:
        # (agent_name, tool_name) -> {"successes": int, "failures": int}
        self._counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"successes": 0, "failures": 0}
        )
        self._lock = asyncio.Lock()
        self._delegations: deque[dict[str, Any]] = deque(maxlen=_DELEGATION_RING_MAX)
        self._route_events: deque[dict[str, Any]] = deque(maxlen=_DELEGATION_RING_MAX)
        self._invocation_store = invocation_store

    async def record_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        success: bool,
        *,
        error_code: str | None = None,
        run_id: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Record one tool invocation outcome."""
        async with self._lock:
            key = (agent_name, tool_name)
            if success:
                self._counts[key]["successes"] += 1
            else:
                self._counts[key]["failures"] += 1
        if self._invocation_store is not None:
            try:
                await self._invocation_store.record(
                    agent=agent_name,
                    tool=tool_name,
                    success=success,
                    error_code=error_code,
                    run_id=run_id,
                    duration_ms=duration_ms,
                )
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                logger.debug("ToolInvocationStore.record failed", exc_info=True)

    async def record_delegation(
        self,
        from_agent: str,
        to_agent: str,
        duration_ms: int,
        *,
        ok: bool,
        run_id: str = "",
    ) -> None:
        """Append one ``ask_agent`` hop for /api/metrics (ring buffer, in-memory only)."""
        async with self._lock:
            self._delegations.append(
                {
                    "from": from_agent,
                    "to": to_agent,
                    "duration_ms": max(0, int(duration_ms)),
                    "ok": ok,
                    "run_id": (run_id or "").strip(),
                }
            )

    async def get_recent_delegations(self, run_id: str = "") -> list[dict[str, Any]]:
        """Recent delegation events oldest-to-newest (capped)."""
        async with self._lock:
            rid = (run_id or "").strip()
            if not rid:
                return list(self._delegations)
            return [d for d in self._delegations if str(d.get("run_id") or "") == rid]

    async def record_route_event(
        self,
        *,
        source: str,
        required: list[str],
        optional: list[str],
        executed: list[str],
        skipped_optional: list[str],
        required_tools: list[str],
        missing_tools: list[str],
        executed_optional: list[str] | None = None,
    ) -> None:
        """Append route-plan execution telemetry (ring buffer)."""
        async with self._lock:
            self._route_events.append(
                {
                    "source": source,
                    "required": list(required),
                    "optional": list(optional),
                    "executed": list(executed),
                    "executed_optional": list(executed_optional or []),
                    "skipped_optional": list(skipped_optional),
                    "required_tools": list(required_tools),
                    "missing_tools": list(missing_tools),
                }
            )

    async def get_recent_route_events(self) -> list[dict[str, Any]]:
        async with self._lock:
            return list(self._route_events)

    async def get_tool_success_rates(
        self,
        *,
        by_tool: bool = True,
    ) -> list[dict[str, Any]]:
        """Return success rate stats.

        If by_tool is True, each row is (agent, tool) with total, successes, failures, rate.
        If by_tool is False, each row is agent-only (aggregated across tools).
        """
        async with self._lock:
            if by_tool:
                rows: list[dict[str, Any]] = []
                for (agent_name, tool_name), counts in self._counts.items():
                    total = counts["successes"] + counts["failures"]
                    if total == 0:
                        continue
                    rows.append(
                        {
                            "agent": agent_name,
                            "tool": tool_name,
                            "total": total,
                            "successes": counts["successes"],
                            "failures": counts["failures"],
                            "rate": round(counts["successes"] / total, 2),
                        }
                    )
                return sorted(rows, key=lambda r: (r["agent"], r["tool"]))
            # Aggregate by agent
            agent_totals: dict[str, dict[str, int | float]] = defaultdict(
                lambda: {"total": 0, "successes": 0, "failures": 0}
            )
            for (agent_name, _), counts in self._counts.items():
                agent_totals[agent_name]["total"] += counts["successes"] + counts["failures"]
                agent_totals[agent_name]["successes"] += counts["successes"]
                agent_totals[agent_name]["failures"] += counts["failures"]
            rows = []
            for agent_name, data in agent_totals.items():
                total = data["total"]
                if total == 0:
                    continue
                rows.append(
                    {
                        "agent": agent_name,
                        "tool": "(all)",
                        "total": total,
                        "successes": data["successes"],
                        "failures": data["failures"],
                        "rate": round(data["successes"] / total, 2),
                    }
                )
            return sorted(rows, key=lambda r: r["agent"])
