"""Metis — Stability Monitor agent.

Tracks response quality across Cursor sessions and determines when
the system has reached a stable mode.  Can auto-adjust max_rounds
(react_max_iterations) when quality is below threshold.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.agents.engineering_tools import register_metis_engineering_tools
from brain_os.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("metis_system")
_STABILITY_THRESHOLD = 75
_CONSECUTIVE_REQUIRED = 10
_WINDOW_SIZE = 20
_REPORT_PROBABILITY = 0.1

_DEFAULT_STABILITY_STATE: dict[str, Any] = {
    "scores": [],
    "consecutive_above": 0,
    "stable": False,
    "max_rounds": 8,
    "last_updated": 0,
}


def stability_scores_path() -> Path:
    """Resolved path for Metis rolling quality scores (under ``BRAIN_DATA_DIR`` / ``./data``)."""
    from brain_os.systems.data_dir_lock import get_data_dir

    return get_data_dir() / "brain" / "stability_scores.json"


def metis_stability_public_snapshot() -> dict[str, Any]:
    """Read-only Metis state for health endpoints (no writes)."""
    path = stability_scores_path()
    base_empty: dict[str, Any] = {
        "persisted": False,
        "readable": True,
        "scores_in_window": 0,
        "rolling_avg": None,
        "stable": False,
        "consecutive_above_threshold": 0,
        "max_rounds_recorded": None,
        "last_updated_unix": None,
    }
    if not path.exists():
        return {**base_empty}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, TypeError, ValueError):
        return {
            "persisted": True,
            "readable": False,
            "scores_in_window": 0,
            "rolling_avg": None,
            "stable": False,
            "consecutive_above_threshold": 0,
            "max_rounds_recorded": None,
            "last_updated_unix": None,
        }
    scores = state.get("scores") or []
    avg = sum(s["score"] for s in scores) / len(scores) if scores else None
    return {
        "persisted": True,
        "readable": True,
        "scores_in_window": len(scores),
        "rolling_avg": round(avg, 1) if avg is not None else None,
        "stable": bool(state.get("stable", False)),
        "consecutive_above_threshold": int(state.get("consecutive_above", 0)),
        "max_rounds_recorded": state.get("max_rounds"),
        "last_updated_unix": state.get("last_updated"),
    }


class Metis(BaseAgent):
    name = "metis"
    role = "Stability Monitor"
    description = "Tracks response quality and determines when Brain OS is stable in Cursor"
    knowledge_categories = []
    timeout = 15

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="score_response",
                description="Score a response on quality (0-100).",
                parameters={
                    "agents_succeeded": "Number of agents that succeeded (no timeout/error)",
                    "agents_total": "Total number of agents invoked",
                    "has_sources": "Whether sources were cited (true/false)",
                    "response_length": "Length of the response in characters",
                    "had_warnings": "Whether provenance/DLP warnings were added (true/false)",
                },
                handler=self._tool_score_response,
            )
        )

        self.register_tool(
            AgentTool(
                name="get_stability_status",
                description="Check current stability score, max_rounds, and stable mode status.",
                parameters={},
                handler=self._tool_get_status,
            )
        )
        register_metis_engineering_tools(self)

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)

    def _read_state_file(self) -> dict[str, Any]:
        """Sync disk read for use via ``asyncio.to_thread`` only."""
        sp = stability_scores_path()
        if sp.exists():
            try:
                return json.loads(sp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                pass
        return dict(_DEFAULT_STABILITY_STATE)

    def _write_state_file(self, state: dict[str, Any]) -> None:
        """Sync disk write for use via ``asyncio.to_thread`` only."""
        sp = stability_scores_path()
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(state, indent=2), encoding="utf-8")

    async def _load_state(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._read_state_file)

    async def _save_state(self, state: dict[str, Any]) -> None:
        await asyncio.to_thread(self._write_state_file, state)

    def _compute_score(
        self,
        agents_succeeded: int,
        agents_total: int,
        has_sources: bool,
        response_length: int,
        had_warnings: bool,
    ) -> int:
        score = 0
        if agents_total > 0 and agents_succeeded == agents_total:
            score += 30
        elif agents_total > 0:
            score += int(30 * (agents_succeeded / agents_total))
        if has_sources:
            score += 20
        if response_length > 200:
            score += 30
        elif response_length > 50:
            score += 15
        if not had_warnings:
            score += 10
        score += 10  # no user correction (assumed at scoring time)
        return min(100, score)

    async def score_and_track(
        self,
        agents_used: list[str],
        raw_response: str,
    ) -> dict[str, Any]:
        """Score a response and update stability tracking. Returns status dict."""
        agents_total = len(agents_used)
        agents_succeeded = sum(
            1 for a in agents_used if a not in ("timeout",) and "timed out" not in raw_response
        )
        has_sources = any(
            marker in raw_response
            for marker in (
                "source:",
                "Source:",
                "Sources:",
                "**Sources**",
                "Confidence:",
                "thread_id",
                "https://",
                "http://",
            )
        )
        had_warnings = any(
            marker in raw_response
            for marker in ("Provenance note:", "Content note:", "could not be traced")
        )

        score = self._compute_score(
            agents_succeeded,
            agents_total,
            has_sources,
            len(raw_response),
            had_warnings,
        )

        state = await self._load_state()
        state["scores"].append({"score": score, "timestamp": time.time()})
        state["scores"] = state["scores"][-_WINDOW_SIZE:]
        state["last_updated"] = time.time()

        avg = sum(s["score"] for s in state["scores"]) / len(state["scores"])

        if avg >= _STABILITY_THRESHOLD:
            state["consecutive_above"] += 1
        else:
            state["consecutive_above"] = 0

        should_announce_stable = (
            not state["stable"] and state["consecutive_above"] >= _CONSECUTIVE_REQUIRED
        )

        should_report = random.random() < _REPORT_PROBABILITY

        from brain_os.config import get_settings

        current_max = get_settings().app.react_max_iterations
        state["max_rounds"] = current_max

        await self._save_state(state)

        return {
            "score": score,
            "rolling_avg": round(avg, 1),
            "consecutive_above": state["consecutive_above"],
            "stable": state["stable"],
            "should_announce_stable": should_announce_stable,
            "should_report": should_report,
            "max_rounds": current_max,
        }

    async def confirm_stable(self) -> None:
        """User confirmed stability. Mark as stable."""
        state = await self._load_state()
        state["stable"] = True
        await self._save_state(state)
        logger.info("METIS | Stability confirmed by user at max_rounds=%d", state["max_rounds"])

    async def reject_stable(self) -> None:
        """User rejected stability. Increase max_rounds by 20%."""
        state = await self._load_state()
        state["consecutive_above"] = 0

        from brain_os.config import get_settings

        current = get_settings().app.react_max_iterations
        new_max = max(current + 1, int(current * 1.2))
        state["max_rounds"] = new_max
        await self._save_state(state)
        logger.info("METIS | Stability rejected. max_rounds %d → %d", current, new_max)

    async def _tool_score_response(
        self,
        agents_succeeded: str = "0",
        agents_total: str = "0",
        has_sources: str = "false",
        response_length: str = "0",
        had_warnings: str = "false",
    ) -> str:
        score = self._compute_score(
            int(agents_succeeded),
            int(agents_total),
            has_sources.lower() == "true",
            int(response_length),
            had_warnings.lower() == "true",
        )
        return f"Response quality score: {score}/100"

    async def _tool_get_status(self) -> str:
        state = await self._load_state()
        scores = state.get("scores", [])
        avg = sum(s["score"] for s in scores) / len(scores) if scores else 0
        return (
            f"Stability status:\n"
            f"- Rolling average: {avg:.1f}/100 (last {len(scores)} requests)\n"
            f"- Consecutive above {_STABILITY_THRESHOLD}: {state.get('consecutive_above', 0)}\n"
            f"- Stable: {state.get('stable', False)}\n"
            f"- Max rounds: {state.get('max_rounds', 8)}"
        )
