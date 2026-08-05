"""Gamified agent performance tracking.

Each agent in the Pantheon accumulates a power-level score based on
successful task completions, failures, and Nemesis training sessions.
Scores map to named tiers that surface in dashboards and leaderboards.

Persistence is via ``{BRAIN_DATA_DIR or ./data}/brain/power_levels.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

import aiofiles

logger = logging.getLogger(__name__)


def default_power_levels_path() -> Path:
    """Default persistence path under ``BRAIN_DATA_DIR`` / ``./data``."""
    from brain_os.systems.data_dir_lock import get_data_dir

    return get_data_dir() / "brain" / "power_levels.json"


class Tier(str, Enum):
    MORTAL = "MORTAL"
    WARRIOR = "WARRIOR"
    HERO = "HERO"
    LEGEND = "LEGEND"


_TIER_THRESHOLDS: list[tuple[int, Tier]] = [
    (601, Tier.LEGEND),
    (301, Tier.HERO),
    (101, Tier.WARRIOR),
    (0, Tier.MORTAL),
]

_TRAINING_MAX_SCORE = 10
_TRAINING_MAX_BOOST = 15
_DEFAULT_TRUST = 0.8
_TRUST_DECREMENT = 0.1
_TRUST_MIN = 0.0
_TRUST_MAX = 1.0

DEFAULT_TRUST = _DEFAULT_TRUST


def _default_pantheon_agent_names() -> set[str]:
    """Best-effort Pantheon roster used for trust graph bootstrapping."""
    try:
        from brain_os.pantheon import _AGENT_CLASSES

        return {str(cls.name).lower() for cls in _AGENT_CLASSES if getattr(cls, "name", None)}
    except (ImportError, AttributeError, TypeError):
        logger.debug("Could not load pantheon registry for power levels", exc_info=True)
        return set()


def _clamp_trust(value: float) -> float:
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


def _tier_for_score(score: int) -> Tier:
    for threshold, tier in _TIER_THRESHOLDS:
        if score >= threshold:
            return tier
    return Tier.MORTAL


class PowerLevelTracker:
    """Track and persist per-agent power-level scores."""

    def __init__(
        self, data_path: Path | None = None, pantheon_agents: Iterable[str] | None = None
    ) -> None:
        self._path = data_path if data_path is not None else default_power_levels_path()
        self._agents: dict[str, dict[str, Any]] = {}
        if pantheon_agents is None:
            self._pantheon_agents = _default_pantheon_agent_names()
        else:
            self._pantheon_agents = {
                str(name).strip().lower() for name in pantheon_agents if str(name).strip()
            }
        self._lock = asyncio.Lock()

    async def _load(self) -> None:
        if not self._path.exists():
            self._agents = {}
        else:
            try:
                async with aiofiles.open(self._path, encoding="utf-8") as f:
                    raw = await f.read()
                data = json.loads(raw)
                self._agents = data.get("agents", {})
                for entry in self._agents.values():
                    if "trust_in" not in entry:
                        entry["trust_in"] = {}
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to read power levels from %s", self._path)
                self._agents = {}
        if self._bootstrap_pantheon_defaults():
            await self._save()
        logger.info("PowerLevels loaded: %d agents", len(self._agents))

    async def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"agents": self._agents}, indent=2, ensure_ascii=False)
        async with aiofiles.open(self._path, mode="w", encoding="utf-8") as f:
            await f.write(payload)

    def _ensure_agent(self, agent_name: str) -> dict[str, Any]:
        agent_name = str(agent_name).strip().lower()
        if agent_name not in self._agents:
            self._agents[agent_name] = {
                "score": 0,
                "successes": 0,
                "failures": 0,
                "trust_in": {},
            }
        entry = self._agents[agent_name]
        if "trust_in" not in entry:
            entry["trust_in"] = {}
        return entry

    def _ensure_trust_defaults(self, agent_name: str, entry: dict[str, Any]) -> bool:
        trust_in = entry.setdefault("trust_in", {})
        changed = False
        if agent_name in trust_in:
            del trust_in[agent_name]
            changed = True

        if agent_name in self._pantheon_agents:
            for peer in sorted(self._pantheon_agents):
                if peer == agent_name:
                    continue
                if peer not in trust_in:
                    trust_in[peer] = _DEFAULT_TRUST
                    changed = True

        for peer, current in list(trust_in.items()):
            try:
                value = _clamp_trust(float(current))
            except (TypeError, ValueError):
                value = _DEFAULT_TRUST
            if trust_in.get(peer) != value:
                trust_in[peer] = value
                changed = True
        return changed

    def _bootstrap_pantheon_defaults(self) -> bool:
        changed = False
        for agent_name in sorted(self._pantheon_agents):
            if agent_name not in self._agents:
                self._agents[agent_name] = {
                    "score": 0,
                    "successes": 0,
                    "failures": 0,
                    "trust_in": {},
                }
                changed = True
            entry = self._agents[agent_name]
            if "trust_in" not in entry:
                entry["trust_in"] = {}
                changed = True
            if self._ensure_trust_defaults(agent_name, entry):
                changed = True
        return changed

    def _normalize_collaboration_agents(self, agents: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for name in agents:
            agent = str(name).strip().lower()
            if not agent or agent in seen:
                continue
            if self._pantheon_agents and agent not in self._pantheon_agents:
                continue
            seen.add(agent)
            ordered.append(agent)
        return ordered

    # ── public API ────────────────────────────────────────────────────────

    async def record_success(self, agent_name: str, boost: int = 10) -> None:
        """Increase *agent_name*'s score after a successful task."""
        async with self._lock:
            self._bootstrap_pantheon_defaults()
            entry = self._ensure_agent(agent_name)
            entry["score"] += boost
            entry["successes"] += 1
            self._ensure_trust_defaults(str(agent_name).strip().lower(), entry)
            await self._save()

    async def record_failure(self, agent_name: str, penalty: int = 5) -> None:
        """Decrease *agent_name*'s score after a failure (floor at 0)."""
        async with self._lock:
            self._bootstrap_pantheon_defaults()
            entry = self._ensure_agent(agent_name)
            entry["score"] = max(0, entry["score"] - penalty)
            entry["failures"] += 1
            self._ensure_trust_defaults(str(agent_name).strip().lower(), entry)
            await self._save()

    async def training_boost(self, agent_name: str, training_score: int) -> None:
        """Apply a Nemesis-training boost.

        *training_score* is clamped to 1-10 and linearly mapped to
        1-15 bonus points.
        """
        clamped = max(1, min(_TRAINING_MAX_SCORE, training_score))
        boost = round(clamped / _TRAINING_MAX_SCORE * _TRAINING_MAX_BOOST)
        async with self._lock:
            self._bootstrap_pantheon_defaults()
            entry = self._ensure_agent(agent_name)
            entry["score"] += boost
            self._ensure_trust_defaults(str(agent_name).strip().lower(), entry)
            await self._save()
        logger.info(
            "Training boost: %s +%d (training_score=%d)",
            agent_name,
            boost,
            training_score,
        )

    def get_level(self, agent_name: str) -> dict[str, Any]:
        """Return the current level info for a single agent."""
        entry = self._ensure_agent(agent_name)
        score = entry["score"]
        leaderboard = self.get_leaderboard()
        rank = next(
            (i + 1 for i, row in enumerate(leaderboard) if row["agent"] == agent_name),
            len(leaderboard),
        )
        return {
            "agent": agent_name,
            "score": score,
            "tier": _tier_for_score(score).value,
            "rank": rank,
        }

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Return all agents sorted by score (descending)."""
        rows: list[dict[str, Any]] = []
        for name, entry in self._agents.items():
            score = entry["score"]
            rows.append(
                {
                    "agent": name,
                    "score": score,
                    "tier": _tier_for_score(score).value,
                    "successes": entry.get("successes", 0),
                    "failures": entry.get("failures", 0),
                }
            )
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows

    @staticmethod
    def get_tier(score: int) -> str:
        """Return the tier name for a given score."""
        return _tier_for_score(score).value

    def get_trust_matrix(self, observer_agent: str) -> dict[str, float]:
        """Return observer_agent's trust score (0–1) toward each other agent."""
        observer = str(observer_agent).strip().lower()
        entry = self._ensure_agent(observer)
        self._ensure_trust_defaults(observer, entry)
        trust_in = entry.get("trust_in", {})
        return dict(trust_in)

    async def record_trust_decrease(
        self,
        observer_agent: str,
        target_agent: str,
        amount: float = _TRUST_DECREMENT,
    ) -> None:
        """Lower observer_agent's trust in target_agent (e.g. after delegation failure)."""
        async with self._lock:
            self._bootstrap_pantheon_defaults()
            observer = str(observer_agent).strip().lower()
            target = str(target_agent).strip().lower()
            entry = self._ensure_agent(observer)
            self._ensure_trust_defaults(observer, entry)
            trust_in = entry.setdefault("trust_in", {})
            current = float(trust_in.get(target, _DEFAULT_TRUST))
            trust_in[target] = _clamp_trust(current - amount)
            await self._save()
        logger.debug(
            "Trust decrease: %s -> %s (now %.2f)",
            observer_agent,
            target_agent,
            trust_in[target],
        )

    async def record_trust_increase(
        self,
        observer_agent: str,
        target_agent: str,
        amount: float = 0.03,
    ) -> None:
        """Increase observer_agent's trust in target_agent after successful collaboration."""
        async with self._lock:
            self._bootstrap_pantheon_defaults()
            observer = str(observer_agent).strip().lower()
            target = str(target_agent).strip().lower()
            entry = self._ensure_agent(observer)
            self._ensure_trust_defaults(observer, entry)
            trust_in = entry.setdefault("trust_in", {})
            current = float(trust_in.get(target, _DEFAULT_TRUST))
            trust_in[target] = _clamp_trust(current + amount)
            await self._save()
        logger.debug(
            "Trust increase: %s -> %s (now %.2f)",
            observer_agent,
            target_agent,
            trust_in[target],
        )

    async def record_turn_outcome(
        self,
        *,
        successful_agents: Iterable[str],
        failed_agents: Iterable[str] = (),
        success_boost: int = 3,
        failure_penalty: int = 2,
        trust_increase: float = 0.03,
    ) -> None:
        """Batch-update scores + mutual trust from one multi-agent turn."""
        successful = self._normalize_collaboration_agents(successful_agents)
        successful_set = set(successful)
        failed = [
            agent
            for agent in self._normalize_collaboration_agents(failed_agents)
            if agent not in successful_set
        ]
        if not successful and not failed:
            return

        async with self._lock:
            changed = self._bootstrap_pantheon_defaults()

            for agent in successful:
                entry = self._ensure_agent(agent)
                entry["score"] += success_boost
                entry["successes"] += 1
                if self._ensure_trust_defaults(agent, entry):
                    changed = True
                changed = True

            for agent in failed:
                entry = self._ensure_agent(agent)
                entry["score"] = max(0, int(entry.get("score", 0)) - failure_penalty)
                entry["failures"] += 1
                if self._ensure_trust_defaults(agent, entry):
                    changed = True
                changed = True

            for observer in successful:
                observer_entry = self._ensure_agent(observer)
                trust_in = observer_entry.setdefault("trust_in", {})
                for target in successful:
                    if target == observer:
                        continue
                    current = float(trust_in.get(target, _DEFAULT_TRUST))
                    boosted = _clamp_trust(current + trust_increase)
                    if boosted != current:
                        trust_in[target] = boosted
                        changed = True

            if changed:
                await self._save()

    async def nudge_trust_toward_default(self, step: float = 0.05) -> None:
        """Nudge all trust_in values toward _DEFAULT_TRUST (gentle recovery during dream)."""
        async with self._lock:
            changed = self._bootstrap_pantheon_defaults()
            for agent_name, entry in self._agents.items():
                if self._ensure_trust_defaults(agent_name, entry):
                    changed = True
                trust_in = entry.get("trust_in", {})
                if not trust_in:
                    continue
                for other, current in list(trust_in.items()):
                    current_f = float(current)
                    if current_f < _DEFAULT_TRUST:
                        trust_in[other] = _clamp_trust(current_f + step)
                        changed = True
                    elif current_f > _DEFAULT_TRUST:
                        trust_in[other] = _clamp_trust(current_f - step)
                        changed = True
            if changed:
                await self._save()
        logger.debug("Trust nudged toward default (step=%.2f)", step)

    async def reload(self) -> None:
        """Re-read the data file from disk."""
        await self._load()


def power_levels_public_probe() -> dict[str, Any]:
    """Read-only summary for health endpoints (sync read; no lock)."""
    path = default_power_levels_path()
    empty: dict[str, Any] = {
        "persisted": False,
        "readable": True,
        "agents_tracked": 0,
        "top_agent": None,
        "top_score": None,
        "top_tier": None,
    }
    if not path.exists():
        return {**empty}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        agents = data.get("agents", {})
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {
            "persisted": True,
            "readable": False,
            "agents_tracked": 0,
            "top_agent": None,
            "top_score": None,
            "top_tier": None,
        }
    rows: list[tuple[str, int, str]] = []
    for name, entry in agents.items():
        score = int(entry.get("score", 0))
        rows.append((name, score, PowerLevelTracker.get_tier(score)))
    rows.sort(key=lambda r: r[1], reverse=True)
    top = rows[0] if rows else None
    return {
        "persisted": True,
        "readable": True,
        "agents_tracked": len(rows),
        "top_agent": top[0] if top else None,
        "top_score": top[1] if top else None,
        "top_tier": top[2] if top else None,
    }
