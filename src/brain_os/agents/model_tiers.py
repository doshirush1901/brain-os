"""Logical LLM model profiles assigned to Pantheon agents at startup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain_os.agents.base_agent import BaseAgent

REASONING_PROFILE_AGENTS: frozenset[str] = frozenset(
    {
        "athena",
        "clio",
        "vera",
        "gapper",
        "prometheus",
        "plutus",
        "sophia",
        "argus",
        "mnemon",
    }
)

FAST_PROFILE_AGENTS: frozenset[str] = frozenset({"sphinx", "delphi"})


def resolve_model_profile_for_agent(
    agent_name: str,
    *,
    explicit_profile: str | None = None,
) -> str | None:
    """Return a logical profile when the agent class has no explicit profile."""
    if explicit_profile:
        return None
    name = (agent_name or "").strip().lower()
    if name in REASONING_PROFILE_AGENTS:
        return "reasoning"
    if name in FAST_PROFILE_AGENTS:
        return "fast"
    return None


def apply_model_profile_to_agent(agent: BaseAgent) -> None:
    """Set instance ``preferred_model_profile`` from tier registry when unset on class."""
    cls_profile = getattr(type(agent), "preferred_model_profile", None)
    profile = resolve_model_profile_for_agent(agent.name, explicit_profile=cls_profile)
    if profile:
        agent.preferred_model_profile = profile
