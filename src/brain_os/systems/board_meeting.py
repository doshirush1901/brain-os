"""Board meeting system — multi-agent collaborative discussions.

Gathers contributions from multiple Pantheon agents on a topic, then
synthesises a final decision via Athena.  Used by the
``POST /api/board-meeting`` endpoint and the CLI.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx

from brain_os.data.models import BoardMeetingMinutes

logger = logging.getLogger(__name__)

_BOARD_AGENT_ERRORS = (
    httpx.HTTPError,
    asyncio.TimeoutError,
    ValueError,
    KeyError,
    TypeError,
)

_DEFAULT_PARTICIPANTS = [
    "clio",
    "prometheus",
    "plutus",
    "hermes",
    "hephaestus",
    "themis",
    "tyche",
    "calliope",
]


class BoardMeeting:
    """Orchestrates a multi-agent board meeting on a given topic."""

    def __init__(
        self,
        agent_handler: Callable[[str, str], Awaitable[str]],
    ) -> None:
        self._agent_handler = agent_handler

    async def run_meeting(
        self,
        topic: str,
        participants: list[str] | None = None,
    ) -> BoardMeetingMinutes:
        agent_names = participants or _DEFAULT_PARTICIPANTS

        async def _contribute(name: str) -> tuple[str, str]:
            try:
                response = await self._agent_handler(name, topic)
                return name, response
            except _BOARD_AGENT_ERRORS:
                logger.exception("Agent '%s' failed during board meeting", name)
                return name, f"(Agent '{name}' encountered an error)"

        tasks = [_contribute(n) for n in agent_names]
        results = await asyncio.gather(*tasks)
        contributions = dict(results)

        synthesis_prompt = (
            f"Board meeting topic: {topic}\n\n"
            + "\n\n".join(f"**{agent}**: {resp}" for agent, resp in contributions.items())
            + "\n\nSynthesise these perspectives into a final decision with action items."
        )

        try:
            synthesis = await self._agent_handler("athena", synthesis_prompt)
        except _BOARD_AGENT_ERRORS:
            logger.exception("Athena synthesis failed")
            synthesis = "Synthesis unavailable — see individual contributions above."

        action_items = self._extract_action_items(synthesis)

        return BoardMeetingMinutes(
            topic=topic,
            participants=["athena"] + list(contributions.keys()),
            contributions=contributions,
            synthesis=synthesis,
            action_items=action_items,
        )

    @staticmethod
    def _extract_action_items(synthesis: str) -> list[str]:
        """Best-effort extraction of bullet-pointed action items."""
        items: list[str] = []
        in_section = False
        for line in synthesis.splitlines():
            stripped = line.strip()
            lower = stripped.lower()
            if "action item" in lower or "next step" in lower or "follow-up" in lower:
                in_section = True
                continue
            if in_section and stripped.startswith(("-", "*", "•")):
                items.append(stripped.lstrip("-*• ").strip())
            elif in_section and not stripped:
                in_section = False
        return items
