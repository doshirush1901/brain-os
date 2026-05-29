"""Phase: perceive (pipeline step 1). Extracted from pipeline.py 2026-05-23.

Resolves sender identity and channel context via SensorySystem. Must not import
from ``brain_os.pipeline``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from brain_os.data.models import Channel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerceiveResult:
    """Output of step 1 (PERCEIVE)."""

    perception: dict[str, Any]
    contact_info: dict[str, Any]
    contact_email: str


async def run_perceive_step(
    *,
    sensory: Any,
    channel: str,
    raw_input: str,
    sender_id: str,
    meta: dict[str, Any],
    record_stage_fn: Any | None = None,
    on_progress: Any | None = None,
) -> PerceiveResult:
    """Step 1: sensory perception — resolve contact and emotional context."""
    if on_progress:
        await on_progress({"type": "perceiving"})

    from brain_os.systems.sensory import PerceptionEvent

    event = PerceptionEvent(
        channel=Channel(channel.upper() if isinstance(channel, str) else channel),
        raw_input=raw_input,
        sender_id=sender_id,
        sender_name=meta.get("sender_name"),
        metadata=meta,
    )
    perception = await sensory.perceive(event)
    contact_info = perception["resolved_contact"]
    contact_email = contact_info["email"]

    if record_stage_fn is not None:
        record_stage_fn("perceive")
    logger.info("PERCEIVE | %s | %s", channel, contact_email)

    return PerceiveResult(
        perception=perception,
        contact_info=contact_info,
        contact_email=contact_email,
    )
