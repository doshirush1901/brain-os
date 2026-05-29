"""Checkpoint resume preamble — REMEMBER without PERCEIVE/SPHINX."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from brain_os.exceptions import DatabaseError
from brain_os.pipeline_checkpoint import build_perception_stub_for_resume
from brain_os.pipeline_runtime import _is_ambiguity_sensitive_query, _should_bypass_cheap_exits

logger = logging.getLogger(__name__)


@dataclass
class CheckpointResumePreamble:
    """State gathered after a clarification checkpoint resume."""

    contact_email: str
    resolved_input: str
    perception: dict[str, Any]
    history: list[dict[str, Any]]
    cross_channel_history: list[dict[str, Any]]
    history_summary: str
    active_goal: Any
    bypass_cheap_exits: bool
    ambiguity_sensitive: bool
    clarification_answer: str
    original_query: str


async def build_checkpoint_resume_preamble(
    pipeline: Any,
    *,
    resume: dict[str, Any],
    raw_input: str,
    channel: str,
    sender_id: str,
    meta: dict[str, Any],
    app_meta: Any,
    bypass_cheap_exits_raw: bool,
) -> CheckpointResumePreamble:
    """Run REMEMBER-equivalent work for a pipeline checkpoint resume."""
    from brain_os.brain.untrusted_input import sanitize_pipeline_query
    from brain_os.config import get_settings as _get_settings_untrusted

    contact_email = str(resume.get("contact_email") or "").strip()
    original_query = str(resume.get("original_query") or "")
    clarification_answer = str(resume.get("clarification_answer") or raw_input)
    merged_query = str(resume.get("merged_query") or raw_input)

    history: list[dict[str, Any]] = []
    try:
        history = await pipeline._conversation.get_history(
            contact_email,
            channel,
            limit=20,
        )
    except (DatabaseError, Exception):
        logger.exception("Checkpoint resume: history load failed")

    resolved_input = merged_query
    if history:
        try:
            resolved_input = await pipeline._conversation.resolve_coreferences(
                merged_query,
                history,
            )
        except (DatabaseError, Exception):
            logger.exception("Checkpoint resume: coreference resolution failed")

    history_summary = ""
    try:
        _recent, history_summary = await pipeline._conversation.get_summarized_history(
            contact_email,
            channel,
            recent_limit=5,
            full_limit=20,
        )
    except (DatabaseError, Exception):
        logger.debug("Checkpoint resume: summarized history unavailable", exc_info=True)

    cross_channel_history: list[dict[str, Any]] = []
    if pipeline._unified_ctx is not None:
        try:
            cross_channel_history = pipeline._unified_ctx.recent_history(
                contact_email,
                limit=10,
            )
        except (DatabaseError, Exception):
            logger.exception("Checkpoint resume: unified context failed")

    active_goal = None
    if pipeline._goals is not None:
        try:
            active_goal = await pipeline._goals.get_active_goal(contact_email)
        except (DatabaseError, Exception):
            logger.exception("Checkpoint resume: goal lookup failed")

    resolved_input = sanitize_pipeline_query(
        resolved_input,
        max_chars=_get_settings_untrusted().app.pipeline_query_max_chars,
    )
    bypass_cheap_exits = bypass_cheap_exits_raw or _should_bypass_cheap_exits(
        resolved_input,
        enabled=app_meta.cheap_exit_bypass_enabled,
        bypass_keywords=app_meta.cheap_exit_bypass_keywords,
        allowlist_keywords=app_meta.cheap_exit_allowlist_keywords,
    )
    ambiguity_sensitive = _is_ambiguity_sensitive_query(resolved_input)

    perception = build_perception_stub_for_resume(
        contact_email=contact_email,
        channel=channel,
        sender_id=sender_id,
        merged_query=resolved_input,
        meta=meta,
    )

    return CheckpointResumePreamble(
        contact_email=contact_email,
        resolved_input=resolved_input,
        perception=perception,
        history=history,
        cross_channel_history=cross_channel_history,
        history_summary=history_summary,
        active_goal=active_goal,
        bypass_cheap_exits=bypass_cheap_exits,
        ambiguity_sensitive=ambiguity_sensitive,
        clarification_answer=clarification_answer,
        original_query=original_query,
    )
