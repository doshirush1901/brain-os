"""Phase: remember (pipeline step 2). Extracted from pipeline.py 2026-05-23.

Loads conversation history, coreference resolution, cross-channel context, and
active goals. Must not import from ``brain_os.pipeline``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from brain_os.exceptions import DatabaseError
from brain_os.pipeline_runtime import _is_ambiguity_sensitive_query, _should_bypass_cheap_exits

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RememberResult:
    """Output of step 2 (REMEMBER)."""

    history: list[dict[str, Any]]
    cross_channel_history: list[dict[str, Any]]
    history_summary: str
    active_goal: Any | None
    resolved_input: str
    bypass_cheap_exits: bool
    ambiguity_sensitive: bool


async def run_remember_step(
    *,
    conversation: Any,
    unified_ctx: Any | None,
    goals: Any | None,
    contact_email: str,
    channel: str,
    raw_input: str,
    bypass_cheap_exits_raw: bool,
    app_meta: Any,
    strict_slack_workflow: bool,
    record_stage_fn: Any | None = None,
    on_progress: Any | None = None,
) -> RememberResult:
    """Step 2: conversation memory, coref, goals, and query sanitization."""
    if on_progress:
        await on_progress({"type": "remembering"})

    history = await conversation.get_history(
        contact_email,
        channel,
        limit=20,
    )

    resolved_input = raw_input
    if history:
        try:
            resolved_input = await conversation.resolve_coreferences(
                raw_input,
                history,
            )
        except (DatabaseError, Exception):
            logger.exception("Coreference resolution failed")

    history_summary = ""
    try:
        _recent_msgs, history_summary = await conversation.get_summarized_history(
            contact_email,
            channel,
            recent_limit=5,
            full_limit=20,
        )
    except (DatabaseError, Exception):
        logger.debug("Summarized history not available", exc_info=True)

    cross_channel_history: list[dict[str, Any]] = []
    if unified_ctx is not None:
        try:
            cross_channel_history = unified_ctx.recent_history(
                contact_email,
                limit=10,
            )
        except (DatabaseError, Exception):
            logger.exception("UnifiedContextManager lookup failed")

    active_goal = None
    if goals is not None:
        try:
            active_goal = await goals.get_active_goal(contact_email)
        except (DatabaseError, Exception):
            logger.exception("GoalManager lookup failed")

    if record_stage_fn is not None:
        record_stage_fn("remember")

    bypass_cheap_exits = bypass_cheap_exits_raw or _should_bypass_cheap_exits(
        resolved_input,
        enabled=app_meta.cheap_exit_bypass_enabled,
        bypass_keywords=app_meta.cheap_exit_bypass_keywords,
        allowlist_keywords=app_meta.cheap_exit_allowlist_keywords,
    )
    if strict_slack_workflow:
        bypass_cheap_exits = True

    logger.info(
        "REMEMBER | history=%d msgs | cross_channel=%d | goal=%s | summary=%s",
        len(history),
        len(cross_channel_history),
        active_goal.goal_type.value if active_goal else "none",
        "yes" if history_summary else "no",
    )

    from brain_os.brain.untrusted_input import sanitize_pipeline_query
    from brain_os.config import get_settings as _get_settings_untrusted

    resolved_input = sanitize_pipeline_query(
        resolved_input,
        max_chars=_get_settings_untrusted().app.pipeline_query_max_chars,
    )
    ambiguity_sensitive = _is_ambiguity_sensitive_query(resolved_input)

    return RememberResult(
        history=history,
        cross_channel_history=cross_channel_history,
        history_summary=history_summary,
        active_goal=active_goal,
        resolved_input=resolved_input,
        bypass_cheap_exits=bypass_cheap_exits,
        ambiguity_sensitive=ambiguity_sensitive,
    )
