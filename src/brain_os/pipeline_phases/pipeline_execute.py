"""RequestPipeline multi-agent execute path (routed agents + Athena synthesis).

Separate from AgentLoop phase execution in ``execute.py``. Must not import
from ``brain_os.pipeline``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from brain_os.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)


async def execute_routed_agents(
    *,
    pantheon: Any,
    agent_names: list[str],
    query: str,
    context: dict[str, Any],
    on_progress: Any | None = None,
) -> tuple[str, list[str]]:
    """Execute one or more routed agents with bounded parallelism."""
    from brain_os.config import get_settings
    from brain_os.deployment_profile import effective_app_config
    from brain_os.pipeline_phases.execute import (
        athena_multi_agent_synthesis_context,
        execute_path_agent_done_event,
        execute_path_agent_started_event,
        execute_path_synthesizing_event,
    )

    app_cfg = effective_app_config(get_settings().app)
    slot_s = max(1, int(app_cfg.agent_timeout))
    max_parallel = max(1, int(app_cfg.max_parallel_agents))
    synthesis_timeout_s = max(1, int(app_cfg.athena_synthesis_timeout))
    used: list[str] = []
    responses: dict[str, str] = {}
    sem = asyncio.Semaphore(max_parallel)

    shared_audit = context.get("_tool_audit")
    if not isinstance(shared_audit, list):
        shared_audit = []
        context["_tool_audit"] = shared_audit

    async def _run_agent(name: str) -> tuple[str, str, list[dict[str, Any]]]:
        agent = pantheon.get_agent(name)
        if agent is None:
            logger.warning("Agent '%s' not found — skipping", name)
            return name, "", []
        if on_progress:
            await on_progress(
                execute_path_agent_started_event(name, role=getattr(agent, "role", ""))
            )
        # Per-agent context copy + private audit list (merge after gather).
        agent_ctx = dict(context)
        local_audit: list[dict[str, Any]] = []
        agent_ctx["_tool_audit"] = local_audit
        try:
            async with sem:
                timeout_s = max(1, int(getattr(agent, "timeout", None) or slot_s))
                resp = await asyncio.wait_for(
                    agent.handle(query, agent_ctx),
                    timeout=timeout_s,
                )
                out = resp
        except TimeoutError:
            logger.warning("Agent '%s' timed out after %ds", name, slot_s)
            out = f"(Agent '{name}' timed out after {slot_s}s)"
        except (ToolExecutionError, Exception):
            logger.exception("Agent '%s' failed during execution", name)
            out = f"(Agent '{name}' encountered an error)"
        if on_progress:
            await on_progress(execute_path_agent_done_event(name, out))
        return name, out, local_audit

    tasks = [asyncio.create_task(_run_agent(name)) for name in agent_names]
    for task in asyncio.as_completed(tasks):
        name, out, local_audit = await task
        if local_audit:
            shared_audit.extend(local_audit)
        if not out:
            continue
        responses[name] = out
        used.append(name)

    if not responses:
        return await pantheon.process(query, context, on_progress=on_progress), ["athena"]

    if len(responses) == 1:
        return next(iter(responses.values())), used

    athena = pantheon.get_agent("athena")
    if athena is not None:
        if on_progress:
            await on_progress(execute_path_synthesizing_event())
        try:
            synthesised = await asyncio.wait_for(
                athena.handle(
                    query,
                    athena_multi_agent_synthesis_context(responses),
                ),
                timeout=synthesis_timeout_s,
            )
        except TimeoutError:
            logger.warning("Athena synthesis timed out after %ds", synthesis_timeout_s)
            return "\n\n".join(responses.values()), used
        return synthesised, used + ["athena"]

    return "\n\n".join(responses.values()), used
