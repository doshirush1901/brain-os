"""Pipeline progress rendering for CLI ask/chat and demo_mode.

L3.3d: relocated from ``brain_os.interfaces.cli._progress`` (same species as
``cli_shutdown`` — process/UX helpers for the composition root). Lives in
``runtime`` rather than ``contracts`` because it wires Rich Progress and
stderr consoles, not a pure protocol/constant surface.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Local stderr console — do not import interfaces.cli.app (would re-invert).
err_console = Console(stderr=True)


def _render_progress_event(event: dict[str, Any]) -> tuple[str, str | None]:
    """Map pipeline progress events to CLI status and detail lines."""
    etype = str(event.get("type", "")).strip()
    agent = str(event.get("agent", "")).strip()
    role = str(event.get("role", "")).strip()

    if etype == "perceiving":
        return "Perceiving input...", "• Perceiving input"
    if etype == "remembering":
        return "Recalling context...", "• Recalling memory and conversation context"
    if etype == "fast_path":
        category = event.get("category") or "general"
        return f"Fast path matched ({category})", f"• Fast path category: {category}"
    if etype == "sphinx_checking":
        return "Checking query clarity...", "• Sphinx is checking clarity"
    if etype == "sphinx_clarifying":
        return "Generating clarification...", "• Sphinx requested clarification"
    if etype == "routing":
        method = event.get("method") or "agent routing"
        return f"Routing request ({method})...", f"• Routing method: {method}"
    if etype == "enriching":
        return "Enriching context...", "• Building enriched execution context"
    if etype == "agent_started":
        role_suffix = f" ({role})" if role else ""
        return f"Consulting {agent}{role_suffix}...", f"• Started: {agent}{role_suffix}"
    if etype == "agent_thinking":
        iteration = event.get("iteration") or "?"
        return (
            f"{agent} thinking (iter {iteration})...",
            f"  ↳ {agent} thinking, iteration {iteration}",
        )
    if etype == "tool_called":
        tool = event.get("tool") or "tool"
        return f"{agent} using {tool}...", f"  ↳ {agent} called tool: {tool}"
    if etype == "agent_done":
        return f"{agent} finished", f"✓ Completed: {agent}"
    if etype == "synthesizing":
        return "Synthesizing agent outputs...", "• Athena is synthesizing responses"
    if etype == "gap_resolving":
        gaps = event.get("gaps")
        if isinstance(gaps, int):
            return f"Resolving gaps ({gaps})...", f"• Gapper resolving {gaps} gap(s)"
        return "Resolving response gaps...", "• Gapper resolving response gaps"
    if etype == "faithfulness_check":
        return "Running fact-faithfulness checks...", "• Verifying response faithfulness"
    if etype == "assessing":
        return "Assessing confidence...", "• Metacognition is assessing confidence"
    if etype == "reflecting":
        return "Reflecting...", "• Sophia reflection pass"
    if etype == "shaping":
        return "Shaping final response...", "• Applying voice and channel shaping"

    fallback = etype.replace("_", " ").strip() or "working"
    return f"{fallback}...", None


def _progress_event_to_step_line(event: dict[str, Any]) -> str:
    """Format a pipeline progress event as a single line for the steps array (--json)."""
    etype = str(event.get("type", "")).strip()
    agent = str(event.get("agent", "")).strip()
    role = str(event.get("role", "")).strip()
    tool = str(event.get("tool", "")).strip()
    iteration = event.get("iteration")
    _pv = event.get("preview") or ""
    preview = _pv[:80] if isinstance(_pv, str) else str(_pv)[:80]

    if etype == "perceiving":
        return "• Perceiving input"
    if etype == "remembering":
        return "• Recalling memory and conversation context"
    if etype == "fast_path":
        cat = event.get("category") or "general"
        return f"• Fast path matched ({cat})"
    if etype == "sphinx_checking":
        return "• Sphinx: checking query clarity"
    if etype == "sphinx_clarifying":
        return "• Sphinx: clarification needed"
    if etype == "routing":
        return "• Routing (Athena)"
    if etype == "enriching":
        return "• Enriching context"
    if etype == "agent_started":
        role_suffix = f" — {role}" if role else ""
        return f"▶ {agent}{role_suffix}"
    if etype == "agent_thinking":
        it = iteration if iteration is not None else "?"
        return f"  … {agent} thinking (iter {it})"
    if etype == "tool_called":
        return f"  ↳ {agent} → {tool}" if tool else f"  ↳ {agent} → tool"
    if etype == "agent_done":
        return f"✓ {agent} done" + (f" — {preview}…" if preview else "")
    if etype == "synthesizing":
        return "• Athena synthesizing"
    if etype == "gap_resolving":
        n = event.get("gaps")
        return f"• Gapper resolving {n} gap(s)" if isinstance(n, int) else "• Gapper resolving gaps"
    if etype == "faithfulness_check":
        return "• Faithfulness check"
    if etype == "assessing":
        return "• Assessing confidence"
    if etype == "reflecting":
        return "• Reflecting (Sophia)"
    if etype == "shaping":
        return "• Shaping response"
    return f"• {etype.replace('_', ' ')}"


async def _process_request_with_live_progress(
    pipeline: Any,
    *,
    raw_input: str,
    sender_id: str,
    channel: str = "cli",
    metadata: dict[str, Any] | None = None,
) -> tuple[str, list[str], str]:
    """Run pipeline with a Manus-style live progress feed in terminal."""
    last_status = ""
    step_no = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=err_console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("[bold green]Initializing request...[/bold green]", total=None)

        async def on_progress(event: dict[str, Any]) -> None:
            nonlocal last_status, step_no
            status, detail = _render_progress_event(event)
            if status and status != last_status:
                step_no += 1
                progress.update(
                    task_id,
                    description=f"[bold green][step {step_no}] {status}[/bold green]",
                )
                last_status = status
            if detail:
                err_console.print(f"[dim]{detail}[/dim]")

        response, agents_used, run_id = await pipeline.process_request(
            raw_input=raw_input,
            channel=channel,
            sender_id=sender_id,
            metadata=metadata,
            on_progress=on_progress,
        )
        progress.update(task_id, description="[bold green]Done[/bold green]")
        return response, agents_used, run_id


_MAX_STEPS_IN_JSON = 80  # Cap so Cursor/UI doesn't get a huge block; tail is preserved


async def _process_request_collect_steps(
    pipeline: Any,
    *,
    raw_input: str,
    sender_id: str,
    channel: str = "cli",
    metadata: dict[str, Any] | None = None,
) -> tuple[str, list[str], list[str], str]:
    """Run pipeline and collect progress events as a list of step lines (for --json)."""
    steps: list[str] = []
    # Manus-style: show that Brain OS is working as soon as we start
    err_console.print("[bold green]Brain OS is thinking…[/bold green]")
    err_console.print("[dim]  Running full pipeline (routing, agents, retrieval).[/dim]")

    async def on_progress(event: dict[str, Any]) -> None:
        line = _progress_event_to_step_line(event)
        if line and (not steps or steps[-1] != line):
            steps.append(line)
            # Show what Brain OS is doing (Manus-style) when using --json so Cursor/scripts see live progress
            err_console.print(f"[dim]  {line}[/dim]")

    response, agents_used, run_id = await pipeline.process_request(
        raw_input=raw_input,
        channel=channel,
        sender_id=sender_id,
        metadata=metadata,
        on_progress=on_progress,
    )
    # Cap steps for Cursor/UI; keep head + tail so start and end of pipeline are visible
    if len(steps) > _MAX_STEPS_IN_JSON:
        keep_tail = _MAX_STEPS_IN_JSON // 3
        steps = (
            steps[: _MAX_STEPS_IN_JSON - keep_tail]
            + [f"… ({len(steps) - _MAX_STEPS_IN_JSON} more steps)"]
            + steps[-keep_tail:]
        )
    return response, agents_used or [], steps, run_id
