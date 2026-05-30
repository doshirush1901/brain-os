"""On-disk workspace helpers for :class:`~ira.systems.task_orchestrator.TaskOrchestrator`.

Each task gets a directory under ``data/tasks/<task_id>/`` with README, PLAN,
MEMORIES, and per-phase dumps. Paths are validated so ``task_id`` cannot escape
the tasks root.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain_os.schemas.llm_outputs import TaskPlan

logger = logging.getLogger(__name__)

_TASK_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_MEMORIES_TAIL_CHARS = 8000
_PHASE_DUMP_MAX_BYTES = 500_000


def validate_task_id(task_id: str) -> None:
    """Reject path components and unexpected ``task_id`` shapes."""
    if not task_id or not _TASK_ID_RE.fullmatch(task_id):
        msg = f"Invalid task_id for workspace: {task_id!r}"
        raise ValueError(msg)


def resolve_task_workspace_dir(tasks_root: Path, task_id: str) -> Path:
    """Return ``tasks_root / task_id`` resolved, ensuring it stays under ``tasks_root``."""
    validate_task_id(task_id)
    root = tasks_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root / task_id).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        msg = f"Workspace path escapes tasks root: {target}"
        raise ValueError(msg) from exc
    return target


def init_workspace_files(workspace: Path, *, task_id: str, goal: str) -> None:
    """Create README, MEMORIES, PLAN stub, and ``phases/`` for a new task."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "phases").mkdir(parents=True, exist_ok=True)

    readme = (
        f"# Task `{task_id}`\n\n"
        f"- **Status:** created\n"
        f"- **Goal:** {goal.strip()}\n\n"
        "## Paths\n\n"
        "- `PLAN.md` — filled after Athena planning\n"
        "- `CONTRACT.json` — validation contract + per-phase verdicts\n"
        "- `MEMORIES.md` — append-only orchestrator log\n"
        "- `phases/` — raw output per phase\n"
        "- Final report: `data/reports/<task_id>.md` (after completion)\n"
    )
    (workspace / "README.md").write_text(readme, encoding="utf-8")

    memories_header = (
        "# Task memory (orchestrator)\n\nAppend-only log written by Brain OS's task orchestrator.\n"
    )
    (workspace / "MEMORIES.md").write_text(memories_header, encoding="utf-8")

    plan_stub = "# Plan\n\n_Pending Athena planning…_\n"
    (workspace / "PLAN.md").write_text(plan_stub, encoding="utf-8")


def update_readme_status(workspace: Path, status: str) -> None:
    """Replace the ``**Status:**`` line in README, or append if missing."""
    path = workspace / "README.md"
    if not path.is_file():
        logger.warning("README missing for workspace %s — skipping status update", workspace)
        return
    text = path.read_text(encoding="utf-8")
    if "**Status:**" in text:
        text_new = re.sub(
            r"(?m)^- \*\*Status:\*\*.*$",
            f"- **Status:** {status}",
            text,
            count=1,
        )
    else:
        text_new = text + f"\n- **Status:** {status}\n"
    path.write_text(text_new, encoding="utf-8")


def append_readme_artifact(workspace: Path, label: str, path_value: str) -> None:
    """Append an artifact line to README if not already present."""
    path = workspace / "README.md"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    needle = f"**{label}:**"
    if needle in text:
        return
    block = f"\n- {needle} `{path_value}`\n"
    path.write_text(text + block, encoding="utf-8")


def append_memory_line(workspace: Path, message: str) -> None:
    """Append a single timestamped bullet to MEMORIES.md."""
    path = workspace / "MEMORIES.md"
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"\n- [{ts}] {message}\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def read_memories_tail(workspace: Path, *, max_chars: int = _MEMORIES_TAIL_CHARS) -> str:
    """Return the tail of MEMORIES.md for prompt injection (bounded)."""
    path = workspace / "MEMORIES.md"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text.strip()
    return text[-max_chars:].strip()


def format_plan_markdown(plan: TaskPlan) -> str:
    """Render ``TaskPlan`` as markdown checklist + descriptions."""
    lines: list[str] = ["# Task plan\n", "\n", f"**Goal:** {plan.goal.strip()}\n"]
    if plan.reasoning.strip():
        lines.extend(["\n## Reasoning\n\n", plan.reasoning.strip(), "\n"])
    lines.append("\n## Phases (checklist)\n\n")
    for i, p in enumerate(plan.phases):
        dep = f" _(depends_on: {p.depends_on})_" if p.depends_on else ""
        lines.append(f"- [ ] **Phase {i + 1}: {p.title}** — agent `{p.agent}`{dep}\n")
        desc = (p.description or "").strip()
        if desc:
            lines.append(f"  - {desc}\n")
        lines.append("\n")
    return "".join(lines)


def write_plan_file(workspace: Path, plan: TaskPlan) -> None:
    """Overwrite PLAN.md with the structured plan."""
    (workspace / "PLAN.md").write_text(format_plan_markdown(plan), encoding="utf-8")


def write_contract_file(workspace: Path, payload: dict[str, Any]) -> None:
    """Write or overwrite ``CONTRACT.json`` (validation contract + phase verdicts)."""
    path = workspace / "CONTRACT.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    append_readme_artifact(workspace, "Contract", "CONTRACT.json")


def append_contract_phase_verdict(
    workspace: Path,
    *,
    phase_index: int,
    verdict: dict[str, Any],
) -> None:
    """Merge one phase validation verdict into ``CONTRACT.json``."""
    path = workspace / "CONTRACT.json"
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except json.JSONDecodeError:
            logger.warning("Corrupt CONTRACT.json in %s — resetting phase_verdicts", workspace)
    phase_verdicts = data.get("phase_verdicts")
    if not isinstance(phase_verdicts, dict):
        phase_verdicts = {}
    phase_verdicts[str(phase_index)] = verdict
    data["phase_verdicts"] = phase_verdicts
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _slug_fragment(text: str, *, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (s[:max_len] or "phase").rstrip("_")


def _truncate_utf8(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    cut = max_bytes - 80
    prefix = data[:cut].decode("utf-8", errors="ignore")
    return prefix + "\n\n… [truncated by orchestrator]\n"


def write_phase_artifact(
    workspace: Path,
    *,
    phase_index: int,
    agent: str,
    title: str,
    body: str,
) -> None:
    """Write ``phases/NN__agent__slug.md`` with optional UTF-8 truncation."""
    phases_dir = workspace / "phases"
    phases_dir.mkdir(parents=True, exist_ok=True)
    name = f"{phase_index:02d}__{_slug_fragment(agent)}__{_slug_fragment(title)}.md"
    path = phases_dir / name
    content = f"# Phase {phase_index + 1}: {title}\n\n**Agent:** `{agent}`\n\n---\n\n"
    content += _truncate_utf8(body, _PHASE_DUMP_MAX_BYTES)
    path.write_text(content, encoding="utf-8")


def record_phase_completed(
    workspace: Path,
    *,
    phase_index: int,
    agent: str,
    title: str,
    result_preview: str,
    result_body: str,
) -> None:
    """Append memory line and write the phase dump file."""
    preview = (result_preview or "").replace("\n", " ")[:200]
    append_memory_line(
        workspace,
        f"Phase {phase_index + 1} complete — `{agent}` / {title}: {preview}",
    )
    write_phase_artifact(
        workspace,
        phase_index=phase_index,
        agent=agent,
        title=title,
        body=result_body,
    )
