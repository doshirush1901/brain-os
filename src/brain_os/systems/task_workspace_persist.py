"""Disk persistence helpers for long-lived task workspaces."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STATE_JSON = "state.json"
_NEXT_MD = "NEXT.md"


def write_state_json(workspace: Path, state: dict[str, Any]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / _STATE_JSON
    payload = json.dumps(state, indent=2, default=str)
    fd, tmp = tempfile.mkstemp(dir=workspace, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    except (OSError, PermissionError):
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_state_json(workspace: Path) -> dict[str, Any] | None:
    path = workspace / _STATE_JSON
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read %s", path)
        return None


def state_to_summary(state: dict[str, Any]) -> dict[str, Any]:
    sg = state.get("standing_goal") if isinstance(state.get("standing_goal"), dict) else {}
    return {
        "task_id": state.get("task_id"),
        "status": state.get("status"),
        "goal": (str(state.get("goal") or ""))[:200],
        "long_lived": bool(state.get("long_lived")),
        "standing_objective": (str(state.get("standing_objective") or ""))[:200],
        "standing_goal_status": sg.get("status") if sg else None,
        "workspace_dir": state.get("workspace_dir"),
        "updated_at": state.get("updated_at"),
        "file_path": state.get("file_path"),
    }


def list_workspace_summaries(tasks_root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    root = tasks_root.expanduser().resolve()
    if not root.is_dir():
        return []
    rows: list[tuple[str, dict[str, Any]]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        state = read_state_json(child)
        if state is None:
            tid = child.name
            if len(tid) == 12 and tid.isalnum():
                state = {"task_id": tid, "status": "unknown", "workspace_dir": str(child)}
            else:
                continue
        updated = str(state.get("updated_at") or "")
        rows.append((updated, state_to_summary(state)))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in rows[:limit]]


def write_next_md(workspace: Path, *, lines: list[str]) -> None:
    body = "# Next wave\n\n" + "\n".join(f"- {line}" for line in lines if line.strip()) + "\n"
    (workspace / _NEXT_MD).write_text(body, encoding="utf-8")


def update_readme_last_wave(workspace: Path, *, wave: int, phases_done: int) -> None:
    path = workspace / "README.md"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    line = f"- **Last wave:** {wave} ({phases_done} phase(s) this wave)"
    if "**Last wave:**" in text:
        import re

        text = re.sub(r"(?m)^- \*\*Last wave:\*\*.*$", line, text, count=1)
    else:
        text = text.rstrip() + f"\n{line}\n"
    path.write_text(text, encoding="utf-8")


def mark_phase_done_in_plan(workspace: Path, phase_index: int) -> None:
    path = workspace / "PLAN.md"
    if not path.is_file():
        return
    import re

    text = path.read_text(encoding="utf-8")
    pattern = rf"(?m)^- \[ \] (\*\*Phase {phase_index + 1}:)"
    if re.search(pattern, text):
        text = re.sub(pattern, r"- [x] \1", text, count=1)
        path.write_text(text, encoding="utf-8")
