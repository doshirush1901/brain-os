"""Usage analytics for agent and tool/function calls.

Builds durable metrics from Graphe's `cursor_sessions` SQLite log.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_json_list(raw: str | None) -> list[Any]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def build_usage_report(
    *,
    db_path: Path | None = None,
    all_agents: list[str] | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Return usage metrics from logged Cursor sessions.

    Parameters
    ----------
    db_path:
        Optional path to Graphe session DB. Defaults to ``data/brain/cursor_sessions.db``.
    all_agents:
        Optional full pantheon agent roster. Used to compute never-used agents.
    top_n:
        Number of rows for top/least lists.
    """
    path = db_path or Path("data/brain/cursor_sessions.db")
    if not path.exists():
        return {
            "db_path": str(path),
            "sessions_count": 0,
            "agents_used_count": 0,
            "tools_used_count": 0,
            "top_agents": [],
            "least_used_agents": [],
            "never_used_agents": sorted(set(all_agents or [])),
            "top_tools": [],
            "least_used_tools": [],
        }

    agent_counter: Counter[str] = Counter()
    tool_counter: Counter[str] = Counter()
    tool_by_agent_counter: Counter[str] = Counter()
    sessions_count = 0

    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(cursor_sessions)")
        columns = {str(row[1]) for row in cur.fetchall()}
        has_tool_calls = "tool_calls" in columns

        if has_tool_calls:
            cur.execute("SELECT agents_used, tool_calls FROM cursor_sessions")
            records = cur.fetchall()
        else:
            cur.execute("SELECT agents_used FROM cursor_sessions")
            records = [(agents_raw, "[]") for (agents_raw,) in cur.fetchall()]

        for agents_raw, tools_raw in records:
            sessions_count += 1

            agents = [
                str(a).strip().lower() for a in _parse_json_list(agents_raw) if str(a).strip()
            ]
            for agent in agents:
                agent_counter[agent] += 1

            for item in _parse_json_list(tools_raw):
                if not isinstance(item, dict):
                    continue
                agent = str(item.get("agent", "")).strip().lower()
                tool = str(item.get("tool", "")).strip()
                if not tool:
                    continue
                tool_counter[tool] += 1
                if agent:
                    tool_by_agent_counter[f"{agent}.{tool}"] += 1

    known_agents = sorted({a.lower() for a in all_agents or []})
    used_agents = sorted(agent_counter.keys())
    never_used = sorted(set(known_agents) - set(used_agents)) if known_agents else []

    def _rows(counter: Counter[str], reverse: bool) -> list[dict[str, Any]]:
        items = sorted(
            counter.items(), key=lambda kv: (-kv[1], kv[0]) if reverse else (kv[1], kv[0])
        )
        return [{"name": name, "count": count} for name, count in items[: max(1, top_n)]]

    return {
        "db_path": str(path),
        "sessions_count": sessions_count,
        "agents_used_count": len(used_agents),
        "tools_used_count": len(tool_counter),
        "top_agents": _rows(agent_counter, reverse=True),
        "least_used_agents": _rows(agent_counter, reverse=False),
        "never_used_agents": never_used,
        "top_tools": _rows(tool_counter, reverse=True),
        "least_used_tools": _rows(tool_counter, reverse=False),
        "top_functions_by_agent": _rows(tool_by_agent_counter, reverse=True),
        "least_used_functions_by_agent": _rows(tool_by_agent_counter, reverse=False),
    }
