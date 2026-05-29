"""Centralised prompt loader for Ira.

Reads ``.txt`` prompt files from the ``prompts/`` directory at the
repository root.  Prompts are cached on first access so the filesystem
is hit only once per process.

**Prefix-cache / prompt stability:** ``load_prompt`` and
``load_soul_preamble`` use process-wide caches. Edits to ``SOUL.md`` or
``prompts/*.txt`` on disk therefore do **not** affect an already-running
Ira process until restart. For a **per-request** fresh read of SOUL (one
parse at pipeline entry, stable for all agents on that request), enable
``APP__REQUEST_PROMPT_SNAPSHOT``; the pipeline stores the result on the
request context as ``_soul_preamble_snapshot`` (see ``BaseAgent``).

``AGENTS.md`` is documentation only; it is not loaded into LLM prompts at
runtime.

Usage::

    from brain_os.prompt_loader import load_prompt

    _SYSTEM_PROMPT = load_prompt("athena_system")
    # -> reads  prompts/athena_system.txt
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _REPO_ROOT / "prompts"
_SOUL_PATH = _REPO_ROOT / "SOUL.md"


_SOUL_PREAMBLE_SECTIONS: tuple[str, ...] = (
    "Identity",
    "Philosophical Foundation",
    "Values",
    "Voice",
    "Behavioral Boundaries",
)


def _soul_section_title_matches(line: str) -> bool:
    if not line.startswith("## "):
        return False
    return any(keyword in line for keyword in _SOUL_PREAMBLE_SECTIONS)


def soul_preamble_from_markdown(text: str) -> str:
    """Extract core SOUL sections for agent system prompts.

    Includes Identity, Philosophical Foundation (Anekantavada, Syadvada, …),
    Values, Voice, and Behavioral Boundaries — in document order.
    """
    sections: list[str] = []
    capture = False
    current: list[str] = []

    for line in text.splitlines():
        if line.startswith("## ") and _soul_section_title_matches(line):
            if current and capture:
                sections.append("\n".join(current))
            current = [line]
            capture = True
        elif line.startswith("## ") and capture:
            sections.append("\n".join(current))
            current = []
            capture = False
        elif capture:
            current.append(line)

    if current and capture:
        sections.append("\n".join(current))

    if not sections:
        return ""

    return "--- IRA CORE IDENTITY ---\n" + "\n\n".join(sections) + "\n--- END CORE IDENTITY ---"


def snapshot_soul_preamble_from_disk() -> str:
    """Read and parse ``SOUL.md`` from disk (no process-level cache).

    Use once per HTTP/CLI request when ``APP__REQUEST_PROMPT_SNAPSHOT`` is
    enabled so all agents on that request share the same preamble even if
    the process cache is stale relative to disk.
    """
    if not _SOUL_PATH.exists():
        logger.warning(
            "SOUL.md not found at %s — agents will run without shared identity", _SOUL_PATH
        )
        return ""
    return soul_preamble_from_markdown(_SOUL_PATH.read_text(encoding="utf-8"))


@functools.cache
def load_prompt(name: str) -> str:
    """Return the contents of ``prompts/{name}.txt``, stripped of trailing whitespace.

    Raises :class:`FileNotFoundError` with a helpful message if the
    file does not exist.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}  (looked in {_PROMPTS_DIR})")
    return path.read_text(encoding="utf-8").rstrip()


@functools.lru_cache(maxsize=1)
def load_soul_preamble() -> str:
    """Load Identity, Philosophical Foundation, Values, Voice, and Behavioral Boundaries from SOUL.md.

    Cached for the lifetime of the process (see module docstring). Returns
    an empty string if SOUL.md is missing so the system degrades gracefully.
    """
    if not _SOUL_PATH.exists():
        logger.warning(
            "SOUL.md not found at %s — agents will run without shared identity", _SOUL_PATH
        )
        return ""
    return soul_preamble_from_markdown(_SOUL_PATH.read_text(encoding="utf-8"))
