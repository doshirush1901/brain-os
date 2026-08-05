"""Canonical data-root helper (MEMBRANE LAW / FB-2 coordination).

Prefer :func:`data_root` (exported from :mod:`brain_os.config`) over hard-coded
``Path("data/...")`` so pytest can redirect via ``BRAIN_DATA_DIR``.

Lives in ``data_paths.py`` so ``from brain_os.config import data_root`` is the
function, not a submodule.
"""

from __future__ import annotations

from pathlib import Path

from brain_os.systems.data_dir_lock import get_data_dir


def data_root() -> Path:
    """Absolute path to Brain OS's mutable data directory."""
    return get_data_dir()


def resolve_under_data_root(raw: str, *, default: str = "") -> Path:
    """Resolve a path that may be absolute or ``data/``-prefixed relative."""
    text = (raw or default).strip() or default
    if not text:
        return data_root()
    if text.startswith("/"):
        return Path(text)
    rel = text.removeprefix("data/").lstrip("/")
    return data_root() / rel


def knowledge_path(filename: str) -> Path:
    """``data_root()/knowledge/<filename>`` (FB-2 / MEMBRANE-safe)."""
    name = (filename or "").strip().removeprefix("data/knowledge/").lstrip("/")
    return data_root() / "knowledge" / name


__all__ = ["data_root", "knowledge_path", "resolve_under_data_root"]
