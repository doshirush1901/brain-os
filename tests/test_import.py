"""Smoke import for Brain OS overlay package."""

from __future__ import annotations

import brain_os


def test_import_brain_os_version() -> None:
    assert brain_os.__version__ == "0.1.0"
