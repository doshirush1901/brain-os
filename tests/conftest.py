"""Pytest bootstrap: import overlay without requiring ``poetry install``."""

from __future__ import annotations

import sys
from pathlib import Path

_OVERLAY_SRC = Path(__file__).resolve().parents[1] / "overlay" / "src"
if _OVERLAY_SRC.is_dir():
    overlay_str = str(_OVERLAY_SRC)
    if overlay_str not in sys.path:
        sys.path.insert(0, overlay_str)
