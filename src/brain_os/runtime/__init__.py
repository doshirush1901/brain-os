"""Composition / CLI bootstrap root (not an HTTP or MCP interface).

``cli_runtime``, ``cli_shutdown``, and ``procedural_factory`` live here so
business layers can depend on pantheon/pipeline wiring without importing
``brain_os.interfaces`` (L3.1).
"""

from __future__ import annotations
