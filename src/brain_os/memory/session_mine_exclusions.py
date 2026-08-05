"""Miner eval/smoke exclusion filters.

Session-mine and the learning compiler both extract "lessons" from raw text
(Cursor session transcripts, or a compiler-input row's ``query``). Smoke
tests, gold-set fixtures, calibration probes, and eval-harness runs are not
real operator sessions — mining them pollutes the pending-memory queue and
the learning-promotion queue with junk candidates (e.g. a literal
``"How old are you today?"`` calibration probe becoming a "learned fact").

This module is the single source of truth for what counts as excluded so
``mine_session`` (session_miner.py) and ``learning_compiler._row_eligible``
apply the *same* filter instead of drifting apart.

Configurable via the ``BRAIN_SESSION_MINE_EXCLUSION_MARKERS`` env var
(comma-separated, appended to the defaults below) so an operator can widen
the filter without a code change or new Settings field.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

#: Literal markers / keywords that flag a transcript or query as smoke/eval
#: rather than a real operator session. Matched case-insensitively as a
#: substring check — cheap and dependency-free.
DEFAULT_EXCLUSION_MARKERS: tuple[str, ...] = (
    "[BRAIN_EMAIL]",
    "[CLEAR]",
    "[CLARIFY]",
    "Reply with exactly",
    "smoke test",
    "gold-set",
    "gold set",
    "calibration",
    "eval-harness",
    "eval harness",
    "How old are you today?",
)

_ENV_VAR = "BRAIN_SESSION_MINE_EXCLUSION_MARKERS"


def exclusion_markers(*, extra: Sequence[str] | None = None) -> tuple[str, ...]:
    """Return the effective marker list: defaults + env var + *extra*."""
    markers = list(DEFAULT_EXCLUSION_MARKERS)
    raw_env = os.environ.get(_ENV_VAR, "")
    if raw_env:
        markers.extend(part.strip() for part in raw_env.split(",") if part.strip())
    if extra:
        markers.extend(extra)
    return tuple(markers)


def find_exclusion_marker(
    text: str | None,
    *,
    markers: Sequence[str] | None = None,
) -> str | None:
    """Return the first configured marker found in *text* (case-insensitive), else ``None``."""
    if not text:
        return None
    haystack = text.lower()
    for marker in markers if markers is not None else exclusion_markers():
        if marker and marker.lower() in haystack:
            return marker
    return None


def exclusion_reason(
    text: str | None,
    *,
    markers: Sequence[str] | None = None,
) -> str | None:
    """Return ``excluded_marker:<marker>`` when *text* matches a smoke/eval marker, else ``None``."""
    marker = find_exclusion_marker(text, markers=markers)
    if marker is None:
        return None
    return f"excluded_marker:{marker}"


__all__ = [
    "DEFAULT_EXCLUSION_MARKERS",
    "exclusion_markers",
    "exclusion_reason",
    "find_exclusion_marker",
]
