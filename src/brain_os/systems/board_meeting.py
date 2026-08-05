"""Board meeting — compatibility shim to v2 decision instrument.

Prefer ``brain_os.systems.board_meeting_v2`` for new call sites.
"""

from __future__ import annotations

from brain_os.systems.board_meeting_v2 import (
    BoardMeeting,
    athena_decide,
    deliver_board_meeting_email,
    format_board_markdown,
    nemesis_dissent,
    run_board_meeting_v2,
    vera_fact_check,
)

__all__ = [
    "BoardMeeting",
    "athena_decide",
    "deliver_board_meeting_email",
    "format_board_markdown",
    "nemesis_dissent",
    "run_board_meeting_v2",
    "vera_fact_check",
]
