"""CLI process shutdown — avoid interpreter hangs on leftover non-daemon threads.

``aiosqlite`` spins a non-daemon ``_connection_worker_thread`` per connection.
Pipeline bootstrap opens many long-lived SQLite handles (episodic, procedural,
corrections, …) that pantheon ``__aexit__`` does not close. After the CLI prints
its result, CPython joins those threads on exit and can sit for many minutes.

Fix: (1) patch aiosqlite so new workers are daemon from construction; (2) at CLI
exit, ``stop()`` any still-open connections found via ``gc`` and briefly join.
"""

from __future__ import annotations

import gc
import logging
import threading
from typing import Final

logger = logging.getLogger(__name__)

_PATCHED: Final[str] = "_ira_aiosqlite_daemon_patched"


def patch_aiosqlite_worker_threads_daemon() -> bool:
    """Make new aiosqlite connection workers daemon threads (idempotent).

    Must run before any ``await aiosqlite.connect(...)`` — daemon can only be
    set before ``Thread.start()``.
    """
    try:
        import aiosqlite.core as core
    except ImportError:  # pragma: no cover
        return False

    if getattr(core.Connection, _PATCHED, False):
        return True

    orig_init = core.Connection.__init__

    def _daemon_init(self: object, *args: object, **kwargs: object) -> None:
        orig_init(self, *args, **kwargs)  # type: ignore[misc]
        thread = getattr(self, "_thread", None)
        if isinstance(thread, threading.Thread):
            thread.daemon = True

    core.Connection.__init__ = _daemon_init  # type: ignore[method-assign]
    setattr(core.Connection, _PATCHED, True)
    logger.debug("aiosqlite Connection workers patched to daemon=True")
    return True


def _stop_open_aiosqlite_connections() -> int:
    """Best-effort ``Connection.stop()`` for leaked handles (no event loop needed)."""
    try:
        from aiosqlite.core import Connection
    except ImportError:  # pragma: no cover
        return 0

    stopped = 0
    for obj in gc.get_objects():
        # Avoid ``isinstance`` on exotic GC objects (e.g. torch) that warn/raise.
        if type(obj) is not Connection:
            continue
        thread = getattr(obj, "_thread", None)
        if not isinstance(thread, threading.Thread) or not thread.is_alive():
            continue
        try:
            obj.stop()
            stopped += 1
        except Exception:
            logger.debug("aiosqlite stop() failed during CLI shutdown", exc_info=True)
    return stopped


def release_cli_non_daemon_threads(*, join_timeout_s: float = 1.5) -> int:
    """Stop leftover aiosqlite workers so interpreter exit cannot block.

    Prefer closing connections at the source; this is the fail-closed exit hatch
    when pantheon/pipeline teardown left handles open (heartbeat drip, ask, …).

    Note: Python forbids flipping ``daemon`` on an already-started thread, so we
    stop workers via aiosqlite's sentinel instead of mutating daemon flags.
    """
    stopped = _stop_open_aiosqlite_connections()
    # Brief join so workers can drain the STOP sentinel before process exit.
    deadline_slices = max(1, int(float(join_timeout_s) / 0.1)) if join_timeout_s > 0 else 0
    remaining = 0
    for _ in range(deadline_slices):
        remaining = 0
        for thread in threading.enumerate():
            if thread is threading.main_thread() or thread.daemon:
                continue
            name = thread.name or ""
            if "_connection_worker_thread" not in name:
                continue
            remaining += 1
            try:
                thread.join(timeout=0.1)
            except RuntimeError:
                pass
        if remaining == 0:
            break
    if stopped or remaining:
        logger.debug(
            "CLI shutdown: stopped=%s aiosqlite connection(s); remaining workers=%s",
            stopped,
            remaining,
        )
    return stopped
