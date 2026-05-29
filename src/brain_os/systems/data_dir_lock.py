"""Single-instance lock on the Brain OS data directory."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR_LOCK_USER_HINT = (
    "What to do: (1) If the API server is running, stop it before using "
    "`brain health` / `brain ask`. (2) Or set BRAIN_DATA_DIR to a separate directory "
    "for a second process."
)


def get_data_dir() -> Path:
    """Resolve data root: BRAIN_DATA_DIR, legacy BRAIN_DATA_DIR, or ./data."""
    raw = os.environ.get("BRAIN_DATA_DIR", "").strip()
    if not raw:
        raw = os.environ.get("BRAIN_DATA_DIR", "").strip()
    if raw:
        return Path(raw).resolve()
    return (Path.cwd() / "data").resolve()


def coerce_config_path(raw: object, *, default: str = "") -> str:
    if raw is None or raw == "":
        return default
    if not isinstance(raw, (str, Path)):
        raise ValueError(f"config path must be str or Path, got {type(raw).__name__!r}")
    val = str(raw).strip()
    if "MagicMock" in val or val.startswith("mock.") or "().app." in val:
        raise ValueError(f"invalid config path (mock leaked): {val!r}")
    return val or default


def _lock_path() -> Path:
    return get_data_dir() / ".brain.lock"


def _make_lock(timeout: float = 10.0):
    lock_path = _lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from filelock import FileLock, Timeout
    except ImportError:
        return None, None
    return FileLock(str(lock_path) + ".lock", timeout=timeout), Timeout


@contextmanager
def data_dir_lock(timeout: float = 10.0):
    lock, Timeout = _make_lock(timeout)
    if lock is None:
        logger.warning(
            "filelock not installed — skipping data-dir single-instance lock. "
            "Install with: pip install filelock"
        )
        yield
        return

    try:
        lock.acquire()
        logger.debug("Acquired data-dir lock at %s", _lock_path())
        yield
    except Timeout:
        lp = _lock_path()
        raise RuntimeError(
            "Another Brain OS process is using this data directory. "
            f"Lock file: {lp}\n{_DATA_DIR_LOCK_USER_HINT}"
        ) from None
    finally:
        try:
            lock.release()
            logger.debug("Released data-dir lock")
        except OSError:
            pass


@asynccontextmanager
async def async_data_dir_lock(timeout: float = 10.0):
    lock, Timeout = _make_lock(timeout)
    if lock is None:
        logger.warning("filelock not installed — skipping data-dir single-instance lock.")
        yield
        return

    try:
        await asyncio.to_thread(lock.acquire)
        logger.debug("Acquired data-dir lock at %s (server)", _lock_path())
        yield
    except Timeout:
        lp = _lock_path()
        raise RuntimeError(
            "Another Brain OS process is using this data directory. "
            f"Lock file: {lp}\n{_DATA_DIR_LOCK_USER_HINT}"
        ) from None
    finally:
        try:
            await asyncio.to_thread(lock.release)
            logger.debug("Released data-dir lock (server)")
        except OSError:
            pass
