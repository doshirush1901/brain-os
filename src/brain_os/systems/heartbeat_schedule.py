"""Heartbeat schedule math, last-run persistence, and due-state helpers.

Extracted from ``heartbeat_runner`` so the orchestrator can stay under the
file-size grandfather budget (L0.1). Public symbols are re-exported from
``brain_os.systems.heartbeat_runner`` for API stability.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import fcntl
except ImportError:  # pragma: no cover — POSIX-only; heartbeat runs on Linux/macOS
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_LAST_RUN_KEY = "heartbeat:last_run:"
_LAST_RUN_FILE = "operations/heartbeat_last_run.json"

DEFAULT_CATCH_UP_WINDOW_HOURS = 6.0
_CATCH_UP_LATE_GRACE_SECONDS = 15 * 60
_DREAM_BUSY_SKIP_FILE = "operations/dream_busy_skip_slots.json"


def _last_run_file_path() -> Path:
    from brain_os.systems.data_dir_lock import get_data_dir

    return get_data_dir() / _LAST_RUN_FILE


def _read_last_run_file(job_name: str) -> float | None:
    path = _last_run_file_path()
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    val = raw.get(job_name)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _last_run_lock_path() -> Path:
    return _last_run_file_path().with_suffix(".lock")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* to *path* via a temp file + ``os.replace`` (atomic on POSIX)."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write_last_run_file(job_name: str, when: float | None = None) -> None:
    """Merge ``{job_name: timestamp}`` into the shared last-run JSON file.

    Multiple heartbeat processes (e.g. an overlapping ``dream_nightly`` run
    plus the regular heartbeat loop) can call this concurrently. The naive
    read-modify-write below was a lost-update race: process A reads the file,
    process B reads+writes a different key, then A's stale-payload write
    clobbers B's key — this is the reason ``dream_nightly`` intermittently
    disappeared from ``data/operations/heartbeat_last_run.json`` even though
    the job had actually run (mop-up item 5). We now hold an ``fcntl``
    advisory lock on a sibling ``.lock`` file around the read-modify-write,
    and write the payload atomically via a temp file + ``os.replace`` so a
    crash mid-write cannot leave a truncated/corrupt JSON file either.
    """
    path = _last_run_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _last_run_lock_path()
    lock_fh = None
    try:
        if fcntl is not None:
            lock_fh = open(lock_path, "w", encoding="utf-8")  # noqa: SIM115
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        payload: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError):
                payload = {}
        payload[job_name] = float(when if when is not None else time.time())
        try:
            _atomic_write_json(path, payload)
        except OSError:
            logger.warning("heartbeat last-run file write failed", exc_info=True)
    finally:
        if lock_fh is not None:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_fh.close()


_WEEKDAY_ALIASES: dict[str, int] = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


def _parse_at_hhmm(at: str) -> tuple[int, int] | None:
    text = (at or "").strip()
    if not text or ":" not in text:
        return None
    hour_s, minute_s = text.split(":", 1)
    try:
        hour = int(hour_s)
        minute = int(minute_s)
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _job_schedule_weekdays(job: dict[str, Any]) -> set[int] | None:
    """Weekday filter for wall-clock jobs; ``None`` = every day."""
    raw = job.get("days")
    if not isinstance(raw, list):
        return None
    out: set[int] = set()
    for item in raw:
        key = str(item or "").strip().lower()[:3]
        if key in _WEEKDAY_ALIASES:
            out.add(_WEEKDAY_ALIASES[key])
    return out or None


def _job_trailing_window_days(job: dict[str, Any], default: int) -> int:
    """Trailing report window (int); distinct from weekday ``days`` list."""
    for key in ("window_days", "report_days"):
        raw = job.get(key)
        if isinstance(raw, (int, float)):
            return max(1, int(raw))
    raw_days = job.get("days")
    if isinstance(raw_days, (int, float)):
        return max(1, int(raw_days))
    return default


def _now_in_tz(tz_name: str, now: datetime | None) -> datetime:
    tz = ZoneInfo((tz_name or "UTC").strip() or "UTC")
    if now is None:
        return datetime.now(tz)
    if now.tzinfo is None:
        return now.replace(tzinfo=tz)
    return now.astimezone(tz)


def _wall_clock_fire_on_date(job: dict[str, Any], day: date) -> datetime | None:
    """Fire datetime for ``day`` in job tz when ``at`` parses and weekday is allowed."""
    at_raw = job.get("at")
    if not at_raw:
        return None
    parsed = _parse_at_hhmm(str(at_raw))
    if parsed is None:
        return None
    hour, minute = parsed
    allowed = _job_schedule_weekdays(job)
    if allowed is not None and day.weekday() not in allowed:
        return None
    tz = ZoneInfo(str(job.get("tz") or "UTC").strip() or "UTC")
    return datetime(day.year, day.month, day.day, hour, minute, 0, 0, tzinfo=tz)


def current_wall_clock_fire_time(
    job: dict[str, Any],
    *,
    now: datetime | None = None,
) -> datetime | None:
    """Today's fire slot in job tz when allowed weekday and ``now`` is past ``at``."""
    moment = _now_in_tz(str(job.get("tz") or "UTC"), now)
    fire = _wall_clock_fire_on_date(job, moment.date())
    if fire is None:
        return None
    if moment < fire:
        return None
    return fire


def _wall_clock_slot_unsatisfied(
    fire: datetime,
    last_run_unix: float | None,
) -> bool:
    """True when ``last_run`` has not satisfied this local fire slot.

    Only a run *at or after* ``fire`` satisfies this slot. The previous
    same-calendar-day shortcut (``last_dt.date() >= fire.date()``) treated
    *any* run earlier the same day as satisfying a later same-day fire time
    — e.g. an early-morning run would wrongly mark an evening
    ``warm_lane_daily`` slot as already done, so it never fired. Comparing
    the timestamps directly still prevents re-firing later the same day
    (once ``last_dt >= fire``) while correctly leaving a not-yet-reached
    same-day slot unsatisfied.
    """
    if last_run_unix is None:
        return True
    try:
        last = float(last_run_unix)
    except (TypeError, ValueError):
        return True
    last_dt = datetime.fromtimestamp(last, tz=fire.tzinfo)
    return last_dt < fire


def missed_wall_clock_fire(
    job: dict[str, Any],
    last_run_unix: float | None,
    *,
    now: datetime | None = None,
    catch_up_window_hours: float = DEFAULT_CATCH_UP_WINDOW_HOURS,
) -> datetime | None:
    """Most recent unsatisfied ``at`` fire within the catch-up window.

    Considers today's fire (once ``now`` is past ``at``) and yesterday's fire
    only when its age is still within ``catch_up_window_hours`` (lid-close /
    daemon downtime). No unbounded multi-day backfill.
    """
    moment = _now_in_tz(str(job.get("tz") or "UTC"), now)
    window_s = max(0.0, float(catch_up_window_hours)) * 3600.0
    candidates: list[datetime] = []

    today_fire = _wall_clock_fire_on_date(job, moment.date())
    if today_fire is not None and moment >= today_fire:
        candidates.append(today_fire)

    yesterday = moment.date() - timedelta(days=1)
    yest_fire = _wall_clock_fire_on_date(job, yesterday)
    if yest_fire is not None:
        age = (moment - yest_fire).total_seconds()
        if 0.0 <= age <= window_s:
            candidates.append(yest_fire)

    unsatisfied = [f for f in candidates if _wall_clock_slot_unsatisfied(f, last_run_unix)]
    if not unsatisfied:
        return None
    return max(unsatisfied)


def wall_clock_due_info(
    job: dict[str, Any],
    last_run_unix: float | None,
    *,
    now: datetime | None = None,
    force: bool = False,
    catch_up_window_hours: float = DEFAULT_CATCH_UP_WINDOW_HOURS,
) -> tuple[bool, bool, datetime | None]:
    """Return ``(due, catch_up, fire)`` for a wall-clock job.

    ``catch_up`` is True when the fire is more than 15 minutes late or is a
    previous calendar day's slot recovered inside the catch-up window.
    """
    if force:
        return True, False, current_wall_clock_fire_time(job, now=now)
    fire = missed_wall_clock_fire(
        job,
        last_run_unix,
        now=now,
        catch_up_window_hours=catch_up_window_hours,
    )
    if fire is None:
        return False, False, None
    moment = _now_in_tz(str(job.get("tz") or "UTC"), now)
    late = (moment - fire).total_seconds() > _CATCH_UP_LATE_GRACE_SECONDS
    cross_day = fire.date() < moment.date()
    return True, bool(late or cross_day), fire


def is_wall_clock_job_due(
    job: dict[str, Any],
    last_run_unix: float | None,
    *,
    now: datetime | None = None,
    force: bool = False,
    catch_up_window_hours: float = DEFAULT_CATCH_UP_WINDOW_HOURS,
) -> bool:
    """True when a wall-clock (``at``/``tz``) job is due to fire.

    Firing rules (mop-up item 5 — documents the ``dream_nightly`` timing
    investigation):

    * The job fires only *after* its local ``at`` time has passed (in ``tz``,
      respecting an optional ``days`` weekday allow-list) — see
      :func:`missed_wall_clock_fire`. There is no "fires at exactly HH:MM";
      it is "due any time on/after HH:MM until the next run records a
      ``last_run`` at/after that fire moment".
    * **Same-day catch-up**: if the heartbeat process was down (or the job
      was skipped) past ``at``, the very next heartbeat tick still sees
      ``fire`` in the past and ``last_run < fire`` — so it runs once, late,
      the same day. Late by more than 15 minutes tags ``catch_up=true``.
    * **Cross-midnight catch-up**: yesterday's fire is due only while
      ``now - fire <= catch_up_window_hours`` (default 6h). No unbounded
      multi-day backfill.
    * Observed pre-22:00 IST fires for ``dream_nightly`` (e.g. 20:59, 21:05)
      were **not** a timezone bug — they were caused by (a) multiple
      concurrent heartbeat processes racing to read/write
      ``heartbeat_last_run.json`` (see ``_write_last_run_file``'s lost-update
      race, now fixed with an ``fcntl`` lock + atomic write) and/or (b)
      ``force=True`` test/operator runs writing a ``last_run`` that looked
      like a legitimate wall-clock fire. The once-per-calendar-day guard
      is a second, independent safety net: even if a stale/missing
      ``last_run`` slips through, a job cannot fire twice for the same local
      calendar date.
    """
    due, _catch_up, _fire = wall_clock_due_info(
        job,
        last_run_unix,
        now=now,
        force=force,
        catch_up_window_hours=catch_up_window_hours,
    )
    return due


def is_interval_job_due(
    last_run_unix: float | None,
    interval_seconds: float,
    *,
    force: bool = False,
) -> bool:
    """Return whether an interval has elapsed since the previous run."""
    if force:
        return True
    if last_run_unix is None:
        return True
    try:
        last = float(last_run_unix)
    except (TypeError, ValueError):
        return True
    return (time.time() - last) >= interval_seconds


def load_heartbeat_jobs(jobs_path: str | Path) -> list[dict[str, Any]]:
    path = Path(jobs_path)
    if not path.is_file():
        logger.warning("Heartbeat jobs file missing: %s", path)
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Heartbeat jobs invalid JSON (%s): %s", path, exc)
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            out.append(item)
    return out


def _interval_seconds(job: dict[str, Any]) -> float:
    try:
        h = float(job.get("interval_hours", 24))
    except (TypeError, ValueError):
        h = 24.0
    return max(60.0, h * 3600.0)


async def _fetch_last_run_unix(redis: Any, job_name: str) -> float | None:
    """Prefer Redis; fall back to on-disk last-run so Redis outages cannot silence jobs."""
    if redis is not None and getattr(redis, "available", False):
        key = f"{_LAST_RUN_KEY}{job_name}"
        raw = await redis.get(key)
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    return _read_last_run_file(job_name)


async def _due_state(
    redis: Any,
    *,
    job: dict[str, Any],
    job_name: str,
    interval_seconds: float,
    force: bool,
    now: datetime | None = None,
    catch_up_window_hours: float = DEFAULT_CATCH_UP_WINDOW_HOURS,
) -> tuple[bool, bool]:
    """Return ``(due, catch_up)`` for interval or wall-clock jobs."""
    if force:
        return True, False
    last = await _fetch_last_run_unix(redis, job_name)
    if job.get("at"):
        due, catch_up, _fire = wall_clock_due_info(
            job,
            last,
            now=now,
            force=False,
            catch_up_window_hours=catch_up_window_hours,
        )
        return due, catch_up
    return is_interval_job_due(last, interval_seconds, force=False), False


async def _is_due(
    redis: Any,
    *,
    job: dict[str, Any],
    job_name: str,
    interval_seconds: float,
    force: bool,
    now: datetime | None = None,
    catch_up_window_hours: float = DEFAULT_CATCH_UP_WINDOW_HOURS,
) -> bool:
    due, _catch_up = await _due_state(
        redis,
        job=job,
        job_name=job_name,
        interval_seconds=interval_seconds,
        force=force,
        now=now,
        catch_up_window_hours=catch_up_window_hours,
    )
    return due


def _dream_busy_dedupe_path() -> Path:
    from brain_os.systems.data_dir_lock import get_data_dir

    return get_data_dir() / _DREAM_BUSY_SKIP_FILE


def _dream_busy_slot_key(
    job_name: str,
    fire: datetime | None,
    now: datetime | None,
) -> str:
    if fire is not None:
        day = fire.date()
    elif now is not None:
        day = now.date() if now.tzinfo is not None else now.replace(tzinfo=UTC).date()
    else:
        day = datetime.now(UTC).date()
    return f"{job_name}:{day.isoformat()}"


def _should_log_dream_busy_skip(
    job_name: str,
    fire: datetime | None,
    now: datetime | None,
) -> bool:
    """Persist slot key; return False if this busy-skip was already logged."""
    path = _dream_busy_dedupe_path()
    key = _dream_busy_slot_key(job_name, fire, now)
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}
    if key in payload:
        return False
    stamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload[key] = stamp
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, payload)
    except OSError:
        logger.debug("dream busy skip dedupe write failed", exc_info=True)
    return True


async def _mark_ran(redis: Any, job_name: str) -> None:
    now = time.time()
    if redis is not None and getattr(redis, "available", False):
        key = f"{_LAST_RUN_KEY}{job_name}"
        await redis.set(key, str(now))
    _write_last_run_file(job_name, now)
