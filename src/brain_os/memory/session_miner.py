"""Per-session mining for Graphe / Cursor sessions.

Replaces dream stage 12's legacy ``SELECT … LIMIT 50`` single-summary window with
one cheap-model extraction per operator session: lessons, corrections, preferences,
procedures, and open loops land in ``learning_meta_json`` and the pending-memory queue.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import random
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.agents.graphe import _DB_PATH
from brain_os.config import get_settings
from brain_os.memory.pending_memory_queue import PendingMemoryQueue
from brain_os.memory.session_mine_exclusions import exclusion_reason
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import SessionMineExtraction

logger = logging.getLogger(__name__)

SESSION_MINE_KEY = "session_mine"
DEFAULT_QUEUE_PATH = Path("data/brain/session_mine_queue.jsonl")
#: Permanent failures (orphans / garbage) — never re-queued.
_DEAD_LETTER_REASONS = frozenset(
    {"session_not_found", "invalid_json", "missing_session_id", "empty_session_id"}
)
_MAX_QUEUE_ATTEMPTS = 3

_SYSTEM_PROMPT = load_prompt("session_miner")


def session_mine_queue_path() -> Path:
    raw = getattr(get_settings().app, "session_mine_queue_path", None)
    return Path(raw) if raw else DEFAULT_QUEUE_PATH


def session_mine_dead_letter_path(queue_path: Path | None = None) -> Path:
    """Sibling dead-letter file for unresolvable queue entries."""
    path = queue_path or session_mine_queue_path()
    return path.with_name(path.stem + "_dead.jsonl")


def _queue_lock_path(queue_path: Path) -> Path:
    return queue_path.with_suffix(queue_path.suffix + ".lock")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, str(path))
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _append_dead_letter(entry: dict[str, Any], *, queue_path: Path) -> None:
    dead = session_mine_dead_letter_path(queue_path)
    dead.parent.mkdir(parents=True, exist_ok=True)
    with dead.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _parse_learning_meta(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _merge_meta(existing: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(existing)
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        elif key == SESSION_MINE_KEY and isinstance(value, dict) and isinstance(out.get(key), dict):
            merged_sm = dict(out[key])
            merged_sm.update(value)
            out[key] = merged_sm
        else:
            out[key] = value
    return out


def is_mined_meta(meta: dict[str, Any] | None) -> bool:
    """True when ``session_mine.mined_at`` is present in learning metadata."""
    if not meta:
        return False
    blob = meta.get(SESSION_MINE_KEY)
    if not isinstance(blob, dict):
        return False
    mined_at = blob.get("mined_at")
    return bool(mined_at)


def _group_key_for_row(row: dict[str, Any]) -> str:
    run_id = (row.get("run_id") or "").strip()
    if run_id:
        return run_id
    return str(row.get("id") or "")


def enqueue_session_mine(
    session_id: str,
    *,
    source: str = "unknown",
    queue_path: Path | None = None,
) -> dict[str, Any]:
    """Append a mining request to the JSONL close-queue."""
    sid = (session_id or "").strip()
    if not sid:
        return {"enqueued": False, "reason": "empty_session_id", "session_id": sid}

    path = queue_path or session_mine_queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": sid,
        "enqueued_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": (source or "unknown")[:64],
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")
    return {"enqueued": True, "session_id": sid, "queue_path": str(path), **payload}


def load_agent_transcript_text(session_id: str) -> str:
    """Best-effort load of Cursor agent transcript JSONL for *session_id*."""
    sid = (session_id or "").strip()
    if not sid:
        return ""

    candidates: list[Path] = []
    env_dir = os.environ.get("CURSOR_AGENT_TRANSCRIPTS_DIR", "").strip()
    if env_dir:
        base = Path(env_dir).expanduser()
        candidates.extend(
            [
                base / sid / f"{sid}.jsonl",
                base / f"{sid}.jsonl",
            ]
        )

    home = Path.home()
    projects_root = home / ".cursor" / "projects"
    if projects_root.is_dir():
        for project_dir in projects_root.iterdir():
            if not project_dir.is_dir():
                continue
            transcript = project_dir / "agent-transcripts" / sid / f"{sid}.jsonl"
            candidates.append(transcript)

    for path in candidates:
        if not path.is_file():
            continue
        try:
            return _flatten_agent_transcript(path)
        except OSError:
            logger.debug("agent transcript read failed path=%s", path, exc_info=True)
    return ""


def _flatten_agent_transcript(path: Path) -> str:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        role = str(obj.get("role") or "unknown")
        message = obj.get("message") or {}
        parts: list[str] = []
        if isinstance(message, dict):
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = str(block.get("text") or "").strip()
                    if text:
                        parts.append(text)
        elif isinstance(message, str):
            parts.append(message.strip())
        if parts:
            lines.append(f"{role}: " + "\n".join(parts))
    return "\n".join(lines)


async def _fetch_rows_for_lookup(
    session_id: str,
    *,
    db_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    sid = (session_id or "").strip()
    if not sid or not db_path.is_file():
        return sid, []

    async with aiosqlite.connect(str(db_path), timeout=30.0) as db:
        await db.execute("PRAGMA busy_timeout=30000")
        cur = await db.execute("SELECT * FROM cursor_sessions WHERE id = ?", (sid,))
        row = await cur.fetchone()
        cols = [d[0] for d in cur.description] if cur.description else []
        await cur.close()

        anchor: dict[str, Any] | None = None
        if row is not None and cols:
            anchor = dict(zip(cols, row))
        else:
            cur = await db.execute(
                "SELECT * FROM cursor_sessions WHERE run_id = ? ORDER BY timestamp ASC",
                (sid,),
            )
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            await cur.close()
            if rows and cols:
                anchor = dict(zip(cols, rows[0]))

        if anchor is None:
            return sid, []

        group_key = _group_key_for_row(anchor)
        run_id = (anchor.get("run_id") or "").strip()
        if run_id:
            cur = await db.execute(
                "SELECT * FROM cursor_sessions WHERE run_id = ? ORDER BY timestamp ASC",
                (run_id,),
            )
        else:
            cur = await db.execute(
                "SELECT * FROM cursor_sessions WHERE id = ? ORDER BY timestamp ASC",
                (anchor.get("id"),),
            )
        rows = await cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        await cur.close()
        group = [dict(zip(cols, r)) for r in rows] if rows and cols else [anchor]
        return group_key, group


def _build_transcript_text(
    rows: list[dict[str, Any]],
    *,
    agent_transcript: str = "",
) -> str:
    parts: list[str] = []
    for row in sorted(rows, key=lambda r: float(r.get("timestamp") or 0.0)):
        q = str(row.get("query") or "").strip()
        a = str(row.get("response_summary") or "").strip()
        agents = row.get("agents_used") or "[]"
        parts.append(f"Turn timestamp={row.get('timestamp')}")
        parts.append(f"Q: {q}")
        parts.append(f"A: {a}")
        parts.append(f"agents_used: {agents}")
        parts.append("")
    if agent_transcript.strip():
        parts.append("--- Cursor agent transcript (best-effort) ---")
        parts.append(agent_transcript.strip())
    return "\n".join(parts).strip()


def _session_mine_blob(extraction: SessionMineExtraction, *, source: str) -> dict[str, Any]:
    return {
        "mined_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": (source or "session_miner")[:64],
        "summary": extraction.summary,
        "lessons": list(extraction.lessons or []),
        "corrections": list(extraction.corrections or []),
        "preferences": list(extraction.preferences or []),
        "procedures": list(extraction.procedures or []),
        "open_loops": list(extraction.open_loops or []),
    }


async def _update_group_learning_meta(
    rows: list[dict[str, Any]],
    *,
    db_path: Path,
    patch: dict[str, Any],
) -> int:
    updated = 0
    async with aiosqlite.connect(str(db_path), timeout=30.0) as db:
        await db.execute("PRAGMA busy_timeout=30000")
        for row in rows:
            row_id = row.get("id")
            if not row_id:
                continue
            existing = _parse_learning_meta(row.get("learning_meta_json"))
            merged = _merge_meta(existing, patch)
            lm_json = json.dumps(merged, default=str)
            cur = await db.execute(
                "UPDATE cursor_sessions SET learning_meta_json = ? WHERE id = ?",
                (lm_json, row_id),
            )
            if cur.rowcount:
                updated += 1
            await cur.close()
        await db.commit()
    return updated


async def _push_extraction_to_pending(
    extraction: SessionMineExtraction,
    session_id: str,
    *,
    enqueue_pending: bool,
) -> dict[str, int]:
    if not enqueue_pending:
        return {"pushed": 0}

    sid = (session_id or "").strip()
    tag = f"[session_mine:{sid}]"
    source = f"session_mine:{sid}"[:80]
    q = PendingMemoryQueue()
    await q.initialize()

    pushed = 0
    buckets: list[tuple[str, list[str]]] = [
        ("lesson", list(extraction.lessons or [])),
        ("correction", list(extraction.corrections or [])),
        ("preference", list(extraction.preferences or [])),
        ("procedure", list(extraction.procedures or [])),
        ("open_loop", list(extraction.open_loops or [])),
    ]
    for label, items in buckets:
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            body = f"{tag} {label}: {text}"
            await q.enqueue(body, source=source)
            pushed += 1
    if extraction.summary.strip():
        await q.enqueue(f"{tag} summary: {extraction.summary.strip()}", source=source)
        pushed += 1
    return {"pushed": pushed}


async def mine_session(
    session_id: str,
    *,
    db_path: Path | None = None,
    llm: Any | None = None,
    enqueue_pending: bool = True,
    force: bool = False,
    source: str = "session_miner",
    use_agent_transcript: bool = True,
) -> dict[str, Any]:
    """Mine one session group (by Graphe row id or Cursor ``run_id``)."""
    path = db_path or _DB_PATH
    sid_lookup = (session_id or "").strip()
    if not sid_lookup:
        return {"status": "error", "reason": "empty_session_id", "session_id": sid_lookup}

    group_key, rows = await _fetch_rows_for_lookup(sid_lookup, db_path=path)
    if not rows:
        logger.error(
            "SESSION_MINE | session_not_found session_id=%s (orphan queue id / never logged)",
            sid_lookup,
        )
        return {
            "status": "error",
            "reason": "session_not_found",
            "session_id": sid_lookup,
            "group_key": group_key,
        }

    representative = rows[-1]
    existing_meta = _parse_learning_meta(representative.get("learning_meta_json"))
    if not force and is_mined_meta(existing_meta):
        return {
            "status": "skipped",
            "reason": "already_mined",
            "session_id": sid_lookup,
            "group_key": group_key,
            "row_count": len(rows),
        }

    agent_text = load_agent_transcript_text(group_key) if use_agent_transcript else ""
    transcript = _build_transcript_text(rows, agent_transcript=agent_text)
    if not transcript.strip():
        return {
            "status": "error",
            "reason": "empty_transcript",
            "session_id": sid_lookup,
            "group_key": group_key,
        }

    # Skip smoke/eval/gold-set/calibration sessions — mining them pollutes
    # the pending-memory queue with fixture junk. See session_mine_exclusions.py.
    excluded = exclusion_reason(transcript)
    if excluded:
        return {
            "status": "skipped",
            "reason": excluded,
            "session_id": sid_lookup,
            "group_key": group_key,
            "row_count": len(rows),
        }

    if llm is None:
        from brain_os.services.llm_client import get_llm_client

        llm = get_llm_client()

    extraction = await llm.generate_structured(
        system=_SYSTEM_PROMPT,
        user=transcript[:120_000],
        response_model=SessionMineExtraction,
        model_profile="fast",
        temperature=0.1,
        max_tokens=2000,
        name="session_miner",
        session_id=group_key,
    )

    mine_blob = _session_mine_blob(extraction, source=source)
    patch = {SESSION_MINE_KEY: mine_blob}
    updated = await _update_group_learning_meta(rows, db_path=path, patch=patch)
    pending_stats = await _push_extraction_to_pending(
        extraction,
        group_key,
        enqueue_pending=enqueue_pending,
    )

    return {
        "status": "ok",
        "session_id": sid_lookup,
        "group_key": group_key,
        "row_count": len(rows),
        "rows_updated": updated,
        "pending": pending_stats,
        "summary": extraction.summary,
        "counts": {
            "lessons": len(extraction.lessons or []),
            "corrections": len(extraction.corrections or []),
            "preferences": len(extraction.preferences or []),
            "procedures": len(extraction.procedures or []),
            "open_loops": len(extraction.open_loops or []),
        },
    }


async def list_unmined_groups(
    *,
    db_path: Path | None = None,
    since_days: float | None = None,
    until_days_ago: float | None = None,
) -> list[dict[str, Any]]:
    """Return unmined session groups oldest-first."""
    path = db_path or _DB_PATH
    if not path.is_file():
        return []

    now = time.time()
    min_ts: float | None = None
    max_ts: float | None = None
    if since_days is not None:
        min_ts = now - float(since_days) * 86400.0
    if until_days_ago is not None:
        max_ts = now - float(until_days_ago) * 86400.0

    async with aiosqlite.connect(str(path), timeout=30.0) as db:
        await db.execute("PRAGMA busy_timeout=30000")
        query = "SELECT * FROM cursor_sessions"
        clauses: list[str] = []
        params: list[Any] = []
        if min_ts is not None:
            clauses.append("timestamp >= ?")
            params.append(min_ts)
        if max_ts is not None:
            clauses.append("timestamp <= ?")
            params.append(max_ts)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp ASC"
        cur = await db.execute(query, params)
        rows = await cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        await cur.close()

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        item = dict(zip(cols, row))
        key = _group_key_for_row(item)
        if not key:
            continue
        grouped.setdefault(key, []).append(item)

    out: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        rep = group_rows[-1]
        meta = _parse_learning_meta(rep.get("learning_meta_json"))
        if is_mined_meta(meta):
            continue
        out.append(
            {
                "group_key": key,
                "session_id": key,
                "timestamp": float(rep.get("timestamp") or 0.0),
                "row_count": len(group_rows),
                "latest_row_id": rep.get("id"),
            }
        )
    out.sort(key=lambda g: g["timestamp"])
    return out


async def mine_unmined(
    *,
    limit: int = 25,
    db_path: Path | None = None,
    llm: Any | None = None,
    enqueue_pending: bool = True,
    since_days: float | None = None,
    until_days_ago: float | None = None,
    sample_fraction: float | None = None,
    sleep_seconds: float = 0.0,
    source: str = "mine_unmined",
    force: bool = False,
    use_agent_transcript: bool = True,
) -> dict[str, Any]:
    """Mine up to *limit* unmined session groups oldest-first."""
    groups = await list_unmined_groups(
        db_path=db_path,
        since_days=since_days,
        until_days_ago=until_days_ago,
    )
    if sample_fraction is not None and groups:
        frac = max(0.0, min(1.0, float(sample_fraction)))
        if frac < 1.0:
            random.shuffle(groups)
            keep = max(1, int(len(groups) * frac)) if frac > 0 else 0
            groups = groups[:keep]

    summary = {"processed": 0, "ok": 0, "skipped": 0, "errors": 0, "results": []}
    for group in groups[: max(1, int(limit))]:
        summary["processed"] += 1
        try:
            result = await mine_session(
                str(group["session_id"]),
                db_path=db_path,
                llm=llm,
                enqueue_pending=enqueue_pending,
                force=force,
                source=source,
                use_agent_transcript=use_agent_transcript,
            )
        except Exception as exc:
            logger.exception("mine_unmined failed session_id=%s", group.get("session_id"))
            summary["errors"] += 1
            summary["results"].append(
                {
                    "session_id": group.get("session_id"),
                    "status": "error",
                    "error": str(exc),
                }
            )
        else:
            status = str(result.get("status") or "error")
            if status == "ok":
                summary["ok"] += 1
            elif status == "skipped":
                summary["skipped"] += 1
            else:
                summary["errors"] += 1
            summary["results"].append(result)
        if sleep_seconds > 0:
            await asyncio.sleep(float(sleep_seconds))
    return summary


async def drain_mine_queue(
    *,
    limit: int = 25,
    db_path: Path | None = None,
    llm: Any | None = None,
    enqueue_pending: bool = True,
    queue_path: Path | None = None,
    source: str = "drain_mine_queue",
    force: bool = False,
    use_agent_transcript: bool = True,
) -> dict[str, Any]:
    """Process close-queue JSONL entries via :func:`mine_session`.

    Holds an exclusive flock around the read-modify-write so concurrent
    heartbeat / dream / CLI drains cannot resurrect drained lines. Terminal
    failures (``session_not_found``, malformed JSON, …) and entries that
    exceed ``_MAX_QUEUE_ATTEMPTS`` are dead-lettered — never silently
    re-queued forever.
    """
    path = queue_path or session_mine_queue_path()
    empty: dict[str, Any] = {
        "processed": 0,
        "ok": 0,
        "skipped": 0,
        "errors": 0,
        "drained": 0,
        "dead_lettered": 0,
        "requeued": 0,
        "remaining": 0,
        "results": [],
    }
    if not path.is_file():
        return empty

    # Claim under lock (short critical section) — never hold flock across LLM.
    def _claim() -> list[str]:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return []
        claimed = lines[: max(1, int(limit))]
        rest = lines[len(claimed) :]
        _atomic_write_text(path, "\n".join(rest) + ("\n" if rest else ""))
        return claimed

    lock_path = _queue_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            to_process = _claim()
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)

    if not to_process:
        return empty

    summary: dict[str, Any] = {**empty, "results": []}
    requeue_lines: list[str] = []

    def _dead_letter(entry: dict[str, Any]) -> None:
        entry = {
            **entry,
            "dead_lettered_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        _append_dead_letter(entry, queue_path=path)
        summary["dead_lettered"] += 1
        summary["errors"] += 1
        logger.error(
            "SESSION_MINE_QUEUE | dead-letter session_id=%s reason=%s attempts=%s",
            entry.get("session_id"),
            entry.get("reason") or entry.get("error"),
            entry.get("attempts"),
        )
        summary["results"].append({**entry, "status": "error", "dead_lettered": True})

    def _requeue(payload: dict[str, Any], *, reason: str) -> None:
        attempts = int(payload.get("attempts") or 0) + 1
        payload = {**payload, "attempts": attempts, "last_error": reason}
        if attempts >= _MAX_QUEUE_ATTEMPTS or reason in _DEAD_LETTER_REASONS:
            _dead_letter(
                {
                    **payload,
                    "reason": reason,
                    "session_id": payload.get("session_id"),
                }
            )
            return
        requeue_lines.append(json.dumps(payload, default=str))
        summary["requeued"] += 1
        summary["errors"] += 1
        logger.warning(
            "SESSION_MINE_QUEUE | requeued session_id=%s reason=%s attempts=%s/%s",
            payload.get("session_id"),
            reason,
            attempts,
            _MAX_QUEUE_ATTEMPTS,
        )
        summary["results"].append(
            {
                "session_id": payload.get("session_id"),
                "status": "error",
                "reason": reason,
                "attempts": attempts,
                "requeued": True,
            }
        )

    for raw in to_process:
        summary["processed"] += 1
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            _dead_letter({"reason": "invalid_json", "raw": raw[:200], "attempts": 1})
            continue
        if not isinstance(payload, dict):
            _dead_letter({"reason": "invalid_json", "raw": raw[:200], "attempts": 1})
            continue
        sid = str(payload.get("session_id") or "").strip()
        if not sid:
            _dead_letter({**payload, "reason": "missing_session_id", "attempts": 1})
            continue
        item_source = str(payload.get("source") or source)
        try:
            result = await mine_session(
                sid,
                db_path=db_path,
                llm=llm,
                enqueue_pending=enqueue_pending,
                force=force,
                source=item_source,
                use_agent_transcript=use_agent_transcript,
            )
        except Exception as exc:
            logger.exception("SESSION_MINE_QUEUE | exception session_id=%s", sid)
            _requeue(payload, reason=f"exception:{type(exc).__name__}:{exc}"[:200])
            continue

        status = str(result.get("status") or "error")
        reason = str(result.get("reason") or status)
        if status == "ok":
            summary["ok"] += 1
            summary["drained"] += 1
            summary["results"].append(result)
        elif status == "skipped":
            summary["skipped"] += 1
            summary["drained"] += 1
            summary["results"].append(result)
        else:
            logger.error(
                "SESSION_MINE_QUEUE | mine failed session_id=%s reason=%s",
                sid,
                reason,
            )
            if reason in _DEAD_LETTER_REASONS:
                attempts = int(payload.get("attempts") or 0) + 1
                _dead_letter(
                    {
                        **payload,
                        "reason": reason,
                        "session_id": sid,
                        "attempts": attempts,
                        "mine_result": {
                            k: result.get(k) for k in ("status", "reason", "session_id")
                        },
                    }
                )
            else:
                _requeue(payload, reason=reason)

    if requeue_lines:

        def _append_requeues() -> None:
            existing = ""
            if path.is_file():
                existing = path.read_text(encoding="utf-8")
            body = existing
            if body and not body.endswith("\n"):
                body += "\n"
            body += "\n".join(requeue_lines) + "\n"
            _atomic_write_text(path, body)

        with open(lock_path, "a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                _append_requeues()
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    remaining_n = 0
    if path.is_file():
        remaining_n = sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
    summary["remaining"] = remaining_n
    if summary["errors"]:
        logger.error(
            "SESSION_MINE_QUEUE | drain done processed=%s drained=%s errors=%s "
            "dead_lettered=%s requeued=%s remaining=%s",
            summary["processed"],
            summary["drained"],
            summary["errors"],
            summary["dead_lettered"],
            summary["requeued"],
            summary["remaining"],
        )
    return summary


async def synthesize_daily_mined_digest(
    *,
    db_path: Path | None = None,
    llm: Any | None = None,
    store_long_term: Any | None = None,
) -> dict[str, Any]:
    """Synthesize today's mined sessions into one digest (cheap model)."""
    path = db_path or _DB_PATH
    if not path.is_file():
        return {
            "status": "skipped",
            "reason": "db_missing",
            "sessions_mined_today": 0,
            "stored": False,
        }

    start_of_day = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    mined_rows: list[dict[str, Any]] = []

    async with aiosqlite.connect(str(path), timeout=30.0) as db:
        await db.execute("PRAGMA busy_timeout=30000")
        cur = await db.execute(
            "SELECT id, timestamp, query, learning_meta_json FROM cursor_sessions"
        )
        rows = await cur.fetchall()
        await cur.close()

    for row_id, ts, query, lm_raw in rows:
        meta = _parse_learning_meta(lm_raw)
        sm = meta.get(SESSION_MINE_KEY)
        if not isinstance(sm, dict) or not sm.get("mined_at"):
            continue
        try:
            mined_dt = datetime.fromisoformat(str(sm["mined_at"]).replace("Z", "+00:00"))
        except ValueError:
            continue
        if mined_dt < start_of_day:
            continue
        mined_rows.append(
            {
                "id": row_id,
                "timestamp": ts,
                "query": query,
                "summary": sm.get("summary") or "",
                "lessons": sm.get("lessons") or [],
                "corrections": sm.get("corrections") or [],
            }
        )

    if not mined_rows:
        return {"status": "ok", "sessions_mined_today": 0, "stored": False, "digest": ""}

    if llm is None:
        from brain_os.services.llm_client import get_llm_client

        llm = get_llm_client()

    blob_lines = []
    for item in mined_rows[:80]:
        blob_lines.append(
            f"- session={item['id']} q={(item.get('query') or '')[:200]} "
            f"summary={(item.get('summary') or '')[:300]} lessons={item.get('lessons')}"
        )
    digest = await llm.generate_text(
        system=(
            "You write a concise operator learning digest from per-session mining results. "
            "4-8 bullets: themes, corrections, procedures worth repeating, open loops. No preamble."
        ),
        user="Today's mined sessions:\n" + "\n".join(blob_lines),
        model_profile="fast",
        temperature=0.2,
        max_tokens=900,
        name="session_mine_daily_digest",
    )

    stored = False
    if store_long_term is not None and digest.strip():
        try:
            await store_long_term.store_gated(
                digest.strip(),
                user_id="global",
                metadata={
                    "type": "session_mine_daily_digest",
                    "sessions": len(mined_rows),
                    "memory_category": "session_digest",
                },
                source="session_miner:daily_digest",
                category="session_digest",
            )
            stored = True
        except Exception:
            logger.exception("session mine daily digest store_gated failed")

    return {
        "status": "ok",
        "sessions_mined_today": len(mined_rows),
        "stored": stored,
        "digest": digest.strip(),
    }


async def coverage_snapshot(
    *,
    since_days: float = 7.0,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Coverage stats for recent Graphe rows."""
    path = db_path or _DB_PATH
    cutoff = time.time() - float(since_days) * 86400.0
    out: dict[str, Any] = {
        "since_days": since_days,
        "db_exists": path.is_file(),
        "rows_total": 0,
        "rows_with_meta": 0,
        "rows_mined": 0,
        "pct_meta": None,
        "pct_mined": None,
    }
    if not path.is_file():
        return out

    async with aiosqlite.connect(str(path), timeout=30.0) as db:
        await db.execute("PRAGMA busy_timeout=30000")
        cur = await db.execute(
            "SELECT learning_meta_json FROM cursor_sessions WHERE timestamp >= ?",
            (cutoff,),
        )
        rows = await cur.fetchall()
        await cur.close()

    total = len(rows)
    with_meta = 0
    mined = 0
    for (lm_raw,) in rows:
        meta = _parse_learning_meta(lm_raw)
        if meta:
            with_meta += 1
        if is_mined_meta(meta):
            mined += 1

    out["rows_total"] = total
    out["rows_with_meta"] = with_meta
    out["rows_mined"] = mined
    if total > 0:
        out["pct_meta"] = round(with_meta / total, 4)
        out["pct_mined"] = round(mined / total, 4)
    return out
