"""Sleep-time memory curation — decay (archive) + episode→fact promotion.

Nightly dream pass (stage ``2_2_sleep_curation``):

(a) **DECAY** — Mem0 rows with 0 accesses in ``idle_days`` (default 180) AND
    no gold-class provenance (``operator_stated`` / ``correction`` /
    ``commercial``) are archived to a local recoverable JSONL and removed
    from the live index. Shrinks the dead tail without hard-deleting truth.

(b) **PROMOTION** — episode clusters mentioning the same entity ≥3 times
    become candidate semantic facts enqueued on the learning approval queue
    (operator taps — same consent gate as procedure/GEPA promotions).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)

GOLD_PROVENANCE_CLASSES: frozenset[str] = frozenset(
    {
        "operator_stated",
        "correction",
        "commercial",
        "commercial_fact",
        "preference",  # operator prefs are gold for decay purposes
        "evidence",  # anchored evidence stays
    }
)
GOLD_CATEGORIES: frozenset[str] = frozenset(
    {
        "correction",
        "commercial_fact",
        "preference",
        "operator_stated",
    }
)
GOLD_SOURCES_PREFIXES: tuple[str, ...] = (
    "operator:",
    "correction:",
    "quote:",
    "mnemon:",
    "feedback:",
)

DEFAULT_IDLE_DAYS = 180
_ENTITY_TOKEN_RX = re.compile(r"\b([A-Z][A-Za-z0-9]{2,}(?:\s+[A-Z][A-Za-z0-9]{2,}){0,2})\b")
_STOP_ENTITIES: frozenset[str] = frozenset(
    {
        "The",
        "This",
        "That",
        "With",
        "From",
        "User",
        "Brain OS",
        "Agent",
        "Email",
        "Quote",
        "Machine",
        "Company",
        "Customer",
        "India",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    }
)


def decay_archive_path(*, data_root: Path | None = None) -> Path:
    root = data_root or get_data_dir()
    path = root / "memory" / "decay_archive.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sleep_curation_report_path(*, data_root: Path | None = None) -> Path:
    root = data_root or get_data_dir()
    path = root / "memory" / "sleep_curation_last.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def is_gold_memory(memory: dict[str, Any]) -> bool:
    """True when provenance/category marks the row as protected from decay."""
    meta = memory.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    prov = str(
        memory.get("provenance_class")
        or meta.get("provenance_class")
        or meta.get("provenance")
        or ""
    ).lower()
    if prov in GOLD_PROVENANCE_CLASSES:
        return True
    cat = str(meta.get("memory_category") or meta.get("category") or meta.get("type") or "").lower()
    if cat in GOLD_CATEGORIES:
        return True
    src = str(meta.get("source") or "").lower()
    if any(src.startswith(p) for p in GOLD_SOURCES_PREFIXES):
        return True
    if meta.get("operator_stated") is True:
        return True
    return False


def is_decay_candidate(
    memory: dict[str, Any],
    *,
    access_count: int,
    age_days: int | None,
    idle_days: int = DEFAULT_IDLE_DAYS,
) -> tuple[bool, str]:
    """Return ``(candidate, reason)`` under 180d zero-access + non-gold rules."""
    if access_count > 0:
        return False, "accessed"
    if is_gold_memory(memory):
        return False, "gold_provenance"
    if age_days is None:
        return False, "unknown_age"
    if age_days < idle_days:
        return False, "too_young"
    return True, "idle_zero_access"


def _memory_age_days(memory: dict[str, Any], now: datetime) -> int | None:
    created = memory.get("created_at")
    if not created:
        return None
    try:
        if isinstance(created, str):
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        else:
            dt = created
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return max(0, (now - dt).days)
    except (ValueError, TypeError):
        return None


def append_decay_archive(
    memory: dict[str, Any],
    *,
    user_id: str,
    reason: str,
    access_count: int = 0,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    """Append one recoverable archive row to local JSONL."""
    path = archive_path or decay_archive_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "memory_id": str(memory.get("id", "") or ""),
        "user_id": user_id,
        "content": str(memory.get("memory", memory.get("content", "")) or ""),
        "metadata": memory.get("metadata") or {},
        "created_at": str(memory.get("created_at", "") or ""),
        "archived_at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "access_count_at_archive": int(access_count),
        "provenance_class": memory.get("provenance_class")
        or (memory.get("metadata") or {}).get("provenance_class"),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row


def read_decay_archive(*, archive_path: Path | None = None) -> list[dict[str, Any]]:
    path = archive_path or decay_archive_path()
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


async def restore_from_decay_archive(
    long_term: Any,
    memory_id: str,
    *,
    archive_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Re-store one archived memory into Mem0 (recoverability roundtrip)."""
    mid = (memory_id or "").strip()
    rows = read_decay_archive(archive_path=archive_path)
    row = next((r for r in rows if str(r.get("memory_id") or "") == mid), None)
    if row is None:
        return {"ok": False, "error": "not_in_archive", "memory_id": mid}
    if dry_run:
        return {"ok": True, "dry_run": True, "memory_id": mid, "content": row.get("content")}
    content = str(row.get("content") or "")
    meta = dict(row.get("metadata") or {})
    meta["restored_from_decay_archive"] = True
    meta["original_memory_id"] = mid
    user_id = str(row.get("user_id") or "global")
    gated = await long_term.store_gated(
        content,
        user_id=user_id,
        source="sleep_curation:restore",
        category=str(meta.get("memory_category") or meta.get("type") or "fact"),
        metadata=meta,
    )
    return {
        "ok": not gated.get("skipped"),
        "memory_id": mid,
        "store_result": gated,
    }


async def run_decay_pass(
    long_term: Any,
    *,
    dry_run: bool = True,
    idle_days: int = DEFAULT_IDLE_DAYS,
    run_cap: int = 200,
    user_id: str = "global",
    archive_path: Path | None = None,
) -> dict[str, Any]:
    """Scan Mem0; archive idle non-gold rows. Returns COUNTS-ONLY-OK counts."""
    from brain_os.memory.mem0_access_tracker import get_mem0_access_tracker

    tracker = get_mem0_access_tracker()
    now = datetime.now(UTC)
    attempted = 0
    written = 0
    failed = 0
    protected = 0
    candidates: list[dict[str, Any]] = []

    page = 1
    page_size = 100
    scanned = 0
    while scanned < 50_000:
        try:
            batch = await long_term.list_memories(user_id, page=page, page_size=page_size)
        except Exception:
            logger.exception("decay list_memories failed page=%s", page)
            failed += 1
            break
        if not batch:
            break
        ids = [str(m.get("id", "")) for m in batch if isinstance(m, dict)]
        counts = await tracker.get_counts(ids)
        for m in batch:
            if not isinstance(m, dict):
                continue
            scanned += 1
            mid = str(m.get("id", "") or "")
            access = int(counts.get(mid, 0))
            age = _memory_age_days(m, now)
            ok, reason = is_decay_candidate(
                m, access_count=access, age_days=age, idle_days=idle_days
            )
            if not ok:
                if reason in {"gold_provenance", "accessed"}:
                    protected += 1
                continue
            candidates.append({"memory": m, "access": access, "reason": reason})
            if len(candidates) >= run_cap:
                break
        if len(candidates) >= run_cap or len(batch) < page_size:
            break
        page += 1

    for item in candidates:
        attempted += 1
        m = item["memory"]
        mid = str(m.get("id", "") or "")
        try:
            if dry_run:
                written += 1
                continue
            append_decay_archive(
                m,
                user_id=user_id,
                reason=item["reason"],
                access_count=item["access"],
                archive_path=archive_path,
            )
            deleted = await long_term.delete_memory(mid)
            if deleted:
                written += 1
            else:
                failed += 1
        except Exception:
            logger.exception("decay archive/delete failed id=%s", mid)
            failed += 1

    status = "ok" if failed == 0 else ("partial_error" if written else "error")
    return {
        "status": status,
        "dry_run": dry_run,
        "scanned": scanned,
        "candidates": len(candidates),
        "protected": protected,
        "idle_days": idle_days,
        "counts": {"attempted": attempted, "written": written, "failed": failed},
    }


def extract_episode_entities(narrative: str) -> list[str]:
    """Cheap proper-noun / token extraction for cluster keys."""
    text = (narrative or "").strip()
    if not text:
        return []
    found: list[str] = []
    for m in _ENTITY_TOKEN_RX.finditer(text):
        ent = m.group(1).strip()
        if ent in _STOP_ENTITIES or len(ent) < 3:
            continue
        found.append(ent)
    # Also catch all-caps company tokens (NRC, KTX, …)
    for tok in re.findall(r"\b([A-Z]{2,8})\b", text):
        if tok not in _STOP_ENTITIES and tok not in found:
            found.append(tok)
    return found[:12]


def build_promotion_candidates(
    narratives: list[str],
    *,
    min_mentions: int = 3,
    max_candidates: int = 10,
) -> list[dict[str, Any]]:
    """Cluster narratives by entity → promotion payloads (no brain import)."""
    entity_mentions: dict[str, list[str]] = defaultdict(list)
    for narrative in narratives:
        for ent in extract_episode_entities(narrative):
            key = ent.strip()
            if key:
                entity_mentions[key].append(narrative[:300])
    ranked = sorted(entity_mentions.items(), key=lambda kv: len(kv[1]), reverse=True)
    out: list[dict[str, Any]] = []
    for entity, mentions in ranked:
        if len(mentions) < min_mentions:
            continue
        if len(out) >= max_candidates:
            break
        uniq = list(dict.fromkeys(mentions))[:5]
        fact = (
            f"Candidate semantic fact (from {len(uniq)} episode mentions of {entity}): "
            f"{entity} appears repeatedly in recent interactions. "
            f"Sample: {uniq[0][:180]}"
        )
        out.append(
            {
                "entity": entity,
                "fact": fact,
                "mention_count": len(mentions),
                "samples": uniq,
                "kind_detail": "episode_cluster_promotion",
                "dedupe_key": f"semantic_fact:{entity.lower()}",
            }
        )
    return out


async def collect_episode_narratives(
    episodic: Any | None,
    *,
    lookback_days: int = 90,
) -> list[str]:
    """Load recent episode narratives for promotion clustering."""
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=lookback_days)
    narratives: list[str] = []
    if episodic is None:
        return narratives
    try:
        if hasattr(episodic, "initialize"):
            await episodic.initialize()
        backend = getattr(episodic, "_backend", None)
        if backend is not None and hasattr(backend, "weave_rows"):
            rows = await backend.weave_rows("global", None, 200)
            for r in rows or []:
                narrative = str(r[1] if isinstance(r, (list, tuple)) else "")
                created = r[2] if isinstance(r, (list, tuple)) and len(r) > 2 else None
                if created:
                    try:
                        dt = (
                            datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                            if not isinstance(created, datetime)
                            else created
                        )
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=UTC)
                        if dt < cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass
                if narrative.strip():
                    narratives.append(narrative)
        else:
            for seed in ("customer", "quote", "visit", "delay", "machine"):
                eps = await episodic.surface_relevant_episodes(seed, "global")
                for e in eps or []:
                    n = str(e.get("narrative", "") or "")
                    if n.strip():
                        narratives.append(n)
    except Exception:
        logger.exception("promotion episode scan failed")
    return narratives


async def run_promotion_pass(
    episodic: Any | None = None,
    *,
    min_mentions: int = 3,
    lookback_days: int = 90,
    dry_run: bool = False,
    max_candidates: int = 10,
    enqueue_fn: Any | None = None,
) -> dict[str, Any]:
    """Cluster episodes by entity; enqueue via *enqueue_fn* (injected; no brain import).

    Dream stage / CLI should pass ``learning_promotion_queue.enqueue_promotion``.
    When *enqueue_fn* is None and not dry_run, candidates are returned without writing.
    """
    narratives_raw = await collect_episode_narratives(episodic, lookback_days=lookback_days)
    # Keep a lightweight entities_seen count from the raw narratives.
    entity_mentions_probe: dict[str, int] = defaultdict(int)
    for narrative in narratives_raw:
        for ent in extract_episode_entities(narrative):
            entity_mentions_probe[ent.strip()] += 1

    candidates = build_promotion_candidates(
        narratives_raw,
        min_mentions=min_mentions,
        max_candidates=max_candidates,
    )

    attempted = 0
    written = 0
    failed = 0
    enqueued: list[dict[str, Any]] = []
    for payload in candidates:
        attempted += 1
        if dry_run or enqueue_fn is None:
            written += 1 if dry_run else 0
            enqueued.append({"dry_run": True, **payload} if dry_run else payload)
            if not dry_run and enqueue_fn is None:
                # Candidates collected for an outer enqueue (dream/CLI).
                written += 1
            continue
        try:
            item = enqueue_fn(
                kind="semantic_fact",
                payload={
                    "entity": payload["entity"],
                    "fact": payload["fact"],
                    "mention_count": payload["mention_count"],
                    "samples": payload["samples"],
                    "kind_detail": payload["kind_detail"],
                },
                evidence={
                    "mention_count": payload["mention_count"],
                    "samples": payload["samples"],
                },
                source="sleep_curation:promotion",
                dedupe_key=payload["dedupe_key"],
            )
            if item is None:
                continue
            written += 1
            enqueued.append(item)
        except Exception:
            logger.exception("promotion enqueue failed entity=%s", payload.get("entity"))
            failed += 1

    status = "ok" if failed == 0 else ("partial_error" if written else "error")
    return {
        "status": status,
        "dry_run": dry_run,
        "entities_seen": len(entity_mentions_probe),
        "clusters_ge_min": sum(1 for c in entity_mentions_probe.values() if c >= min_mentions),
        "enqueued": enqueued,
        "counts": {"attempted": attempted, "written": written, "failed": failed},
    }


async def run_sleep_curation(
    long_term: Any,
    episodic: Any | None = None,
    *,
    dry_run: bool = False,
    idle_days: int = DEFAULT_IDLE_DAYS,
    promote: bool = True,
    enqueue_fn: Any | None = None,
) -> dict[str, Any]:
    """Full nightly pass: decay then promotion. Writes last-run report."""
    decay = await run_decay_pass(long_term, dry_run=dry_run, idle_days=idle_days)
    promotion: dict[str, Any] = {
        "status": "skipped",
        "counts": {"attempted": 0, "written": 0, "failed": 0},
    }
    if promote:
        promotion = await run_promotion_pass(episodic, dry_run=dry_run, enqueue_fn=enqueue_fn)
    report = {
        "ran_at": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
        "decay": decay,
        "promotion": promotion,
        "counts": {
            "attempted": int(decay["counts"]["attempted"]) + int(promotion["counts"]["attempted"]),
            "written": int(decay["counts"]["written"]) + int(promotion["counts"]["written"]),
            "failed": int(decay["counts"]["failed"]) + int(promotion["counts"]["failed"]),
        },
        "status": (
            "ok"
            if decay.get("status") == "ok" and promotion.get("status") in {"ok", "skipped"}
            else "partial_error"
        ),
    }
    try:
        sleep_curation_report_path().write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
    except OSError:
        logger.debug("sleep curation report write failed", exc_info=True)
    return report
