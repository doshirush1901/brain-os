"""Episode hygiene — empty/spam skips at write time + dream-time cleanup.

Dream pass deletes ``No conversation took place`` empties and near-duplicate
pairs; write-time guards prevent new spam-email episodes.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

_EMPTY_NARRATIVE_RX = re.compile(
    r"no\s+conversation\s+took\s+place",
    re.IGNORECASE,
)
_SPAM_MARKERS = re.compile(
    r"\b(spam|newsletter|unsubscribe|marketing\s+blast|bulk\s+email)\b",
    re.IGNORECASE,
)
_TOKEN_RX = re.compile(r"[a-z0-9]{3,}")


def is_empty_episode_narrative(narrative: str) -> bool:
    text = (narrative or "").strip()
    if not text:
        return True
    if _EMPTY_NARRATIVE_RX.search(text):
        return True
    if text in {"(Consolidation failed)", "(no content)"}:
        return True
    return False


def is_spam_episode_payload(transcript: list[dict[str, Any]] | None, narrative: str) -> bool:
    """Heuristic + Delphi spam-class markers on transcript metadata/content."""
    if _SPAM_MARKERS.search(narrative or ""):
        return True
    for msg in transcript or []:
        if not isinstance(msg, dict):
            continue
        meta = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else {}
        cls = str(
            meta.get("email_class")
            or meta.get("classification")
            or meta.get("intent")
            or msg.get("email_class")
            or ""
        ).lower()
        if cls in {"spam", "newsletter", "marketing"}:
            return True
        content = str(msg.get("content") or "")
        if _SPAM_MARKERS.search(content[:500]):
            return True
    return False


def should_skip_episode_write(
    *,
    narrative: str,
    transcript: list[dict[str, Any]] | None = None,
    user_id: str = "",
) -> str:
    """Return skip reason or empty string when write should proceed."""
    if is_empty_episode_narrative(narrative):
        return "empty_narrative"
    if is_spam_episode_payload(transcript, narrative):
        return "spam_email"
    return ""


def _token_sig(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RX.findall((text or "").lower()))


def narratives_near_duplicate(a: str, b: str, *, jaccard: float = 0.92) -> bool:
    """True when two narratives share nearly the same token set (1290≈1843 class)."""
    sa, sb = _token_sig(a), _token_sig(b)
    if not sa or not sb:
        return False
    if a.strip() == b.strip():
        return True
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return False
    return (inter / union) >= jaccard


async def run_episode_hygiene(
    db: Any,
    *,
    dry_run: bool = True,
    jaccard: float = 0.92,
    limit: int = 50_000,
) -> dict[str, Any]:
    """Dream-time episode pass: delete empties + dedupe near-identical pairs.

    When *dry_run* is True, only counts/samples are returned (no DELETE).
    Conservative: on ambiguity keep the older row.
    """
    if db is None:
        return {"status": "skipped", "reason": "no_db", "empties": 0, "deduped": 0}

    cursor = await db.execute(
        "SELECT id, user_id, narrative, created_at FROM episodes ORDER BY created_at ASC LIMIT ?",
        (int(limit),),
    )
    rows = await cursor.fetchall()
    await cursor.close()

    empty_ids: list[int] = []
    empty_samples: list[dict[str, Any]] = []
    keep_by_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    dedupe_ids: list[int] = []
    dedupe_samples: list[dict[str, Any]] = []

    for row in rows:
        eid = int(row[0])
        user_id = str(row[1] or "")
        narrative = str(row[2] or "")
        created = str(row[3] or "")
        if is_empty_episode_narrative(narrative):
            empty_ids.append(eid)
            if len(empty_samples) < 20:
                empty_samples.append(
                    {
                        "id": eid,
                        "user_id": user_id,
                        "narrative": narrative[:120],
                        "created_at": created,
                    }
                )
            continue
        kept = keep_by_user[user_id]
        dup_of: int | None = None
        for kid, knarr in kept:
            if narratives_near_duplicate(narrative, knarr, jaccard=jaccard):
                dup_of = kid
                break
        if dup_of is not None:
            dedupe_ids.append(eid)
            if len(dedupe_samples) < 20:
                dedupe_samples.append(
                    {
                        "id": eid,
                        "duplicate_of": dup_of,
                        "user_id": user_id,
                        "narrative": narrative[:120],
                    }
                )
        else:
            kept.append((eid, narrative))

    deleted_empty = 0
    deleted_dup = 0
    if not dry_run:
        for batch_name, ids in (("empty", empty_ids), ("dedupe", dedupe_ids)):
            for i in range(0, len(ids), 200):
                chunk = ids[i : i + 200]
                placeholders = ",".join("?" * len(chunk))
                await db.execute(f"DELETE FROM episodes WHERE id IN ({placeholders})", chunk)
                if batch_name == "empty":
                    deleted_empty += len(chunk)
                else:
                    deleted_dup += len(chunk)
        await db.commit()

    summary = {
        "status": "ok",
        "dry_run": dry_run,
        "scanned": len(rows),
        "empties": len(empty_ids),
        "deduped": len(dedupe_ids),
        "deleted_empties": deleted_empty,
        "deleted_duplicates": deleted_dup,
        "empty_samples": empty_samples,
        "dedupe_samples": dedupe_samples,
    }
    logger.info(
        "Episode hygiene: scanned=%d empties=%d deduped=%d dry_run=%s",
        len(rows),
        len(empty_ids),
        len(dedupe_ids),
        dry_run,
    )
    return summary
