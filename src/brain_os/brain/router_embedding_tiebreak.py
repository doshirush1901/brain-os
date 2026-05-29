"""Optional embedding-based override when deterministic regex margin is thin.

Reads short anchor phrases from ``data/brain/router_intent_anchors.json`` and
prefers the intent whose anchor is closest (cosine) to the query embedding.

Only runs when ``APP__ROUTER_EMBEDDING_TIEBREAK_ENABLED=true`` — disabled by default.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from brain_os.exceptions import LLMError
from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)

_ANCHORS_NAME = "router_intent_anchors.json"
_EMBED_CACHE: dict[str, list[float]] = {}


def _anchors_path() -> Path:
    return get_data_dir() / "brain" / _ANCHORS_NAME


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def load_intent_anchor_texts() -> dict[str, str]:
    path = _anchors_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("Router intent anchors JSON unreadable at %s", path, exc_info=True)
        return {}
    anchors = raw.get("anchors") if isinstance(raw, dict) else None
    if not isinstance(anchors, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in anchors.items() if str(v).strip()}


async def pick_intent_via_embedding(
    query: str,
    *,
    current_intent: str,
    embeddings: Any,
) -> tuple[str | None, dict[str, Any]]:
    """Return overriding intent key (ROUTING_TABLE / ``IntentCategory`` value) or None.

    *embeddings* must provide ``embed_query(query: str)`` and ``embed_texts(texts: list[str])``.
    """
    anchors = load_intent_anchor_texts()
    detail: dict[str, Any] = {"anchors_loaded": len(anchors), "override": None}
    if len(anchors) < 2:
        return None, detail

    if current_intent not in anchors:
        # Still allow embedding pick among anchored intents if current is brittle.
        pass

    try:
        qvec = await embeddings.embed_query(query)
    except (LLMError, OSError, ValueError, TypeError) as exc:
        logger.warning("Embedding tie-break: query embed failed", exc_info=True)
        return None, {**detail, "error": "query_embed"}

    keys: list[str] = []
    vectors: list[list[float]] = []
    for key, txt in sorted(anchors.items(), key=lambda kv: kv[0]):
        cache_key = f"{key}:{txt}"
        cached = _EMBED_CACHE.get(cache_key)
        if cached is None:
            try:
                batch = await embeddings.embed_texts([txt])
            except (LLMError, OSError, ValueError, TypeError) as exc:
                logger.warning(
                    "Embedding tie-break: anchor embed failed for %s", key, exc_info=True
                )
                continue
            if batch and isinstance(batch[0], list):
                cached = batch[0]
                _EMBED_CACHE[cache_key] = cached
        if cached:
            keys.append(key)
            vectors.append(cached)

    if len(keys) < 2:
        return None, {**detail, "error": "insufficient_anchor_vectors"}

    best_key: str | None = None
    best_score = -1.0
    scores: dict[str, float] = {}
    for key, vec in zip(keys, vectors, strict=False):
        s = _cosine_sim(qvec, vec)
        scores[key] = round(float(s), 4)
        if s > best_score:
            best_score = s
            best_key = key

    detail["scores"] = scores
    detail["winner"] = best_key
    detail["winner_sim"] = round(float(best_score), 4)

    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
    detail["runner_up_sim"] = round(float(second), 4)

    # Require modest separation vs runner-up before overriding regex winner.
    if best_key is not None and best_score - second >= 0.04 and best_key != current_intent:
        detail["override"] = best_key
        return best_key, detail

    return None, detail
