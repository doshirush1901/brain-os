"""Sparse vector generation for hybrid (dense + sparse) search in Qdrant.

Produces (indices, values) from text using token-hash and optional tf weighting.
Used when APP__USE_SPARSE_HYBRID is True to improve exact match (model numbers, names).
"""

from __future__ import annotations

import re
from collections import Counter

# Index space size for sparse (keep under 2^20 to avoid huge vectors).
_SPARSE_INDEX_MOD = 1 << 20
_MIN_TOKEN_LEN = 2
_MAX_NONZERO = 1024


def text_to_sparse(
    text: str,
    *,
    max_nonzero: int = _MAX_NONZERO,
    use_tf: bool = True,
) -> tuple[list[int], list[float]]:
    """Convert text to a sparse vector (indices, values) for Qdrant.

    Tokens are alphanumeric runs of length >= 2. Each token maps to an index
    via hash(token) % mod. Values are 1.0 or sqrt(tf) when use_tf is True.
    Returns at most max_nonzero entries (top by value).
    """
    if not text or not text.strip():
        return [], []
    tokens = re.findall(r"\b\w{" + str(_MIN_TOKEN_LEN) + r",}\b", text.lower())
    if not tokens:
        return [], []
    counter: Counter[str] = Counter(tokens)
    idx_val: dict[int, float] = {}
    for t, count in counter.items():
        idx = hash(t) % _SPARSE_INDEX_MOD
        if idx < 0:
            idx += _SPARSE_INDEX_MOD
        val = (count**0.5) if use_tf else 1.0
        idx_val[idx] = idx_val.get(idx, 0) + val
    # Sort by value desc and cap
    sorted_pairs = sorted(idx_val.items(), key=lambda x: -x[1])[:max_nonzero]
    indices = [p[0] for p in sorted_pairs]
    values = [round(p[1], 6) for p in sorted_pairs]
    return indices, values
