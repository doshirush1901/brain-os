"""Sparse vector generation for hybrid (dense + sparse) search in Qdrant.

Produces (indices, values) from text using token-hash and optional tf weighting.
Used when APP__USE_SPARSE_HYBRID is True to improve exact match (model numbers, names).

Code-aware tokenization (VECTOR REPAIR V2): alphanumeric part/model/quote codes
like ``DEMO-X-0707``, ``MT2026009``, ``MCT-2026-...`` are kept as whole tokens
(plus a lowercase copy) so sparse search does not shred them on hyphens.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

# Index space size for sparse (keep under 2^20 to avoid huge vectors).
_SPARSE_INDEX_MOD = 1 << 20
_MIN_TOKEN_LEN = 2
_MAX_NONZERO = 1024

# Whole codes: start alnum, then alnum/hyphen, length ≥ 4 (e.g. DEMO-X-0707, MCT-2026).
_CODE_RE = re.compile(r"\b([A-Za-z0-9][A-Za-z0-9\-]{3,})\b")
# Ordinary word tokens (letters/digits/underscore); hyphens handled by code pass.
_WORD_RE = re.compile(r"\b\w{" + str(_MIN_TOKEN_LEN) + r",}\b")


def _stable_token_index(token: str) -> int:
    """Process-stable index (never use salted builtin ``hash()``)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SPARSE_INDEX_MOD


def _code_variants(raw: str) -> list[str]:
    """Full code + hyphen prefixes so ``MCT-2026`` matches ``MCT-2026-001``."""
    upper = raw.upper()
    out: list[str] = []
    parts = upper.split("-")
    if len(parts) >= 2:
        for i in range(2, len(parts) + 1):
            prefix = "-".join(parts[:i])
            if len(prefix) >= 4:
                out.append(prefix)
    else:
        out.append(upper)
    # Always include the full upper form once.
    if upper not in out:
        out.append(upper)
    # Lowercase copies for case-insensitive sparse match.
    expanded: list[str] = []
    seen: set[str] = set()
    for tok in out:
        for variant in (tok, tok.lower()):
            if variant not in seen:
                seen.add(variant)
                expanded.append(variant)
    return expanded


def looks_like_code_token(token: str) -> bool:
    """True for part/model/quote-like tokens (digit + letters and/or hyphen)."""
    t = token.strip()
    if len(t) < 4:
        return False
    has_digit = any(ch.isdigit() for ch in t)
    has_hyphen = "-" in t
    return has_digit and (has_hyphen or any(ch.isalpha() for ch in t))


# Back-compat alias used by older call sites / tests.
_looks_like_code_token = looks_like_code_token


def extract_sparse_tokens(text: str) -> list[str]:
    """Return sparse tokens, preserving alphanumeric product/quote codes intact."""
    if not text or not text.strip():
        return []
    tokens: list[str] = []
    seen_codes: set[str] = set()
    for m in _CODE_RE.finditer(text):
        raw = m.group(1)
        if not _looks_like_code_token(raw):
            continue
        for variant in _code_variants(raw):
            if variant not in seen_codes:
                tokens.append(variant)
                seen_codes.add(variant)
    for t in _WORD_RE.findall(text.lower()):
        tokens.append(t)
    return tokens


def text_to_sparse(
    text: str,
    *,
    max_nonzero: int = _MAX_NONZERO,
    use_tf: bool = True,
) -> tuple[list[int], list[float]]:
    """Convert text to a sparse vector (indices, values) for Qdrant.

    Tokens include ordinary alphanumeric runs plus intact product/quote codes.
    Each token maps to a **stable** blake2b index % mod (not Python's salted
    ``hash()``, which breaks query/doc matching across processes). Values are
    1.0 or sqrt(tf) when use_tf is True; code-like tokens get a 3× boost.
    Returns at most max_nonzero entries.
    """
    tokens = extract_sparse_tokens(text)
    if not tokens:
        return [], []
    counter: Counter[str] = Counter(tokens)
    idx_val: dict[int, float] = {}
    for t, count in counter.items():
        idx = _stable_token_index(t)
        val = (count**0.5) if use_tf else 1.0
        if _looks_like_code_token(t):
            val *= 3.0
        idx_val[idx] = idx_val.get(idx, 0) + val
    sorted_pairs = sorted(idx_val.items(), key=lambda x: -x[1])[:max_nonzero]
    indices = [p[0] for p in sorted_pairs]
    values = [round(p[1], 6) for p in sorted_pairs]
    return indices, values
