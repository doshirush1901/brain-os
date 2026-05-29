"""Structured KB evidence for faithfulness and downstream consumers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceChunk(BaseModel):
    """One retrieval row prepared for verification."""

    content: str = ""
    source: str = ""
    source_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    """Unified evidence package built from retriever rows."""

    query: str
    chunks: list[EvidenceChunk] = Field(default_factory=list)

    def texts_for_faithfulness(self) -> list[str]:
        return [c.content for c in self.chunks if c.content]


def bundle_from_rows(query: str, rows: list[dict[str, Any]]) -> EvidenceBundle:
    chunks: list[EvidenceChunk] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        meta = r.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        chunks.append(
            EvidenceChunk(
                content=str(r.get("content") or ""),
                source=str(r.get("source") or ""),
                source_type=str(r.get("source_type") or ""),
                metadata=meta,
            )
        )
    return EvidenceBundle(query=query, chunks=chunks)


def trim_bundle_by_chars(
    bundle: EvidenceBundle,
    *,
    max_chars: int,
) -> tuple[EvidenceBundle, dict[str, Any]]:
    """Drop lowest-index chunks from the tail until total content length <= max_chars.

    When *max_chars* <= 0, returns the bundle unchanged.
    """
    if max_chars <= 0:
        return bundle, {
            "trimmed": False,
            "removed": 0,
            "total_chars": sum(len(c.content) for c in bundle.chunks),
        }
    kept: list[EvidenceChunk] = []
    total = 0
    removed = 0
    for c in bundle.chunks:
        ln = len(c.content)
        if total + ln <= max_chars:
            kept.append(c)
            total += ln
        else:
            removed += 1
    trimmed = removed > 0
    return EvidenceBundle(query=bundle.query, chunks=kept), {
        "trimmed": trimmed,
        "removed": removed,
        "total_chars": total,
        "max_chars": max_chars,
    }


def apply_bundle_pii(
    bundle: EvidenceBundle,
    *,
    mode: str,
) -> tuple[EvidenceBundle, dict[str, Any]]:
    """Scan or redact PII in chunk text. ``off`` is a no-op; ``flag`` counts hits only."""
    m = (mode or "off").strip().lower()
    if m not in ("off", "flag", "redact"):
        m = "off"
    if m == "off":
        return bundle, {"mode": m, "hits": 0, "redacted": False}
    from brain_os.brain.pii_redact import redact_pii

    hits = 0
    if m == "flag":
        for c in bundle.chunks:
            _, stats = redact_pii(c.content)
            hits += int(stats.get("emails_redacted", 0)) + int(stats.get("phones_redacted", 0))
        return bundle, {"mode": m, "hits": hits, "redacted": hits > 0}

    new_chunks: list[EvidenceChunk] = []
    for c in bundle.chunks:
        out, stats = redact_pii(c.content)
        hits += int(stats.get("emails_redacted", 0)) + int(stats.get("phones_redacted", 0))
        new_chunks.append(c.model_copy(update={"content": out}))
    return EvidenceBundle(query=bundle.query, chunks=new_chunks), {
        "mode": m,
        "hits": hits,
        "redacted": hits > 0,
    }
