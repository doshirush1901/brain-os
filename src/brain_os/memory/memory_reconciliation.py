"""Dream stage 3i — cross-memory contradiction reconciliation (Gap #2).

Conference-style "dream looks at the memory store and corrects mismatches"
for Brain OS, beyond domain-specific stages (predictions / pricing / pending).

Safety model (mirrors Mem0 forgetting):
* Always samples a **bounded** slice of episodic + Mem0 + ledger + relationships.
* Always appends candidates to ``data/brain/memory_reconciliation_events.jsonl``.
* **Does not** mutate Mem0 / Qdrant / ledger directly.
* Optional enqueue (``APP__DREAM_MEMORY_RECONCILIATION_ENQUEUE``) writes HIGH-
  confidence rows with a non-empty ``correct_value`` into CorrectionStore so
  Stage 0.5 SleepTrainer can apply them on the next cycle.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain_os.config import get_settings
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import MemoryReconciliationResult
from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)

_PROMPT = load_prompt("dream_memory_reconciliation")
_SOURCE = "dream_memory_reconciliation"
_MAX_CLAIM_CHARS = 400


@dataclass
class MemoryReconciliationReport:
    """Outcome counters and candidates from a cross-store reconciliation pass."""

    status: str = "ok"
    claims_sampled: int = 0
    contradictions_found: int = 0
    ledger_hits: int = 0
    llm_contradictions: int = 0
    enqueued: int = 0
    events_written: int = 0
    skipped_reason: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)


def reconciliation_events_path() -> Path:
    """Append-only audit log for dream memory-mismatch candidates."""
    p = get_data_dir() / "brain" / "memory_reconciliation_events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _append_event(event: dict[str, Any]) -> None:
    try:
        path = reconciliation_events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True, default=str) + "\n")
    except OSError:
        logger.warning("Failed to append memory reconciliation event", exc_info=True)


def _clip(text: str, limit: int = _MAX_CLAIM_CHARS) -> str:
    t = " ".join(str(text or "").split())
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


def _ledger_claims(ledger: dict[str, Any], *, limit: int) -> list[dict[str, str]]:
    entities = ledger.get("entities") or {}
    out: list[dict[str, str]] = []
    for name, raw in list(entities.items())[:limit]:
        if not isinstance(raw, dict):
            continue
        correct = str(raw.get("correct_value") or raw.get("current_status") or "").strip()
        if not correct:
            continue
        stale = raw.get("stale_values") or []
        if isinstance(stale, str):
            stale = [stale]
        stale_s = "; ".join(str(s).strip() for s in stale if str(s).strip())[:200]
        claim = f"{name}: {correct}"
        if stale_s:
            claim += f" (stale: {stale_s})"
        out.append(
            {
                "store": "ledger",
                "entity": str(name),
                "claim": _clip(claim),
            }
        )
    return out


def _episode_claims(episodes: list[dict[str, Any]], *, limit: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for ep in episodes[:limit]:
        narrative = str(ep.get("narrative") or "").strip()
        if not narrative:
            continue
        uid = str(ep.get("user_id") or ep.get("id") or "episode")
        out.append(
            {
                "store": "episodic",
                "entity": uid,
                "claim": _clip(f"[{uid}] {narrative}"),
            }
        )
    return out


def _insight_claims(insights: dict[str, Any] | None, *, limit: int) -> list[dict[str, str]]:
    if not insights:
        return []
    out: list[dict[str, str]] = []
    for item in (insights.get("contradictions") or [])[:limit]:
        if isinstance(item, dict):
            desc = str(item.get("description") or "").strip()
            sources = item.get("sources") or []
        else:
            desc = str(getattr(item, "description", "") or "").strip()
            sources = getattr(item, "sources", []) or []
        if not desc:
            continue
        src = ",".join(str(s) for s in sources[:4]) if sources else "insight"
        out.append(
            {
                "store": "insight",
                "entity": _clip(desc, 80),
                "claim": _clip(f"[{src}] {desc}"),
            }
        )
    return out


def _relationship_claims(relationships: list[Any], *, limit: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for rel in relationships[:limit]:
        contact = str(getattr(rel, "contact_id", "") or "").strip()
        if not contact:
            continue
        moments = getattr(rel, "memorable_moments", None) or []
        prefs = getattr(rel, "learned_preferences", None) or {}
        bits: list[str] = []
        for m in list(moments)[:3]:
            bits.append(str(m))
        if isinstance(prefs, dict):
            for k, v in list(prefs.items())[:4]:
                bits.append(f"{k}={v}")
        elif prefs:
            bits.append(str(prefs))
        if not bits:
            continue
        warmth = getattr(getattr(rel, "warmth_level", None), "value", None) or getattr(
            rel, "warmth_level", ""
        )
        out.append(
            {
                "store": "relationship",
                "entity": contact,
                "claim": _clip(f"{contact} warmth={warmth}; " + "; ".join(bits)),
            }
        )
    return out


async def _mem0_claims(
    long_term: Any,
    *,
    entity_hints: list[str],
    limit: int,
    per_query: int,
) -> list[dict[str, str]]:
    if long_term is None or not hasattr(long_term, "search"):
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    queries = [h for h in entity_hints if h.strip()][:8]
    if not queries:
        queries = ["customer status pricing lead time machine quote"]
    for q in queries:
        if len(out) >= limit:
            break
        try:
            rows = await long_term.search(q, user_id="global", limit=per_query)
        except Exception:
            logger.debug("Mem0 search failed for reconciliation query=%r", q, exc_info=True)
            continue
        for row in rows or []:
            if len(out) >= limit:
                break
            text = str(row.get("memory") or row.get("text") or "").strip()
            if not text:
                continue
            sig = text[:120].lower()
            if sig in seen:
                continue
            seen.add(sig)
            mid = str(row.get("id") or "")
            out.append(
                {
                    "store": "mem0",
                    "entity": mid or _clip(text, 60),
                    "claim": _clip(text),
                }
            )
    return out


def _entity_hints_from_claims(claims: list[dict[str, str]], *, limit: int = 12) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    for c in claims:
        ent = str(c.get("entity") or "").strip()
        if ent and "@" not in ent and len(ent) < 80:
            key = ent.lower()
            if key not in seen:
                seen.add(key)
                hints.append(ent)
        if len(hints) >= limit:
            break
    return hints


def _ledger_vs_store_hits(
    ledger_claims: list[dict[str, str]],
    other_claims: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Deterministic mismatches: ledger correct_value appears alongside a stale_value echo."""
    hits: list[dict[str, Any]] = []
    for lc in ledger_claims:
        entity = lc.get("entity") or ""
        claim = lc.get("claim") or ""
        # claim format: "Name: correct (stale: a; b)"
        correct = ""
        stale_blob = ""
        if ": " in claim:
            rest = claim.split(": ", 1)[1]
            if " (stale: " in rest:
                correct, stale_blob = rest.split(" (stale: ", 1)
                stale_blob = stale_blob.rstrip(")")
            else:
                correct = rest
        correct = correct.strip()
        if not correct or not stale_blob:
            continue
        stale_parts = [p.strip() for p in stale_blob.split(";") if p.strip()]
        for oc in other_claims:
            text = (oc.get("claim") or "").lower()
            for stale in stale_parts:
                if len(stale) < 4:
                    continue
                if stale.lower() in text and correct.lower() not in text:
                    hits.append(
                        {
                            "entity": entity,
                            "claim_a": lc["claim"],
                            "claim_b": oc["claim"],
                            "source_a": "ledger",
                            "source_b": oc.get("store") or "other",
                            "correct_value": correct,
                            "wrong_value": stale,
                            "confidence": "HIGH",
                            "category": "GENERAL",
                            "rationale": "Mnemon ledger stale_value echoed in another memory store",
                            "detection": "ledger_hit",
                        }
                    )
                    break
    return hits


def _format_claims_for_llm(claims: list[dict[str, str]]) -> str:
    lines = []
    for i, c in enumerate(claims, 1):
        lines.append(f"{i}. [{c.get('store')}] entity={c.get('entity')}: {c.get('claim')}")
    return "\n".join(lines)


def _normalize_candidate(raw: dict[str, Any]) -> dict[str, Any] | None:
    entity = str(raw.get("entity") or "").strip()
    correct = str(raw.get("correct_value") or "").strip()
    wrong = str(raw.get("wrong_value") or "").strip()
    claim_a = str(raw.get("claim_a") or "").strip()
    claim_b = str(raw.get("claim_b") or "").strip()
    if not entity and not (claim_a and claim_b):
        return None
    conf = str(raw.get("confidence") or "LOW").strip().upper()
    if conf not in {"HIGH", "MEDIUM", "LOW"}:
        conf = "LOW"
    cat = str(raw.get("category") or "GENERAL").strip().upper()
    if cat not in {"PRICING", "SPECS", "CUSTOMER", "COMPETITOR", "GENERAL"}:
        cat = "GENERAL"
    return {
        "entity": entity or _clip(claim_a or claim_b, 80),
        "claim_a": claim_a,
        "claim_b": claim_b,
        "source_a": str(raw.get("source_a") or ""),
        "source_b": str(raw.get("source_b") or ""),
        "correct_value": correct,
        "wrong_value": wrong or claim_b or claim_a,
        "confidence": conf,
        "category": cat,
        "rationale": str(raw.get("rationale") or ""),
        "detection": str(raw.get("detection") or "llm"),
    }


async def _enqueue_corrections_via(
    enqueue_fn: Any,
    candidates: list[dict[str, Any]],
    *,
    max_enqueue: int,
) -> int:
    """Delegate CorrectionStore writes to a brain-layer callback (import contract)."""
    return int(await enqueue_fn(candidates, max_enqueue=max_enqueue))


async def run_memory_reconciliation_cycle(
    *,
    llm: Any | None = None,
    long_term: Any | None = None,
    relationship_memory: Any | None = None,
    episodes: list[dict[str, Any]] | None = None,
    insights: dict[str, Any] | None = None,
    ledger: dict[str, Any] | None = None,
    skip_llm: bool = False,
    enqueue_fn: Any | None = None,
) -> MemoryReconciliationReport:
    """Sample cross-store claims, detect contradictions, log (+ optional enqueue).

    ``enqueue_fn`` must live in ``brain_os.brain`` (e.g. CorrectionStore write) — memory
    must not import brain (``memory_no_brain`` import contract).
    """
    cfg = get_settings().app
    report = MemoryReconciliationReport()

    if not getattr(cfg, "dream_memory_reconciliation_enabled", True):
        report.status = "skipped"
        report.skipped_reason = "dream_memory_reconciliation_disabled"
        return report

    max_claims = int(getattr(cfg, "dream_memory_reconciliation_max_claims", 40))
    max_candidates = int(getattr(cfg, "dream_memory_reconciliation_max_candidates", 20))
    enqueue = bool(getattr(cfg, "dream_memory_reconciliation_enqueue", False))
    max_enqueue = int(getattr(cfg, "dream_memory_reconciliation_max_enqueue", 10))

    if ledger is None:
        try:
            ledger_path = get_data_dir() / "brain" / "correction_ledger.json"
            if ledger_path.is_file():
                ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            else:
                ledger = {"entities": {}}
        except Exception:
            logger.debug("Could not load correction ledger for reconciliation", exc_info=True)
            ledger = {"entities": {}}

    claims: list[dict[str, str]] = []
    ledger_slice = _ledger_claims(ledger, limit=min(25, max_claims // 2 or 1))
    claims.extend(ledger_slice)

    ep_budget = max(5, max_claims // 4)
    claims.extend(_episode_claims(episodes or [], limit=ep_budget))
    claims.extend(_insight_claims(insights, limit=min(10, ep_budget)))

    relationships: list[Any] = []
    if relationship_memory is not None and hasattr(relationship_memory, "get_all_relationships"):
        try:
            relationships = await relationship_memory.get_all_relationships()
        except Exception:
            logger.debug("Relationship fetch failed for reconciliation", exc_info=True)
            relationships = []
    claims.extend(_relationship_claims(relationships, limit=min(15, ep_budget)))

    hints = _entity_hints_from_claims(ledger_slice + claims)
    mem0_budget = max(5, max_claims - len(claims))
    mem0_rows = await _mem0_claims(
        long_term,
        entity_hints=hints,
        limit=mem0_budget,
        per_query=min(5, mem0_budget),
    )
    claims.extend(mem0_rows)
    claims = claims[:max_claims]
    report.claims_sampled = len(claims)

    if len(claims) < 2:
        report.status = "skipped"
        report.skipped_reason = "insufficient_claims"
        return report

    ledger_only = [c for c in claims if c.get("store") == "ledger"]
    others = [c for c in claims if c.get("store") != "ledger"]
    candidates = _ledger_vs_store_hits(ledger_only, others)
    report.ledger_hits = len(candidates)

    if not skip_llm and llm is not None and hasattr(llm, "generate_structured"):
        try:
            result = await llm.generate_structured(
                _PROMPT,
                _format_claims_for_llm(claims),
                MemoryReconciliationResult,
                temperature=0.1,
                max_user_chars=120_000,
                name="dream.memory_reconciliation",
            )
            for item in result.contradictions:
                norm = _normalize_candidate(item.model_dump())
                if norm:
                    candidates.append(norm)
            report.llm_contradictions = len(result.contradictions)
        except Exception:
            logger.exception("Dream memory reconciliation LLM pass failed")

    # Dedupe by entity+wrong_value
    deduped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for cand in candidates:
        key = f"{cand.get('entity', '')}|{cand.get('wrong_value', '')}|{cand.get('correct_value', '')}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(cand)
    candidates = deduped[:max_candidates]
    report.candidates = candidates
    report.contradictions_found = len(candidates)

    now = datetime.now(UTC).isoformat()
    for cand in candidates:
        _append_event(
            {
                "ts": now,
                "source": _SOURCE,
                "enqueued": False,
                **cand,
            }
        )
        report.events_written += 1

    if enqueue and candidates:
        if enqueue_fn is None:
            logger.warning(
                "Memory reconciliation enqueue requested but no enqueue_fn provided; "
                "candidates logged only"
            )
        else:
            try:
                n = await _enqueue_corrections_via(enqueue_fn, candidates, max_enqueue=max_enqueue)
                report.enqueued = n
                if n:
                    _append_event(
                        {
                            "ts": now,
                            "source": _SOURCE,
                            "event": "enqueue_summary",
                            "enqueued": n,
                            "candidates": len(candidates),
                        }
                    )
            except Exception:
                logger.exception("Failed to enqueue dream reconciliation corrections")
                report.status = "error"
                report.skipped_reason = "enqueue_failed"

    logger.info(
        "Memory reconciliation: claims=%d contradictions=%d ledger_hits=%d enqueued=%d",
        report.claims_sampled,
        report.contradictions_found,
        report.ledger_hits,
        report.enqueued,
    )
    return report
