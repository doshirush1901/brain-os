"""Golden recall harness — labeled queries + P@3 + corpus/noise gauges.

15 queries = audit's 10 (Mem0 landfill repair) + 5 covering each store type
(fact / episode / procedure / relationship / commercial).

Ledger: ``data/memory/recall_eval.jsonl``
Snapshot for vitals: ``data/memory/recall_eval_latest.json``
CLI / script: ``scripts/memory_eval.py`` and heartbeat ``memory_recall_eval_weekly``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)

GOLDEN_RECALL_QUERIES: list[tuple[str, list[str], str]] = [
    ("AcmeVIP Misha VIP quote Russia", ["metakam", "misha", "metecam", "mvzet"], "mem0"),
    (
        "morning brief heat lanes orders housekeeping",
        ["heat", "orders", "housekeeping", "lane"],
        "mem0",
    ),
    (
        "quoted DEMO machine price status",
        ["quoted", "pf1", "status", "inr", "usd", "quote_number"],
        "commercial",
    ),
    ("Liebherr visit follow-up", ["liebherr"], "episode"),
    ("KTX Japan IMG RFQ", ["ktx", "img"], "mem0"),
    ("Paracoat car mats order", ["paracoat"], "mem0"),
    ("NRC Canada DEMO production", ["nrc", "canada"], "episode"),
    (
        "operator preference morning brief",
        ["morning", "brief", "preference", "lane"],
        "fact",
    ),
    ("correction lead time weeks", ["correction", "lead", "week"], "fact"),
    (
        "Acme Corp quote commercial fact",
        ["quoted", "quote", "machine", "quote_number"],
        "commercial",
    ),
    ("NRC transformer delay", ["nrc", "transformer", "delay"], "episode"),
    ("DEMO-C-1510 price", ["pf1", "quote_number", "registry", "1510"], "commercial"),
    (
        "how to draft a Tim Urban outbound email",
        ["procedure", "draft", "email", "calliope", "steps"],
        "procedure",
    ),
    (
        "warmth relationship AcmeVIP Misha",
        ["warmth", "metakam", "misha", "relationship", "vip"],
        "relationship",
    ),
    (
        "remembered fact about Acme Corp lead time",
        ["lead", "week", "fact", "acme-corp"],
        "fact",
    ),
]


@dataclass
class RecallQueryGrade:
    query: str
    expected_store: str
    hit_at_3: bool
    keywords_hit: list[str]
    keywords_missing: list[str]
    top_sources: list[str] = field(default_factory=list)
    intent: str = ""
    preview: str = ""


def recall_eval_ledger_path(*, data_root: Path | None = None) -> Path:
    root = data_root or get_data_dir()
    path = root / "memory" / "recall_eval.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def recall_eval_latest_path(*, data_root: Path | None = None) -> Path:
    root = data_root or get_data_dir()
    path = root / "memory" / "recall_eval_latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def noise_inflow_gauge_path(*, data_root: Path | None = None) -> Path:
    root = data_root or get_data_dir()
    return root / "memory" / "noise_inflow_gauge.json"


def _keyword_hit(text: str, keywords: list[str]) -> tuple[list[str], list[str]]:
    blob = (text or "").lower()
    hit = [k for k in keywords if k.lower() in blob]
    missing = [k for k in keywords if k.lower() not in blob]
    return hit, missing


def grade_recall_text(
    query: str,
    response_text: str,
    expected_keywords: list[str],
    expected_store: str,
    *,
    intent: str = "",
    top_sources: list[str] | None = None,
) -> RecallQueryGrade:
    """P@3 proxy: any expected keyword appears in the top response body."""
    lines = [ln for ln in (response_text or "").splitlines() if ln.strip().startswith("-")]
    top = "\n".join(lines[:3]) if lines else (response_text or "")[:1200]
    hit_kw, miss_kw = _keyword_hit(top, expected_keywords)
    return RecallQueryGrade(
        query=query,
        expected_store=expected_store,
        hit_at_3=bool(hit_kw),
        keywords_hit=hit_kw,
        keywords_missing=miss_kw,
        top_sources=list(top_sources or []),
        intent=intent,
        preview=top[:400],
    )


async def run_golden_recall_eval(
    *,
    services: dict[str, Any] | None = None,
    user_id: str = "global",
    persist: bool = True,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Run the 15-query harness via typed router (or Mem0-only fallback)."""
    from brain_os.services.memory_router import routed_recall

    svc = dict(services or {})
    if not svc.get("long_term_memory"):
        try:
            from brain_os.memory.long_term import LongTermMemory

            svc["long_term_memory"] = LongTermMemory()
        except Exception:
            logger.debug("LTM init for memory_eval failed", exc_info=True)

    grades: list[RecallQueryGrade] = []
    for query, keywords, store_hint in GOLDEN_RECALL_QUERIES:
        try:
            result = await routed_recall(query, services=svc, user_id=user_id)
            text = result.format_tool_response()
            sources = [h.source for h in result.hits[:3]]
            intent = result.intent
        except Exception as exc:
            text = f"ERROR: {exc}"
            sources = []
            intent = ""
        grades.append(
            grade_recall_text(
                query,
                text,
                keywords,
                store_hint,
                intent=intent,
                top_sources=sources,
            )
        )

    hits = sum(1 for g in grades if g.hit_at_3)
    total = len(grades) or 1
    p_at_3 = round(hits / total, 4)

    corpus_size = await _corpus_size(svc, data_root=data_root)
    noise_rate = _read_noise_inflow_rate(data_root=data_root)

    report = {
        "ran_at": datetime.now(UTC).isoformat(),
        "p_at_3": p_at_3,
        "hits": hits,
        "total": total,
        "corpus_size": corpus_size,
        "noise_inflow_rate": noise_rate,
        "grades": [asdict(g) for g in grades],
        "status": "ok",
        "counts": {"attempted": total, "written": 1 if persist else 0, "failed": 0},
    }
    if persist:
        try:
            latest = recall_eval_latest_path(data_root=data_root)
            latest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            ledger = recall_eval_ledger_path(data_root=data_root)
            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "ran_at": report["ran_at"],
                            "p_at_3": p_at_3,
                            "hits": hits,
                            "total": total,
                            "corpus_size": corpus_size,
                            "noise_inflow_rate": noise_rate,
                        },
                        default=str,
                    )
                    + "\n"
                )
        except OSError:
            logger.exception("persist recall eval failed")
            report["status"] = "partial_error"
            report["counts"]["failed"] = 1
    return report


async def _corpus_size(services: dict[str, Any], *, data_root: Path | None = None) -> int | None:
    gauge = noise_inflow_gauge_path(data_root=data_root)
    if gauge.is_file():
        try:
            payload = json.loads(gauge.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("corpus_size") is not None:
                return int(payload["corpus_size"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return None


def _read_noise_inflow_rate(*, data_root: Path | None = None) -> float | None:
    path = noise_inflow_gauge_path(data_root=data_root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rate = payload.get("noise_inflow_rate")
        return float(rate) if rate is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def write_noise_inflow_gauge(
    *,
    corpus_size: int,
    blocked_or_purged: int,
    stores_attempted: int,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Update noise-inflow gauge (blocked/purged ÷ attempted writes)."""
    rate = round(blocked_or_purged / stores_attempted, 4) if stores_attempted > 0 else 0.0
    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
        "corpus_size": corpus_size,
        "blocked_or_purged": blocked_or_purged,
        "stores_attempted": stores_attempted,
        "noise_inflow_rate": rate,
    }
    path = noise_inflow_gauge_path(data_root=data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def memory_vitals_snapshot(*, data_root: Path | None = None) -> dict[str, Any]:
    """Best-effort vitals organ readout for ``brain vitals`` memory line."""
    latest_path = recall_eval_latest_path(data_root=data_root)
    noise = _read_noise_inflow_rate(data_root=data_root)
    corpus_size = None
    p_at_3 = None
    ran_at = None
    if latest_path.is_file():
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
            p_at_3 = payload.get("p_at_3")
            corpus_size = payload.get("corpus_size")
            ran_at = payload.get("ran_at")
            if noise is None:
                noise = payload.get("noise_inflow_rate")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    gauge_path = noise_inflow_gauge_path(data_root=data_root)
    if corpus_size is None and gauge_path.is_file():
        try:
            g = json.loads(gauge_path.read_text(encoding="utf-8"))
            corpus_size = g.get("corpus_size")
        except (OSError, json.JSONDecodeError):
            pass

    status = "yellow"
    if isinstance(p_at_3, (int, float)):
        if p_at_3 >= 0.7:
            status = "green"
        elif p_at_3 < 0.4:
            status = "red"
    reasons: list[str] = []
    if p_at_3 is None:
        reasons.append("run poetry run python scripts/memory_eval.py")
    return {
        "status": status,
        "green": status == "green",
        "p_at_3": p_at_3,
        "corpus_size": corpus_size,
        "noise_inflow_rate": noise,
        "ran_at": ran_at,
        "reasons": reasons,
        "detail": _format_memory_detail(p_at_3, corpus_size, noise),
    }


def _format_memory_detail(
    p_at_3: float | None,
    corpus_size: int | None,
    noise: float | None,
) -> str:
    p_txt = f"{float(p_at_3):.2f}" if isinstance(p_at_3, (int, float)) else "—"
    c_txt = f"{int(corpus_size):,}" if isinstance(corpus_size, int) else "—"
    n_txt = f"{float(noise) * 100:.0f}%" if isinstance(noise, (int, float)) else "—"
    return f"recall P@3={p_txt} · corpus={c_txt} · noise-inflow={n_txt}"


async def run_memory_recall_eval_weekly_job() -> dict[str, Any]:
    """Heartbeat entrypoint — COUNTS-ONLY-OK payload."""
    report = await run_golden_recall_eval(persist=True)
    return {
        "status": report.get("status") or "ok",
        "p_at_3": report.get("p_at_3"),
        "corpus_size": report.get("corpus_size"),
        "noise_inflow_rate": report.get("noise_inflow_rate"),
        "counts": report.get("counts")
        or {"attempted": report.get("total", 0), "written": 1, "failed": 0},
    }
