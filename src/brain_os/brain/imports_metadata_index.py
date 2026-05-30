"""LLM-summarised metadata index for every file in ``examples/acme/docs/``.

Scans the imports directory, extracts a short text preview from each file,
and uses GPT-4.1-mini to generate structured metadata (summary, doc_type,
machines, topics, entities, keywords).  The index is persisted to
``data/brain/imports_metadata.json`` and powers Alexandros's hybrid search.

The index is idempotent — files are fingerprinted by name+size+mtime and
only re-indexed when they change.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langfuse.decorators import observe

from brain_os.brain.document_ingestor import (
    read_csv,
    read_docx,
    read_pdf,
    read_pptx,
    read_txt,
    read_xls,
    read_xlsx,
)
from brain_os.brain.imports_intents import (
    INTENT_TAGS,
    infer_document_role,
    infer_intents_from_text,
    normalize_intent_tags,
)
from brain_os.exceptions import IngestionError, BrainOSError, LLMError
from brain_os.schemas.llm_outputs import DocumentMetadata
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMPORTS_DIR = _PROJECT_ROOT / "data" / "imports"
INDEX_PATH = _PROJECT_ROOT / "data" / "brain" / "imports_metadata.json"
INDEX_PROGRESS_PATH = _PROJECT_ROOT / "data" / "brain" / "imports_index_progress.json"

SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".csv", ".docx", ".txt", ".pptx", ".json", ".md"}
_TEXT_PREVIEW_CHARS = 2000

_READERS: dict[str, Callable[[Path], str]] = {
    ".pdf": read_pdf,
    ".xlsx": read_xlsx,
    ".xls": read_xls,
    ".docx": read_docx,
    ".txt": read_txt,
    ".csv": read_csv,
    ".pptx": read_pptx,
}


# ── text extraction (lightweight preview) ────────────────────────────────


def _extract_preview(filepath: Path) -> str:
    """Extract the first ~2000 chars of text from a file."""
    reader = _READERS.get(filepath.suffix.lower())
    if reader is not None:
        try:
            return reader(filepath)[:_TEXT_PREVIEW_CHARS]
        except (IngestionError, OSError, ValueError, TypeError):
            logger.debug("Reader failed for %s, trying plain text", filepath.name)

    if filepath.suffix.lower() in (".txt", ".json", ".csv", ".md"):
        try:
            return filepath.read_text(errors="ignore")[:_TEXT_PREVIEW_CHARS]
        except (BrainOSError, OSError, UnicodeDecodeError):
            logger.debug("Plain-text read failed for %s", filepath.name)
    return ""


def _file_fingerprint(filepath: Path) -> str:
    """Quick hash based on name+size+mtime (not content — too slow for 700+ files)."""
    stat = filepath.stat()
    key = f"{filepath.name}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ── LLM metadata generation ─────────────────────────────────────────────


@observe()
async def _generate_metadata_llm(filename: str, text_preview: str) -> dict[str, Any] | None:
    """Use Brain OS's fast model profile to produce structured metadata from a file preview."""
    system = "Extract structured metadata from documents. Return only valid JSON."
    intents_csv = ", ".join(INTENT_TAGS)
    user = f"""Analyze this document and return structured metadata as JSON.

FILENAME: {filename}

TEXT PREVIEW (first {_TEXT_PREVIEW_CHARS} chars):
{text_preview[:_TEXT_PREVIEW_CHARS]}

Return ONLY valid JSON with these fields:
{{
    "summary": "1-2 sentence description of what this document is about",
    "doc_type": "one of: quote, catalogue, order, presentation, email, spreadsheet, report, manual, contract, lead_list, customer_data, technical_spec, brochure, invoice, other",
    "machines": ["list of machine models mentioned, e.g. DEMO-C-2015, AM-5060"],
    "topics": ["list from: pricing, specs, customer, application, lead, order, contract, presentation, marketing, technical, installation, warranty, shipping, competitor, market_research, training, quote_sent (add quote_sent for outbound quotes — we sent this to a customer; follow-up status not in document)"],
    "entities": ["company names, person names, countries mentioned"],
    "keywords": ["5-10 important searchable terms from the document"],
    "intent_tags": ["zero or more from: {intents_csv}"],
    "counterparty_type": "one of: customer, vendor, internal, unknown",
    "document_role": "single best role using intent taxonomy",
    "intent_confidence": {{"intent_tag": 0.0}}
}}"""

    try:
        llm = get_llm_client()
        result = await llm.generate_structured(
            system,
            user,
            DocumentMetadata,
            model_profile="fast",
            name="imports.metadata",
        )
        payload = result.model_dump()
        normalized_intents = normalize_intent_tags(payload.get("intent_tags", []))
        if not normalized_intents:
            normalized_intents, inferred_counterparty, inferred_role, inferred_conf = (
                infer_intents_from_text(
                    f"{filename} {text_preview[:800]}",
                    doc_type=payload.get("doc_type", ""),
                )
            )
        else:
            inferred_counterparty = "unknown"
            inferred_role = infer_document_role(normalized_intents)
            inferred_conf = {}
        payload["intent_tags"] = normalized_intents
        payload["counterparty_type"] = payload.get("counterparty_type", "") or inferred_counterparty
        payload["document_role"] = payload.get("document_role", "") or inferred_role
        payload["intent_confidence"] = {
            k: float(v)
            for k, v in (payload.get("intent_confidence", {}) or {}).items()
            if k in normalized_intents
        }
        if not payload["intent_confidence"]:
            payload["intent_confidence"] = inferred_conf
        return payload
    except LLMError as exc:
        logger.warning("LLM metadata failed for %s: %s", filename, exc)
        return None


def _generate_metadata_local(filename: str, text_preview: str) -> dict[str, Any]:
    """Fast local metadata extraction without LLM (fallback)."""
    name_lower = filename.lower()

    machines = re.findall(
        r"(PF1[-\s]?[A-Z]?[-\s]?\d+[-\s]?\d*|AM[-\s]?\w+\d+|IMG[-\s]?\d+|FCS[-\s]?\w+|UNO[-\s]?\w+|DUO[-\s]?\w+)",
        filename + " " + text_preview[:500],
        re.IGNORECASE,
    )
    machines = list({m.upper().replace(" ", "-") for m in machines})

    doc_type = "other"
    type_keywords: dict[str, list[str]] = {
        "quote": ["quote", "quotation", "offer", "price"],
        "catalogue": ["catalogue", "catalog", "brochure"],
        "order": ["order", "po", "purchase"],
        "presentation": ["ppt", "presentation", "pptx"],
        "email": ["gmail", "email", "mail"],
        "spreadsheet": ["xlsx", "xls", "csv"],
        "manual": ["manual", "instruction", "operating"],
        "contract": ["contract", "nda", "agreement"],
        "lead_list": ["lead", "contact", "inquiry", "visitor"],
        "technical_spec": ["spec", "technical", "table"],
    }
    for dtype, kws in type_keywords.items():
        if any(kw in name_lower for kw in kws):
            doc_type = dtype
            break

    merged = filename + " " + text_preview[:500]
    intent_tags, counterparty_type, document_role, intent_confidence = infer_intents_from_text(
        merged,
        doc_type=doc_type,
    )

    words = re.findall(r"\b\w{4,}\b", (filename + " " + text_preview[:300]).lower())
    keywords = list(set(words))[:10]

    return {
        "summary": f"Document: {filename}",
        "doc_type": doc_type,
        "machines": machines,
        "topics": [],
        "entities": [],
        "keywords": keywords,
        "intent_tags": intent_tags,
        "counterparty_type": counterparty_type,
        "document_role": document_role,
        "intent_confidence": intent_confidence,
    }


# ── index persistence ────────────────────────────────────────────────────


async def load_index() -> dict[str, Any]:
    """Load the metadata index from disk."""
    if INDEX_PATH.exists():
        try:
            raw = await asyncio.to_thread(INDEX_PATH.read_text)
            return json.loads(raw)
        except (OSError, json.JSONDecodeError):
            logger.error("Corrupt metadata index at %s", INDEX_PATH)
    return {"files": {}, "built_at": None, "total_files": 0, "version": 2}


async def _save_index(index: dict[str, Any]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(
        INDEX_PATH.write_text,
        json.dumps(index, indent=2, ensure_ascii=False),
    )


async def _save_progress(done: int, total: int, current_file: str) -> None:
    INDEX_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(
        INDEX_PROGRESS_PATH.write_text,
        json.dumps(
            {
                "done": done,
                "total": total,
                "current": current_file,
                "percent": round(done / total * 100, 1) if total else 0,
                "updated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
    )


# ── index building ───────────────────────────────────────────────────────


async def build_index(
    *,
    use_llm: bool = True,
    force: bool = False,
    include_prefixes: tuple[str, ...] = (),
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    """Build or incrementally update the metadata index.

    If *include_prefixes* is non-empty, only files whose relative path
    starts with one of those prefixes are considered (e.g. 01_Quotes_and_Proposals/).

    Returns stats: ``{"total", "new", "skipped", "errors"}``.
    """
    normalized_includes = tuple(
        p.strip().lower().rstrip("/") + "/" for p in include_prefixes if p and p.strip()
    )
    if force and not normalized_includes:
        index = {"files": {}, "built_at": None, "total_files": 0, "version": 2}
    else:
        index = await load_index()

    all_files = [
        fp
        for fp in IMPORTS_DIR.rglob("*")
        if fp.is_file()
        and not fp.name.startswith(".")
        and fp.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if normalized_includes:
        all_files = [
            fp
            for fp in all_files
            if any(
                str(fp.relative_to(IMPORTS_DIR)).lower().startswith(prefix)
                for prefix in normalized_includes
            )
        ]

    total = len(all_files)
    new_count = 0
    skipped = 0
    errors = 0

    for i, fp in enumerate(all_files):
        fhash = _file_fingerprint(fp)
        rel_path = str(fp.relative_to(IMPORTS_DIR))

        if (
            not force
            and rel_path in index["files"]
            and index["files"][rel_path].get("hash") == fhash
        ):
            skipped += 1
            continue

        if progress_callback:
            progress_callback(i + 1, total, fp.name)
        await _save_progress(i + 1, total, fp.name)

        try:
            preview = await asyncio.to_thread(_extract_preview, fp)

            metadata: dict[str, Any] | None = None
            if use_llm and preview and len(preview) > 50:
                metadata = await _generate_metadata_llm(fp.name, preview)
            if metadata is None:
                metadata = _generate_metadata_local(fp.name, preview)

            index["files"][rel_path] = {
                "name": fp.name,
                "path": str(fp),
                "hash": fhash,
                "size_kb": fp.stat().st_size // 1024,
                "extension": fp.suffix.lower(),
                "indexed_at": datetime.now(UTC).isoformat(),
                **metadata,
            }
            new_count += 1

            if new_count % 20 == 0:
                await _save_index(index)

            if use_llm and preview and len(preview) > 50:
                await asyncio.sleep(0.3)

        except (IngestionError, OSError, ValueError, TypeError) as exc:
            logger.warning("Error indexing %s: %s", fp.name, exc)
            errors += 1

    index["built_at"] = datetime.now(UTC).isoformat()
    index["total_files"] = len(index["files"])
    await _save_index(index)

    if INDEX_PROGRESS_PATH.exists():
        INDEX_PROGRESS_PATH.unlink()

    stats = {"total": total, "new": new_count, "skipped": skipped, "errors": errors}
    logger.info("Index built: %s", stats)
    return stats


# ── keyword search ───────────────────────────────────────────────────────


_STEM_MAP: dict[str, str] = {
    "quotes": "quote",
    "quotations": "quote",
    "quotation": "quote",
    "machines": "machine",
    "specifications": "spec",
    "specs": "spec",
    "proposals": "proposal",
    "orders": "order",
    "invoices": "invoice",
    "contracts": "contract",
    "presentations": "presentation",
    "customers": "customer",
    "companies": "company",
    "documents": "document",
    "files": "file",
    "brochures": "brochure",
    "catalogues": "catalogue",
    "catalogs": "catalogue",
    "manuals": "manual",
    "reports": "report",
    "emails": "email",
}

_TOPIC_TRIGGERS: dict[str, set[str]] = {
    "pricing": {"price", "cost", "quote", "lakh", "usd", "inr", "euro", "budget"},
    "specs": {"spec", "specification", "technical", "heater", "vacuum", "forming", "dimension"},
    "customer": {"customer", "order", "client", "company", "buyer"},
    "application": {"application", "automotive", "bathtub", "packaging", "food", "medical"},
    "lead": {"lead", "prospect", "inquiry", "visitor", "contact"},
    "marketing": {"marketing", "campaign", "drip", "newsletter", "linkedin"},
    "shipping": {"shipping", "freight", "logistics", "delivery", "transport"},
    "installation": {"installation", "commissioning", "setup", "assembly"},
    "competitor": {"competitor", "competition", "illig", "kiefel", "brown", "maac"},
}


def _stem_words(words: set[str]) -> set[str]:
    """Expand a word set with basic stem normalisations."""
    expanded = set(words)
    for w in words:
        if w in _STEM_MAP:
            expanded.add(_STEM_MAP[w])
        for full, stem in _STEM_MAP.items():
            if stem == w:
                expanded.add(full)
    return expanded


# Machine / path scoring — exact set intersection misses DEMO-1325 vs DEMO-X-1325-PWB-UMS
# or documents whose parent folder carries the model (generic PDF filenames).
_MACHINE_MODEL_RE = re.compile(
    r"(PF1[-\s]?[A-Z]?[-\s]?\d+[-\s]?\d*|AM[-\s]?\w+|IMG[-\s]?\d+|FCS[-\s]?\w+|UNO[-\s]?\w+|DUO[-\s]?\w+)",
    re.IGNORECASE,
)


def _alnum_upper(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _pf1_plate_key(machine: str) -> str | None:
    """Return the main 4-digit plate size from a model string (e.g. 1325), if present."""
    m = re.search(r"DEMO.*?(\d{4})\b", machine.upper().replace(" ", ""))
    return m.group(1) if m else None


def _machine_match_score(query_machines: set[str], file_machines: list[str]) -> float:
    """Score alignment between query machine tokens and indexed machine strings.

    Handles variants like DEMO-1325 vs DEMO-X-1325 vs DEMO-1325-PWB-UMS that never match as
    exact strings.
    """
    if not query_machines or not file_machines:
        return 0.0
    score = 0.0
    fms = [str(f).upper().strip() for f in file_machines if f]
    for qm in query_machines:
        qu = qm.upper().strip()
        q_alnum = _alnum_upper(qm)
        q_key = _pf1_plate_key(qm)
        for fm in fms:
            fm_alnum = _alnum_upper(fm)
            f_key = _pf1_plate_key(fm)
            if qu == fm or qu in fm or fm in qu:
                score += 5.0
                break
            if len(q_alnum) >= 6 and (q_alnum in fm_alnum or fm_alnum in q_alnum):
                score += 5.0
                break
            if q_key and f_key and q_key == f_key and "PF1" in fm and "PF1" in qu:
                score += 4.5
                break
    return score


def _path_and_folder_machine_score(rel_path: str, query_machines: set[str]) -> float:
    """Boost when project folders / filenames contain model tokens (even if PDF text doesn't)."""
    if not query_machines:
        return 0.0
    blob = rel_path.lower().replace("\\", "/")
    score = 0.0
    for qm in query_machines:
        q_compact = qm.lower().replace(" ", "")
        if q_compact and q_compact in blob.replace(" ", ""):
            score += 8.0
            continue
        q_key = _pf1_plate_key(qm)
        if q_key and "pf1" in blob and q_key in blob:
            score += 7.0
    return score


async def search_index(
    query: str,
    limit: int = 10,
    doc_type_filter: str = "",
    intent_filters: list[str] | None = None,
    counterparty_filter: str = "",
    role_filter: str = "",
) -> list[dict[str, Any]]:
    """Score files against a query using metadata fields.

    Weights: machine match (5), entity match (3), topic match (2),
    keyword overlap (1), summary hit (0.5), filename hit (0.3),
    relative path / folder hit (up to 8).

    *doc_type_filter*: if set, only return files of this doc_type.
    """
    index = await load_index()
    if not index.get("files"):
        return []

    query_lower = query.lower()
    query_words = _stem_words(set(re.findall(r"\b\w{3,}\b", query_lower)))
    intent_filters = normalize_intent_tags(intent_filters)

    query_machines = {m.upper().replace(" ", "-") for m in _MACHINE_MODEL_RE.findall(query)}

    results: list[dict[str, Any]] = []
    for rel_path, meta in index["files"].items():
        if doc_type_filter and meta.get("doc_type", "") != doc_type_filter:
            continue
        file_intents = normalize_intent_tags(meta.get("intent_tags", []))
        if intent_filters and not any(intent in file_intents for intent in intent_filters):
            continue
        if counterparty_filter and meta.get("counterparty_type", "unknown") != counterparty_filter:
            continue
        if role_filter and meta.get("document_role", "other") != role_filter:
            continue

        score = 0.0
        if intent_filters or counterparty_filter or role_filter:
            # Ensure intent-filtered lookups return relevant candidates even with sparse query terms.
            score += 0.4

        file_machines_list = meta.get("machines", []) or []
        score += _machine_match_score(query_machines, file_machines_list)
        score += _path_and_folder_machine_score(rel_path, query_machines)

        file_keywords = {k.lower() for k in meta.get("keywords", [])}
        score += len(query_words & file_keywords) * 1.0

        file_topics = {t.lower() for t in meta.get("topics", [])}
        for topic, triggers in _TOPIC_TRIGGERS.items():
            if triggers & query_words and topic in file_topics:
                score += 2.0

        for entity in meta.get("entities", []):
            if entity.lower() in query_lower:
                score += 3.0

        if intent_filters:
            score += len(set(intent_filters) & set(file_intents)) * 4.0

        summary = meta.get("summary", "").lower()
        score += sum(0.5 for w in query_words if w in summary and len(w) > 3)

        name_lower = meta.get("name", "").lower()
        path_lower = rel_path.lower()
        score += sum(0.3 for w in query_words if w in name_lower and len(w) > 3)
        score += sum(0.25 for w in query_words if w in path_lower and len(w) > 3)

        if score > 0.3:
            results.append(
                {
                    "path": meta.get("path", ""),
                    "name": meta.get("name", ""),
                    "score": round(score, 2),
                    "summary": meta.get("summary", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "machines": meta.get("machines", []),
                    "topics": meta.get("topics", []),
                    "intent_tags": file_intents,
                    "counterparty_type": meta.get("counterparty_type", "unknown"),
                    "document_role": meta.get("document_role", "other"),
                }
            )

    results.sort(key=lambda x: -x["score"])
    return results[:limit]


# ── stats ────────────────────────────────────────────────────────────────


async def get_index_stats() -> dict[str, Any]:
    """Return summary statistics about the current index."""
    index = await load_index()
    files = index.get("files", {})
    if not files:
        return {"indexed": 0, "built_at": None}

    doc_types: dict[str, int] = {}
    all_machines: set[str] = set()
    intent_covered = 0
    for meta in files.values():
        dt = meta.get("doc_type", "other")
        doc_types[dt] = doc_types.get(dt, 0) + 1
        all_machines.update(meta.get("machines", []))
        if normalize_intent_tags(meta.get("intent_tags", [])):
            intent_covered += 1

    on_disk = (
        sum(
            1
            for f in IMPORTS_DIR.rglob("*")
            if f.is_file()
            and not f.name.startswith(".")
            and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if IMPORTS_DIR.exists()
        else 0
    )

    return {
        "indexed": len(files),
        "total_on_disk": on_disk,
        "unindexed": on_disk - len(files),
        "built_at": index.get("built_at"),
        "doc_types": doc_types,
        "intent_coverage_pct": round((intent_covered * 100 / len(files)), 1),
        "unique_machines": len(all_machines),
        "top_machines": sorted(all_machines)[:20],
    }
