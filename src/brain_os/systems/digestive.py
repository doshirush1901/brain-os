"""Digestive system — data ingestion with nutrient extraction.

Orchestrates the full data ingestion pipeline using a biological metaphor:
MOUTH (receive) → STOMACH (LLM nutrient extraction) → DUODENUM (LLM
summarization into searchable statements) → SMALL INTESTINE (chunk +
embed + upsert) → LIVER (entity extraction into knowledge graph).

The key insight: not all data is equal.  The nutrient extraction step ensures
only high-value information is stored, and the summarization step rewrites
raw facts into clean, self-contained statements that embed well for retrieval.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from typing import Any

import httpx
from anthropic import AnthropicError as _AnthropicError
from langfuse.decorators import observe
from openai import OpenAIError as _OpenAIError
from pydantic import ValidationError

from brain_os.brain.document_ingestor import (
    DocumentIngestor,
    chunk_text,
    get_chunk_params_for_doc_type,
)
from brain_os.brain.embeddings import EmbeddingService
from brain_os.brain.knowledge_graph import KnowledgeGraph
from brain_os.brain.qdrant_manager import QdrantManager
from brain_os.brain.quality_filter import QualityFilter
from brain_os.brain.write_contract import build_receipt, enqueue_retry, record_receipt
from brain_os.config import get_settings
from brain_os.data.models import Email, KnowledgeItem
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import (
    DigestiveSummary,
    EmailMetadata,
    ExtractedContacts,
    NutrientClassification,
)
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_NUTRIENT_SYSTEM_PROMPT = load_prompt("digestive_nutrient")
_SUMMARIZE_SYSTEM_PROMPT = load_prompt("digestive_summarize")
_EMAIL_META_SYSTEM_PROMPT = load_prompt("digestive_email_meta")
_EXTRACT_CONTACTS_SYSTEM_PROMPT = load_prompt("digestive_extract_contacts")

_GRAPH_WRITE_ERRORS = (RuntimeError, ValueError, TypeError, AttributeError, OSError)
_DIGESTIVE_INGEST_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
    AttributeError,
    KeyError,
    json.JSONDecodeError,
    UnicodeDecodeError,
    UnicodeEncodeError,
    _OpenAIError,
    _AnthropicError,
    ValidationError,
)


class DigestiveSystem:
    """Orchestrates the full data ingestion pipeline."""

    _WINDOW_SIZE = 10_000
    _WINDOW_OVERLAP = 500
    _CLASSIFY_MAX_CHARS = 1800
    _CLASSIFY_MAX_LINES = 60

    def __init__(
        self,
        ingestor: DocumentIngestor,
        knowledge_graph: KnowledgeGraph,
        embedding_service: EmbeddingService,
        qdrant: QdrantManager,
    ) -> None:
        self._ingestor = ingestor
        self._graph = knowledge_graph
        self._embeddings = embedding_service
        self._qdrant = qdrant
        self._quality_filter = QualityFilter(
            qdrant_manager=qdrant,
            embedding_service=embedding_service,
        )
        self._llm = get_llm_client()
        _app = get_settings().app
        self._structured_provider: str = _app.digestive_llm_provider
        self._digestive_anthropic_model: str = (_app.digestive_anthropic_model or "").strip()

    def _digestive_model_kw(self) -> dict[str, str]:
        """Structured-ingest model override when billing Anthropic (empty → LLMClient default)."""
        if self._structured_provider != "anthropic" or not self._digestive_anthropic_model:
            return {}
        return {"model": self._digestive_anthropic_model}

    # ── STOMACH: nutrient extraction ──────────────────────────────────────

    async def _extract_nutrients(self, raw_data: str) -> dict[str, list[str]]:
        """Use an LLM to classify text into protein / carbs / waste.

        Long documents are split into overlapping windows so that every
        part of the text is seen by the model.  Results from each window
        are merged into a single nutrient dict.
        """
        if not self._llm.is_provider_configured(self._structured_provider):
            logger.warning(
                "No %s LLM client — skipping nutrient extraction",
                self._structured_provider,
            )
            return {"protein": [raw_data], "carbs": [], "waste": []}

        windows = self._split_windows(raw_data)
        merged: dict[str, list[str]] = {"protein": [], "carbs": [], "waste": []}

        for i, window in enumerate(windows):
            result = await self._classify_window(window)
            for key in ("protein", "carbs", "waste"):
                merged[key].extend(result.get(key, []))
            if len(windows) > 1:
                logger.debug(
                    "Nutrient window %d/%d: protein=%d carbs=%d waste=%d",
                    i + 1,
                    len(windows),
                    len(result.get("protein", [])),
                    len(result.get("carbs", [])),
                    len(result.get("waste", [])),
                )

        return merged

    def _split_windows(self, text: str) -> list[str]:
        """Split *text* into overlapping character windows for nutrient extraction."""
        if len(text) <= self._WINDOW_SIZE:
            return [text]

        windows: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self._WINDOW_SIZE, len(text))
            windows.append(text[start:end])
            if end == len(text):
                break
            start += self._WINDOW_SIZE - self._WINDOW_OVERLAP
        return windows

    def _is_mostly_markup(self, text: str) -> bool:
        """True if text is mostly HTML/CSS/Office markup; skip LLM to avoid huge waste output."""
        markers = (
            "mso-",
            "font-family",
            "behavior:url",
            "WordSection1",
            "@page",
            "@font-face",
            ".shape {",
            "JFIF",
            "MsoNormal",
        )
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return True
        markup_count = sum(1 for ln in lines if any(m in ln for m in markers))
        return markup_count / len(lines) >= 0.5

    async def _classify_window(self, text: str) -> dict[str, list[str]]:
        """Send a single text window to the LLM for nutrient classification."""
        prepared = self._prepare_classification_input(text)
        if not prepared or len(prepared.strip()) < 20:
            return {"protein": [], "carbs": [], "waste": []}
        if self._is_mostly_markup(prepared):
            return {"protein": [], "carbs": [], "waste": ["[HTML/CSS and binary data]"]}
        # NutrientClassification can be large; cap output via prompt + schema validator.
        result = await self._llm.generate_structured(
            _NUTRIENT_SYSTEM_PROMPT,
            prepared,
            NutrientClassification,
            name="digestive.classify",
            max_tokens=8192,
            provider=self._structured_provider,
            **self._digestive_model_kw(),
        )
        return result.model_dump()

    def _is_noise_line(self, line: str) -> bool:
        """True if line is HTML/XML/binary junk that would blow LLM output (e.g. Office markup, PNG dump)."""
        if len(line) > 800:
            # Long lines that look like binary or repeated padding often blow completion tokens.
            non_printable = sum(
                1 for c in line if c not in " \t\n\r" and (ord(c) < 32 or ord(c) > 126)
            )
            if non_printable > 0.25 * len(line):
                return True
            # Repeated short pattern (e.g. "E@ P E@ P E@ ") from binary.
            if len(line) > 200:
                for chunk_len in (4, 5, 6, 8):
                    if chunk_len * 20 > len(line):
                        break
                    first = line[:chunk_len]
                    if line.count(first) * chunk_len > 0.7 * len(line):
                        return True
        # Office/HTML/XML markup: model puts whole blob in waste and hits max_tokens.
        markup_markers = (
            "<!--[if ",
            "<![endif]-->",
            "<o:shapelayout",
            "<o:idmap",
            'v:ext="edit"',
            "</xml>",
            "tEXtSoftware",
            "Microsoft Office",
            "mso-style",
            "font-family",
            "behavior:url",
            "WordSection1",
            "@page ",
            "@font-face",
            ".shape {",
            "JFIF",
            "mso-",
        )
        lower = line.lower()
        for m in markup_markers:
            if m.lower() in lower or m in line:
                return True
        # PNG/binary dump only in long lines (avoid dropping "We support PNG").
        if len(line) > 200 and (" IHDR " in line or " IDAT" in line or "PNG" in line):
            return True
        if line.startswith("data:image/"):
            return True
        return False

    def _prepare_classification_input(self, text: str) -> str:
        """Bound high-cardinality input to avoid oversized structured outputs."""
        if not text:
            return ""

        # Drop noisy placeholders and HTML/binary blobs that explode waste output size.
        filtered_lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line in {"<!-- image -->", "![image]", "[image]"}:
                continue
            if line.startswith("data:image/"):
                continue
            if self._is_noise_line(line):
                continue
            filtered_lines.append(line)

        # Deduplicate repeated lines while preserving first occurrence order.
        seen: set[str] = set()
        deduped_lines: list[str] = []
        for line in filtered_lines:
            if line in seen:
                continue
            seen.add(line)
            deduped_lines.append(line)

        bounded = "\n".join(deduped_lines)
        if len(bounded) > self._CLASSIFY_MAX_CHARS:
            head = bounded[:1200]
            tail = bounded[-500:]
            omitted = len(bounded) - (len(head) + len(tail))
            bounded = f"{head}\n\n[... omitted {omitted} characters ...]\n\n{tail}"

        lines = bounded.splitlines()
        if len(lines) > self._CLASSIFY_MAX_LINES:
            keep_head = 45
            keep_tail = 10
            omitted = max(0, len(lines) - (keep_head + keep_tail))
            lines = lines[:keep_head] + [f"[... omitted {omitted} lines ...]"] + lines[-keep_tail:]

        # If lines are still repetitive variants, keep only most informative ones.
        counts = Counter(lines)
        lines = [line for line in lines if counts[line] == 1 or len(line) > 24]

        return "\n".join(lines)

    # ── DUODENUM: summarization ─────────────────────────────────────────

    async def _summarize_nutrients(self, nutrients: dict[str, list[str]]) -> dict[str, list[str]]:
        """Rewrite raw protein + carbs items as clean, searchable statements.

        This bridges the gap between raw extracted facts and high-quality
        embeddings.  Each statement is self-contained so it retrieves well
        in semantic search.  Waste is passed through unchanged.
        """
        if not self._llm.is_provider_configured(self._structured_provider):
            return nutrients

        summarized: dict[str, list[str]] = {
            "protein": [],
            "carbs": [],
            "waste": nutrients.get("waste", []),
        }

        for nutrient_type in ("protein", "carbs"):
            items = nutrients.get(nutrient_type, [])
            if not items:
                continue

            batch_text = "\n".join(f"- {item}" for item in items)
            for window in self._split_windows(batch_text):
                statements = await self._call_summarize(window)
                summarized[nutrient_type].extend(statements)

        logger.debug(
            "Summarized: protein %d->%d, carbs %d->%d statements",
            len(nutrients.get("protein", [])),
            len(summarized["protein"]),
            len(nutrients.get("carbs", [])),
            len(summarized["carbs"]),
        )
        return summarized

    async def _call_summarize(self, text: str) -> list[str]:
        """Send nutrient items to the LLM for summarization into searchable statements."""
        # DigestiveSummary.statements can be long; use 8192 to avoid truncation.
        result = await self._llm.generate_structured(
            _SUMMARIZE_SYSTEM_PROMPT,
            text,
            DigestiveSummary,
            name="digestive.summarize",
            max_tokens=8192,
            provider=self._structured_provider,
            **self._digestive_model_kw(),
        )
        return [s for s in result.statements if isinstance(s, str) and s.strip()]

    # ── SMALL INTESTINE: absorption ───────────────────────────────────────

    async def _absorb(
        self,
        nutrients: dict[str, list[str]],
        source: str,
        source_category: str,
        source_id: str = "",
        entity_refs: list[tuple[str, str]] | None = None,
        doc_type: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[int, list[KnowledgeItem]]:
        """Chunk protein + carbs and upsert into Qdrant. Returns (count, upserted items).

        If entity_refs is provided, each item's metadata gets graph_entity_ids for
        Qdrant–Neo4j linking (denser retrieval). *doc_type* selects chunk_size/overlap.
        """
        protein_text = "\n".join(nutrients.get("protein", []))
        carbs_text = "\n".join(nutrients.get("carbs", []))
        chunk_size, overlap = get_chunk_params_for_doc_type(doc_type)

        items: list[KnowledgeItem] = []
        graph_entity_ids = [f"{label}:{key}" for label, key in (entity_refs or [])]

        def _first_sentence(text: str, max_chars: int = 200) -> str:
            """First sentence or first max_chars for parent_summary."""
            s = (text or "").strip()
            if not s:
                return ""
            for sep in (". ", ".\n", "\n"):
                idx = s.find(sep)
                if idx != -1:
                    s = s[: idx + 1].strip()
                    break
            return s[:max_chars].strip() or ""

        from brain_os.brain.pii_redact import redact_pii, should_redact_pii

        do_redact = should_redact_pii()

        if protein_text.strip():
            for i, chunk in enumerate(
                chunk_text(protein_text, chunk_size=chunk_size, overlap=overlap)
            ):
                content = chunk
                if do_redact:
                    content, _ = redact_pii(chunk)
                meta: dict[str, Any] = {
                    "nutrient": "protein",
                    "chunk_index": i,
                    "source_id": source_id,
                    "parent_summary": _first_sentence(content),
                    "window": content,
                }
                if graph_entity_ids:
                    meta["graph_entity_ids"] = graph_entity_ids
                if extra_metadata:
                    meta.update(extra_metadata)
                items.append(
                    KnowledgeItem(
                        source=source,
                        source_category=source_category,
                        content=content,
                        metadata=meta,
                    )
                )

        if carbs_text.strip():
            for i, chunk in enumerate(
                chunk_text(carbs_text, chunk_size=chunk_size, overlap=overlap)
            ):
                content = chunk
                if do_redact:
                    content, _ = redact_pii(chunk)
                meta = {
                    "nutrient": "carbs",
                    "chunk_index": i,
                    "source_id": source_id,
                    "parent_summary": _first_sentence(content),
                    "window": content,
                }
                if graph_entity_ids:
                    meta["graph_entity_ids"] = graph_entity_ids
                if extra_metadata:
                    meta.update(extra_metadata)
                items.append(
                    KnowledgeItem(
                        source=source,
                        source_category=source_category,
                        content=content,
                        metadata=meta,
                    )
                )

        if not items:
            return (0, [])

        filtered = []
        for item in items:
            result = self._quality_filter.filter_chunk(item.content)
            if result["pass"]:
                filtered.append(item)
            else:
                logger.debug(
                    "Quality filter rejected chunk (score=%.2f): %s",
                    result["quality_score"],
                    result["reason"],
                )
        items = filtered

        if not items:
            return (0, [])

        count = await self._qdrant.upsert_items(items)
        return (count, items)

    # ── LIVER: entity extraction ──────────────────────────────────────────

    async def _extract_entities(
        self, protein_text: str, *, source_id: str = ""
    ) -> tuple[dict[str, int], list[tuple[str, str]]]:
        """Extract entities from protein content, add to the knowledge graph, return (counts, entity_refs)."""
        empty_counts = {"companies": 0, "people": 0, "machines": 0, "relationships": 0}
        if not protein_text.strip():
            return (empty_counts, [])

        extracted = await self._graph.extract_entities_from_text(protein_text)

        counts: dict[str, int] = {"companies": 0, "people": 0, "machines": 0, "relationships": 0}
        entity_refs: list[tuple[str, str]] = []

        for company in extracted.get("companies", []):
            if not company.get("name"):
                continue
            try:
                await self._graph.add_company(
                    name=company["name"],
                    region=company.get("region", ""),
                    industry=company.get("industry", ""),
                    website=company.get("website", ""),
                    source_id=source_id,
                )
            except _GRAPH_WRITE_ERRORS:
                logger.warning(
                    "Failed to add company %s during ingest",
                    company.get("name"),
                    exc_info=True,
                )
                continue
            counts["companies"] += 1
            entity_refs.append(("Company", company["name"]))

        for person in extracted.get("people", []):
            if person.get("name"):
                await self._graph.add_person(
                    name=person["name"],
                    email=person.get("email", ""),
                    company_name=person.get("company", ""),
                    role=person.get("role", ""),
                    source_id=source_id,
                )
                counts["people"] += 1
                if person.get("email"):
                    entity_refs.append(("Person", person["email"]))

        for machine in extracted.get("machines", []):
            if machine.get("model"):
                await self._graph.add_machine(
                    model=machine["model"],
                    category=machine.get("category", ""),
                    description=machine.get("description", ""),
                    source_id=source_id,
                )
                counts["machines"] += 1
                entity_refs.append(("Machine", machine["model"]))

        rel_count = 0
        for rel in extracted.get("relationships", []):
            try:
                ok = await self._graph.add_relationship(
                    from_type=rel.get("from_type", ""),
                    from_key=rel.get("from_key", ""),
                    rel_type=rel.get("rel", ""),
                    to_type=rel.get("to_type", ""),
                    to_key=rel.get("to_key", ""),
                    properties={
                        k: v
                        for k, v in rel.items()
                        if k not in ("from_type", "from_key", "rel", "to_type", "to_key")
                    },
                    source_id=source_id,
                )
                if ok:
                    rel_count += 1
            except _GRAPH_WRITE_ERRORS:
                logger.debug("Failed to add relationship: %s", rel, exc_info=True)
        counts["relationships"] = rel_count

        return (counts, entity_refs)

    # ── public API ────────────────────────────────────────────────────────

    @observe()
    async def ingest(
        self,
        raw_data: str,
        source: str,
        source_category: str,
        source_id: str = "",
        doc_type: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the full ingestion pipeline on raw text.

        *doc_type* (e.g. quote, manual, contract) is used to choose chunk size and overlap.
        *extra_metadata* is merged into each Qdrant chunk (e.g. teacher canon tags).
        """
        start = time.monotonic()

        if not raw_data or not raw_data.strip():
            return {
                "nutrients_extracted": {"protein": 0, "carbs": 0, "waste": 0},
                "chunks_created": 0,
                "entities_found": {"companies": 0, "people": 0, "machines": 0},
                "processing_time": 0.0,
            }

        try:
            # STOMACH
            nutrients = await self._extract_nutrients(raw_data)

            nutrient_counts = {k: len(v) for k, v in nutrients.items()}

            # DUODENUM — rewrite raw facts as searchable statements
            nutrients = await self._summarize_nutrients(nutrients)

            # LIVER (before absorb so we can link chunks to entities)
            protein_text = "\n".join(nutrients.get("protein", []))
            entities_found, entity_refs = await self._extract_entities(
                protein_text,
                source_id=source_id,
            )

            # SMALL INTESTINE — absorb with graph_entity_ids for denser Qdrant–Neo4j linking
            chunks_created, upserted_items = await self._absorb(
                nutrients,
                source,
                source_category,
                source_id=source_id,
                entity_refs=entity_refs,
                doc_type=doc_type,
                extra_metadata=extra_metadata,
            )
            record_receipt(
                build_receipt(
                    store="qdrant",
                    attempted=True,
                    success=chunks_created > 0,
                    operation="digestive.absorb",
                    run_id=source_id,
                    reason="" if chunks_created > 0 else "no_chunks_created",
                    ids=[item.id.hex for item in upserted_items[:50]],
                    metadata={"source": source, "source_category": source_category},
                )
            )
            if chunks_created == 0:
                enqueue_retry(
                    build_receipt(
                        store="qdrant",
                        attempted=True,
                        success=False,
                        operation="digestive.absorb",
                        run_id=source_id,
                        reason="no_chunks_created",
                        metadata={"source": source, "source_category": source_category},
                    )
                )

            # Link each Qdrant chunk to Neo4j (Chunk node + DESCRIBES edges)
            link_attempts = 0
            link_success = 0
            for item in upserted_items:
                if entity_refs:
                    try:
                        link_attempts += 1
                        await self._graph.add_chunk_and_describes(
                            qdrant_point_id=item.id.hex,
                            source=item.source,
                            source_category=item.source_category,
                            content_preview=item.content[:500],
                            entity_refs=entity_refs,
                        )
                        link_success += 1
                    except _GRAPH_WRITE_ERRORS:
                        logger.debug("Chunk–graph link failed for %s", item.id.hex, exc_info=True)
            if entity_refs:
                neo_receipt = build_receipt(
                    store="neo4j",
                    attempted=link_attempts > 0,
                    success=link_attempts > 0 and link_success == link_attempts,
                    operation="digestive.chunk_links",
                    run_id=source_id,
                    reason="" if link_attempts == link_success else "partial_or_failed_chunk_links",
                    metadata={
                        "source": source,
                        "source_category": source_category,
                        "link_attempts": link_attempts,
                        "link_success": link_success,
                    },
                )
            else:
                neo_receipt = build_receipt(
                    store="neo4j",
                    attempted=False,
                    success=False,
                    operation="digestive.chunk_links",
                    run_id=source_id,
                    reason="no_entity_refs",
                    metadata={"source": source, "source_category": source_category},
                )
            record_receipt(neo_receipt)
            if neo_receipt.attempted and not neo_receipt.success:
                enqueue_retry(neo_receipt)

            elapsed = time.monotonic() - start
            logger.info(
                "DIGEST complete: source=%s nutrients=%s chunks=%d entities=%s time=%.2fs",
                source,
                nutrient_counts,
                chunks_created,
                entities_found,
                elapsed,
            )

            return {
                "nutrients_extracted": nutrient_counts,
                "chunks_created": chunks_created,
                "entities_found": entities_found,
                "processing_time": round(elapsed, 3),
                "source_id": source_id,
            }
        except _DIGESTIVE_INGEST_ERRORS as exc:
            logger.warning(
                "Nutrient extraction or summarization failed for source=%s — saving raw body to KB and graph: %s",
                source,
                exc,
                exc_info=True,
            )
            fallback_nutrients: dict[str, list[str]] = {
                "protein": [raw_data.strip()],
                "carbs": [],
                "waste": [],
            }
            entities_found, entity_refs = await self._extract_entities(raw_data.strip())
            chunks_created, upserted_items = await self._absorb(
                fallback_nutrients,
                source,
                source_category,
                source_id=source_id,
                entity_refs=entity_refs,
                doc_type=doc_type,
            )
            record_receipt(
                build_receipt(
                    store="qdrant",
                    attempted=True,
                    success=chunks_created > 0,
                    operation="digestive.absorb",
                    run_id=source_id,
                    reason="" if chunks_created > 0 else "no_chunks_created",
                    ids=[item.id.hex for item in upserted_items[:50]],
                    metadata={
                        "source": source,
                        "source_category": source_category,
                        "fallback_raw": True,
                    },
                )
            )
            link_attempts = 0
            link_success = 0
            for item in upserted_items:
                if entity_refs:
                    try:
                        link_attempts += 1
                        await self._graph.add_chunk_and_describes(
                            qdrant_point_id=item.id.hex,
                            source=item.source,
                            source_category=item.source_category,
                            content_preview=item.content[:500],
                            entity_refs=entity_refs,
                        )
                        link_success += 1
                    except _GRAPH_WRITE_ERRORS:
                        logger.debug("Chunk–graph link failed for %s", item.id.hex, exc_info=True)
            if entity_refs:
                neo_receipt = build_receipt(
                    store="neo4j",
                    attempted=link_attempts > 0,
                    success=link_attempts > 0 and link_success == link_attempts,
                    operation="digestive.chunk_links",
                    run_id=source_id,
                    reason="" if link_attempts == link_success else "partial_or_failed_chunk_links",
                    metadata={
                        "source": source,
                        "source_category": source_category,
                        "link_attempts": link_attempts,
                        "link_success": link_success,
                        "fallback_raw": True,
                    },
                )
            else:
                neo_receipt = build_receipt(
                    store="neo4j",
                    attempted=False,
                    success=False,
                    operation="digestive.chunk_links",
                    run_id=source_id,
                    reason="no_entity_refs",
                    metadata={
                        "source": source,
                        "source_category": source_category,
                        "fallback_raw": True,
                    },
                )
            record_receipt(neo_receipt)
            if neo_receipt.attempted and not neo_receipt.success:
                enqueue_retry(neo_receipt)
            elapsed = time.monotonic() - start
            return {
                "nutrients_extracted": {"protein": 1, "carbs": 0, "waste": 0},
                "chunks_created": chunks_created,
                "entities_found": entities_found,
                "processing_time": round(elapsed, 3),
                "source_id": source_id,
                "fallback_raw": True,
            }

    @observe()
    async def ingest_email(self, email: Email) -> dict[str, Any]:
        """Ingest an email through the full pipeline with extra metadata extraction."""
        result = await self.ingest(
            raw_data=email.body,
            source=email.from_address,
            source_category="email",
        )

        email_metadata = await self._extract_email_metadata(email.body)
        result["email_metadata"] = email_metadata
        result["email_id"] = email.id
        result["subject"] = email.subject
        return result

    async def _extract_email_metadata(self, body: str) -> dict[str, Any]:
        """Extract email-specific metadata (sender info, mentions, dates)."""
        # EmailMetadata lists can be long for long emails; use 8192 to avoid truncation.
        result = await self._llm.generate_structured(
            _EMAIL_META_SYSTEM_PROMPT,
            body[:12_000],
            EmailMetadata,
            name="digestive.email_meta",
            max_tokens=8192,
            provider=self._structured_provider,
            **self._digestive_model_kw(),
        )
        return result.model_dump()

    @observe()
    async def extract_contacts_from_text(self, text: str) -> list[dict[str, Any]]:
        """Use the LLM to find names, emails, companies, and machine models in raw text.

        For use with unstructured documents (e.g. PDFs, memos in 08_Sales_and_CRM)
        where regex/table parsing fails. Returns a list of contact dicts suitable
        for the CRM populator (keys: name, email, company, machine_model).
        """
        if not text or len(text.strip()) < 20:
            return []
        if not self._llm.is_provider_configured(self._structured_provider):
            logger.warning(
                "No %s LLM client — skipping contact extraction",
                self._structured_provider,
            )
            return []
        # Cap input size to avoid token overflow
        prepared = text[:15_000].strip()
        result = await self._llm.generate_structured(
            _EXTRACT_CONTACTS_SYSTEM_PROMPT,
            prepared,
            ExtractedContacts,
            name="digestive.extract_contacts",
            max_tokens=4096,
            provider=self._structured_provider,
            **self._digestive_model_kw(),
        )
        out: list[dict[str, Any]] = []
        for c in result.contacts:
            email = (c.email or "").strip().lower()
            if not email or "@" not in email:
                continue
            row: dict[str, Any] = {
                "name": (c.name or "").strip() or email.split("@")[0],
                "email": email,
                "company": (c.company or "").strip() or email.split("@")[1].split(".")[0].title(),
            }
            machine = (c.machine_model or "").strip()
            if machine:
                row["machine_model"] = machine
            out.append(row)
        return out

    async def batch_ingest(self, items: list[dict[str, str]]) -> dict[str, Any]:
        """Process multiple items, logging progress and handling errors."""
        total_processed = 0
        total_failed = 0
        total_chunks = 0
        total_entities: dict[str, int] = {
            "companies": 0,
            "people": 0,
            "machines": 0,
            "relationships": 0,
        }
        errors: list[str] = []

        for i, item in enumerate(items):
            try:
                result = await self.ingest(
                    raw_data=item.get("raw_data", ""),
                    source=item.get("source", "unknown"),
                    source_category=item.get("source_category", "uncategorised"),
                )
                total_processed += 1
                total_chunks += result["chunks_created"]
                for key in total_entities:
                    total_entities[key] += result["entities_found"].get(key, 0)
            except _DIGESTIVE_INGEST_ERRORS:
                total_failed += 1
                errors.append(item.get("source", f"item_{i}"))
                logger.exception("Batch ingest failed for item %d", i)

            if (i + 1) % 10 == 0:
                logger.info("Batch progress: %d/%d processed", i + 1, len(items))

        return {
            "total_processed": total_processed,
            "total_failed": total_failed,
            "total_chunks": total_chunks,
            "total_entities": total_entities,
            "errors": errors,
        }
