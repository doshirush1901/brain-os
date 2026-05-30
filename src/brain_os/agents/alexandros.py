"""Alexandros — The Librarian and Ingestion Gatekeeper.

Gatekeeper of ``examples/acme/docs/``.  Every file in the archive has been
catalogued with an LLM-generated summary, entities, machines, topics,
and keywords.  Alexandros holds this catalogue in memory and serves
three functions:

1. **ask** — hybrid search (keyword + Voyage semantic) over the
   catalogue, extract text from the best matching files, and synthesise
   a focused answer via LLM.
2. **browse** — list files in a folder with summaries.
3. **read_file** — read a specific file by name.

As Ingestion Gatekeeper, Alexandros also owns the ingestion log and
decides which files need to be ingested (or re-ingested) into Qdrant
through the DigestiveSystem pipeline.  The respiratory system's inhale
cycle delegates to ``run_ingestion_cycle()`` rather than calling the
raw ingestor directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.agents.mnemon import _load_ledger as _load_correction_ledger
from brain_os.brain.imports_fallback_retriever import (
    extract_file_text,
    hybrid_search,
    queue_for_deferred_ingestion,
)
from brain_os.brain.imports_intents import infer_query_intents, normalize_intent_tags
from brain_os.brain.imports_metadata_index import load_index
from brain_os.brain.ingestion_gatekeeper import (
    run_ingestion_cycle,
    scan_for_undigested,
)
from brain_os.exceptions import LLMError, ToolExecutionError
from brain_os.prompt_loader import load_prompt
from brain_os.service_keys import ServiceKey as SK
from brain_os.services.llm_client import reset_llm_circuit_breakers

logger = logging.getLogger(__name__)

_PDFCO_ERRORS = (httpx.HTTPError, OSError, ToolExecutionError, RuntimeError, ValueError)
_LLM_SYNTHESIS_ERRORS = (LLMError, httpx.HTTPError, asyncio.TimeoutError, ToolExecutionError)
_CORRECTION_OVERLAY_ERRORS = (
    OSError,
    KeyError,
    ValueError,
    TypeError,
    AttributeError,
    json.JSONDecodeError,
)

_SYSTEM_PROMPT = load_prompt("alexandros_system")

_DOC_TYPE_HINTS: dict[str, str] = {
    "quote": {
        "price",
        "cost",
        "quote",
        "quotation",
        "offer",
        "pricing",
        "lakh",
        "inr",
        "usd",
        "euro",
    },
    "technical_spec": {"spec", "specification", "technical", "dimension", "capacity", "heater"},
    "order": {"order", "purchase", "po"},
    "contract": {"contract", "agreement", "nda", "terms"},
    "lead_list": {"lead", "prospect", "contact", "visitor"},
    "presentation": {"presentation", "ppt", "slide", "deck"},
    "manual": {"manual", "instruction", "operating", "guide"},
    "catalogue": {"catalogue", "catalog", "brochure"},
}

_MANUAL_INGEST_TRIGGER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(i\s+added|i\s+uploaded|i\s+dropped|new\s+file|new\s+document)\b.*\b(ingest|index|sync|update)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(ingest|index|sync|update)\b.*\b(new\s+file|new\s+document|imports?|archive)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bplease\s+(ingest|index|sync|update)\b",
        re.IGNORECASE,
    ),
)


def _infer_doc_type(query: str) -> str:
    """Infer a doc_type filter from query keywords, or return empty."""
    words = set(re.findall(r"\b\w{3,}\b", query.lower()))
    best, best_overlap = "", 0
    for dtype, triggers in _DOC_TYPE_HINTS.items():
        overlap = len(words & triggers)
        if overlap > best_overlap:
            best, best_overlap = dtype, overlap
    return best if best_overlap >= 2 else ""


class Alexandros(BaseAgent):
    name = "alexandros"
    role = "Librarian"
    description = (
        "Gatekeeper of the raw document archive. Searches, browses, "
        "and reads files from examples/acme/docs/ using LLM-generated metadata."
    )
    knowledge_categories = [
        "company_internal",
        "product_catalogues",
        "project_case_studies",
        "contracts_and_legal",
    ]

    # ── tool registration ─────────────────────────────────────────────────

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="search_archive",
                description="Hybrid search the document archive by query. Optionally filter by folder.",
                parameters={
                    "query": "Search query",
                    "folder": "Optional folder name to restrict search to",
                },
                handler=self._tool_search_archive,
            )
        )
        self.register_tool(
            AgentTool(
                name="browse_folder",
                description="List files in an archive folder with summaries and metadata.",
                parameters={"path": "Folder name or search term (empty for root listing)"},
                handler=self._tool_browse_folder,
            )
        )
        self.register_tool(
            AgentTool(
                name="read_file",
                description="Read the full text of a specific file from the archive.",
                parameters={"file_path": "Filename or relative path in the archive"},
                handler=self._tool_read_file,
            )
        )
        self.register_tool(
            AgentTool(
                name="get_archive_stats",
                description="Get statistics about the archive (file counts, types, folders).",
                parameters={},
                handler=self._tool_get_archive_stats,
            )
        )
        self.register_tool(
            AgentTool(
                name="extract_document_tables",
                description=(
                    "Extract tables and structured data from a PDF in the archive "
                    "(e.g. PO copies, quotes, order confirmations). Returns CSV-formatted "
                    "table data with pricing, specs, delivery dates, and payment terms. "
                    "Use this when you need exact numbers from a document."
                ),
                parameters={"file_path": "Filename or path of the PDF in the archive"},
                handler=self._tool_extract_document_tables,
            )
        )
        self.register_tool(
            AgentTool(
                name="scan_undigested",
                description="Scan for files not yet ingested into the vector store.",
                parameters={},
                handler=self._tool_scan_undigested,
            )
        )
        self.register_tool(
            AgentTool(
                name="queue_for_ingestion",
                description="Queue a file for ingestion into the vector store.",
                parameters={"file_path": "Path of the file to ingest"},
                handler=self._tool_queue_for_ingestion,
            )
        )
        self.register_tool(
            AgentTool(
                name="search_knowledge_base_skill",
                description="Run canonical skill search over internal knowledge for archive questions.",
                parameters={"query": "Search query"},
                handler=self._tool_search_knowledge_base_skill,
            )
        )
        self.register_tool(
            AgentTool(
                name="extract_key_facts_skill",
                description="Extract structured entities and key facts from archive text.",
                parameters={"text": "Raw archive text"},
                handler=self._tool_extract_key_facts_skill,
            )
        )
        self.register_tool(
            AgentTool(
                name="summarize_document_skill",
                description="Summarize archive document text for quick review.",
                parameters={"text": "Raw document text"},
                handler=self._tool_summarize_document_skill,
            )
        )

    # ── tool handlers ─────────────────────────────────────────────────────

    async def _tool_search_archive(self, query: str, folder: str = "") -> str:
        ctx: dict[str, Any] = {}
        if folder:
            ctx["doc_type"] = ""
        result = await self.ask(query, ctx)
        return result

    async def _tool_browse_folder(self, path: str = "") -> str:
        return await self.browse(path or "")

    async def _tool_read_file(self, file_path: str) -> str:
        return await self.read_file(file_path)

    async def _tool_get_archive_stats(self) -> str:
        return await self.stats()

    async def _tool_extract_document_tables(self, file_path: str) -> str:
        """Extract tables from a PDF using PDF.co or Docling fallback."""
        index = await load_index()
        files = index.get("files", {})

        target: dict[str, Any] | None = None
        fn_lower = file_path.lower().strip()
        for rel_path, meta in files.items():
            if fn_lower in rel_path.lower() or fn_lower in meta.get("name", "").lower():
                target = meta
                break

        if not target:
            return f"File '{file_path}' not found in the archive."

        filepath = target.get("path", "")
        if not filepath or not Path(filepath).exists():
            return f"File '{target.get('name', '')}' is indexed but missing from disk."

        pdfco = self._services.get(SK.PDFCO)
        if pdfco and filepath.lower().endswith(".pdf"):
            try:
                file_bytes = Path(filepath).read_bytes()
                csv_data = await pdfco.extract_tables_csv(file_bytes=file_bytes)
                if csv_data and len(csv_data.strip()) > 10:
                    return f"Tables extracted from {target.get('name', '')}:\n\n{csv_data[:4000]}"
            except _PDFCO_ERRORS as exc:
                logger.warning("PDF.co table extraction failed for %s: %s", filepath, exc)

        text = await extract_file_text(filepath)
        if not text:
            return f"Could not extract content from '{target.get('name', '')}'."

        raw = (
            f"Full text of {target.get('name', '')} "
            f"(table extraction unavailable, returning raw text):\n\n"
            f"{text[:4000]}"
        )
        return await self._apply_corrections_overlay(raw)

    async def _tool_scan_undigested(self) -> str:
        results = await self.scan_for_undigested_files()
        if not results:
            return "All files are up to date — nothing to ingest."
        return json.dumps(results[:20], default=str)

    async def _tool_queue_for_ingestion(self, file_path: str) -> str:
        result = await self.run_ingestion(batch_size=1)
        return json.dumps(result, default=str)

    async def _tool_search_knowledge_base_skill(self, query: str) -> str:
        return await self.use_skill("search_knowledge_base", query=query)

    async def _tool_extract_key_facts_skill(self, text: str) -> str:
        return await self.use_skill("extract_key_facts", text=text)

    async def _tool_summarize_document_skill(self, text: str) -> str:
        return await self.use_skill("summarize_document", text=text)

    # ── main handler (orchestrator entry point) ──────────────────────────

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        context = context or {}
        action = context.get("action", "")

        if action in {"ingest", "manual_ingest"} or self._looks_like_manual_ingest_trigger(query):
            return await self.run_manual_ingest_trigger(query=query, context=context)

        if action == "browse":
            return await self.browse(
                context.get("folder", query),
                doc_type=context.get("doc_type", ""),
            )
        if action == "read_file":
            return await self.read_file(context.get("filename", query))
        if action == "stats":
            return await self.stats()

        if action == "ask":
            return await self.ask(query, context)

        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)

    # ── Tool 1: ask — the main search interface ─────────────────────────

    async def ask(self, query: str, context: dict[str, Any] | None = None) -> str:
        """Hybrid search the catalogue, extract text, optionally synthesise."""
        context = context or {}

        doc_type_filter = context.get("doc_type", "") or _infer_doc_type(query)
        inferred_intents, inferred_counterparty, inferred_role = infer_query_intents(query)
        intent_filters = normalize_intent_tags(context.get("intent_tags", inferred_intents))
        counterparty_filter = context.get("counterparty_type", "") or inferred_counterparty
        role_filter = context.get("document_role", "") or inferred_role
        synthesize = context.get("synthesize", True)
        limit = 5 if context.get("full_text_in_body") else 3

        candidates = await hybrid_search(
            query,
            limit=limit,
            doc_type_filter=doc_type_filter,
            intent_filters=intent_filters,
            counterparty_filter=counterparty_filter,
            role_filter=role_filter,
        )

        if not candidates and (
            doc_type_filter or intent_filters or role_filter or counterparty_filter
        ):
            candidates = await hybrid_search(query, limit=limit, doc_type_filter="")

        if context.get("full_text_in_body") and candidates:
            query_terms = set(re.findall(r"\w+", query.lower())) - {
                "a",
                "an",
                "the",
                "is",
                "are",
                "for",
                "from",
                "with",
                "and",
                "or",
                "in",
                "on",
                "to",
            }
            query_terms = {t for t in query_terms if len(t) > 2}
            filtered = []
            for c in candidates:
                path = c.get("path", "")
                if not path or not Path(path).exists():
                    continue
                text = (await extract_file_text(path)).lower()
                if query_terms and any(term in text for term in query_terms):
                    filtered.append(c)
            if filtered:
                candidates = filtered[:3]

        if not candidates:
            folder = await self._match_folder(query)
            if folder:
                return await self.browse(folder)
            index = await load_index()
            total = len(index.get("files", {}))
            return (
                f"Alexandros: I searched the archive ({total} files) "
                f"but found nothing matching '{query}'. Try different keywords, "
                f"a customer name, machine model, or project number."
            )

        doc_texts: list[str] = []
        citations: list[str] = []
        for candidate in candidates:
            filepath = candidate.get("path", "")
            filename = candidate.get("name", "")
            if not filepath or not Path(filepath).exists():
                continue

            text = await extract_file_text(filepath)
            if not text or len(text.strip()) < 30:
                continue

            header = (
                f"━━━ {filename} ━━━\n"
                f"Type: {candidate.get('doc_type', 'unknown')} | "
                f"Role: {candidate.get('document_role', 'other')} | "
                f"Summary: {candidate.get('summary', 'N/A')[:200]}\n"
            )
            doc_texts.append(header + text)
            citations.append(
                f"- **{filename}** ({candidate.get('doc_type', 'unknown')}): "
                f"{candidate.get('summary', 'N/A')[:150]}"
            )

            await queue_for_deferred_ingestion(
                filepath,
                filename,
                query,
                candidate.get("doc_type", "other"),
            )

        if not doc_texts:
            return "Alexandros: Found candidates but couldn't extract text. Files may be images or corrupted."

        if not synthesize:
            preamble = (
                f"**Alexandros retrieved {len(doc_texts)} document(s) from the archive "
                f'for: "{query}"**\n\n'
            )
            return preamble + "\n\n".join(doc_texts)

        return await self._synthesize(query, doc_texts, citations)

    async def _synthesize(
        self,
        query: str,
        doc_texts: list[str],
        citations: list[str],
    ) -> str:
        """Use LLM to produce a focused answer from retrieved documents."""
        combined = "\n\n---\n\n".join(doc_texts)
        if len(combined) > 12000:
            combined = combined[:12000] + "\n\n[... truncated ...]"

        system = (
            "You are Alexandros, the Librarian of the Brain OS Pantheon. "
            "You have retrieved documents from the archive. Synthesise a clear, "
            "focused answer to the user's question using ONLY the document content below. "
            "Always cite the specific filename when referencing information. "
            "If the documents don't contain the answer, say so honestly."
        )
        user_msg = f"QUESTION: {query}\n\nRETRIEVED DOCUMENTS:\n\n{combined}"

        try:
            answer = await self.call_llm(
                system_prompt=system,
                user_message=user_msg,
                temperature=0.2,
            )
        except _LLM_SYNTHESIS_ERRORS:
            logger.warning("LLM synthesis failed, returning raw documents")
            preamble = (
                f"**Alexandros retrieved {len(doc_texts)} document(s) from the archive "
                f'for: "{query}"**\n\n'
            )
            return preamble + "\n\n".join(doc_texts)

        citation_block = "\n".join(citations)
        result = f"{answer}\n\n---\n**Sources ({len(citations)} documents):**\n{citation_block}"
        return await self._apply_corrections_overlay(result)

    # ── Tool 2: browse — list files in a folder ─────────────────────────

    async def browse(self, folder_or_query: str, doc_type: str = "") -> str:
        """List files in a folder with summaries."""
        index = await load_index()
        files = index.get("files", {})

        folder = await self._match_folder(folder_or_query)
        q_lower = folder_or_query.lower()

        matching: list[tuple[str, dict[str, Any]]] = []
        for rel_path, meta in files.items():
            include = False
            if folder and rel_path.startswith(folder + "/"):
                include = True
            elif not folder:
                if q_lower in rel_path.lower() or q_lower in meta.get("summary", "").lower():
                    include = True

            if include and doc_type and meta.get("doc_type", "") != doc_type:
                include = False

            if include:
                matching.append((rel_path, meta))

        if not matching:
            folders = sorted({rp.split("/")[0] for rp in files if "/" in rp})
            return (
                f"Alexandros: No files found matching '{folder_or_query}'.\n\n"
                f"Available folders:\n" + "\n".join(f"  - {f}" for f in folders)
            )

        matching.sort(key=lambda x: x[0])

        lines = [f"**Archive: {folder or folder_or_query}** ({len(matching)} files)\n"]

        by_subfolder: dict[str, list[dict[str, Any]]] = {}
        for rel_path, meta in matching:
            parts = rel_path.split("/")
            subfolder = parts[1] if len(parts) > 2 else "(root)"
            by_subfolder.setdefault(subfolder, []).append(meta)

        for subfolder, metas in sorted(by_subfolder.items()):
            if subfolder != "(root)":
                lines.append(f"\n**{subfolder}/**")
            for meta in metas:
                name = meta.get("name", "?")
                summary = meta.get("summary", "")[:120]
                doc_t = meta.get("doc_type", "")
                machines = ", ".join(meta.get("machines", [])[:3])
                size = meta.get("size_kb", 0)

                line = f"- **{name}** ({doc_t}, {size}KB)"
                if machines:
                    line += f" [{machines}]"
                if summary and not summary.startswith("Document:"):
                    line += f"\n  _{summary}_"
                lines.append(line)

        return "\n".join(lines)

    # ── Tool 3: read_file — read a specific file ────────────────────────

    async def read_file(self, filename_or_path: str) -> str:
        """Read a specific file from the archive by name or path."""
        index = await load_index()
        files = index.get("files", {})

        target: dict[str, Any] | None = None
        fn_lower = filename_or_path.lower().strip()

        for rel_path, meta in files.items():
            if rel_path.lower() == fn_lower or meta.get("name", "").lower() == fn_lower:
                target = meta
                break

        if not target:
            for _rel_path, meta in files.items():
                if fn_lower in meta.get("name", "").lower():
                    target = meta
                    break

        if not target:
            return f"Alexandros: File '{filename_or_path}' not found in the archive."

        filepath = target.get("path", "")
        if not filepath or not Path(filepath).exists():
            return f"Alexandros: File '{target.get('name', '')}' is indexed but missing from disk."

        text = await extract_file_text(filepath)
        if not text or len(text.strip()) < 10:
            return f"Alexandros: Could not extract text from '{target.get('name', '')}'. It may be an image or corrupted PDF."

        await queue_for_deferred_ingestion(
            filepath,
            target.get("name", ""),
            f"file_detail:{filename_or_path}",
            target.get("doc_type", "other"),
        )

        header = (
            f"━━━ {target.get('name', '')} ━━━\n"
            f"Type: {target.get('doc_type', 'unknown')} | "
            f"Size: {target.get('size_kb', 0)}KB | "
            f"Summary: {target.get('summary', 'N/A')[:200]}\n"
            f"Machines: {', '.join(target.get('machines', []))}\n"
            f"Entities: {', '.join(target.get('entities', []))}\n\n"
        )
        return await self._apply_corrections_overlay(header + text)

    # ── correction overlay ─────────────────────────────────────────────

    @staticmethod
    async def _apply_corrections_overlay(text: str) -> str:
        """Append Mnemon corrections for any entities found in the text."""
        try:
            ledger = await _load_correction_ledger()
            entities = ledger.get("entities", {})
            if not entities:
                return text

            text_lower = text.lower()
            relevant: list[str] = []
            for key, entry in entities.items():
                if key in text_lower:
                    has_stale = any(s.lower() in text_lower for s in entry.get("stale_values", []))
                    if has_stale:
                        relevant.append(
                            f"- {key}: {entry['current_status']} "
                            f"(corrected {entry.get('corrected_at', '?')})"
                        )

            if not relevant:
                return text

            return (
                text
                + "\n\n--- MNEMON CORRECTIONS (override stale data above) ---\n"
                + "\n".join(relevant)
                + "\n--- END CORRECTIONS ---"
            )
        except _CORRECTION_OVERLAY_ERRORS:
            return text

    # ── stats ────────────────────────────────────────────────────────────

    async def stats(self) -> str:
        """Quick stats about the archive."""
        index = await load_index()
        files = index.get("files", {})
        total = len(files)

        if total == 0:
            return (
                "Alexandros: Archive is empty. Files uploaded via /api/ingest are indexed "
                "automatically. For files added directly to examples/acme/docs/, run `brain index-imports`."
            )

        by_folder: dict[str, int] = {}
        by_type: dict[str, int] = {}
        llm_count = 0
        for rel_path, meta in files.items():
            folder = rel_path.split("/")[0] if "/" in rel_path else "(root)"
            by_folder[folder] = by_folder.get(folder, 0) + 1
            dt = meta.get("doc_type", "other")
            by_type[dt] = by_type.get(dt, 0) + 1
            s = meta.get("summary", "")
            if not s.startswith("Document:") and len(s) > 20:
                llm_count += 1

        pct = llm_count * 100 // total if total else 0
        lines = [
            "**Alexandros Archive Stats**",
            f"Total files: {total} | LLM-summarized: {llm_count} ({pct}%)",
            f"Built at: {index.get('built_at', 'unknown')}",
            "",
            "**By folder:**",
        ]
        for folder in sorted(by_folder, key=lambda f: -by_folder[f]):
            lines.append(f"  {folder}: {by_folder[folder]}")

        lines.append("\n**By type:**")
        for dt in sorted(by_type, key=lambda t: -by_type[t]):
            lines.append(f"  {dt}: {by_type[dt]}")

        return "\n".join(lines)

    # ── Ingestion Gatekeeper (delegates to ingestion_gatekeeper module) ──

    async def scan_for_undigested_files(self, *, force: bool = False) -> list[dict[str, Any]]:
        """Compare the metadata index against the ingestion log."""
        return await scan_for_undigested(force=force)

    async def run_ingestion(
        self,
        *,
        force: bool = False,
        batch_size: int = 50,
    ) -> dict[str, Any]:
        """Run a gated ingestion cycle through the DigestiveSystem."""
        return await run_ingestion_cycle(force=force, batch_size=batch_size)

    async def run_manual_ingest_trigger(
        self,
        *,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Manual activation path for Cursor prompt: ingest newly added imports."""
        context = context or {}
        include_prefixes = tuple(context.get("include_prefixes") or ())
        exclude_prefixes = tuple(context.get("exclude_prefixes") or ())
        force = bool(context.get("force", False))
        concurrency = int(context.get("workers", 2) or 2)
        max_batch = int(context.get("batch_size", 40) or 40)
        max_batch = max(1, min(max_batch, 300))
        concurrency = max(1, min(concurrency, 5))

        reset_llm_circuit_breakers()
        queue = await scan_for_undigested(
            force=force,
            include_prefixes=include_prefixes,
            exclude_prefixes=exclude_prefixes,
        )
        pending = len(queue)
        if pending == 0:
            return (
                "Alexandros Sentinel: No pending files in data/imports. "
                "Archive is already up to date."
            )

        result = await run_ingestion_cycle(
            force=force,
            batch_size=min(max_batch, pending),
            concurrency=concurrency,
            include_prefixes=include_prefixes,
            exclude_prefixes=exclude_prefixes,
        )
        processed = int(result.get("files_processed", 0) or 0)
        skipped = int(result.get("files_skipped", 0) or 0)
        failed = int(result.get("files_failed", 0) or 0)
        remaining = int(result.get("files_remaining", 0) or 0)
        chunks = int(result.get("total_chunks", 0) or 0)
        return (
            "Alexandros Sentinel completed manual ingest activation.\n"
            f"- Trigger query: {query[:140]}\n"
            f"- Pending before run: {pending}\n"
            f"- Processed: {processed}\n"
            f"- Skipped: {skipped}\n"
            f"- Failed: {failed}\n"
            f"- Chunks created: {chunks}\n"
            f"- Remaining after batch: {remaining}"
        )

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _looks_like_manual_ingest_trigger(query: str) -> bool:
        q = (query or "").strip()
        if not q:
            return False
        return any(p.search(q) for p in _MANUAL_INGEST_TRIGGER_PATTERNS)

    async def _match_folder(self, query: str) -> str | None:
        """Try to match a query to a folder name in the index."""
        q = query.lower().strip()
        index = await load_index()
        folders: set[str] = set()
        for rel_path in index.get("files", {}):
            if "/" in rel_path:
                folders.add(rel_path.split("/")[0])

        for folder in folders:
            if q in folder.lower() or folder.lower() in q:
                return folder
            parts = folder.lower().replace("_", " ").split()
            if any(p in q for p in parts if len(p) > 3):
                return folder
        return None
