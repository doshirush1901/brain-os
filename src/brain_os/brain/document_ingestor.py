"""Document ingestion pipeline for Ira's knowledge base.

Walks the ``data/imports/`` directory tree, reads files in every supported
format (PDF, XLSX, DOCX, CSV, TXT), splits them into token-counted
overlapping chunks, and upserts the resulting :class:`KnowledgeItem` objects
into Qdrant via :class:`QdrantManager`.

A lightweight SQLite ledger (``data/ingested_files.db``) tracks which files
have already been processed so that re-running ingestion is idempotent.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import csv
import errno
import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import tiktoken

from brain_os.brain.qdrant_manager import QdrantManager
from brain_os.data.models import KnowledgeItem
from brain_os.exceptions import DatabaseError, IngestionError, LLMError, PathTraversalError

if TYPE_CHECKING:
    from brain_os.brain.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

_INGEST_GRAPH_WRITE_ERRORS = (DatabaseError, OSError, ValueError, TypeError)


def _source_to_id(source: str) -> str:
    return re.sub(r"[^a-z0-9:_-]+", "_", (source or "").strip().lower()).strip("_")[:180]


_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".xlsx",
    ".docx",
    ".txt",
    ".csv",
    ".pptx",
    ".html",
    ".md",
    ".eml",
    ".json",
}
_DEFAULT_CHUNK_SIZE = 512
_DEFAULT_OVERLAP = 128
_TIKTOKEN_ENCODING = "cl100k_base"


def get_chunk_params_for_doc_type(doc_type: str) -> tuple[int, int]:
    """Return (chunk_size, overlap) for the given doc_type.

    Sizes are driven by ``AppConfig.ingest_chunk_*`` so operators can tune without code changes.
    """
    from brain_os.config import get_settings

    a = get_settings().app
    key = (doc_type or "other").strip().lower()
    quote = (a.ingest_chunk_quote_tokens, a.ingest_chunk_quote_overlap)
    tech = (a.ingest_chunk_technical_tokens, a.ingest_chunk_technical_overlap)
    default = (a.ingest_chunk_default_tokens, a.ingest_chunk_default_overlap)
    if key in {"quote", "contract", "invoice", "order", "lead_list", "customer_data"}:
        return quote
    if key in {"technical_spec", "spreadsheet"}:
        return tech
    if key in {"email"}:
        return tech
    if key in {
        "manual",
        "report",
        "presentation",
        "catalogue",
        "brochure",
        "other",
    }:
        return default
    return default


_LEDGER_PATH = Path("data/ingested_files.db")

_CATEGORY_PATTERN = re.compile(r"^\d{2}_(.+)$")


# ── SQLite ledger ────────────────────────────────────────────────────────────


def _init_ledger(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingested_files (
            path        TEXT PRIMARY KEY,
            hash        TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            ingested_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


# ── file readers ─────────────────────────────────────────────────────────────


_MIN_USEFUL_CHARS = 50


@contextmanager
def _suppress_pypdf_warnings() -> Iterator[None]:
    """Vendor PDFs often have xref quirks; pypdf logs WARNING but still extracts text."""
    names = ("pypdf", "pypdf._reader", "pypdf.generic", "pypdf.pdf", "pypdf.xref")
    previous: dict[str, int] = {}
    for name in names:
        lg = logging.getLogger(name)
        previous[name] = lg.level
        lg.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for name in names:
            logging.getLogger(name).setLevel(previous[name])


def read_pdf(path: Path, *, use_document_ai_fallback: bool = True) -> str:
    """Extract PDF text via the legacy pypdf chain (no Docling/Unstructured).

    Order when *use_document_ai_fallback* is True: pypdf → PDF.co (if configured) →
    Document AI OCR. When False, only pypdf is used (avoids slow/costly fallbacks during
    bulk passes). For the full ingestion-quality cascade, use :func:`extract_pdf_text`.
    """
    text = _read_pdf_pypdf(path)
    if len(text.strip()) >= _MIN_USEFUL_CHARS:
        return text
    if use_document_ai_fallback:
        pdfco_text = _read_pdf_pdfco(path)
        if len(pdfco_text.strip()) >= _MIN_USEFUL_CHARS:
            logger.info("PDF.co recovered text for %s (%d chars)", path, len(pdfco_text.strip()))
            return pdfco_text
        ocr_text = _read_pdf_document_ai(path)
        if ocr_text:
            return ocr_text
    return text


def _read_pdf_pypdf(path: Path) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        with _suppress_pypdf_warnings():
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except PdfReadError as exc:
        logger.warning("pypdf could not read %s: %s", path, exc)
        return ""


def _read_pdf_pdfco(path: Path) -> str:
    """Call PDF.co text extraction when pypdf output is too thin (before Document AI OCR)."""
    try:
        from brain_os.config import get_settings
        from brain_os.systems.pdfco import PdfCoError, PdfCoService

        if not get_settings().pdfco.api_key.get_secret_value().strip():
            return ""

        svc = PdfCoService()
        if not svc.available:
            return ""

        async def _extract() -> str:
            try:
                return await svc.extract_text(path.read_bytes())
            except PdfCoError:
                return ""

        wait_t = 75.0

        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_extract(), loop)
            try:
                return future.result(timeout=wait_t)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "PDF.co text extraction exceeded %.0fs wait for %s",
                    wait_t,
                    path,
                )
                return ""
        try:
            return asyncio.run(asyncio.wait_for(_extract(), timeout=wait_t))
        except TimeoutError:
            logger.warning(
                "PDF.co text extraction exceeded %.0fs wait for %s",
                wait_t,
                path,
            )
            return ""
    except (
        httpx.HTTPError,
        LLMError,
        OSError,
        TimeoutError,
        concurrent.futures.TimeoutError,
        ValueError,
        TypeError,
    ) as exc:
        logger.warning("PDF.co text extraction failed for %s", path, exc_info=True)
        return ""


def _read_pdf_document_ai(path: Path) -> str:
    """OCR fallback via Document AI for scanned/image-heavy PDFs."""
    try:
        from brain_os.config import get_settings
        from brain_os.systems.document_ai import DocumentAIService

        settings = get_settings()
        if not settings.document_ai.processor_id:
            return ""

        svc = DocumentAIService()

        async def _ocr() -> str:
            await svc.connect()
            return await svc.extract_text(path.read_bytes())

        # Match httpx read timeout in DocumentAIService plus margin for connect/token.
        req_t = float(settings.document_ai.request_timeout_seconds)
        wait_t = req_t + 45.0

        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_ocr(), loop)
            try:
                return future.result(timeout=wait_t)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "Document AI OCR exceeded %.0fs wait for %s (raise DOCUMENT_AI_REQUEST_TIMEOUT_SECONDS if needed)",
                    wait_t,
                    path,
                )
                return ""
        else:
            try:
                return asyncio.run(asyncio.wait_for(_ocr(), timeout=wait_t))
            except TimeoutError:
                logger.warning(
                    "Document AI OCR exceeded %.0fs wait for %s (raise DOCUMENT_AI_REQUEST_TIMEOUT_SECONDS if needed)",
                    wait_t,
                    path,
                )
                return ""
    except (LLMError, DatabaseError, httpx.HTTPError, OSError, TimeoutError) as exc:
        logger.warning("Document AI OCR fallback failed for %s", path, exc_info=True)
        return ""


def read_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    try:
        parts: list[str] = []
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            parts.append(f"[Sheet: {sheet}]")
            for row in ws.iter_rows(values_only=True):
                parts.append("\t".join(str(cell) if cell is not None else "" for cell in row))
        return "\n".join(parts)
    finally:
        wb.close()


def read_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(para.text for para in doc.paragraphs)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_md(path: Path) -> str:
    return read_txt(path)


def read_json(path: Path) -> str:
    """Flatten Firecrawl / manifest JSON into searchable text for ingestion."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    lines: list[str] = []

    def _walk(obj: Any, prefix: str = "") -> None:
        if len(lines) > 500:
            return
        if isinstance(obj, dict):
            for key, val in obj.items():
                key_s = str(key)
                next_prefix = f"{prefix}.{key_s}" if prefix else key_s
                _walk(val, next_prefix)
        elif isinstance(obj, list):
            for idx, val in enumerate(obj[:40]):
                _walk(val, f"{prefix}[{idx}]")
        elif isinstance(obj, (str, int, float, bool)):
            text = str(obj).strip()
            if not text or (isinstance(obj, str) and len(text) < 3):
                return
            if prefix:
                lines.append(f"{prefix}: {text}")
            else:
                lines.append(text)

    _walk(data)
    return "\n".join(lines) if lines else raw[:50_000]


def read_csv(path: Path) -> str:
    buf = io.StringIO()
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            buf.write("\t".join(row) + "\n")
    return buf.getvalue()


def read_pptx(path: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(path))
    parts: list[str] = []
    for i, slide in enumerate(prs.slides, 1):
        texts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = para.text.strip()
                    if text:
                        texts.append(text)
        if texts:
            parts.append(f"[Slide {i}]")
            parts.extend(texts)
    return "\n".join(parts)


def read_xls(path: Path) -> str:
    import xlrd

    wb = xlrd.open_workbook(str(path))
    parts: list[str] = []
    for sheet in wb.sheets():
        parts.append(f"[Sheet: {sheet.name}]")
        for row_idx in range(sheet.nrows):
            cells = [str(sheet.cell_value(row_idx, col)) for col in range(sheet.ncols)]
            parts.append("\t".join(cells))
    return "\n".join(parts)


def _read_with_unstructured(path: Path) -> str:
    """Parse a document via the Unstructured.io API.

    Handles PDF, DOCX, PPTX, XLSX, images, HTML, and EML with automatic
    OCR, table extraction, and layout detection.  Returns empty string
    when the API key is not configured or the call fails.
    """
    try:
        from brain_os.config import get_settings

        settings = get_settings()
        api_key = settings.unstructured.api_key.get_secret_value()
        api_url = settings.unstructured.api_url
        if not api_key:
            return ""

        import httpx

        with open(path, "rb") as f:
            resp = httpx.post(
                api_url,
                headers={"unstructured-api-key": api_key},
                files={"files": (path.name, f)},
                data={"strategy": "auto"},
                timeout=120,
            )
        resp.raise_for_status()
        elements = resp.json()
        parts: list[str] = []
        for el in elements:
            el_type = el.get("type", "")
            text = el.get("text", "").strip()
            if not text:
                continue
            if el_type == "Title":
                parts.append(f"## {text}")
            elif el_type == "Table":
                html = el.get("metadata", {}).get("text_as_html", "")
                parts.append(html if html else text)
            else:
                parts.append(text)
        result = "\n\n".join(parts)
        if len(result.strip()) >= _MIN_USEFUL_CHARS:
            return result
    except (httpx.HTTPError, json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
        logger.debug("Unstructured API failed for %s — falling back", path, exc_info=True)
    return ""


def _read_with_docling(path: Path) -> str:
    """Parse any supported document via Docling for high-fidelity extraction.

    Handles tables, reading order, formulas, and complex layouts far better
    than the legacy per-format readers.  Falls back to legacy readers on error.
    """
    try:
        from brain_os.config import get_settings

        hf_cache = get_settings().app.hf_cache_dir.strip()
        if hf_cache:
            os.makedirs(hf_cache, exist_ok=True)
            os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(path))
        md = result.document.export_to_markdown()
        # When Docling exposes figure/caption text (e.g. PictureItem captions), append
        # "Figure N: <caption>" here for better retrieval of figure references.
        if md and len(md.strip()) >= _MIN_USEFUL_CHARS:
            return md
    except OSError as e:
        if e.errno == errno.ENOSPC:
            logger.warning(
                "No space left on device (Docling/HF cache). Free disk space or set APP__HF_CACHE_DIR to a path with space. Path: %s",
                path,
            )
        else:
            logger.debug(
                "Docling failed for %s — falling back to legacy reader", path, exc_info=True
            )
    except (ImportError, ValueError, TypeError, RuntimeError) as exc:
        logger.debug("Docling failed for %s — falling back to legacy reader", path, exc_info=True)
    return ""


def extract_pdf_text(path: Path, *, use_document_ai_fallback: bool = True) -> str:
    """Extract PDF text using the same cascade as knowledge ingestion.

    Order: Docling → Unstructured (when configured) → :func:`read_pdf`
    (pypdf, then PDF.co / Document AI when *use_document_ai_fallback* is True).

    For lightweight previews over many files (e.g. imports metadata snippets),
    call :func:`read_pdf` directly to avoid Docling/Unstructured latency and cost.
    """
    text = _read_with_docling(path)
    if text and len(text.strip()) >= _MIN_USEFUL_CHARS:
        return text
    text = _read_with_unstructured(path)
    if text:
        return text
    return read_pdf(path, use_document_ai_fallback=use_document_ai_fallback)


_UNSTRUCTURED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".eml"}
_DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md"}

_LEGACY_READERS = {
    ".pdf": read_pdf,
    ".xlsx": read_xlsx,
    ".xls": read_xls,
    ".docx": read_docx,
    ".txt": read_txt,
    ".md": read_md,
    ".json": read_json,
    ".csv": read_csv,
    ".pptx": read_pptx,
}


def _get_reader(ext: str):
    """Return a reader function. PDF uses Docling first (tables/layout); others use Unstructured → Docling → legacy."""
    # PDF: Docling first for better tables and reading order, then Unstructured, then pypdf + Document AI.
    if ext == ".pdf":

        def _pdf_reader(path: Path) -> str:
            return extract_pdf_text(path, use_document_ai_fallback=True)

        return _pdf_reader
    if ext in _UNSTRUCTURED_EXTENSIONS:

        def _cascading_reader(path: Path) -> str:
            text = _read_with_unstructured(path)
            if text:
                return text
            if ext in _DOCLING_EXTENSIONS:
                text = _read_with_docling(path)
                if text:
                    return text
            legacy = _LEGACY_READERS.get(ext)
            return legacy(path) if legacy else ""

        return _cascading_reader
    if ext in _DOCLING_EXTENSIONS:

        def _docling_then_legacy(path: Path) -> str:
            text = _read_with_docling(path)
            if text:
                return text
            legacy = _LEGACY_READERS.get(ext)
            return legacy(path) if legacy else ""

        return _docling_then_legacy
    return _LEGACY_READERS.get(ext)


_READERS = {ext: _get_reader(ext) or fn for ext, fn in _LEGACY_READERS.items()}
_READERS.update(
    {
        ext: _get_reader(ext)
        for ext in _UNSTRUCTURED_EXTENSIONS | _DOCLING_EXTENSIONS
        if ext not in _READERS
    }
)


def extract_text_from_upload_bytes(filename: str, data: bytes) -> str:
    """Extract plaintext from uploaded file bytes using the same reader cascade as filesystem ingest.

    Writes to a temporary file (same suffix as ``filename``) so PDF/DOCX/XLSX are never UTF-8 mangled.
    Returns empty string if the extension has no reader or extraction yields nothing.
    """
    ext = Path(filename).suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        return ""

    fd, tmppath = None, ""
    path: Path | None = None
    try:
        import tempfile

        fd, tmppath = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        fd = None
        path = Path(tmppath)
        path.write_bytes(data)

        reader = _get_reader(ext)
        if reader is None:
            return ""
        return (reader(path) or "").strip()
    finally:
        try:
            if path is not None:
                path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to unlink temp ingest path %s", path, exc_info=True)


# ── chunking ─────────────────────────────────────────────────────────────────


def _chunk_text_tiktoken(
    text: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
) -> list[str]:
    """Legacy fixed-size token chunking with tiktoken.  Used as fallback."""
    enc = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
    tokens = enc.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks


_TABLE_PATTERN = re.compile(
    r"((?:^\|.+\|$\n?){2,})",
    re.MULTILINE,
)
_TABLE_SENTINEL_START = "\n\n<!-- TABLE_BOUNDARY -->\n"
_TABLE_SENTINEL_END = "\n<!-- /TABLE_BOUNDARY -->\n\n"


def _protect_tables(text: str) -> tuple[str, list[str]]:
    """Wrap Markdown tables in sentinels so the chunker won't split them."""
    tables: list[str] = []

    def _replace(m: re.Match) -> str:
        tables.append(m.group(1))
        idx = len(tables) - 1
        return f"{_TABLE_SENTINEL_START}__TABLE_{idx}__{_TABLE_SENTINEL_END}"

    return _TABLE_PATTERN.sub(_replace, text), tables


def _restore_tables(chunks: list[str], tables: list[str]) -> list[str]:
    """Replace table placeholders with original table text."""
    restored: list[str] = []
    for chunk in chunks:
        for idx, table in enumerate(tables):
            placeholder = f"__TABLE_{idx}__"
            if placeholder in chunk:
                chunk = chunk.replace(
                    f"{_TABLE_SENTINEL_START}{placeholder}{_TABLE_SENTINEL_END}",
                    f"\n\n{table}\n\n",
                )
        restored.append(chunk.strip())
    return [c for c in restored if c]


def _get_voyage_embeddings():
    """Build a Chonkie VoyageAIEmbeddings instance using our configured key."""
    try:
        from chonkie.embeddings import VoyageAIEmbeddings

        from brain_os.config import get_settings

        settings = get_settings()
        api_key = settings.embedding.api_key.get_secret_value()
        model = settings.embedding.model
        if not api_key:
            return None
        return VoyageAIEmbeddings(model=model, api_key=api_key)
    except (ImportError, OSError, ValueError, TypeError) as exc:
        logger.debug("VoyageAI embeddings for Chonkie unavailable", exc_info=True)
        return None


def chunk_text(
    text: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
) -> list[str]:
    """Split *text* into semantically coherent chunks via Chonkie.

    Uses SemanticChunker with VoyageAI embeddings (matching our retrieval
    embedding space) when available.  Markdown tables are protected from
    being split across chunk boundaries.  Falls back to the legacy
    fixed-size tiktoken chunker on error.
    """
    if text == "":
        return [""]

    protected_text, tables = _protect_tables(text)

    try:
        from chonkie import SemanticChunker

        from brain_os.config import get_settings as _gs_chunk

        _appc = _gs_chunk().app
        embedding_model = _get_voyage_embeddings() or "minishlab/potion-base-32M"
        # Chonkie may warn when model2vec is not installed and it falls back to SentenceTransformer; suppress.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*model2vec is not available.*",
                category=UserWarning,
            )
            chunker = SemanticChunker(
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                threshold=float(_appc.ingest_semantic_similarity_threshold),
                similarity_window=int(_appc.ingest_semantic_similarity_window),
            )
            chunks = chunker.chunk(protected_text)
        result = [c.text for c in chunks if c.text.strip()]
        if result:
            enc = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
            # Repetitive or very uniform text often becomes one huge semantic chunk;
            # fall back to tiktoken so callers that pass *chunk_size* get bounded segments.
            if len(result) == 1 and len(enc.encode(result[0])) > chunk_size:
                pass
            else:
                return _restore_tables(result, tables)
    except (ImportError, OSError, ValueError, TypeError, RuntimeError) as exc:
        logger.debug("Chonkie semantic chunking failed — using tiktoken fallback", exc_info=True)

    fallback = _chunk_text_tiktoken(protected_text, chunk_size, overlap)
    return _restore_tables(fallback, tables)


# ── category extraction ──────────────────────────────────────────────────────


def _category_from_path(file_path: Path, base_path: Path) -> str:
    """Derive a source_category slug from the first directory under *base_path*.

    Expects folder names like ``01_Quotes_and_Proposals``.  Returns the part
    after the numeric prefix, lowercased (e.g. ``quotes_and_proposals``).
    Falls back to ``"uncategorised"`` if the pattern doesn't match.
    """
    try:
        relative = file_path.relative_to(base_path)
        top_dir = relative.parts[0] if relative.parts else ""
    except ValueError:
        return "uncategorised"

    m = _CATEGORY_PATTERN.match(top_dir)
    return m.group(1).lower() if m else top_dir.lower() or "uncategorised"


# ── ingestor class ───────────────────────────────────────────────────────────


class DocumentIngestor:
    """Reads, chunks, and upserts documents from the imports directory."""

    def __init__(
        self,
        qdrant: QdrantManager,
        *,
        knowledge_graph: KnowledgeGraph | None = None,
        ledger_path: Path = _LEDGER_PATH,
    ) -> None:
        self._qdrant = qdrant
        self._graph = knowledge_graph
        self._ledger = _init_ledger(ledger_path)

    async def __aenter__(self) -> DocumentIngestor:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ── discovery ────────────────────────────────────────────────────────

    def discover_files(self, base_path: str = "data/imports") -> list[dict[str, Any]]:
        """Walk *base_path* and return metadata for every supported file."""
        base = Path(base_path).resolve()
        project_root = Path(__file__).resolve().parents[3]
        data_root = project_root / "data"
        if not base.is_relative_to(data_root):
            raise PathTraversalError(f"Path {base_path} is outside the data directory")

        root = Path(base_path)
        if not root.exists():
            logger.warning("Import directory does not exist: %s", root)
            return []

        files: list[dict[str, Any]] = []
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext not in _SUPPORTED_EXTENSIONS:
                continue
            files.append(
                {
                    "path": str(p),
                    "category": _category_from_path(p, root),
                    "extension": ext,
                    "size": p.stat().st_size,
                }
            )

        logger.info("Discovered %d importable files under %s", len(files), root)
        return files

    # ── contextual retrieval ─────────────────────────────────────────────

    _CONTEXT_PREVIEW_CHARS = 3000

    async def _generate_document_context(self, text: str, filename: str) -> str:
        """Generate a 2-sentence document summary for contextual retrieval.

        Prepended to every chunk so isolated chunks retain the global
        context of their source document (Anthropic's Contextual Retrieval).
        """
        try:
            from brain_os.config import get_settings
            from brain_os.prompt_loader import load_prompt
            from brain_os.services.llm_client import get_llm_client

            app = get_settings().app
            provider = app.digestive_llm_provider
            model_kw: dict[str, str] = {}
            if provider == "anthropic":
                dm = (app.digestive_anthropic_model or "").strip()
                if dm:
                    model_kw["model"] = dm

            system = load_prompt("document_context")
            preview = text[: self._CONTEXT_PREVIEW_CHARS]
            user_msg = f"Filename: {filename}\n\n{preview}"
            summary = await get_llm_client().generate_text(
                system,
                user_msg,
                temperature=0.0,
                max_tokens=200,
                name="document_context",
                provider=provider,
                **model_kw,
            )
            return summary.strip()
        except (LLMError, httpx.HTTPError, OSError, ValueError, TypeError) as exc:
            logger.warning(
                "Document context generation failed for %s — proceeding without context",
                filename,
                exc_info=True,
            )
            return ""

    # ── single-file ingestion ────────────────────────────────────────────

    def is_already_ingested(self, file_info: dict[str, Any]) -> bool:
        """Check whether a file has already been ingested with the same hash."""
        path = Path(file_info["path"])
        return self._already_ingested(str(path), _file_hash(path))

    async def ingest_file(
        self,
        file_info: dict[str, Any],
        *,
        force: bool = False,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        overlap: int = _DEFAULT_OVERLAP,
    ) -> int:
        """Read, chunk, and upsert one file.  Returns the chunk count.

        When *force* is ``True`` the ledger check is skipped and any
        existing Qdrant points for this source path are deleted first.
        """
        path = Path(file_info["path"])
        category = file_info["category"]
        ext = file_info["extension"]

        current_hash = await asyncio.to_thread(_file_hash, path)
        if not force and self._already_ingested(str(path), current_hash):
            logger.debug("Skipping already-ingested file: %s", path)
            return 0

        if force:
            await self._qdrant.delete_by_source(str(path))

        reader = _get_reader(ext) or _READERS.get(ext)
        if reader is None:
            logger.warning("No reader for extension '%s': %s", ext, path)
            return 0

        text = await asyncio.to_thread(reader, path)

        if ext == ".pdf" and len(text.strip()) < _MIN_USEFUL_CHARS:
            ocr_text = await self._ocr_fallback(path)
            if ocr_text:
                text = ocr_text

        if not text.strip():
            logger.warning("Empty content after reading: %s", path)
            return 0

        doc_context = await self._generate_document_context(text, path.name)

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        chunk_meta: dict[str, Any] = {
            "chunk_index": 0,
            "total_chunks": len(chunks),
            "extension": ext,
            "document_context": doc_context or "",
        }
        teacher_extra = file_info.get("teacher_metadata")
        if isinstance(teacher_extra, dict):
            chunk_meta.update(teacher_extra)

        items = [
            KnowledgeItem(
                source=str(path),
                source_category=category,
                content=(f"[Source Context: {doc_context}]\n\n{chunk}" if doc_context else chunk),
                metadata={
                    **chunk_meta,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        upserted = await self._qdrant.upsert_items(items)

        if self._graph is not None:
            await self._extract_and_store_entities(text, str(path))

        self._record_ingestion(str(path), current_hash, upserted)
        logger.info("Ingested %s -> %d chunks (category: %s)", path, upserted, category)
        return upserted

    # ── bulk ingestion ───────────────────────────────────────────────────

    async def ingest_all(
        self,
        base_path: str = "data/imports",
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Process every supported file under *base_path*.

        Returns a summary dict with keys ``files_processed``,
        ``files_skipped``, ``total_chunks``, ``per_category``, and
        ``errors`` (a list of ``{"path": ..., "error": ...}`` dicts).
        """
        files = self.discover_files(base_path)

        files_processed = 0
        files_skipped = 0
        total_chunks = 0
        per_category: dict[str, int] = {}
        errors: list[dict[str, str]] = []

        for file_info in files:
            try:
                n = await self.ingest_file(file_info, force=force)
                if n > 0:
                    files_processed += 1
                    total_chunks += n
                    cat = file_info["category"]
                    per_category[cat] = per_category.get(cat, 0) + n
                else:
                    files_skipped += 1
            except (IngestionError, DatabaseError, httpx.HTTPError, OSError, LLMError) as exc:
                logger.exception("Failed to ingest %s", file_info["path"])
                errors.append({"path": file_info["path"], "error": str(exc)})

        summary: dict[str, Any] = {
            "files_processed": files_processed,
            "files_skipped": files_skipped,
            "total_chunks": total_chunks,
            "per_category": per_category,
            "errors": errors,
        }
        logger.info("Ingestion complete: %s", summary)
        return summary

    # ── Document AI OCR fallback ─────────────────────────────────────────

    async def _ocr_fallback(self, path: Path) -> str:
        """Try Document AI OCR when pypdf yields insufficient text."""
        try:
            from brain_os.systems.document_ai import DocumentAIService

            svc = DocumentAIService()
            await svc.connect()
            if not svc.available:
                return ""
            text = await svc.extract_text(path.read_bytes())
            if text.strip():
                logger.info("Document AI OCR recovered text for %s (%d chars)", path, len(text))
            return text
        except (LLMError, DatabaseError, httpx.HTTPError, OSError) as exc:
            logger.warning("Document AI OCR fallback failed for %s", path, exc_info=True)
            return ""

    async def reingest_scanned_pdfs(
        self,
        base_path: str = "data/imports",
        *,
        min_file_size: int = 5 * 1024 * 1024,
        min_chars_per_page: int = 25,
    ) -> dict[str, Any]:
        """Re-ingest PDFs that are likely scanned (large files with poor text).

        Finds PDFs over *min_file_size* bytes, checks if pypdf yields little
        text per page (below *min_chars_per_page*), and re-ingests with Document AI OCR.
        """
        files = self.discover_files(base_path)
        candidates = [f for f in files if f["extension"] == ".pdf" and f["size"] >= min_file_size]
        logger.info(
            "Found %d PDF candidates for OCR re-ingestion (>%d bytes, <%d chars/page)",
            len(candidates),
            min_file_size,
            min_chars_per_page,
        )

        reingested = 0
        skipped = 0
        errors: list[dict[str, str]] = []

        for file_info in candidates:
            path = Path(file_info["path"])
            try:
                pypdf_text = await asyncio.to_thread(_read_pdf_pypdf, path)
                text_len = len(pypdf_text.strip())
                if text_len >= _MIN_USEFUL_CHARS:
                    try:
                        from pypdf import PdfReader

                        with _suppress_pypdf_warnings():
                            reader = PdfReader(str(path))
                            page_count = max(1, len(reader.pages))
                        if text_len / page_count >= min_chars_per_page:
                            skipped += 1
                            continue
                    except (OSError, ImportError, ValueError, TypeError):
                        pass
                    skipped += 1
                    continue  # enough text overall, skip OCR

                ocr_text = await self._ocr_fallback(path)
                if not ocr_text.strip():
                    skipped += 1
                    continue

                n = await self.ingest_file(file_info, force=True)
                if n > 0:
                    reingested += 1
                    logger.info("Re-ingested scanned PDF: %s -> %d chunks", path, n)
            except (IngestionError, DatabaseError, OSError, LLMError) as exc:
                logger.exception("Failed to re-ingest %s", path)
                errors.append({"path": str(path), "error": str(exc)})

        summary = {
            "candidates": len(candidates),
            "reingested": reingested,
            "skipped": skipped,
            "errors": errors,
        }
        logger.info("Scanned PDF re-ingestion complete: %s", summary)
        return summary

    # ── entity extraction ─────────────────────────────────────────────────

    async def _extract_and_store_entities(self, text: str, source: str) -> None:
        """Extract entities from document text and store them in Neo4j.

        Extraction priority: GraphRAG (schema-bound, with entity resolution)
        > legacy LLM extraction > GLiNER (fast local fallback).
        Results from all available extractors are merged and deduplicated.
        """
        assert self._graph is not None

        try:
            from brain_os.brain.entity_extractor import extract_entities_gliner

            gliner_entities = await asyncio.to_thread(extract_entities_gliner, text)
        except (OSError, ImportError, RuntimeError, ValueError, TypeError) as exc:
            logger.debug("GLiNER extraction failed for %s — using LLM only", source)
            gliner_entities = {"companies": [], "people": [], "machines": [], "relationships": []}

        try:
            entities = await self._graph.extract_entities_from_text(text)
        except (LLMError, httpx.HTTPError, OSError, ValueError, TypeError):
            logger.warning("Entity extraction failed for %s — using GLiNER results only", source)
            entities = gliner_entities

        entities = self._merge_entity_results(gliner_entities, entities)
        source_id = _source_to_id(source)

        for company in entities.get("companies", []):
            try:
                await self._graph.add_company(
                    name=company.get("name", ""),
                    region=company.get("region", ""),
                    industry=company.get("industry", ""),
                    source_id=source_id,
                )
            except _INGEST_GRAPH_WRITE_ERRORS:
                logger.warning("Failed to add company from %s: %s", source, company)

        for person in entities.get("people", []):
            try:
                await self._graph.add_person(
                    name=person.get("name", ""),
                    email=person.get("email", ""),
                    company_name=person.get("company", ""),
                    role=person.get("role", ""),
                    source_id=source_id,
                )
            except _INGEST_GRAPH_WRITE_ERRORS:
                logger.warning("Failed to add person from %s: %s", source, person)

        for machine in entities.get("machines", []):
            try:
                await self._graph.add_machine(
                    model=machine.get("model", ""),
                    category=machine.get("category", ""),
                    description=machine.get("description", ""),
                    source_id=source_id,
                )
            except _INGEST_GRAPH_WRITE_ERRORS:
                logger.warning("Failed to add machine from %s: %s", source, machine)

        rel_count = 0
        for rel in entities.get("relationships", []):
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
            except _INGEST_GRAPH_WRITE_ERRORS:
                logger.warning("Failed to add relationship from %s: %s", source, rel)

        logger.info(
            "Extracted entities from %s: %d companies, %d people, %d machines, %d relationships",
            source,
            len(entities.get("companies", [])),
            len(entities.get("people", [])),
            len(entities.get("machines", [])),
            rel_count,
        )

    @staticmethod
    def _merge_entity_results(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        """Merge two entity extraction results, deduplicating by key fields."""

        def _dedup(items: list[dict], key_field: str) -> list[dict]:
            seen: set[str] = set()
            out: list[dict] = []
            for item in items:
                k = item.get(key_field, "").lower().strip()
                if k and k not in seen:
                    seen.add(k)
                    out.append(item)
            return out

        return {
            "companies": _dedup(a.get("companies", []) + b.get("companies", []), "name"),
            "people": _dedup(a.get("people", []) + b.get("people", []), "name"),
            "machines": _dedup(a.get("machines", []) + b.get("machines", []), "model"),
            "relationships": a.get("relationships", []) + b.get("relationships", []),
        }

    # ── ledger helpers ───────────────────────────────────────────────────

    def _file_hash_for(self, file_info: dict[str, Any]) -> str:
        """Return the SHA-256 hash for a discovered file."""
        return _file_hash(Path(file_info["path"]))

    def _already_ingested(self, path: str, file_hash: str) -> bool:
        row = self._ledger.execute(
            "SELECT hash FROM ingested_files WHERE path = ?", (path,)
        ).fetchone()
        return row is not None and row[0] == file_hash

    def _record_ingestion(self, path: str, file_hash: str, chunk_count: int) -> None:
        self._ledger.execute(
            """
            INSERT INTO ingested_files (path, hash, chunk_count, ingested_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                hash = excluded.hash,
                chunk_count = excluded.chunk_count,
                ingested_at = excluded.ingested_at
            """,
            (path, file_hash, chunk_count, datetime.now(UTC).isoformat()),
        )
        self._ledger.commit()

    def close(self) -> None:
        self._ledger.close()
