"""Google Document AI integration for Brain OS.

Provides :class:`DocumentAIService`, an async wrapper around the
Document AI REST API for OCR, invoice parsing, and form extraction.

Authenticates via the same OAuth credentials used by Gmail and Google
Docs.  Requires a Document AI processor to be created in the GCP
console and its ID set in ``DOCUMENT_AI_PROCESSOR_ID``.

Constructed once at startup and injected via the service locator
(``ServiceKey.DOCUMENT_AI``).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from brain_os.config import DocumentAIConfig, GoogleConfig, get_settings

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_TOKEN_FILE = "token_document_ai.json"


class DocumentAIError(Exception):
    """Raised when a Document AI operation fails."""


def mime_type_for_document_ai_filename(filename: str) -> str:
    """Infer ``mimeType`` for Document AI ``rawDocument`` from basename or path."""
    ext = Path(filename).suffix.lower()
    mapping: dict[str, str] = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    return mapping.get(ext, "application/pdf")


async def parse_bytes_by_kind(
    file_bytes: bytes,
    *,
    filename: str,
    kind: str,
    service: DocumentAIService | None = None,
) -> dict[str, Any]:
    """Run OCR, invoice parser, or form parser and return a JSON-serializable dict.

    *kind* must be ``ocr``, ``invoice``, or ``form`` (case-insensitive).
    """
    k = kind.strip().lower()
    if k not in ("ocr", "invoice", "form"):
        raise DocumentAIError(f"Invalid kind {kind!r}; use ocr, invoice, or form")
    mime = mime_type_for_document_ai_filename(filename)
    svc = service or DocumentAIService()
    await svc.connect()
    if k == "ocr":
        text = await svc.extract_text(file_bytes, mime)
        return {"kind": "ocr", "mime_type": mime, "text": text}
    if k == "invoice":
        parsed = await svc.parse_invoice(file_bytes, mime)
        return {"kind": "invoice", "mime_type": mime, **parsed}
    parsed = await svc.parse_form(file_bytes, mime)
    return {"kind": "form", "mime_type": mime, **parsed}


class DocumentAIService:
    """Async Google Document AI client for OCR and structured extraction."""

    def __init__(
        self,
        config: DocumentAIConfig | None = None,
        google_config: GoogleConfig | None = None,
    ) -> None:
        settings = get_settings()
        cfg = config or settings.document_ai
        gcfg = google_config or settings.google

        self._project_id = cfg.project_id
        self._location = cfg.location
        self._processors = {
            "ocr": cfg.processor_id,
            "invoice": cfg.invoice_processor_id,
            "form": cfg.form_processor_id,
        }
        self._request_timeout = float(cfg.request_timeout_seconds)
        self._creds_path = Path(gcfg.credentials_path)
        self._token_path = Path(_TOKEN_FILE)
        self._creds: Credentials | None = None

    @property
    def available(self) -> bool:
        return bool(self._project_id and any(self._processors.values()) and self._creds)

    @property
    def configured(self) -> bool:
        return bool(self._project_id and any(self._processors.values()))

    # ── authentication ─────────────────────────────────────────────────

    async def connect(self) -> None:
        """Authenticate with Google Cloud for Document AI access."""
        if self._creds is not None and self._creds.valid:
            return
        if not self.configured:
            logger.warning(
                "Document AI not configured (project=%s, processor=%s)",
                self._project_id,
                self._processors.get("ocr", ""),
            )
            return

        def _authenticate() -> Credentials:
            creds: Credentials | None = None

            if self._token_path.exists():
                creds = Credentials.from_authorized_user_file(
                    str(self._token_path),
                    _SCOPES,
                )

            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            elif not creds or not creds.valid:
                if not self._creds_path.exists():
                    raise DocumentAIError(f"Credentials file not found: {self._creds_path}")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self._creds_path),
                    _SCOPES,
                )
                creds = flow.run_local_server(port=0)

            self._token_path.write_text(creds.to_json())
            return creds

        try:
            self._creds = await asyncio.to_thread(_authenticate)
            logger.info("Document AI service connected (project=%s)", self._project_id)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            logger.error("Document AI authentication failed: %s", exc)
            raise DocumentAIError(str(exc)) from exc

    async def close(self) -> None:
        self._creds = None
        logger.info("Document AI service closed")

    async def _ensure_connected(self) -> None:
        if self._creds is None or not self._creds.valid:
            await self.connect()
        if self._creds is None:
            raise DocumentAIError("Document AI service is not configured or authentication failed")

    def _endpoint(self, processor_type: str = "ocr") -> str:
        pid = self._processors.get(processor_type) or self._processors.get("ocr", "")
        if not pid:
            raise DocumentAIError(f"No processor configured for type '{processor_type}'")
        return (
            f"https://{self._location}-documentai.googleapis.com/v1/"
            f"projects/{self._project_id}/locations/{self._location}/"
            f"processors/{pid}:process"
        )

    # ── document parsing ───────────────────────────────────────────────

    async def parse_document(
        self,
        file_bytes: bytes,
        mime_type: str = "application/pdf",
        *,
        processor_type: str = "ocr",
    ) -> dict[str, Any]:
        """Parse a document and return the full Document AI response.

        *processor_type* selects which processor to use:
        ``"ocr"`` (default), ``"invoice"``, or ``"form"``.

        Supports PDF, TIFF, GIF, JPEG, PNG, BMP, and WEBP.
        """
        await self._ensure_connected()

        encoded = base64.standard_b64encode(file_bytes).decode("utf-8")
        payload = {
            "rawDocument": {
                "content": encoded,
                "mimeType": mime_type,
            }
        }

        timeout = httpx.Timeout(
            connect=30.0,
            read=self._request_timeout,
            write=120.0,
            pool=30.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(
                    self._endpoint(processor_type),
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._creds.token}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                return resp.json()
            except (
                httpx.HTTPError,
                TimeoutError,
                ValueError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                raise DocumentAIError(f"Document AI request failed: {exc}") from exc

    async def extract_text(
        self,
        file_bytes: bytes,
        mime_type: str = "application/pdf",
    ) -> str:
        """Extract plain text from a document via the OCR processor."""
        result = await self.parse_document(file_bytes, mime_type, processor_type="ocr")
        document = result.get("document", {})
        return document.get("text", "")

    async def parse_invoice(
        self,
        file_bytes: bytes,
        mime_type: str = "application/pdf",
    ) -> dict[str, Any]:
        """Parse an invoice and return structured fields (vendor, amount, line items, etc.)."""
        result = await self.parse_document(file_bytes, mime_type, processor_type="invoice")
        document = result.get("document", {})
        entities = document.get("entities", [])
        return {
            "text": document.get("text", ""),
            "fields": [
                {
                    "type": e.get("type", ""),
                    "mention_text": e.get("mentionText", ""),
                    "confidence": e.get("confidence", 0.0),
                    "normalized_value": e.get("normalizedValue", {}),
                }
                for e in entities
            ],
        }

    async def parse_form(
        self,
        file_bytes: bytes,
        mime_type: str = "application/pdf",
    ) -> dict[str, Any]:
        """Parse a form and return key-value pairs and tables."""
        result = await self.parse_document(file_bytes, mime_type, processor_type="form")
        document = result.get("document", {})
        pages = document.get("pages", [])

        form_fields: list[dict[str, Any]] = []
        for page in pages:
            for field in page.get("formFields", []):
                name = _layout_text(
                    field.get("fieldName", {}).get("layout", {}),
                    document.get("text", ""),
                )
                value = _layout_text(
                    field.get("fieldValue", {}).get("layout", {}),
                    document.get("text", ""),
                )
                form_fields.append(
                    {
                        "name": name,
                        "value": value,
                        "confidence": field.get("fieldValue", {}).get("confidence", 0.0),
                    }
                )

        return {
            "text": document.get("text", ""),
            "form_fields": form_fields,
            "tables": self._extract_tables_from_doc(document),
        }

    async def extract_entities(
        self,
        file_bytes: bytes,
        mime_type: str = "application/pdf",
        *,
        processor_type: str = "ocr",
    ) -> list[dict[str, Any]]:
        """Extract structured entities (fields) from a document."""
        result = await self.parse_document(file_bytes, mime_type, processor_type=processor_type)
        document = result.get("document", {})
        entities = document.get("entities", [])
        return [
            {
                "type": e.get("type", ""),
                "mention_text": e.get("mentionText", ""),
                "confidence": e.get("confidence", 0.0),
                "normalized_value": e.get("normalizedValue", {}),
            }
            for e in entities
        ]

    async def extract_tables(
        self,
        file_bytes: bytes,
        mime_type: str = "application/pdf",
    ) -> list[list[list[str]]]:
        """Extract tables from a document via the form processor."""
        result = await self.parse_document(file_bytes, mime_type, processor_type="form")
        document = result.get("document", {})
        return self._extract_tables_from_doc(document)

    @staticmethod
    def _extract_tables_from_doc(document: dict[str, Any]) -> list[list[list[str]]]:
        pages = document.get("pages", [])
        full_text = document.get("text", "")
        tables: list[list[list[str]]] = []
        for page in pages:
            for table in page.get("tables", []):
                rows: list[list[str]] = []
                for header_row in table.get("headerRows", []):
                    cells = [
                        _layout_text(cell.get("layout", {}), full_text)
                        for cell in header_row.get("cells", [])
                    ]
                    rows.append(cells)
                for body_row in table.get("bodyRows", []):
                    cells = [
                        _layout_text(cell.get("layout", {}), full_text)
                        for cell in body_row.get("cells", [])
                    ]
                    rows.append(cells)
                tables.append(rows)
        return tables

    # ── health ─────────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        if not self.configured:
            return {"status": "not_configured", "available": False}
        if not self.available:
            return {"status": "disconnected", "available": False}
        active = {k: v for k, v in self._processors.items() if v}
        return {
            "status": "connected",
            "available": True,
            "project": self._project_id,
            "location": self._location,
            "processors": active,
        }


def _layout_text(layout: dict[str, Any], full_text: str) -> str:
    """Extract text from a Document AI layout using text anchors."""
    segments = layout.get("textAnchor", {}).get("textSegments", [])
    parts: list[str] = []
    for seg in segments:
        start = int(seg.get("startIndex", 0))
        end = int(seg.get("endIndex", 0))
        parts.append(full_text[start:end])
    return "".join(parts).strip()
