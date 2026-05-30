"""Google Cloud DLP (Sensitive Data Protection) integration for Brain OS.

Provides :class:`DlpService`, an async wrapper around the DLP REST API
for detecting and redacting PII/NDA-sensitive content.

Used primarily by Cadmus for NDA-safe case study generation and by
Delphi for email classification with PII awareness.

Authenticates via the same OAuth credentials used by other Google
services.  Requires the "Sensitive Data Protection" API to be enabled
in the GCP console.

Constructed once at startup and injected via the service locator
(``ServiceKey.DLP``).
"""

from __future__ import annotations

import asyncio
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

_DLP_CONNECT_ERRORS = (
    httpx.HTTPError,
    asyncio.TimeoutError,
    OSError,
    ValueError,
    TypeError,
)
_DLP_HTTP_ERRORS = (httpx.HTTPError, asyncio.TimeoutError)
_DLP_JSON_BODY_ERRORS = (json.JSONDecodeError, ValueError, KeyError, TypeError)

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_TOKEN_FILE = "token_dlp.json"

_DEFAULT_INFO_TYPES = [
    {"name": "PERSON_NAME"},
    {"name": "EMAIL_ADDRESS"},
    {"name": "PHONE_NUMBER"},
    {"name": "STREET_ADDRESS"},
    {"name": "CREDIT_CARD_NUMBER"},
    {"name": "DATE_OF_BIRTH"},
    {"name": "IP_ADDRESS"},
    {"name": "PASSPORT"},
    {"name": "INDIA_PAN_INDIVIDUAL"},
    {"name": "INDIA_AADHAAR_INDIVIDUAL"},
]


class DlpError(Exception):
    """Raised when a DLP operation fails."""


class DlpService:
    """Async Google Cloud DLP client for PII detection and redaction."""

    def __init__(
        self,
        config: DocumentAIConfig | None = None,
        google_config: GoogleConfig | None = None,
    ) -> None:
        settings = get_settings()
        cfg = config or settings.document_ai
        gcfg = google_config or settings.google

        self._project_id = cfg.project_id
        self._creds_path = Path(gcfg.credentials_path)
        self._token_path = Path(_TOKEN_FILE)
        self._creds: Credentials | None = None

    @property
    def available(self) -> bool:
        return bool(self._project_id and self._creds)

    @property
    def configured(self) -> bool:
        return bool(self._project_id)

    # ── authentication ─────────────────────────────────────────────────

    async def connect(self) -> None:
        """Authenticate with Google Cloud for DLP access."""
        if self._creds is not None and self._creds.valid:
            return
        if not self.configured:
            logger.warning("DLP not configured (no project ID)")
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
                    raise DlpError(f"Credentials file not found: {self._creds_path}")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self._creds_path),
                    _SCOPES,
                )
                creds = flow.run_local_server(port=0)

            self._token_path.write_text(creds.to_json())
            return creds

        try:
            self._creds = await asyncio.to_thread(_authenticate)
            logger.info("DLP service connected (project=%s)", self._project_id)
        except _DLP_CONNECT_ERRORS as exc:
            logger.error("DLP authentication failed: %s", exc)
            raise DlpError(str(exc)) from exc

    async def close(self) -> None:
        self._creds = None
        logger.info("DLP service closed")

    async def _ensure_connected(self) -> None:
        if self._creds is None or not self._creds.valid:
            await self.connect()
        if self._creds is None:
            raise DlpError("DLP service is not configured or authentication failed")

    def _base_url(self) -> str:
        return f"https://dlp.googleapis.com/v2/projects/{self._project_id}"

    # ── inspect ────────────────────────────────────────────────────────

    async def inspect_text(
        self,
        text: str,
        *,
        info_types: list[dict[str, str]] | None = None,
        min_likelihood: str = "LIKELY",
    ) -> list[dict[str, Any]]:
        """Detect PII in text. Returns a list of findings."""
        await self._ensure_connected()

        payload = {
            "item": {"value": text},
            "inspectConfig": {
                "infoTypes": info_types or _DEFAULT_INFO_TYPES,
                "minLikelihood": min_likelihood,
                "includeQuote": True,
            },
        }

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.post(
                    f"{self._base_url()}/content:inspect",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._creds.token}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
            except DlpError:
                raise
            except _DLP_HTTP_ERRORS as exc:
                raise DlpError(f"DLP inspect HTTP failed: {exc}") from exc

        try:
            body = resp.json()
        except _DLP_JSON_BODY_ERRORS as exc:
            raise DlpError(f"DLP inspect response parse failed: {exc}") from exc
        if not isinstance(body, dict):
            raise DlpError("DLP inspect: response body is not a JSON object")
        raw_result = body.get("result", {})
        result = raw_result if isinstance(raw_result, dict) else {}
        findings = result.get("findings", [])

        return [
            {
                "info_type": f.get("infoType", {}).get("name", ""),
                "likelihood": f.get("likelihood", ""),
                "quote": f.get("quote", ""),
                "location": {
                    "start": f.get("location", {}).get("codepointRange", {}).get("start", 0),
                    "end": f.get("location", {}).get("codepointRange", {}).get("end", 0),
                },
            }
            for f in findings
        ]

    # ── deidentify ─────────────────────────────────────────────────────

    async def deidentify_text(
        self,
        text: str,
        *,
        info_types: list[dict[str, str]] | None = None,
        masking_char: str = "*",
    ) -> str:
        """Redact PII from text by replacing it with masking characters."""
        await self._ensure_connected()

        payload = {
            "item": {"value": text},
            "deidentifyConfig": {
                "infoTypeTransformations": {
                    "transformations": [
                        {
                            "primitiveTransformation": {
                                "characterMaskConfig": {
                                    "maskingCharacter": masking_char,
                                }
                            }
                        }
                    ]
                }
            },
            "inspectConfig": {
                "infoTypes": info_types or _DEFAULT_INFO_TYPES,
            },
        }

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.post(
                    f"{self._base_url()}/content:deidentify",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._creds.token}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
            except DlpError:
                raise
            except _DLP_HTTP_ERRORS as exc:
                raise DlpError(f"DLP deidentify HTTP failed: {exc}") from exc

        try:
            body = resp.json()
        except _DLP_JSON_BODY_ERRORS as exc:
            raise DlpError(f"DLP deidentify response parse failed: {exc}") from exc
        if not isinstance(body, dict):
            return text
        return body.get("item", {}).get("value", text)

    # ── convenience: inspect + report ──────────────────────────────────

    async def inspect_and_report(self, text: str) -> dict[str, Any]:
        """Inspect text and return a structured report with counts by type."""
        findings = await self.inspect_text(text)

        type_counts: dict[str, int] = {}
        for f in findings:
            info_type = f["info_type"]
            type_counts[info_type] = type_counts.get(info_type, 0) + 1

        return {
            "total_findings": len(findings),
            "findings_by_type": type_counts,
            "findings": findings,
            "has_pii": len(findings) > 0,
        }

    # ── health ─────────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        if not self.configured:
            return {"status": "not_configured", "available": False}
        if not self.available:
            return {"status": "disconnected", "available": False}
        return {
            "status": "connected",
            "available": True,
            "project": self._project_id,
        }
