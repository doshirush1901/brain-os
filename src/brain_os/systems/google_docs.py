"""Google Docs, Drive & Calendar integration for Ira.

Provides :class:`GoogleDocsService`, an async wrapper around the Google
Docs API (v1), Drive API (v3), and Calendar API (v3) that any agent or
subsystem can use to:

* **Read** existing Google Docs (full text or structured content).
* **Create** new Google Docs with initial content.
* **Update** existing Docs (append, insert, or replace text).
* **Search** Google Drive for files by name, type, or folder.
* **List** files in a specific Drive folder.
* **List** upcoming Calendar events (primary calendar).

The service authenticates via OAuth2 using the same credential files as
the EmailProcessor.  It requests Docs + Drive + Calendar scopes and
stores a separate token file (``token_docs.json``) so it doesn't
invalidate the Gmail token.  If you add Calendar after existing tokens,
delete ``token_docs.json`` and re-run so the new scope is granted.

Constructed once at startup and injected via the service locator
(``ServiceKey.GOOGLE_DOCS``).  All operations degrade gracefully — a
missing or invalid credential set logs a warning and raises
:class:`GoogleDocsError` so callers can handle it.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC
from pathlib import Path
from typing import Any

from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from brain_os.config import GoogleConfig, get_settings

logger = logging.getLogger(__name__)

_SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/calendar.readonly",
]

_TOKEN_FILE = "token_docs.json"


class GoogleDocsError(Exception):
    """Raised when a Google Docs/Drive operation fails."""


class GoogleDocsService:
    """Async Google Docs, Drive and Calendar client."""

    def __init__(self, config: GoogleConfig | None = None) -> None:
        cfg = config or get_settings().google
        self._creds_path = Path(cfg.credentials_path)
        # Store token next to credentials (e.g. .credentials/token_docs.json) so one place to delete.
        self._token_path = self._creds_path.parent / _TOKEN_FILE
        self._docs_service: Any | None = None
        self._drive_service: Any | None = None
        self._calendar_service: Any | None = None

    # ── authentication ─────────────────────────────────────────────────

    async def connect(self) -> None:
        """Authenticate and build Docs, Drive and Calendar service objects."""
        if self._docs_service is not None:
            return

        def _authenticate() -> tuple[Any, Any, Any]:
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
                    raise GoogleDocsError(f"Credentials file not found: {self._creds_path}")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self._creds_path),
                    _SCOPES,
                )
                # Use fixed port 8080 so Google Cloud OAuth client redirect URI (http://localhost:8080/) matches.
                creds = flow.run_local_server(port=8080)

            self._token_path.write_text(creds.to_json())

            docs = build("docs", "v1", credentials=creds)
            drive = build("drive", "v3", credentials=creds)
            calendar = build("calendar", "v3", credentials=creds)
            return docs, drive, calendar

        try:
            (
                self._docs_service,
                self._drive_service,
                self._calendar_service,
            ) = await asyncio.to_thread(_authenticate)
            logger.info("Google Docs/Drive/Calendar services connected")
        except (
            HttpError,
            GoogleAuthError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            err_str = str(exc)
            logger.error("Google Docs/Drive/Calendar authentication failed: %s", exc)
            if "invalid_scope" in err_str.lower():
                logger.error(
                    "Ensure the OAuth client in %s has Docs, Drive, Calendar scopes on the "
                    "consent screen (same project). Delete %s and restart to re-authorize.",
                    self._creds_path,
                    self._token_path,
                )
            raise GoogleDocsError(str(exc)) from exc

    @property
    def available(self) -> bool:
        return (
            self._docs_service is not None
            and self._drive_service is not None
            and self._calendar_service is not None
        )

    @property
    def calendar_available(self) -> bool:
        return self._calendar_service is not None

    async def close(self) -> None:
        """Release service objects."""
        self._docs_service = None
        self._drive_service = None
        self._calendar_service = None
        logger.info("Google Docs/Drive/Calendar services closed")

    async def _ensure_connected(self) -> None:
        if not self.available:
            await self.connect()

    # ── read ───────────────────────────────────────────────────────────

    async def get_document(self, document_id: str) -> dict[str, Any]:
        """Fetch the full Google Docs document resource."""
        await self._ensure_connected()

        def _get() -> dict[str, Any]:
            return (
                self._docs_service.documents()
                .get(
                    documentId=document_id,
                )
                .execute()
            )

        try:
            return await asyncio.to_thread(_get)
        except (
            HttpError,
            GoogleAuthError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            raise GoogleDocsError(f"Failed to get document {document_id}: {exc}") from exc

    async def read_document_text(self, document_id: str) -> str:
        """Extract all plain text from a Google Doc."""
        doc = await self.get_document(document_id)
        return _extract_text(doc)

    # ── create ─────────────────────────────────────────────────────────

    async def create_document(
        self,
        title: str,
        body_text: str | None = None,
        *,
        folder_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new Google Doc, optionally with initial text and in a folder.

        Returns the full document resource including ``documentId``.
        """
        await self._ensure_connected()

        def _create() -> dict[str, Any]:
            doc = (
                self._docs_service.documents()
                .create(
                    body={"title": title},
                )
                .execute()
            )
            return doc

        try:
            doc = await asyncio.to_thread(_create)
        except (
            HttpError,
            GoogleAuthError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            raise GoogleDocsError(f"Failed to create document '{title}': {exc}") from exc

        doc_id = doc["documentId"]

        if folder_id:
            await self._move_to_folder(doc_id, folder_id)

        if body_text:
            await self.append_text(doc_id, body_text)
            doc = await self.get_document(doc_id)

        logger.info("Created Google Doc '%s' (%s)", title, doc_id)
        return doc

    # ── update ─────────────────────────────────────────────────────────

    async def append_text(self, document_id: str, text: str) -> None:
        """Append text to the end of a Google Doc."""
        await self._ensure_connected()

        def _append() -> None:
            self._docs_service.documents().batchUpdate(
                documentId=document_id,
                body={
                    "requests": [
                        {
                            "insertText": {
                                "endOfSegmentLocation": {},
                                "text": text,
                            }
                        }
                    ]
                },
            ).execute()

        try:
            await asyncio.to_thread(_append)
        except (
            HttpError,
            GoogleAuthError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            raise GoogleDocsError(f"Failed to append text to {document_id}: {exc}") from exc

    async def replace_text(
        self,
        document_id: str,
        old_text: str,
        new_text: str,
    ) -> None:
        """Find and replace text in a Google Doc."""
        await self._ensure_connected()

        def _replace() -> None:
            self._docs_service.documents().batchUpdate(
                documentId=document_id,
                body={
                    "requests": [
                        {
                            "replaceAllText": {
                                "containsText": {
                                    "text": old_text,
                                    "matchCase": True,
                                },
                                "replaceText": new_text,
                            }
                        }
                    ]
                },
            ).execute()

        try:
            await asyncio.to_thread(_replace)
        except (
            HttpError,
            GoogleAuthError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            raise GoogleDocsError(f"Failed to replace text in {document_id}: {exc}") from exc

    async def insert_text(
        self,
        document_id: str,
        text: str,
        index: int = 1,
    ) -> None:
        """Insert text at a specific index in a Google Doc."""
        await self._ensure_connected()

        def _insert() -> None:
            self._docs_service.documents().batchUpdate(
                documentId=document_id,
                body={
                    "requests": [
                        {
                            "insertText": {
                                "location": {"index": index},
                                "text": text,
                            }
                        }
                    ]
                },
            ).execute()

        try:
            await asyncio.to_thread(_insert)
        except (
            HttpError,
            GoogleAuthError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            raise GoogleDocsError(
                f"Failed to insert text at index {index} in {document_id}: {exc}"
            ) from exc

    # ── drive: search & list ───────────────────────────────────────────

    async def search_files(
        self,
        query: str,
        *,
        mime_type: str | None = None,
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """Search Google Drive files. Supports Drive query syntax.

        If *mime_type* is provided it is ANDed with the query.
        """
        await self._ensure_connected()
        q = query
        if mime_type:
            q = f"{q} and mimeType='{mime_type}'"

        def _search() -> list[dict[str, Any]]:
            resp = (
                self._drive_service.files()
                .list(
                    q=q,
                    pageSize=max_results,
                    fields="files(id, name, mimeType, modifiedTime, webViewLink)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            return resp.get("files", [])

        try:
            return await asyncio.to_thread(_search)
        except (
            HttpError,
            GoogleAuthError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            raise GoogleDocsError(f"Drive search failed: {exc}") from exc

    async def search_docs(
        self,
        name_contains: str,
        *,
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """Search for Google Docs by name."""
        return await self.search_files(
            f"name contains '{name_contains}'",
            mime_type="application/vnd.google-apps.document",
            max_results=max_results,
        )

    async def list_folder(
        self,
        folder_id: str,
        *,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """List all files in a Google Drive folder."""
        return await self.search_files(
            f"'{folder_id}' in parents and trashed = false",
            max_results=max_results,
        )

    async def get_file_metadata(self, file_id: str) -> dict[str, Any]:
        """Get metadata for a single Drive file."""
        await self._ensure_connected()

        def _get() -> dict[str, Any]:
            return (
                self._drive_service.files()
                .get(
                    fileId=file_id,
                    fields="id, name, mimeType, modifiedTime, webViewLink, parents",
                    supportsAllDrives=True,
                )
                .execute()
            )

        try:
            return await asyncio.to_thread(_get)
        except (
            HttpError,
            GoogleAuthError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            raise GoogleDocsError(f"Failed to get file metadata for {file_id}: {exc}") from exc

    # ── drive: move ────────────────────────────────────────────────────

    async def _move_to_folder(self, file_id: str, folder_id: str) -> None:
        """Move a file into a specific Drive folder."""

        def _move() -> None:
            file_meta = (
                self._drive_service.files()
                .get(
                    fileId=file_id,
                    fields="parents",
                )
                .execute()
            )
            previous_parents = ",".join(file_meta.get("parents", []))
            self._drive_service.files().update(
                fileId=file_id,
                addParents=folder_id,
                removeParents=previous_parents,
                fields="id, parents",
            ).execute()

        try:
            await asyncio.to_thread(_move)
        except (
            HttpError,
            GoogleAuthError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            raise GoogleDocsError(f"Failed to move {file_id} to folder {folder_id}: {exc}") from exc

    # ── calendar ───────────────────────────────────────────────────────

    async def list_upcoming_events(
        self,
        *,
        days: int = 7,
        max_results: int = 20,
        calendar_id: str = "primary",
    ) -> list[dict[str, Any]]:
        """List upcoming events from the primary (or given) calendar.

        Returns a list of event dicts with summary, start, end, and optional
        location/link. All-day events have date; timed events have datetime.
        """
        await self._ensure_connected()
        if not self._calendar_service:
            return []

        from datetime import datetime, timedelta

        now = datetime.now(UTC)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=days)).isoformat()

        def _list() -> list[dict[str, Any]]:
            resp = (
                self._calendar_service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            return resp.get("items", [])

        try:
            raw = await asyncio.to_thread(_list)
        except (
            HttpError,
            GoogleAuthError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            raise GoogleDocsError(f"Calendar list failed: {exc}") from exc

        events: list[dict[str, Any]] = []
        for ev in raw:
            start = ev.get("start", {}) or {}
            end = ev.get("end", {}) or {}
            events.append(
                {
                    "summary": ev.get("summary", "(No title)"),
                    "start": start.get("dateTime") or start.get("date", ""),
                    "end": end.get("dateTime") or end.get("date", ""),
                    "location": ev.get("location", ""),
                    "htmlLink": ev.get("htmlLink", ""),
                }
            )
        return events

    # ── health ─────────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        if not self.available:
            return {"status": "disconnected", "available": False}
        try:
            about = await asyncio.to_thread(
                lambda: (
                    self._drive_service.about()
                    .get(
                        fields="user(displayName, emailAddress)",
                    )
                    .execute()
                ),
            )
            user = about.get("user", {})
            return {
                "status": "connected",
                "available": True,
                "user": user.get("displayName", "?"),
                "email": user.get("emailAddress", "?"),
            }
        except (
            HttpError,
            GoogleAuthError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            return {"status": "error", "available": False, "error": str(exc)}


def _extract_text(doc: dict[str, Any]) -> str:
    """Walk the Docs structural elements and concatenate all text runs."""
    parts: list[str] = []
    for element in doc.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        for pe in paragraph.get("elements", []):
            text_run = pe.get("textRun")
            if text_run:
                parts.append(text_run.get("content", ""))
    return "".join(parts)
