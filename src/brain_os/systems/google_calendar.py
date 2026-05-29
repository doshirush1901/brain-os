"""Google Calendar integration (optional).

Provides auth + event creation for MCP/CLI flows that need to schedule
Google Meet invites.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from brain_os.config import GoogleConfig, get_settings

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/calendar"]


class GoogleCalendarError(Exception):
    """Raised when Google Calendar operations fail."""


def _resolve_google_path(path: Path | str) -> Path:
    """Resolve credentials/token path; relative paths are repo-root relative."""
    p = Path(path)
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / p).resolve()


class GoogleCalendarService:
    """Google Calendar client with Meet invite creation support."""

    def __init__(self, config: GoogleConfig | None = None) -> None:
        cfg = config or get_settings().google
        self._creds_path = _resolve_google_path(cfg.credentials_path)
        self._token_path = (self._creds_path.parent / "token_calendar.json").resolve()
        self._oauth_client_id = (cfg.oauth_client_id or "").strip()
        self._oauth_client_secret = (cfg.oauth_client_secret.get_secret_value() or "").strip()
        self._calendar_service: Any | None = None
        self._last_error: str | None = None

    async def connect(self) -> None:
        """Authenticate and initialize Calendar API client.

        Degrades gracefully when credentials are not configured.
        """
        if self._calendar_service is not None:
            return

        def _authenticate() -> Any:
            creds: Credentials | None = None
            if self._token_path.exists():
                creds = Credentials.from_authorized_user_file(str(self._token_path), _SCOPES)

            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            elif not creds or not creds.valid:
                if self._oauth_client_id and self._oauth_client_secret:
                    flow = InstalledAppFlow.from_client_config(
                        {
                            "installed": {
                                "client_id": self._oauth_client_id,
                                "client_secret": self._oauth_client_secret,
                                "redirect_uris": ["http://localhost"],
                                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                "token_uri": "https://oauth2.googleapis.com/token",
                                "auth_provider_x509_cert_url": (
                                    "https://www.googleapis.com/oauth2/v1/certs"
                                ),
                                "client_x509_cert_url": (
                                    "https://www.googleapis.com/oauth2/v1/certs"
                                ),
                            }
                        },
                        _SCOPES,
                    )
                else:
                    if not self._creds_path.exists():
                        raise GoogleCalendarError(
                            "Google Calendar credentials missing. Set GOOGLE_CREDENTIALS_PATH "
                            f"or provide oauth client id/secret. Expected: {self._creds_path}"
                        )
                    flow = InstalledAppFlow.from_client_secrets_file(str(self._creds_path), _SCOPES)
                creds = flow.run_local_server(port=0)

            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_path.write_text(creds.to_json(), encoding="utf-8")
            return build("calendar", "v3", credentials=creds)

        try:
            self._calendar_service = await asyncio.to_thread(_authenticate)
            self._last_error = None
        except (OSError, ValueError, RuntimeError) as exc:
            self._calendar_service = None
            self._last_error = str(exc)
            logger.warning("Google Calendar unavailable: %s", exc)

    async def close(self) -> None:
        self._calendar_service = None

    @property
    def available(self) -> bool:
        return self._calendar_service is not None

    async def create_meeting_event(
        self,
        *,
        summary: str,
        start_iso: str,
        end_iso: str,
        timezone: str,
        attendee_emails: list[str],
        description: str = "",
        location: str = "",
        calendar_id: str = "primary",
        send_updates: bool = True,
        add_google_meet: bool = True,
    ) -> dict[str, Any]:
        """Create a Calendar event with optional Google Meet and optional attendee emails.

        When ``add_google_meet`` is false, no Meet conference is created (use for
        Microsoft Teams / external links in ``description`` or ``location``).
        """
        if not self.available:
            await self.connect()
        if self._calendar_service is None:
            raise GoogleCalendarError(self._last_error or "Google Calendar not connected")

        attendees = [{"email": e.strip().lower()} for e in attendee_emails if e.strip()]
        body: dict[str, Any] = {
            "summary": summary.strip() or "Meeting",
            "description": description or "",
            "start": {"dateTime": start_iso, "timeZone": timezone},
            "end": {"dateTime": end_iso, "timeZone": timezone},
        }
        if add_google_meet:
            body["conferenceData"] = {
                "createRequest": {
                    "requestId": f"ira-meet-{uuid.uuid4().hex[:12]}",
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }
        if attendees:
            body["attendees"] = attendees
        if location.strip():
            body["location"] = location.strip()

        def _create() -> dict[str, Any]:
            return (
                self._calendar_service.events()
                .insert(
                    calendarId=calendar_id or "primary",
                    body=body,
                    conferenceDataVersion=1 if add_google_meet else 0,
                    sendUpdates="all" if send_updates else "none",
                )
                .execute()
            )

        try:
            created = await asyncio.to_thread(_create)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise GoogleCalendarError(f"Failed to create meeting event: {exc}") from exc

        return {
            "event_id": created.get("id"),
            "event_link": created.get("htmlLink"),
            "meet_link": created.get("hangoutLink"),
            "start": created.get("start"),
            "end": created.get("end"),
            "attendees": created.get("attendees", []),
        }

    async def check_availability(
        self,
        *,
        start_iso: str,
        end_iso: str,
        timezone: str = "UTC",
        calendar_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Check free/busy status for one or more calendars in a time window."""
        if not self.available:
            await self.connect()
        if self._calendar_service is None:
            raise GoogleCalendarError(self._last_error or "Google Calendar not connected")

        ids = [c.strip() for c in (calendar_ids or ["primary"]) if c.strip()] or ["primary"]
        body = {
            "timeMin": start_iso,
            "timeMax": end_iso,
            "timeZone": timezone,
            "items": [{"id": cid} for cid in ids],
        }

        def _query() -> dict[str, Any]:
            return self._calendar_service.freebusy().query(body=body).execute()

        try:
            raw = await asyncio.to_thread(_query)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise GoogleCalendarError(f"Failed to check availability: {exc}") from exc

        calendars = raw.get("calendars", {}) if isinstance(raw, dict) else {}
        by_calendar: dict[str, Any] = {}
        all_busy: list[dict[str, str]] = []
        for cid in ids:
            row = calendars.get(cid, {}) if isinstance(calendars, dict) else {}
            busy = row.get("busy", []) if isinstance(row, dict) else []
            cleaned: list[dict[str, str]] = []
            for item in busy:
                if not isinstance(item, dict):
                    continue
                start = str(item.get("start", ""))
                end = str(item.get("end", ""))
                cleaned.append({"start": start, "end": end})
                all_busy.append({"calendar_id": cid, "start": start, "end": end})
            by_calendar[cid] = {"busy": cleaned, "is_free": len(cleaned) == 0}

        return {
            "window": {"start": start_iso, "end": end_iso, "timezone": timezone},
            "calendar_ids": ids,
            "is_free_all": len(all_busy) == 0,
            "busy_slots": all_busy,
            "by_calendar": by_calendar,
        }

    async def health_check(self) -> dict[str, Any]:
        if not self.available:
            return {
                "status": "disabled",
                "detail": self._last_error or "Google Calendar not connected",
            }
        try:
            now = (
                self._calendar_service.calendarList().list(maxResults=1).execute().get("items", [])
            )
            return {
                "status": "healthy",
                "available": True,
                "calendar_count_hint": len(now),
                "token_path": str(self._token_path),
            }
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            return {"status": "error", "available": False, "detail": str(exc)}
