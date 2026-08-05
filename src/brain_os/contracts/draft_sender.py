"""DraftSender protocol — outbound draft/stage surface without importing interfaces.

Concrete implementation: ``brain_os.interfaces.email_processor_draft_sender.GmailDraftSender``.
Callers in the interfaces layer construct the concrete class and inject it into
services/systems typed against this protocol (L3.3b).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DraftSender(Protocol):
    """Structural type matching ``GmailDraftSender`` method signatures."""

    async def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        thread_id: str | None = None,
        *,
        campaign_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a Gmail draft (never sends)."""
        ...

    async def send_notification(self, message: str) -> None:
        """Log/notify drip progress (no outbound mail)."""
        ...

    async def check_replies(self, thread_id: str) -> list[dict[str, Any]]:
        """List messages on a Gmail thread (metadata)."""
        ...


__all__ = ["DraftSender"]
