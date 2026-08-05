"""SystemNotifier protocol — operator/system mail send without importing interfaces.

Concrete implementation: ``brain_os.interfaces.email_processor.EmailProcessor.send_message``.
Callers in the interfaces/runtime layer inject the concrete class into services/systems
typed against this protocol (L3.3c).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SystemNotifier(Protocol):
    """Structural type matching ``EmailProcessor.send_message``."""

    async def send_message(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str | None = None,
        thread_id: str | None = None,
        *,
        from_mailbox: str | None = None,
        user_initiated: bool = False,
        archive_from_inbox: bool | None = None,
        attachment_paths: list[str] | None = None,
        extra_headers: dict[str, str] | None = None,
        html_body: str | None = None,
    ) -> dict[str, Any]:
        """Send an email via Gmail."""
        ...


__all__ = ["SystemNotifier"]
