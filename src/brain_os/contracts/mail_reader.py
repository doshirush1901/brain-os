"""MailReader protocol — Gmail search/thread read without importing interfaces.

Concrete implementation: ``brain_os.interfaces.email_processor.EmailProcessor``
(``search_emails`` / ``get_thread``). Callers in the interfaces/runtime layer
inject the concrete class into services/systems typed against this protocol (L3.3c).

Return type is ``list[Any]`` (concrete returns ``list[Email]``) so contracts
does not depend on ``brain_os.data`` — structural typing still accepts EmailProcessor.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MailReader(Protocol):
    """Structural type matching ``EmailProcessor`` search/thread methods."""

    async def search_emails(
        self,
        *,
        from_address: str = "",
        to_address: str = "",
        subject: str = "",
        label: str = "",
        query: str = "",
        after: str = "",
        before: str = "",
        max_results: int = 10,
        mailbox_scope: str = "both",
    ) -> list[Any]:
        """Search Gmail using native query syntax and return parsed Email models."""
        ...

    async def get_thread(self, thread_id: str, mailbox: str | None = None) -> list[Any]:
        """Fetch a full email thread and return as a sorted list of Email models."""
        ...


__all__ = ["MailReader"]
