"""Lightweight PII detection and redaction for ingestion.

Used before chunks are stored in Qdrant so that sensitive data can be
redacted or tagged. Regex-based; no external model.
"""

from __future__ import annotations

import re
from typing import Any

# Simple patterns: email, phone (international and common US/IN).
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
)
_PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}\b",
)


def redact_pii(text: str) -> tuple[str, dict[str, Any]]:
    """Redact email and phone in *text*. Return (redacted_text, stats).

    stats has keys: emails_redacted, phones_redacted, was_redacted.
    """
    if not text or not isinstance(text, str):
        return text, {"emails_redacted": 0, "phones_redacted": 0, "was_redacted": False}
    out = text
    emails = _EMAIL_RE.findall(out)
    phones = _PHONE_RE.findall(out)
    for e in emails:
        out = out.replace(e, "[EMAIL_REDACTED]", 1)
    for p in phones:
        out = out.replace(p, "[PHONE_REDACTED]", 1)
    was = len(emails) > 0 or len(phones) > 0
    return out, {
        "emails_redacted": len(emails),
        "phones_redacted": len(phones),
        "was_redacted": was,
    }


def should_redact_pii() -> bool:
    """Whether to apply PII redaction during ingestion (env: APP__REDACT_PII_AT_INGEST)."""
    try:
        from brain_os.config import get_settings

        return get_settings().app.redact_pii_at_ingest
    except (ImportError, AttributeError, RuntimeError, ValueError):
        return False
