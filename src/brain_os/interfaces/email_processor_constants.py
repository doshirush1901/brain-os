"""Module-level constants and pure helpers carved out of email_processor.

Holds:
- OAuth scope lists (``_TRAINING_SCOPES``, ``_SEND_SCOPES``, ``_OPERATIONAL_SCOPES``)
- Intent / domain / prefix frozensets used by classification + filtering
- ``_DEAL_SUBJECT_PATTERNS`` regex
- Path resolvers (``_resolve_google_path``, ``_resolve_repo_attachment``)
- MIME builder (``_build_mime_outbound``)
- OAuth client config builder (``_client_config_from_env``)
- ``deep_scan_bar_progress`` (Rich progress callback)

``email_processor.py`` re-exports the externally-imported subset
(``_TRAINING_SCOPES``, ``_OPERATIONAL_SCOPES``, ``_DEAL_SUBJECT_PATTERNS``,
``_is_non_business_sender``, ``deep_scan_bar_progress``) for backward
compatibility with tests and downstream importers.
"""

from __future__ import annotations

import mimetypes
import os
import re
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from brain_os.exceptions import BrainOSError

_REPO_ROOT = Path(__file__).resolve().parents[3]

_TRAINING_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
_SEND_SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
]
_OPERATIONAL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
]

_REPLY_INTENTS = frozenset(
    {
        "QUOTE_REQUEST",
        "SUPPORT",
        "GENERAL_INQUIRY",
        "PARTNERSHIP",
        "COMPLAINT",
        "FOLLOW_UP",
    }
)

_DEAL_INTENTS = frozenset(
    {
        "QUOTE_REQUEST",
        "FOLLOW_UP",
        "PARTNERSHIP",
    }
)

_DEAL_SUBJECT_PATTERNS = re.compile(
    r"(?i)(quote|proposal|offer|DEMO|DEMO2|ATF|AM[-\s]|IMG|FCS|"
    r"thermoform|vacuum\s*form|machine\s+inquiry|pricing|"
    r"techno.?commercial)",
)

# Calendar invites, newsletters, PR — never auto-create CRM deals from these subjects.
_DEAL_SUBJECT_NOISE_PATTERNS = re.compile(
    r"(?i)(^accepted:\s|^declined:\s|invitation:\s|updated invitation|"
    r"automatic reply|out of office|ooo\b|unsubscribe|newsletter|"
    r"subscription purchase|mid-?cap|story pitch|webinar|"
    r"failed-?payments?@|calendar\.google\.com)",
)

# Quote number / formal quotation filename cues (Prometheus: discussion alone ≠ PROPOSAL).
_QUOTE_EVIDENCE_SUBJECT_PATTERNS = re.compile(
    r"(?i)(quotation\s*[-–]\s*MCT-|quote\s*[-–]\s*MCT-|MCT-\d{4}-\d+|"
    r"offer\s+DEMO|updated offer|techno[-\s]?commercial|"
    r"\.pdf\b.*quot|quot.*\.pdf)",
)

_NON_BUSINESS_DOMAINS = frozenset(
    {
        "instagram.com",
        "facebook.com",
        "twitter.com",
        "linkedin.com",
        "netflix.com",
        "spotify.com",
        "amazon.com",
        "google.com",
        "apple.com",
        "microsoft.com",
        "github.com",
        "openai.com",
        "moneycontrol.com",
        "squarespace.com",
        "medium.com",
        "substack.com",
        "mailchimp.com",
        "hubspot.com",
    }
)

_NON_BUSINESS_PREFIXES = frozenset(
    {
        "noreply",
        "no-reply",
        "donotreply",
        "do-not-reply",
        "notifications",
        "notify",
        "mailer",
        "newsletter",
        "marketing",
        "promo",
        "alerts",
        "billing",
    }
)

# Phase 1: vendors, SaaS, media, logistics-only — never auto-create deals (override via env).
_DEFAULT_CRM_DEAL_BLOCKLIST = (
    "yourstory.com,psa.gov.in,indusind.com,icicilombard.com,"
    "economictimesnews.com,thomasnet.com,eventbrite.com,"
    "stripe.com,cursor.com,newsdata.io,mailer.dsij.in,msmegrowthhub.com,"
    "impactguru.com,alibaba.com,openai.com,anthropic.com,mapyourshow.com,"
    "informa.com,linkedin.com,mail.ru,dsij.in,aubank.in,mayankraja.in,"
    "google.com,calendar.google.com,ups.com,nnr-g.com"
)


def _require_quote_evidence_for_proposal() -> bool:
    """When true (default), auto-created deals use PROPOSAL only with quote evidence."""
    raw = (os.environ.get("BRAIN_CRM_REQUIRE_QUOTE_EVIDENCE_FOR_PROPOSAL") or "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _crm_extra_deal_block_domains() -> frozenset[str]:
    """Domains that must never auto-create CRM deals from inbound mail (comma env)."""
    raw = (os.environ.get("BRAIN_CRM_DEAL_BLOCKLIST_DOMAINS") or _DEFAULT_CRM_DEAL_BLOCKLIST).strip()
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return frozenset(parts)


def _is_non_business_sender(email_addr: str) -> bool:
    """Fast pre-filter to skip consumer newsletters and automated senders."""
    local, _, domain = email_addr.lower().partition("@")
    if not domain:
        return False
    parent = ".".join(domain.split(".")[-2:])
    if domain in _NON_BUSINESS_DOMAINS or parent in _NON_BUSINESS_DOMAINS:
        return True
    return any(local.startswith(p) for p in _NON_BUSINESS_PREFIXES)


def _resolve_google_path(path: Path | str) -> Path:
    """Resolve credentials/token path: if relative, resolve against repo root."""
    p = Path(path)
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / p).resolve()


def _resolve_repo_attachment(rel_or_abs: str) -> Path:
    """Resolve an attachment path for outbound mail (repo-relative or absolute)."""
    p = Path(rel_or_abs).expanduser()
    out = p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve()
    if not out.is_file():
        raise BrainOSError(f"Attachment not found or not a file: {out}")
    return out


def _build_mime_outbound(
    to: str,
    subject: str,
    body: str,
    *,
    cc: str | None = None,
    attachment_paths: list[str] | None = None,
) -> MIMEMultipart | MIMEText:
    """Build a RFC 822 message, optionally with file attachments."""
    if not attachment_paths:
        msg = MIMEText(body, "plain", "utf-8")
        msg["To"] = to
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = cc
        return msg
    outer = MIMEMultipart("mixed")
    outer["To"] = to
    outer["Subject"] = subject
    if cc:
        outer["Cc"] = cc
    outer.attach(MIMEText(body, "plain", "utf-8"))
    for rel in attachment_paths:
        path = _resolve_repo_attachment(rel)
        ctype, _enc = mimetypes.guess_type(str(path))
        if not ctype:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with path.open("rb") as fp:
            part = MIMEBase(maintype, subtype)
            part.set_payload(fp.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename=path.name)
        outer.attach(part)
    return outer


def _client_config_from_env(client_id: str, client_secret: str) -> dict[str, Any]:
    """Build OAuth client config dict for InstalledAppFlow.from_client_config."""
    return {
        "installed": {
            "client_id": client_id.strip(),
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        }
    }


def deep_scan_bar_progress(stats: dict[str, Any]) -> tuple[int, int]:
    """Return ``(completed, total)`` for mailbox deep-scan Rich progress.

    Artemis triage increments ``triaged_*`` and ``skipped_non_business``; the deep loop then
    walks every listed stub. Summing ``triaged_business_high + processed`` double-counts
    business rows — use ``deep_stub_index`` (0-based index into ``all_stubs``) plus triage
    totals instead.
    """
    total = int(stats.get("total_listed") or 0)
    if total < 1:
        total = 1
    tri_sum = (
        int(stats.get("triaged_business_high", 0))
        + int(stats.get("triaged_noise", 0))
        + int(stats.get("skipped_non_business", 0))
    )
    idx = stats.get("deep_stub_index")
    if idx is None:
        return min(total, tri_sum), total
    return min(total, tri_sum + int(idx) + 1), total
