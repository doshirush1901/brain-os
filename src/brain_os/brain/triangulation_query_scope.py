"""Deterministic query classifier for pipeline triangulation gaps appendix."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_HEX_KEYWORDS = re.compile(
    r"\b(hex|hexagon|outbound|persuad|send\b|quote\b|production\s+status|atlas)\b",
    re.IGNORECASE,
)
_POSITIVE = re.compile(
    r"\b("
    r"triangulat|hexagon|hex\b|missing\s+legs?|gaps\s+only|account\s+brief|"
    r"\bleads?\b|\baccount\b|\bcompany\b|\bdeal\b|\bprospect\b|\bcustomer\b|"
    r"know\s+about|what\s+do\s+we\s+know|"
    r"pipeline\s+status|hot\s+leads?|"
    r"outbound|persuasion|persuade|draft\s+email|follow[\s-]?up|cold\s+email|"
    r"crm\s+stage|quote\s+sent|thread\s+with"
    r")\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b("
    r"^(hi|hello|hey|thanks|thank\s+you|good\s+(morning|afternoon|evening))\b|"
    r"who\s+are\s+you|what\s+is\s+ira|"
    r"no\s+triangulation|skip\s+triangulation|"
    r"\bpytest\b|\bdocker\b|src/brain_os|\.py\b|git\s+commit|"
    r"lead\s+time\b|delivery\s+time\b|machine\s+spec"
    r")\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_QUOTED_RE = re.compile(
    r'(?:for|about|on|regarding|re:?)\s+["\u201c]([^"\u201d]{2,120})["\u201d]',
    re.IGNORECASE,
)
_BRIEF_COMPANY_RE = re.compile(
    r"(?:brief|triangulat\w*|account)\s+(?:on|for|about)\s+([A-Za-z0-9][\w\s&.-]{1,80})",
    re.IGNORECASE,
)
_ABOUT_COMPANY_RE = re.compile(
    r"(?i)(?:know\s+about|about)\s+"
    r"([A-Z][\w][\w\s&.-]{1,79}?)"
    r"(?:\s+as\s+(?:a\s+)?lead|[?.!,]|$)",
)
_SHAPE_COMPANY_RE = re.compile(
    r"Pipeline\s+task:\s*Shape\s+the\s+(.+?)\s+(?:July|email|dates)\b",
    re.IGNORECASE,
)
_APPROVED_COPY_SPLIT = re.compile(r"\bApproved\s+copy\s*:", re.IGNORECASE)


@dataclass(frozen=True)
class TriangulationScope:
    applies: bool
    mode: Literal["triangle", "hex"]
    company_hint: str | None
    contact_email_hint: str | None
    reason: str


def _domain_to_company_hint(domain: str) -> str:
    parts = domain.lower().split(".")
    if len(parts) >= 2 and parts[0] not in ("www", "mail"):
        return parts[0].replace("-", " ").title()
    return domain.split(".")[0].title()


def extract_company_and_contact(
    query: str,
    *,
    fallback_contact: str = "",
    fallback_company: str = "",
) -> tuple[str | None, str | None]:
    """Best-effort company and email from query text."""
    text = (query or "").strip()
    header = text
    approved_split = _APPROVED_COPY_SPLIT.search(text)
    if approved_split:
        header = text[: approved_split.start()]

    contact: str | None = None
    company: str | None = None

    shape_m = _SHAPE_COMPANY_RE.search(header)
    if shape_m:
        company = shape_m.group(1).strip().rstrip(".,;")

    email_m = _EMAIL_RE.search(header)
    if email_m:
        contact = email_m.group(0).lower()
        if not company:
            dom = contact.split("@", 1)[-1]
            if dom and not dom.endswith(("gmail.com", "yahoo.com", "hotmail.com", "outlook.com")):
                company = _domain_to_company_hint(dom)

    qm = _QUOTED_RE.search(text)
    if qm:
        company = qm.group(1).strip()

    bm = _BRIEF_COMPANY_RE.search(text)
    if bm:
        company = bm.group(1).strip().rstrip(".,;")

    am = _ABOUT_COMPANY_RE.search(text)
    if am:
        cand = am.group(1).strip().rstrip(".,;")
        first = (cand.split() or [""])[0].lower()
        if first not in {"the", "this", "that", "our", "your", "their", "a", "an"}:
            company = cand

    if fallback_contact and not contact:
        contact = fallback_contact.strip() or None
    if fallback_company and not company:
        company = fallback_company.strip() or None

    if company:
        company = " ".join(company.split())[:120]
    return company or None, contact


def classify_triangulation_scope(
    query: str,
    *,
    hex_enabled: bool = False,
    skip_requested: bool = False,
) -> TriangulationScope:
    """Return whether pipeline should append a gaps-only triangulation block."""
    if skip_requested:
        return TriangulationScope(False, "triangle", None, None, "skip_requested")

    text = (query or "").strip()
    if not text or len(text) < 4:
        return TriangulationScope(False, "triangle", None, None, "query_too_short")

    if _NEGATIVE.search(text):
        return TriangulationScope(False, "triangle", None, None, "negative_pattern")

    if not _POSITIVE.search(text):
        return TriangulationScope(False, "triangle", None, None, "no_positive_signal")

    use_hex = hex_enabled and bool(_HEX_KEYWORDS.search(text))
    mode: Literal["triangle", "hex"] = "hex" if use_hex else "triangle"
    company, contact = extract_company_and_contact(text)
    reason = "hex_keywords" if use_hex else "account_outbound_signal"
    return TriangulationScope(True, mode, company, contact, reason)
