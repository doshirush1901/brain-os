"""Text normalization and Cypher property sanitization for the knowledge graph."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import urlparse

_VALID_LABEL = re.compile(r"^[A-Z][A-Za-z_]{0,30}$")
_VALID_PROP_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_ENTITY_SUFFIXES = re.compile(
    r",?\s*\b(Inc\.?|LLC|Ltd\.?|Corp\.?|Co\.?|PLC|GmbH|SA|AG|NV|BV)\s*$",
    re.IGNORECASE,
)


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name for consistent graph storage."""
    name = name.strip()
    name = _ENTITY_SUFFIXES.sub("", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def company_search_token(name: str) -> str:
    """Compact token for matching related :Company nodes (e.g. DEMO-CO ↔ Demo Regional)."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def company_domain_from_host(raw: str) -> str:
    """Normalize email host, bare domain, or URL to lowercase hostname (no www.)."""
    s = (raw or "").strip().lower()
    if not s:
        return ""
    if "@" in s:
        s = s.split("@", 1)[1]
    if "://" in s:
        try:
            host = urlparse(s).netloc or s
        except ValueError:
            host = s
        s = host
    return s.removeprefix("www.").split("/")[0].strip()


def company_domain_label_key(domain: str) -> str:
    """Alphanumeric token from the hostname label (e.g. forma3d.pt → forma3d)."""
    host = company_domain_from_host(domain)
    if not host:
        return ""
    return company_search_token(host.split(".", 1)[0])


def company_name_key(name: str) -> str:
    """Canonical merge key for :Company nodes.

    Casefolded alphanumeric form of the suffix-stripped display name, so
    "WALTERPACK", "Walterpack" and "WalterPack" all resolve to one node.
    Non-Latin names (Cyrillic, CJK, etc.) fall back to a stable ``uid_`` hash
    of the normalized display name. This is THE single key every Company MERGE
    must use (unique constraint on ``name_key``); display ``name`` is set only
    ON CREATE.
    """
    norm = normalize_entity_name(name or "")
    token = company_search_token(norm)
    if token:
        return token
    if not norm:
        return ""
    digest = hashlib.sha256(norm.casefold().encode("utf-8")).hexdigest()[:16]
    return f"uid_{digest}"


def company_id_from_name(name: str) -> str:
    """Stable ``company_id`` slug; uses ``uid_`` hash when ASCII slug is empty."""
    norm = normalize_entity_name(name or "")
    slug = re.sub(r"[^a-z0-9]+", "_", norm.lower()).strip("_")
    if slug:
        return f"company::{slug}"
    key = company_name_key(name)
    if key.startswith("uid_"):
        return f"company::{key}"
    return ""


def normalize_source_id(source_id: str) -> str:
    token = re.sub(r"[^a-z0-9:_-]+", "_", (source_id or "").strip().lower()).strip("_")
    return token[:180]


def sanitize_relationship_props(props: dict[str, Any]) -> dict[str, Any]:
    """Ensure all relationship property values are Neo4j-primitive (no nested Map)."""
    out: dict[str, Any] = {}
    for k, v in props.items():
        if v is None:
            out[k] = None
        elif isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [
                x
                if isinstance(x, (bool, int, float, str))
                else json.dumps(x)
                if isinstance(x, dict)
                else str(x)
                for x in v
            ]
        elif isinstance(v, dict):
            out[k] = json.dumps(v)
        else:
            out[k] = str(v)
    return out


def valid_graph_label(label: str) -> bool:
    return bool(_VALID_LABEL.match(label))


def valid_prop_key(key: str) -> bool:
    return bool(_VALID_PROP_KEY.match(key))
