"""Text normalization and Cypher property sanitization for the knowledge graph."""

from __future__ import annotations

import json
import re
from typing import Any

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
    """Compact token for matching related :Company nodes (e.g. NAFFCO ↔ Naffco KSA)."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


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
