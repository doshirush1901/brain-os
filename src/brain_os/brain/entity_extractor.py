"""GLiNER-based entity extraction for knowledge graph population.

Extracts companies, people, machines, and relationships from text using
the GLiNER model — a lightweight NER model that runs on CPU without
requiring LLM API calls.  Falls back to the existing LLM-based extraction
when GLiNER is unavailable.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_ENTITY_LABELS = [
    "company",
    "organization",
    "person",
    "email",
    "machine",
    "product",
    "model number",
    "location",
    "region",
    "country",
    "job title",
    "role",
]

_RELATION_PATTERNS = [
    (
        r"(?i)(\b\w[\w\s]+)\s+(?:works?\s+(?:at|for)|employed\s+(?:at|by))\s+(\b\w[\w\s]+)",
        "WORKS_AT",
        "person",
        "company",
    ),
    (
        r"(?i)(\b\w[\w\s]+)\s+(?:manufactures?|produces?|builds?)\s+(\b\w[\w\s]+)",
        "MANUFACTURES",
        "company",
        "machine",
    ),
    (
        r"(?i)(\b\w[\w\s]+)\s+(?:interested\s+in|inquir(?:ed|ing)\s+about|requested?)\s+(\b\w[\w\s]+)",
        "INTERESTED_IN",
        "company",
        "machine",
    ),
    (
        r"(?i)(\b\w[\w\s]+)\s+(?:supplied?\s+by|purchased?\s+from|bought?\s+from)\s+(\b\w[\w\s]+)",
        "SUPPLIED_BY",
        "company",
        "company",
    ),
]

_gliner_model = None


def _get_gliner():
    """Lazy-load the GLiNER model (downloads on first use, ~500MB)."""
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model
    try:
        from gliner import GLiNER

        _gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        logger.info("GLiNER model loaded successfully")
        return _gliner_model
    except (ImportError, OSError, RuntimeError, ValueError):
        logger.warning(
            "GLiNER not available — entity extraction will use LLM fallback", exc_info=True
        )
        return None


def extract_entities_gliner(text: str, *, max_chars: int = 15_000) -> dict[str, Any]:
    """Extract entities from text using GLiNER.

    Returns the same schema as KnowledgeGraph.extract_entities_from_text:
    ``{"companies": [...], "people": [...], "machines": [...], "relationships": [...]}``
    """
    model = _get_gliner()
    if model is None:
        return {"companies": [], "people": [], "machines": [], "relationships": []}

    truncated = text[:max_chars]
    predictions = model.predict_entities(truncated, _ENTITY_LABELS, threshold=0.4)

    companies: list[dict[str, str]] = []
    people: list[dict[str, str]] = []
    machines: list[dict[str, str]] = []
    seen: set[str] = set()

    for ent in predictions:
        name = ent["text"].strip()
        label = ent["label"].lower()
        if not name or len(name) < 2 or name.lower() in seen:
            continue
        seen.add(name.lower())

        if label in ("company", "organization"):
            companies.append({"name": name, "region": "", "industry": ""})
        elif label in ("person",):
            companies_nearby = [c["name"] for c in companies[-3:]]
            people.append(
                {
                    "name": name,
                    "email": "",
                    "company": companies_nearby[0] if companies_nearby else "",
                    "role": "",
                }
            )
        elif label in ("email",):
            if people:
                people[-1]["email"] = name
        elif label in ("machine", "product", "model number"):
            machines.append({"model": name, "category": "", "description": ""})
        elif label in ("job title", "role"):
            if people:
                people[-1]["role"] = name
        elif label in ("location", "region", "country"):
            if companies:
                companies[-1]["region"] = name

    relationships = _extract_relationships(truncated)

    return {
        "companies": companies,
        "people": people,
        "machines": machines,
        "relationships": relationships,
    }


def _extract_relationships(text: str) -> list[dict[str, str]]:
    """Extract relationships via regex patterns over the text."""
    rels: list[dict[str, str]] = []
    seen: set[str] = set()

    for pattern, rel_type, from_type, to_type in _RELATION_PATTERNS:
        for match in re.finditer(pattern, text):
            from_key = match.group(1).strip()
            to_key = match.group(2).strip()
            key = f"{from_key}:{rel_type}:{to_key}".lower()
            if key not in seen and len(from_key) > 1 and len(to_key) > 1:
                seen.add(key)
                rels.append(
                    {
                        "from_type": from_type,
                        "from_key": from_key,
                        "rel": rel_type,
                        "to_type": to_type,
                        "to_key": to_key,
                    }
                )

    return rels
