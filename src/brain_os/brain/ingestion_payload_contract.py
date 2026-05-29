"""Static contract for Qdrant ingestion payloads vs hybrid search filters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brain_os.data.models import KnowledgeItem

_DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "retrieval" / "payload_contract.json"
)


def default_payload_contract_path() -> Path:
    return _DEFAULT_CONTRACT_PATH


def load_payload_contract(path: Path | None = None) -> dict[str, Any]:
    p = path or _DEFAULT_CONTRACT_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def required_canonical_payload_keys(path: Path | None = None) -> tuple[str, ...]:
    data = load_payload_contract(path)
    keys = data.get("canonical_payload_required_keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError("payload_contract.json missing canonical_payload_required_keys")
    return tuple(str(k) for k in keys)


def assert_canonical_payload_keys_present(
    payload: dict[str, Any], *, path: Path | None = None
) -> None:
    """Raise AssertionError if *payload* is missing contract keys (used by tests)."""
    required = required_canonical_payload_keys(path)
    missing = [k for k in required if k not in payload]
    if missing:
        raise AssertionError(
            f"canonical payload missing keys {missing!r}; required={list(required)}"
        )


def sample_knowledge_item_for_contract() -> KnowledgeItem:
    """Minimal KnowledgeItem matching document ingestor shape (category + metadata)."""
    return KnowledgeItem(
        source="/tmp/sample_spec.pdf",
        source_category="machine_specs",
        content="Sample chunk about DEMO industrial forming line.",
        metadata={"doc_type": "technical", "chunk_index": 0, "total_chunks": 1},
    )
