"""Canonical source identity helpers for ingestion traceability."""

from __future__ import annotations

import hashlib
from pathlib import Path


def make_source_id(source_path: str, file_hash: str) -> str:
    """Create a stable source identifier from absolute path + file hash."""
    normalized = str(Path(source_path).resolve())
    digest = hashlib.sha256(f"{normalized}:{file_hash}".encode()).hexdigest()
    return digest[:24]
