"""Brain OS license tier and Community caps."""

from __future__ import annotations

from brain_os.licensing.caps import (
    LicenseCapError,
    assert_ingest_allowed,
    community_document_limit,
    get_indexed_document_count,
    record_document_ingested,
)
from brain_os.licensing.store import LicenseRecord, load_license, save_license
from brain_os.licensing.tiers import Tier, effective_tier, is_pro, tier_label

__all__ = [
    "LicenseCapError",
    "LicenseRecord",
    "Tier",
    "assert_ingest_allowed",
    "community_document_limit",
    "effective_tier",
    "get_indexed_document_count",
    "is_pro",
    "load_license",
    "record_document_ingested",
    "save_license",
    "tier_label",
]
