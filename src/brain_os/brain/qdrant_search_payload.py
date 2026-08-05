"""Default Qdrant search payload field allowlist.

Extracted from ``qdrant_manager`` (L0.1 file-size peel).
"""

from __future__ import annotations

# Payload fields returned by search/hybrid_search.  Fetching only these avoids
# transferring large raw payloads over the network — critical for Qdrant Cloud
# where full-payload responses can be 10-30x slower than selective ones.
SEARCH_PAYLOAD_FIELDS: list[str] = [
    "content",
    "text",
    "raw_text",
    "source",
    "filename",
    "source_category",
    "doc_type",
    "metadata",
    "machines",
    "prices",
    "customer",
    "chunk",
    "total_chunks",
    "source_group",
    "ingested_at",
    "subject",
    "from_email",
    "to_email",
    "direction",
    "thread_key",
    "company_domain",
    "has_quote",
    "has_price",
    "canonical_company",
    "email_date",
    "content_hash",
    "embedding_model",
    "ingest_version",
]
