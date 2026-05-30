"""Qdrant vector-database manager for Brain OS's knowledge base.

Provides collection lifecycle management, batched upserts of
:class:`~brain_os.data.models.KnowledgeItem` objects, dense vector search,
and hybrid (dense + keyword filter) retrieval.  All operations are async
and go through :class:`~brain_os.brain.embeddings.EmbeddingService` for
embedding generation.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar
from uuid import UUID

import httpx
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from brain_os.brain.embeddings import EmbeddingService
from brain_os.config import QdrantConfig, get_settings
from brain_os.data.models import KnowledgeItem
from brain_os.exceptions import DatabaseError, BrainOSError, LLMError
from brain_os.services.resilience import RetryPolicy, run_with_retry

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Typed groups for Qdrant / embedding call sites (BLE001 narrowing).
_QDRANT_INIT_ERRORS = (httpx.HTTPError, OSError, ValueError, TypeError, UnexpectedResponse)
_QDRANT_HTTP_IO_ERRORS = (httpx.HTTPError, OSError, UnexpectedResponse, TypeError, AttributeError)
_QDRANT_STORE_ERRORS = (
    DatabaseError,
    httpx.HTTPError,
    OSError,
    LLMError,
    ValueError,
    TypeError,
    UnexpectedResponse,
)
_QDRANT_EVENT_ERRORS = (BrainOSError, OSError, TypeError, ValueError)

_UPSERT_BATCH_SIZE = 100
# Smaller batches for Qdrant Cloud to avoid request/payload size limits (e.g. 32MB).
_CLOUD_UPSERT_BATCH_SIZE = 50
# Max concurrent search requests to Qdrant.  Qdrant Cloud resets connections
# when too many arrive simultaneously (ResponseHandlingException); a semaphore
# of 4 keeps throughput high while avoiding connection drops.
_MAX_CONCURRENT_SEARCHES = 4

# Payload fields returned by search/hybrid_search.  Fetching only these avoids
# transferring large raw payloads over the network — critical for Qdrant Cloud
# where full-payload responses can be 10-30x slower than selective ones.
_SEARCH_PAYLOAD_FIELDS: list[str] = [
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
]

_ENTITY_SUFFIXES = re.compile(
    r",?\s*\b(Inc\.?|LLC|Ltd\.?|Corp\.?|Co\.?|PLC|GmbH|SA|AG|NV|BV)\s*$",
    re.IGNORECASE,
)


def _normalize_company_name(name: str) -> str:
    cleaned = (name or "").strip()
    cleaned = _ENTITY_SUFFIXES.sub("", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned)


def _slug_token(value: str, *, fallback: str = "unknown") -> str:
    token = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return token or fallback


def _canonicalize_payload(item: KnowledgeItem) -> dict[str, Any]:
    metadata = dict(item.metadata or {})
    source = str(item.source or "").strip()
    source_category = _slug_token(str(item.source_category or ""), fallback="unknown")
    doc_type = _slug_token(
        str(metadata.get("doc_type") or source_category),
        fallback=source_category,
    )
    company_raw = (
        metadata.get("canonical_company")
        or metadata.get("company_name")
        or metadata.get("company")
        or metadata.get("customer")
        or ""
    )
    canonical_company = _normalize_company_name(str(company_raw))
    company_id = ""
    if canonical_company:
        company_id = f"company::{_slug_token(canonical_company)}"
        metadata.setdefault("canonical_company", canonical_company)
        metadata.setdefault("company_id", company_id)
    payload = {
        "content": item.content,
        "source": source,
        "source_category": source_category,
        "doc_type": doc_type,
        "metadata": metadata,
        "created_at": item.created_at.isoformat(),
    }
    if canonical_company:
        payload["canonical_company"] = canonical_company
    if company_id:
        payload["company_id"] = company_id
    return payload


def _is_collection_not_found_error(exc: Exception) -> bool:
    """True when Qdrant error indicates missing target collection."""
    text = str(exc).lower()
    return (
        "doesn't exist" in text or "not found" in text or ("collection" in text and "404" in text)
    )


def _qdrant_base_url(url: str) -> str:
    return url.strip().rstrip("/")


def _qdrant_urls_equivalent(a: str, b: str) -> bool:
    return _qdrant_base_url(a) == _qdrant_base_url(b)


def _is_qdrant_transport_error(exc: BaseException) -> bool:
    """True when failing the primary Qdrant node is likely transient (try fallback)."""
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            ConnectionResetError,
            BrokenPipeError,
        ),
    ):
        return True
    if isinstance(exc, OSError) and exc.errno is not None:
        # e.g. connection refused, network unreachable
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and isinstance(cause, BaseException):
        return _is_qdrant_transport_error(cause)
    if type(exc).__name__ == "ResponseHandlingException":
        inner = getattr(exc, "__cause__", None)
        if inner is not None and isinstance(inner, BaseException):
            return _is_qdrant_transport_error(inner)
    low = str(exc).lower()
    if any(
        s in low
        for s in (
            "connection reset",
            "broken pipe",
            "connection refused",
            "name or service not known",
            "nodename nor servname",
            "timed out",
            "timeout",
            "unreachable",
            "network is unreachable",
        )
    ):
        return True
    return False


class QdrantManager:
    """Async wrapper around :class:`AsyncQdrantClient`."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        config: QdrantConfig | None = None,
        event_bus: Any | None = None,
    ) -> None:
        cfg = config or get_settings().qdrant
        self._embeddings = embedding_service
        self._config: QdrantConfig = cfg
        app = get_settings().app
        self._use_sparse_hybrid = getattr(app, "use_sparse_hybrid", False)
        self._default_collection = (
            cfg.collection_hybrid if self._use_sparse_hybrid else cfg.collection
        )
        self._event_bus = event_bus
        self._search_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SEARCHES)
        self._attach_clients(cfg)

    def _attach_clients(self, cfg: QdrantConfig) -> None:
        """Build primary, optional fallback, and optional cloud-mirror clients."""
        api_key = cfg.api_key.get_secret_value() or None
        self._client = AsyncQdrantClient(
            url=cfg.url,
            api_key=api_key,
            check_compatibility=False,
            timeout=cfg.timeout,
        )
        self._client_fallback: AsyncQdrantClient | None = None
        fu = cfg.fallback_url.strip()
        if fu:
            fk = cfg.fallback_api_key.get_secret_value() or None
            try:
                self._client_fallback = AsyncQdrantClient(
                    url=fu,
                    api_key=fk or None,
                    check_compatibility=False,
                    timeout=cfg.timeout,
                )
                logger.info("Qdrant fallback enabled: %s", fu[:72])
            except _QDRANT_INIT_ERRORS:
                logger.warning(
                    "Qdrant fallback client init failed — running without fallback", exc_info=True
                )

        self._client_cloud: AsyncQdrantClient | None = None
        cu = cfg.cloud_url.strip()
        if cu and _qdrant_urls_equivalent(cu, cfg.url):
            logger.debug("Skipping QDRANT_CLOUD_URL: identical to QDRANT_URL (no duplicate mirror)")
        elif cu:
            cloud_key = cfg.cloud_api_key.get_secret_value() or None
            try:
                self._client_cloud = AsyncQdrantClient(
                    url=cu,
                    api_key=cloud_key,
                    check_compatibility=False,
                    timeout=cfg.timeout,
                )
                logger.info("Qdrant dual-write mirror: %s", cu[:50])
            except _QDRANT_INIT_ERRORS:
                logger.warning(
                    "Qdrant cloud mirror client init failed — mirror disabled", exc_info=True
                )

    async def _run_with_fallback(
        self,
        op_name: str,
        runner: Callable[[AsyncQdrantClient], Awaitable[_T]],
    ) -> _T:
        """Run *runner* on the primary client; on transport failure retry on fallback."""
        try:
            return await runner(self._client)
        except Exception as exc:  # intentional — classify via _is_qdrant_transport_error
            if self._client_fallback is None or not _is_qdrant_transport_error(exc):
                raise
            logger.warning(
                "Qdrant primary failed for %s (%s); using fallback",
                op_name,
                type(exc).__name__,
            )
            return await runner(self._client_fallback)

    async def reopen_clients(self) -> None:
        """Close and recreate all Qdrant HTTP clients (self-healing)."""
        await self._close_inner_clients()
        self._attach_clients(self._config)

    async def _close_inner_clients(self) -> None:
        for name, cl in (
            ("primary", self._client),
            ("fallback", self._client_fallback),
            ("mirror", self._client_cloud),
        ):
            if cl is None:
                continue
            try:
                await cl.close()
            except _QDRANT_HTTP_IO_ERRORS:
                logger.debug("Qdrant %s client close failed", name, exc_info=True)

    def set_event_bus(self, event_bus: Any) -> None:
        self._event_bus = event_bus

    async def cloud_health_check(self) -> dict[str, Any] | None:
        """When Qdrant cloud mirror is configured, verify connectivity.

        Returns ``None`` when cloud sync is not configured.
        """
        if self._client_cloud is None:
            return None
        start = time.monotonic()
        try:
            await self._client_cloud.get_collections()
            latency = (time.monotonic() - start) * 1000
            return {"status": "healthy", "latency_ms": round(latency, 1), "error": None}
        except _QDRANT_HTTP_IO_ERRORS as exc:
            latency = (time.monotonic() - start) * 1000
            logger.warning("Qdrant cloud health check failed: %s", exc)
            return {"status": "unhealthy", "latency_ms": round(latency, 1), "error": str(exc)}

    # ── collection lifecycle ─────────────────────────────────────────────

    _PAYLOAD_INDEXES: list[tuple[str, str]] = [
        ("source_category", "keyword"),
        ("doc_type", "keyword"),
        ("source", "keyword"),
        ("canonical_company", "keyword"),
        ("company_id", "keyword"),
        ("source_group", "keyword"),
        ("filename", "keyword"),
        ("from_email", "keyword"),
        ("to_email", "keyword"),
        ("direction", "keyword"),
        ("company_domain", "keyword"),
        ("thread_key", "keyword"),
        ("has_quote", "bool"),
        ("has_price", "bool"),
    ]

    async def ensure_collection(
        self,
        name: str | None = None,
        vector_size: int = 1024,
    ) -> None:
        """Create the collection if it does not already exist.

        Defaults to cosine distance, which matches the normalised Voyage
        embeddings.  Also ensures payload indexes exist for commonly
        filtered fields (required by Qdrant Cloud).
        """
        collection = name or self._default_collection

        async def ensure_on(client: AsyncQdrantClient) -> None:
            if await client.collection_exists(collection):
                logger.debug("Collection '%s' already exists", collection)
            else:
                if self._use_sparse_hybrid:
                    await client.create_collection(
                        collection_name=collection,
                        vectors_config={
                            "dense": models.VectorParams(
                                size=vector_size,
                                distance=models.Distance.COSINE,
                            ),
                        },
                        sparse_vectors_config={
                            "sparse": models.SparseVectorParams(),
                        },
                    )
                    logger.info(
                        "Created Qdrant collection '%s' (dense+sparse hybrid)",
                        collection,
                    )
                else:
                    await client.create_collection(
                        collection_name=collection,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        ),
                    )
                    logger.info("Created Qdrant collection '%s' (dim=%d)", collection, vector_size)
            await self._ensure_payload_indexes(client, collection)

        try:
            await self._run_with_fallback("ensure_collection", ensure_on)
        except _QDRANT_STORE_ERRORS:
            logger.exception("Failed to ensure collection '%s'", collection)
            raise

        # Mirror to cloud if sync is enabled, but do not block startup if cloud is unavailable.
        if self._client_cloud is not None:
            try:
                if await self._client_cloud.collection_exists(collection):
                    logger.debug("Cloud collection '%s' already exists", collection)
                else:
                    if self._use_sparse_hybrid:
                        await self._client_cloud.create_collection(
                            collection_name=collection,
                            vectors_config={
                                "dense": models.VectorParams(
                                    size=vector_size,
                                    distance=models.Distance.COSINE,
                                ),
                            },
                            sparse_vectors_config={
                                "sparse": models.SparseVectorParams(),
                            },
                        )
                    else:
                        await self._client_cloud.create_collection(
                            collection_name=collection,
                            vectors_config=models.VectorParams(
                                size=vector_size,
                                distance=models.Distance.COSINE,
                            ),
                        )
                    logger.info("Created Qdrant cloud collection '%s'", collection)
                await self._ensure_payload_indexes(self._client_cloud, collection)
            except _QDRANT_HTTP_IO_ERRORS:
                logger.warning(
                    "Qdrant mirror unavailable while ensuring '%s' — primary/fallback only",
                    collection,
                    exc_info=True,
                )

    async def _ensure_payload_indexes(
        self,
        client: AsyncQdrantClient,
        collection: str,
    ) -> None:
        """Create payload indexes if they don't already exist.

        Required for Qdrant Cloud where filtered queries fail without
        explicit indexes.  Errors are logged but not raised so that
        startup is not blocked by index creation on large collections.
        """
        try:
            info = await client.get_collection(collection)
            existing = set((info.payload_schema or {}).keys())
        except (*_QDRANT_HTTP_IO_ERRORS, TypeError, AttributeError):
            existing = set()

        for field, schema_type in self._PAYLOAD_INDEXES:
            if field in existing:
                continue
            try:
                schema = (
                    models.PayloadSchemaType.KEYWORD
                    if schema_type == "keyword"
                    else models.PayloadSchemaType.BOOL
                )
                await client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=schema,
                )
                logger.info(
                    "Created payload index '%s' (%s) on '%s'", field, schema_type, collection
                )
            except _QDRANT_HTTP_IO_ERRORS:
                logger.debug(
                    "Payload index '%s' on '%s' not created (may already exist or cluster busy)",
                    field,
                    collection,
                    exc_info=True,
                )

    # ── upsert ───────────────────────────────────────────────────────────

    async def upsert_items(
        self,
        items: Sequence[KnowledgeItem],
        collection: str | None = None,
    ) -> int:
        """Embed and upsert knowledge items, returning the count stored.

        Items are embedded in one call (the embedding service handles its
        own batching) and then upserted to Qdrant in batches of
        ``_UPSERT_BATCH_SIZE``.
        """
        if not items:
            return 0

        col = collection or self._default_collection
        texts = [item.content for item in items]

        try:
            vectors = await self._embeddings.embed_texts(texts)
        except (LLMError, httpx.HTTPError, OSError, ValueError, TypeError):
            logger.exception("Embedding failed for %d items", len(items))
            raise

        if self._use_sparse_hybrid:
            from brain_os.brain.sparse_vectors import text_to_sparse

            points = []
            for item, vector in zip(items, vectors):
                indices, values = text_to_sparse(item.content)
                payload = _canonicalize_payload(item)
                points.append(
                    models.PointStruct(
                        id=_uuid_to_hex(item.id),
                        vector={
                            "dense": vector,
                            "sparse": models.SparseVector(
                                indices=indices,
                                values=values,
                            ),
                        },
                        payload=payload,
                    )
                )
        else:
            points = [
                models.PointStruct(
                    id=_uuid_to_hex(item.id),
                    vector=vector,
                    payload=_canonicalize_payload(item),
                )
                for item, vector in zip(items, vectors)
            ]

        upserted = 0
        for start in range(0, len(points), _UPSERT_BATCH_SIZE):
            batch = points[start : start + _UPSERT_BATCH_SIZE]
            try:

                async def _upsert_batch(c: AsyncQdrantClient, pts: list[Any] = batch) -> None:
                    await c.upsert(collection_name=col, points=pts)

                await self._run_with_fallback(f"upsert@{start}", _upsert_batch)
                upserted += len(batch)
            except _QDRANT_STORE_ERRORS as exc:
                if _is_collection_not_found_error(exc):
                    logger.warning(
                        "Qdrant collection '%s' missing during upsert; creating and retrying batch",
                        col,
                    )
                    await self.ensure_collection(name=col, vector_size=len(vectors[0]))

                    async def _retry_batch(c: AsyncQdrantClient, pts: list[Any] = batch) -> None:
                        await c.upsert(collection_name=col, points=pts)

                    await self._run_with_fallback(f"upsert_retry@{start}", _retry_batch)
                    upserted += len(batch)
                    continue
                logger.exception(
                    "Qdrant upsert failed at offset %d (batch size %d)",
                    start,
                    len(batch),
                )
                raise
            except (
                Exception
            ) as exc:  # intentional — Qdrant client may raise plain Exception for 404
                if not _is_collection_not_found_error(exc):
                    logger.exception(
                        "Qdrant upsert failed at offset %d (batch size %d)",
                        start,
                        len(batch),
                    )
                    raise
                logger.warning(
                    "Qdrant collection '%s' missing during upsert; creating and retrying batch",
                    col,
                )
                await self.ensure_collection(name=col, vector_size=len(vectors[0]))

                async def _retry_batch_plain(c: AsyncQdrantClient, pts: list[Any] = batch) -> None:
                    await c.upsert(collection_name=col, points=pts)

                await self._run_with_fallback(f"upsert_retry@{start}", _retry_batch_plain)
                upserted += len(batch)
                continue
        # Mirror to cloud if sync is enabled (same points, same collection); use smaller batches for Cloud limits
        if self._client_cloud is not None:
            for start in range(0, len(points), _UPSERT_BATCH_SIZE):
                batch = points[start : start + _UPSERT_BATCH_SIZE]
                for cloud_start in range(0, len(batch), _CLOUD_UPSERT_BATCH_SIZE):
                    cloud_batch = batch[cloud_start : cloud_start + _CLOUD_UPSERT_BATCH_SIZE]
                    try:
                        await self._client_cloud.upsert(collection_name=col, points=cloud_batch)
                    except _QDRANT_HTTP_IO_ERRORS:
                        try:
                            import asyncio

                            await asyncio.sleep(0.5)
                            await self._client_cloud.upsert(collection_name=col, points=cloud_batch)
                        except _QDRANT_HTTP_IO_ERRORS:
                            logger.warning(
                                "Qdrant cloud sync upsert failed at offset %d (batch size %d) — primary succeeded",
                                start + cloud_start,
                                len(cloud_batch),
                                exc_info=True,
                            )

        logger.info(
            "Upserted %d items into '%s'%s",
            upserted,
            col,
            " (synced to cloud)" if self._client_cloud else "",
        )

        if self._event_bus is not None and upserted > 0:
            from brain_os.systems.data_event_bus import DataEvent, EventType, SourceStore

            try:
                await self._event_bus.emit(
                    DataEvent(
                        event_type=EventType.CHUNK_UPSERTED,
                        entity_type="knowledge_chunk",
                        entity_id=col,
                        payload={
                            "collection": col,
                            "count": upserted,
                            "sources": list({it.source for it in items[:10]}),
                        },
                        source_store=SourceStore.QDRANT,
                    )
                )
            except _QDRANT_EVENT_ERRORS:
                logger.debug("Qdrant event emission failed", exc_info=True)

        return upserted

    # ── retrieve by id (for graph–vector stitch) ─────────────────────────

    async def get_points(
        self,
        point_ids: Sequence[str],
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve points by ID; returns list of payload dicts (id, content, source, metadata)."""
        if not point_ids:
            return []
        col = collection or self._default_collection
        try:

            async def _retrieve(c: AsyncQdrantClient) -> Any:
                return await c.retrieve(
                    collection_name=col,
                    ids=list(point_ids),
                    with_payload=_SEARCH_PAYLOAD_FIELDS,
                    with_vectors=False,
                )

            points = await self._run_with_fallback("get_points", _retrieve)
        except _QDRANT_STORE_ERRORS:
            logger.exception("get_points failed for %d ids", len(point_ids))
            raise
        return [_record_to_dict(p) for p in points if getattr(p, "payload", None)]

    # ── search ───────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        collection: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Dense vector search — embed the query and find nearest neighbours."""
        col = collection or self._default_collection

        async with self._search_semaphore:
            try:
                query_vector = await self._embeddings.embed_query(query)

                async def _vector_search(c: AsyncQdrantClient) -> Any:
                    if self._use_sparse_hybrid:
                        return await c.query_points(
                            collection_name=col,
                            query=query_vector,
                            using="dense",
                            limit=limit,
                            score_threshold=score_threshold,
                            with_payload=_SEARCH_PAYLOAD_FIELDS,
                        )
                    return await c.query_points(
                        collection_name=col,
                        query=query_vector,
                        limit=limit,
                        score_threshold=score_threshold,
                        with_payload=_SEARCH_PAYLOAD_FIELDS,
                    )

                result = await self._run_with_fallback("search", _vector_search)
                hits = result.points
            except _QDRANT_STORE_ERRORS:
                logger.exception("Search failed in '%s'", col)
                raise

        return [_hit_to_dict(hit) for hit in hits]

    async def hybrid_search(
        self,
        query: str,
        collection: str | None = None,
        limit: int = 10,
        keyword_filter: models.Filter | None = None,
        source_category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Dense vector search combined with Qdrant payload filtering.

        If *source_category* is provided a ``must`` match condition is
        built automatically.  For more complex predicates pass a full
        :class:`qdrant_client.models.Filter` via *keyword_filter*.
        """
        col = collection or self._default_collection

        if keyword_filter is None and source_category is not None:
            keyword_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="source_category",
                        match=models.MatchValue(value=source_category),
                    ),
                    models.FieldCondition(
                        key="doc_type",
                        match=models.MatchValue(value=source_category),
                    ),
                ]
            )

        async with self._search_semaphore:
            query_vector = await self._embeddings.embed_query(query)
            si, sv = ([], [])
            if self._use_sparse_hybrid:
                from brain_os.brain.sparse_vectors import text_to_sparse

                si, sv = text_to_sparse(query)

            async def _hybrid_qp(
                c: AsyncQdrantClient,
                qf: models.Filter | None = keyword_filter,
            ) -> Any:
                if self._use_sparse_hybrid:
                    return await c.query_points(
                        collection_name=col,
                        prefetch=[
                            models.Prefetch(
                                query=models.SparseVector(indices=si, values=sv),
                                using="sparse",
                                limit=limit * 2,
                            ),
                            models.Prefetch(
                                query=query_vector,
                                using="dense",
                                limit=limit * 2,
                            ),
                        ],
                        query=models.FusionQuery(fusion=models.Fusion.RRF),
                        query_filter=qf,
                        limit=limit,
                        with_payload=_SEARCH_PAYLOAD_FIELDS,
                    )
                return await c.query_points(
                    collection_name=col,
                    query=query_vector,
                    query_filter=qf,
                    limit=limit,
                    with_payload=_SEARCH_PAYLOAD_FIELDS,
                )

            try:
                result = await self._run_with_fallback(
                    "hybrid_search",
                    lambda c: _hybrid_qp(c, keyword_filter),
                )
                hits = result.points
            except _QDRANT_STORE_ERRORS as e:
                _err_str = str(e)
                if keyword_filter is not None and (
                    "Index required" in _err_str
                    or "400" in _err_str
                    or "Bad request" in _err_str.lower()
                ):
                    logger.warning(
                        "Hybrid search filter failed in '%s' (missing index?) — "
                        "retrying without filter: %s",
                        col,
                        type(e).__name__,
                    )
                    try:
                        result = await self._run_with_fallback(
                            "hybrid_search_nofilter",
                            lambda c: _hybrid_qp(c, None),
                        )
                        hits = result.points
                    except _QDRANT_STORE_ERRORS as e2:
                        logger.warning(
                            "Hybrid search fallback also failed in '%s': %s",
                            col,
                            e2,
                        )
                        return []
                else:
                    logger.warning(
                        "Hybrid search failed in '%s' (%s: %s) — returning no results",
                        col,
                        type(e).__name__,
                        e,
                    )
                    return []

        return [_hit_to_dict(hit) for hit in hits]

    # ── count by category ────────────────────────────────────────────────

    async def count_by_source_category(
        self,
        source_category: str,
        collection: str | None = None,
    ) -> int:
        """Count points whose source_category (or doc_type) payload matches."""
        col = collection or self._default_collection
        scroll_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="source_category",
                    match=models.MatchValue(value=source_category),
                ),
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchValue(value=source_category),
                ),
            ]
        )

        async def _count_on(c: AsyncQdrantClient) -> int:
            total = 0
            offset: models.PointId | None = None
            while True:
                points, offset = await c.scroll(
                    collection_name=col,
                    scroll_filter=scroll_filter,
                    limit=500,
                    with_payload=False,
                    with_vectors=False,
                    offset=offset,
                )
                total += len(points)
                if offset is None or not points:
                    break
            return total

        try:
            return await self._run_with_fallback("count_by_source_category", _count_on)
        except _QDRANT_STORE_ERRORS as exc:
            if "Index required" in str(exc) or "400" in str(exc):
                logger.warning(
                    "count_by_source_category: payload index missing for '%s' — returning 0",
                    source_category,
                )
                return 0
            logger.exception("count_by_source_category failed for %s", source_category)
            raise

    async def scroll_by_source_category(
        self,
        source_category: str,
        collection: str | None = None,
        *,
        limit: int = 10_000,
        batch_size: int = 500,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Scroll all points whose source_category or doc_type matches; return (point_id, payload) list."""
        col = collection or self._default_collection
        scroll_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="source_category",
                    match=models.MatchValue(value=source_category),
                ),
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchValue(value=source_category),
                ),
            ]
        )
        out: list[tuple[str, dict[str, Any]]] = []

        async def _scroll_cat(c: AsyncQdrantClient) -> list[tuple[str, dict[str, Any]]]:
            rows: list[tuple[str, dict[str, Any]]] = []
            offset: models.PointId | None = None
            while len(rows) < limit:
                points, offset = await c.scroll(
                    collection_name=col,
                    scroll_filter=scroll_filter,
                    limit=min(batch_size, limit - len(rows)),
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
                for pt in points:
                    rows.append((str(pt.id), (pt.payload or {})))
                if offset is None or not points:
                    break
            return rows

        try:
            return await self._run_with_fallback(
                "scroll_by_source_category",
                _scroll_cat,
            )
        except _QDRANT_STORE_ERRORS:
            logger.exception("scroll_by_source_category failed for %s", source_category)
            raise

    async def sync_collection_to_cloud(
        self,
        collection: str | None = None,
        *,
        batch_size: int = 100,
        max_points: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Copy all points from local Qdrant to cloud (requires QDRANT_CLOUD_URL set).

        Scrolls the local collection with vectors and payload, upserts each batch
        to the cloud client. Use for one-time migration. Returns total points synced.
        """
        if self._client_cloud is None:
            raise BrainOSError(
                "Qdrant cloud not configured. Set QDRANT_CLOUD_URL and QDRANT_CLOUD_API_KEY in .env"
            )
        col = collection or self._default_collection
        await self.ensure_collection(name=col)
        total = 0
        offset: models.PointId | None = None
        try:
            while True:
                points, offset = await self._client.scroll(
                    collection_name=col,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=True,
                    offset=offset,
                )
                if not points:
                    break
                # Rebuild PointStruct for cloud upsert (id, vector, payload)
                batch = []
                for pt in points:
                    vec = pt.vector
                    if isinstance(vec, dict):
                        vec = vec.get("") or (list(vec.values())[0] if vec else None)
                    if vec is None:
                        continue
                    batch.append(
                        models.PointStruct(
                            id=pt.id,
                            vector=vec,
                            payload=pt.payload or {},
                        )
                    )
                # Upsert to cloud in smaller chunks to stay under request size limits
                for sub_start in range(0, len(batch), _CLOUD_UPSERT_BATCH_SIZE):
                    sub = batch[sub_start : sub_start + _CLOUD_UPSERT_BATCH_SIZE]
                    await self._client_cloud.upsert(collection_name=col, points=sub)
                total += len(batch)
                logger.info("Synced %d points to cloud (total so far: %d)", len(batch), total)
                if progress_callback is not None:
                    progress_callback(len(batch), total)
                if max_points is not None and total >= max_points:
                    break
                if offset is None:
                    break
        except _QDRANT_STORE_ERRORS:
            logger.exception("sync_collection_to_cloud failed")
            raise
        return total

    async def create_staging_collection(
        self,
        *,
        staging_collection: str,
        vector_size: int = 1024,
    ) -> None:
        """Create/ensure a staging collection for safe large ingest waves."""
        await self.ensure_collection(name=staging_collection, vector_size=vector_size)

    async def swap_collection_alias(
        self,
        *,
        alias_name: str,
        new_collection: str,
        backup_alias_name: str = "",
    ) -> None:
        """Atomically point alias_name to new_collection on active clients."""

        async def _swap_on(client: AsyncQdrantClient) -> None:
            aliases = await client.get_aliases()
            current_targets = {
                a.collection_name
                for a in getattr(aliases, "aliases", [])
                if a.alias_name == alias_name
            }
            backup_targets = {
                a.collection_name
                for a in getattr(aliases, "aliases", [])
                if backup_alias_name and a.alias_name == backup_alias_name
            }
            ops: list[Any] = []
            for old_backup in backup_targets:
                ops.append(
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=backup_alias_name)
                    )
                )
                logger.info(
                    "Removing previous backup alias '%s' from '%s'",
                    backup_alias_name,
                    old_backup,
                )
                break
            for old_collection in current_targets:
                ops.append(
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=alias_name)
                    )
                )
                if backup_alias_name:
                    ops.append(
                        models.CreateAliasOperation(
                            create_alias=models.CreateAlias(
                                collection_name=old_collection,
                                alias_name=backup_alias_name,
                            )
                        )
                    )
                logger.info(
                    "Removing alias '%s' from collection '%s'",
                    alias_name,
                    old_collection,
                )
                break
            ops.append(
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=new_collection,
                        alias_name=alias_name,
                    )
                )
            )
            await client.update_aliases(change_aliases_operations=ops)

        await self._run_with_fallback("swap_collection_alias", _swap_on)
        if self._client_cloud is not None:
            try:
                await _swap_on(self._client_cloud)
            except _QDRANT_HTTP_IO_ERRORS:
                logger.warning("Cloud alias swap failed for '%s'", alias_name, exc_info=True)

    async def scroll_collection_payloads(
        self,
        collection: str | None = None,
        *,
        batch_size: int = 200,
        max_points: int | None = None,
        source_category: str | None = None,
        start_after_point_id: str | None = None,
    ):
        """Async generator: scroll collection and yield batches of payload dicts.

        Does not load vectors. Optional filter by source_category (or doc_type).
        Each item includes "point_id", "content", "source", "source_category", "payload".
        If start_after_point_id is set, scrolling starts after that point (for resume).
        """
        col = collection or self._default_collection
        scroll_filter: models.Filter | None = None
        if source_category:
            scroll_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="source_category",
                        match=models.MatchValue(value=source_category),
                    ),
                    models.FieldCondition(
                        key="doc_type",
                        match=models.MatchValue(value=source_category),
                    ),
                ]
            )

        def _is_retryable_scroll_error(exc: Exception) -> bool:
            if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
                return True
            cause = getattr(exc, "__cause__", None)
            if cause is not None and isinstance(
                cause, (httpx.ConnectError, httpx.TimeoutException)
            ):
                return True
            if type(exc).__name__ == "ResponseHandlingException" and cause is not None:
                return isinstance(cause, (httpx.ConnectError, httpx.TimeoutException))
            return False

        scroll_retry = RetryPolicy(max_attempts=5, base_delay_seconds=2.0)
        total_yielded = 0
        offset: models.PointId | None = start_after_point_id if start_after_point_id else None
        try:
            while True:

                async def _do_scroll() -> tuple[Any, Any]:
                    off = offset

                    async def _sc(c: AsyncQdrantClient) -> tuple[Any, Any]:
                        return await c.scroll(
                            collection_name=col,
                            limit=batch_size,
                            with_payload=True,
                            with_vectors=False,
                            offset=off,
                            scroll_filter=scroll_filter,
                        )

                    return await self._run_with_fallback("scroll_collection_payloads", _sc)

                points, offset = await run_with_retry(
                    _do_scroll,
                    policy=scroll_retry,
                    is_retryable=_is_retryable_scroll_error,
                )
                batch: list[dict[str, Any]] = []
                for pt in points:
                    payload = pt.payload or {}
                    content = payload.get("content") or payload.get("text") or ""
                    if not content or not isinstance(content, str):
                        continue
                    batch.append(
                        {
                            "point_id": str(pt.id),
                            "content": content,
                            "source": payload.get("source", ""),
                            "source_category": payload.get("source_category")
                            or payload.get("doc_type", ""),
                            "payload": payload,
                        }
                    )
                if batch:
                    yield batch
                    total_yielded += len(batch)
                    if max_points is not None and total_yielded >= max_points:
                        break
                if offset is None or not points:
                    break
        except _QDRANT_STORE_ERRORS:
            logger.exception("scroll_collection_payloads failed")
            raise

    # ── deletion ─────────────────────────────────────────────────────────

    async def delete_by_source(
        self,
        source: str,
        collection: str | None = None,
    ) -> None:
        """Delete all points whose ``source`` payload matches *source*."""
        col = collection or self._default_collection
        try:

            async def _delete_src(c: AsyncQdrantClient) -> None:
                await c.delete(
                    collection_name=col,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="source",
                                    match=models.MatchValue(value=source),
                                )
                            ]
                        )
                    ),
                )

            await self._run_with_fallback("delete_by_source", _delete_src)
            logger.info("Deleted points with source='%s' from '%s'", source, col)
        except _QDRANT_STORE_ERRORS:
            logger.exception("Failed to delete points for source '%s'", source)
            raise

    async def delete_by_source_category(
        self,
        source_category: str,
        collection: str | None = None,
    ) -> None:
        """Delete all points whose source_category (or doc_type) payload matches *source_category*."""
        col = collection or self._default_collection
        try:

            async def _delete_cat(c: AsyncQdrantClient) -> None:
                await c.delete(
                    collection_name=col,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            should=[
                                models.FieldCondition(
                                    key="source_category",
                                    match=models.MatchValue(value=source_category),
                                ),
                                models.FieldCondition(
                                    key="doc_type",
                                    match=models.MatchValue(value=source_category),
                                ),
                            ]
                        )
                    ),
                )

            await self._run_with_fallback("delete_by_source_category", _delete_cat)
            logger.info(
                "Deleted points with source_category/doc_type='%s' from '%s'",
                source_category,
                col,
            )
        except _QDRANT_STORE_ERRORS:
            logger.exception(
                "Failed to delete points for source_category '%s'",
                source_category,
            )
            raise

    # ── payload updates ─────────────────────────────────────────────────

    async def set_payload(
        self,
        point_id: str,
        payload: dict[str, Any],
        collection: str | None = None,
    ) -> None:
        """Update payload fields on an existing point without re-embedding."""
        col = collection or self._default_collection
        try:

            async def _set_pay(c: AsyncQdrantClient) -> None:
                await c.set_payload(
                    collection_name=col,
                    payload=payload,
                    points=[point_id],
                )

            await self._run_with_fallback("set_payload", _set_pay)
        except _QDRANT_STORE_ERRORS:
            logger.warning("set_payload failed for point %s", point_id, exc_info=True)
            raise

    # ── cleanup ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._close_inner_clients()

    async def __aenter__(self) -> QdrantManager:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


# ── helpers ──────────────────────────────────────────────────────────────────


def _uuid_to_hex(uid: UUID) -> str:
    """Qdrant accepts string or int point IDs; we use the UUID hex."""
    return uid.hex


def _record_to_dict(record: Any) -> dict[str, Any]:
    """Build canonical result dict from a retrieved point (id + payload, no score)."""
    payload = getattr(record, "payload", None) or {}
    content = payload.get("content") or payload.get("text") or payload.get("raw_text", "")
    source = payload.get("source") or payload.get("filename", "")
    source_category = payload.get("source_category") or payload.get("doc_type", "")
    metadata = payload.get("metadata", {})
    if not metadata:
        extra_keys = {
            "machines",
            "prices",
            "customer",
            "chunk",
            "total_chunks",
            "source_group",
            "doc_type",
            "filename",
            "ingested_at",
            "subject",
            "from_email",
            "to_email",
            "direction",
            "thread_key",
            "company_domain",
            "has_quote",
            "has_price",
        }
        metadata = {k: v for k, v in payload.items() if k in extra_keys and v}
    return {
        "id": str(getattr(record, "id", "")),
        "content": content,
        "score": 1.0,
        "source": source,
        "metadata": metadata,
        "source_category": source_category,
    }


def _hit_to_dict(hit: models.ScoredPoint) -> dict[str, Any]:
    """Normalise a Qdrant hit into maintainer-export's canonical result schema.

    Handles both the old ira payload layout (``text``, ``doc_type``,
    ``filename``, ``machines``, ``prices``, …) and the new maintainer-export layout
    (``content``, ``source_category``, ``metadata``).
    """
    payload = hit.payload or {}

    content = payload.get("content") or payload.get("text") or payload.get("raw_text", "")
    source = payload.get("source") or payload.get("filename", "")
    source_category = payload.get("source_category") or payload.get("doc_type", "")

    metadata = payload.get("metadata", {})
    if not metadata:
        extra_keys = {
            "machines",
            "prices",
            "customer",
            "chunk",
            "total_chunks",
            "source_group",
            "doc_type",
            "filename",
            "ingested_at",
            "subject",
            "from_email",
            "to_email",
            "direction",
            "thread_key",
            "company_domain",
            "has_quote",
            "has_price",
        }
        metadata = {k: v for k, v in payload.items() if k in extra_keys and v}

    return {
        "id": str(hit.id),
        "content": content,
        "score": hit.score,
        "source": source,
        "metadata": metadata,
        "source_category": source_category,
    }
