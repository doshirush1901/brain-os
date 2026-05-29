"""Ingest MCP tools with Community document cap."""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool
from brain_os.licensing.caps import LicenseCapError, assert_ingest_allowed, record_document_ingested

logger = logging.getLogger(__name__)


async def ingest_document(file_path: str) -> str:
    """Ingest a document into the Brain OS knowledge base (Community: 500 doc cap)."""
    from pathlib import Path

    from brain_os.interfaces import mcp_server as srv

    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"

    try:
        assert_ingest_allowed()
    except LicenseCapError as exc:
        return str(exc)

    await srv._ensure_initialized()

    try:
        ingestor = srv._ingestor
        if ingestor is None:
            from brain_os.brain.document_ingestor import DocumentIngestor
            from brain_os.brain.embeddings import EmbeddingService
            from brain_os.brain.knowledge_graph import KnowledgeGraph
            from brain_os.brain.qdrant_manager import QdrantManager

            embedding = EmbeddingService()
            qdrant = QdrantManager(embedding_service=embedding)
            await qdrant.ensure_collection()
            graph = KnowledgeGraph()
            ingestor = DocumentIngestor(qdrant=qdrant, knowledge_graph=graph)

        file_info = {
            "path": str(path),
            "name": path.name,
            "extension": path.suffix.lower(),
            "size": path.stat().st_size,
            "category": "mcp_upload",
        }
        chunks = await ingestor.ingest_file(file_info)
        record_document_ingested()
        return f"Ingested {path.name}: {chunks} chunks stored."
    except LicenseCapError as exc:
        return str(exc)
    except Exception as exc:
        logger.exception("MCP ingest_document failed")
        return f"Error: {exc}"


async def parse_document_ai(file_path: str, kind: str = "ocr") -> str:
    """Parse a PDF or image with Google Document AI (does not count toward doc cap)."""
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()

    from pathlib import Path

    from brain_os.systems.document_ai import DocumentAIError, parse_bytes_by_kind

    try:
        path = Path(file_path).expanduser()
        if not path.is_file():
            return json.dumps({"error": f"file not found or not a file: {file_path}"}, indent=2)

        size = path.stat().st_size
        if size > srv._MCP_DOCAI_MAX_BYTES:
            return json.dumps(
                {
                    "error": (
                        f"file too large ({size} bytes); max {srv._MCP_DOCAI_MAX_BYTES} "
                        f"({srv._MCP_DOCAI_MAX_BYTES // (1024 * 1024)} MiB)"
                    )
                },
                indent=2,
            )

        data = path.read_bytes()
        result = await parse_bytes_by_kind(data, filename=path.name, kind=kind)
        return json.dumps(result, indent=2, default=str)
    except DocumentAIError as exc:
        return json.dumps({"error": str(exc)}, indent=2)
    except Exception as exc:
        logger.exception("MCP parse_document_ai failed")
        return json.dumps({"error": str(exc)}, indent=2)


def register(mcp: FastMCP) -> None:
    mcp.tool()(hardened_mcp_tool(ingest_document))
    mcp.tool()(hardened_mcp_tool(parse_document_ai))
