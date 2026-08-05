"""System status MCP tool.

``get_system_status`` is Brain OS's introspection surface — 41 grouped snapshots
covering services, agents, retrieval config, data directories, store
clients, observability, repo layout, runtime capabilities, and feature
flags. Reads no secrets; returns presence flags and public values only.

All snapshot helpers are lazy-imported inside the function body to avoid
paying their cost on MCP server cold-start when ``get_system_status``
isn't called.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool
from brain_os.interfaces.public_snapshot_utils import capture_optional_snapshots

logger = logging.getLogger(__name__)


async def get_system_status() -> str:
    """Check Brain OS's system health and agent power levels.

    Returns:
    1. ``service_health`` — Qdrant, Neo4j, PostgreSQL, OpenAI, Voyage, etc.
    2. ``agent_leaderboard`` — top agents by performance / tier.
    3. ``rerank_and_retrieval`` — FlashRank ONNX cache probe and Voyage rerank config (no secrets).
    4. ``data_directory`` — resolved ``BRAIN_DATA_DIR`` path (or ``./data``).
    5. ``pipeline_runtime`` — active timeout/limit settings from ``AppConfig`` (no secrets).
    6. ``metis_stability`` — rolling quality window from Metis persistence.
    7. ``agent_power_summary`` — top agent from ``power_levels`` JSON (disk probe).
    8. ``email_operations`` — TRAINING vs OPERATIONAL, poll flag, oauth files present (no secrets).
    9. ``knowledge_store`` — active Qdrant collection, hybrid flag, optional cloud dual-write.
    10. ``graph_store`` — Neo4j Bolt netloc and configured user (no password).
    11. ``crm_database`` — Postgres host:port and database name from DSN (no credentials).
    12. ``mem0_client`` — whether a Mem0 API key is configured (key never returned).
    13. ``redis_cache`` — whether Redis URL is set and public host:port (no password).
    14. ``llm_stack`` — OpenAI / Anthropic key presence and configured model names (no secrets).
    15. ``tracing_config`` — Langfuse, Sentry, Helicone flags (incl. property env/app set), Langfuse host (no secrets).
    16. ``vendor_apis`` — optional search, enrichment, NewsData, OCR-related API key presence, Document AI project flag, and ``iris_web_search_configured`` (any of Tavily/Serper/SearchAPI).
    17. ``http_api`` — whether ``API_SECRET_KEY`` is set and a short CORS origins preview (no secret values).
    18. ``ingestion_brain`` — presence/size of ``brain/imports_metadata.json``, ``ingestion_log.json``, Graphe DB under ``BRAIN_DATA_DIR``, plus ingest flags (PII redaction, Mnemon semantic check).
    19. ``runtime_build`` — installed ``brain`` package version, Python version, and OS platform.
    20. ``secondary_mailbox`` — optional second Gmail OAuth paths configured and files present.
    21. ``governance_brain`` — correction ledger/DB and related files under ``brain/``, plus ``legacy_quarantine_strict``.
    22. ``storage_clients`` — Qdrant timeout, API-key presence (primary + cloud), Neo4j driver pool size.
    23. ``workspace_content`` — repo ``data/imports`` directory exists (imports metadata index source).
    24. ``email_training`` — whether a TRAINING mailbox address is set in config (not the address).
    25. ``pipeline_observability`` — log format, HF cache override, Sentry trace sample rate, Google OAuth client id presence, Document AI processor id flags, Unstructured API host.
    26. ``repo_brain_aux`` — repo ``data/brain`` auxiliary files: embedding cache DB, retrieval log, Asclepius DB, agent scores JSON.
    27. ``process_context`` — whether ``BRAIN_DATA_DIR`` env is set, resolved cwd label, Python executable basename.
    28. ``sales_coaching_brain`` — repo ``data/brain/sales_training.json`` (Chiron) presence and size.
    29. ``secondary_google_oauth`` — secondary Gmail OAuth *client id* configured (not secrets).
    30. ``neo4j_connection_flags`` — password field / NEO4J_AUTH string presence, localhost Bolt URI (no secrets).
    31. ``repo_truth_hints`` — repo ``data/brain/truth_hints.json`` and ``learned_truth_hints.json`` presence/size.
    32. ``runtime_capabilities`` — import probes for ``filelock``, ``apscheduler``, ``httpx``, ``fastapi``, ``typer``, ``qdrant_client``, ``neo4j``, ``sqlalchemy`` in this interpreter.
    33. ``repo_framework`` — presence of repo-root framework files: ``web-ui/package.json``, ``docker-compose.local.yml``, ``.env.example``, ``alembic.ini``, ``pyproject.toml`` (truncated root label, no secrets).
    34. ``repo_sources`` — ``src/brain_os`` package, ``prompts/`` (``.txt`` count), ``tests/``, ``alembic/``, ``scripts/`` directory flags.
    35. ``app_runtime_imports`` — import probes for LLM/server stack: ``openai``, ``anthropic``, ``langfuse``, ``redis``, ``uvicorn``, ``voyageai``, ``mem0``, ``mcp``, ``tiktoken``, ``jinja2``, ``pydantic_settings``.
    36. ``optional_tooling_imports`` — heavier optional imports: ``sentry_sdk``, ``alembic``, ``asyncpg``, ``aiosqlite``, ``google.auth``, ``googleapiclient``, ``sse_starlette``, ``crawl4ai``, ``docling``, ``neo4j_graphrag``, ``gliner``, ``instructor``, ``chonkie``, ``pypdf``.
    37. ``repo_vcs`` — git present at repo root, current branch label (abbrev), working tree clean vs dirty (no hashes or paths).
    38. ``app_feature_flags`` — ingest/governance toggles: Mnemon semantic check, legacy quarantine strict, PII redaction at ingest, FlashRank cache path override (booleans only).
    39. ``pantheon_registry`` — static specialist agent registration count and truncated comma-separated agent names.
    40. ``quality_stack_imports`` — import probes for ``flashrank``, ``deepeval``, ``ragas``, ``guardrails`` (optional quality/eval stack).
    41. ``skills_registry`` — count of entries in :data:`SKILL_MATRIX` and truncated skill name preview.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()

    result: dict[str, Any] = {}

    try:
        from brain_os.brain.power_levels import PowerLevelTracker

        tracker = PowerLevelTracker()
        await tracker._load()
        leaderboard = tracker.get_leaderboard()
        result["agent_leaderboard"] = leaderboard[:10]
    except Exception as exc:
        logger.warning("Power level check failed: %s", exc)
        result["agent_leaderboard"] = f"Error: {exc}"

    try:
        from brain_os.brain.embeddings import EmbeddingService
        from brain_os.brain.knowledge_graph import KnowledgeGraph
        from brain_os.brain.qdrant_manager import QdrantManager
        from brain_os.systems.immune import ImmuneSystem

        embedding = EmbeddingService()
        qdrant = QdrantManager(embedding_service=embedding)
        graph = KnowledgeGraph()
        immune = ImmuneSystem(
            qdrant=qdrant,
            knowledge_graph=graph,
            embedding_service=embedding,
        )
        health = await immune.run_startup_validation()
        result["service_health"] = {
            name: {"status": info["status"], "latency_ms": info.get("latency_ms")}
            for name, info in health.items()
        }
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        result["service_health"] = f"Error: {exc}"

    try:
        from brain_os.brain.retriever import flashrank_cache_probe
        from brain_os.config import get_settings

        _cfg = get_settings()
        from brain_os.systems.data_dir_lock import get_data_dir

        result["rerank_and_retrieval"] = {
            "flashrank_cache": flashrank_cache_probe(),
            "voyage_rerank_model": _cfg.embedding.rerank_model,
            "voyage_embedding_model": _cfg.embedding.model,
            "voyage_api_configured": bool(_cfg.embedding.api_key.get_secret_value().strip()),
        }
        result["data_directory"] = {"path": str(get_data_dir())}
    except Exception as exc:
        logger.warning("Rerank probe failed: %s", exc)
        result["rerank_and_retrieval"] = f"Error: {exc}"

    from brain_os.agents.metis import metis_stability_public_snapshot
    from brain_os.brain.power_levels import power_levels_public_probe
    from brain_os.config import (
        get_app_feature_flags_public_snapshot,
        get_app_runtime_imports_public_snapshot,
        get_crm_database_public_snapshot,
        get_email_ops_public_snapshot,
        get_email_training_public_snapshot,
        get_event_driven_ingestion_public_snapshot,
        get_governance_brain_public_snapshot,
        get_graph_store_public_snapshot,
        get_http_api_public_snapshot,
        get_ingestion_brain_public_snapshot,
        get_knowledge_store_public_snapshot,
        get_llm_stack_public_snapshot,
        get_mem0_public_snapshot,
        get_neo4j_connection_flags_public_snapshot,
        get_optional_tooling_imports_public_snapshot,
        get_pipeline_observability_public_snapshot,
        get_pipeline_runtime_public_snapshot,
        get_process_context_public_snapshot,
        get_quality_stack_imports_public_snapshot,
        get_redis_cache_public_snapshot,
        get_repo_brain_aux_public_snapshot,
        get_repo_framework_public_snapshot,
        get_repo_sources_public_snapshot,
        get_repo_truth_hints_public_snapshot,
        get_repo_vcs_public_snapshot,
        get_runtime_build_public_snapshot,
        get_runtime_capabilities_public_snapshot,
        get_sales_coaching_brain_public_snapshot,
        get_secondary_google_oauth_public_snapshot,
        get_secondary_mail_public_snapshot,
        get_skills_registry_public_snapshot,
        get_storage_clients_public_snapshot,
        get_tracing_public_snapshot,
        get_vendor_apis_public_snapshot,
        get_workspace_content_public_snapshot,
    )
    from brain_os.services.harness_review import harness_health_public_snapshot

    def _pantheon_registry_snapshot() -> Any:
        from brain_os.pantheon import get_pantheon_registry_public_snapshot

        return get_pantheon_registry_public_snapshot()

    capture_optional_snapshots(
        result,
        snapshots=[
            ("pipeline_runtime", "Pipeline runtime", get_pipeline_runtime_public_snapshot),
            ("metis_stability", "Metis (response quality)", metis_stability_public_snapshot),
            ("harness_governance", "Harness (governance)", harness_health_public_snapshot),
            ("agent_power_summary", "Agent power probe", power_levels_public_probe),
            ("email_operations", "Email ops", get_email_ops_public_snapshot),
            ("knowledge_store", "Knowledge store", get_knowledge_store_public_snapshot),
            ("graph_store", "Graph store", get_graph_store_public_snapshot),
            ("crm_database", "CRM database", get_crm_database_public_snapshot),
            ("mem0_client", "Mem0 client", get_mem0_public_snapshot),
            ("redis_cache", "Redis cache", get_redis_cache_public_snapshot),
            ("llm_stack", "LLM stack", get_llm_stack_public_snapshot),
            ("tracing_config", "Tracing config", get_tracing_public_snapshot),
            ("vendor_apis", "Vendor APIs", get_vendor_apis_public_snapshot),
            ("http_api", "HTTP API", get_http_api_public_snapshot),
            ("ingestion_brain", "Ingestion brain", get_ingestion_brain_public_snapshot),
            ("runtime_build", "Runtime build", get_runtime_build_public_snapshot),
            ("secondary_mailbox", "Secondary mailbox", get_secondary_mail_public_snapshot),
            ("governance_brain", "Governance brain", get_governance_brain_public_snapshot),
            ("storage_clients", "Storage clients", get_storage_clients_public_snapshot),
            ("workspace_content", "Workspace content", get_workspace_content_public_snapshot),
            ("email_training", "Email training", get_email_training_public_snapshot),
            (
                "pipeline_observability",
                "Pipeline observability",
                get_pipeline_observability_public_snapshot,
            ),
            ("repo_brain_aux", "Repo brain aux", get_repo_brain_aux_public_snapshot),
            ("process_context", "Process context", get_process_context_public_snapshot),
            (
                "sales_coaching_brain",
                "Sales coaching brain",
                get_sales_coaching_brain_public_snapshot,
            ),
            (
                "secondary_google_oauth",
                "Secondary Google OAuth",
                get_secondary_google_oauth_public_snapshot,
            ),
            (
                "neo4j_connection_flags",
                "Neo4j connection flags",
                get_neo4j_connection_flags_public_snapshot,
            ),
            ("repo_truth_hints", "Repo truth hints", get_repo_truth_hints_public_snapshot),
            (
                "runtime_capabilities",
                "Runtime capabilities",
                get_runtime_capabilities_public_snapshot,
            ),
            ("repo_framework", "Repo framework", get_repo_framework_public_snapshot),
            ("repo_sources", "Repo sources", get_repo_sources_public_snapshot),
            ("app_runtime_imports", "App runtime imports", get_app_runtime_imports_public_snapshot),
            (
                "optional_tooling_imports",
                "Optional tooling imports",
                get_optional_tooling_imports_public_snapshot,
            ),
            ("repo_vcs", "Repo VCS", get_repo_vcs_public_snapshot),
            ("app_feature_flags", "App feature flags", get_app_feature_flags_public_snapshot),
            ("pantheon_registry", "Pantheon registry", _pantheon_registry_snapshot),
            (
                "quality_stack_imports",
                "Quality stack imports",
                get_quality_stack_imports_public_snapshot,
            ),
            ("skills_registry", "Skills registry", get_skills_registry_public_snapshot),
            (
                "event_driven_ingestion",
                "Event-driven ingestion",
                lambda: get_event_driven_ingestion_public_snapshot(
                    services=dict(getattr(srv, "_shared_services", None) or {}),
                ),
            ),
        ],
        logger=logger,
        error_as_dict=False,
    )

    return json.dumps(result, indent=2, default=str)


def register(mcp: FastMCP) -> None:
    """Register the system status tool on the given FastMCP instance."""
    mcp.tool()(hardened_mcp_tool(get_system_status))
