"""Public health/ops snapshots and health-status helpers (no secrets).

Split from ``config.py`` (Phase 6) to shrink the settings module. Import from
``brain_os.config`` for backward compatibility.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from brain_os.config import get_settings


def infer_http_health_status(services: dict[str, Any]) -> str:
    """Derive top-level ``/api/health`` status from per-service probes (no secrets).

    * ``healthy`` / ``skipped`` / ``disabled`` — OK for that probe.
    * Optional Qdrant cloud mirror: when ``enabled`` is false, ``unavailable`` / ``unknown`` / ``disabled`` do not degrade.
    * Any other ``status`` on a structured probe → ``degraded``.
    """
    for _name, probe in services.items():
        if not isinstance(probe, dict):
            continue
        st = probe.get("status")
        if st is None:
            continue
        if st in ("healthy", "skipped", "disabled"):
            continue
        if _name == "qdrant_cloud_sync":
            if probe.get("enabled") is False and st in ("unavailable", "unknown", "disabled"):
                continue
            if st == "ok":
                continue
        if st == "ok":
            continue
        return "degraded"
    return "ok"


@lru_cache(maxsize=64)
def _public_postgres_host_db(url: str) -> tuple[str, str]:
    """Host:port and database name from a SQLAlchemy-style URL; never returns credentials."""
    raw = url.strip()
    if not raw or "://" not in raw:
        return "", ""
    _ignored_scheme, rest = raw.split("://", 1)
    if "@" in rest:
        _, host_and_path = rest.rsplit("@", 1)
    else:
        host_and_path = rest
    hostport, _, path = host_and_path.partition("/")
    hostport = hostport.split("?")[0].strip()
    db = path.split("?")[0].strip() if path else ""
    return hostport, db


@lru_cache(maxsize=64)
def _public_redis_netloc(url: str) -> str:
    """Redis host:port for display (drops password, db index)."""
    from urllib.parse import urlparse

    u = url.strip()
    if not u:
        return ""
    parsed = urlparse(u)
    host = parsed.hostname or ""
    if not host:
        return ""
    port = parsed.port
    if port:
        return f"{host}:{port}"
    return host


def get_pipeline_runtime_public_snapshot() -> dict[str, str | int | float | bool]:
    """Non-secret pipeline and guardrail knobs for health endpoints and ops (matches active `.env`)."""
    a = get_settings().app
    snap: dict[str, str | int | float | bool] = {
        "environment": a.environment,
        "log_level": a.log_level,
        "pipeline_timeout_s": a.pipeline_timeout,
        "agent_timeout_s": a.agent_timeout,
        "max_parallel_agents": a.max_parallel_agents,
        "athena_synthesis_timeout_s": a.athena_synthesis_timeout,
        "react_max_iterations": a.react_max_iterations,
        "mem0_timeout_s": a.mem0_timeout,
        "faithfulness_threshold": a.faithfulness_threshold,
        "faithfulness_hard_threshold": a.faithfulness_hard_threshold,
        "faithfulness_mode": a.faithfulness_mode,
        "confidence_floor": a.confidence_floor,
        "max_delegation_depth": a.max_delegation_depth,
        "use_sparse_hybrid": a.use_sparse_hybrid,
        "guardrails_fail_closed": a.guardrails_fail_closed,
        "drip_llm_insights": a.drip_llm_insights,
        "drip_llm_adjust": a.drip_llm_adjust,
        "progressive_tool_discovery": a.progressive_tool_discovery,
        "progressive_tool_discovery_top_k": a.progressive_tool_discovery_top_k,
        "progressive_tool_discovery_expand_on_low_confidence": a.progressive_tool_discovery_expand_on_low_confidence,
        "tool_selection_telemetry": a.tool_selection_telemetry,
        "mcp_fastlane_enabled": a.mcp_fastlane_enabled,
        "mcp_fastlane_min_score": a.mcp_fastlane_min_score,
        "mcp_fastlane_max_snippets": a.mcp_fastlane_max_snippets,
        "mcp_email_default_scope": a.mcp_email_default_scope,
        "mcp_route_prefer_cloud_fast": a.mcp_route_prefer_cloud_fast,
        "mcp_ollama_retry_budget": a.mcp_ollama_retry_budget,
        "mcp_fast_fail_to_openai": a.mcp_fast_fail_to_openai,
        "request_prompt_snapshot": a.request_prompt_snapshot,
        "pipeline_query_max_chars": a.pipeline_query_max_chars,
        "llm_monthly_token_budget": a.llm_monthly_token_budget,
        "llm_budget_scope": a.llm_budget_scope,
        "router_deterministic_margin_min": a.router_deterministic_margin_min,
        "router_embedding_tiebreak_enabled": a.router_embedding_tiebreak_enabled,
        "router_optional_keyword_ranking": a.router_optional_keyword_ranking,
        "pipeline_attach_observability_trace": a.pipeline_attach_observability_trace,
        "run_record_enabled": a.run_record_enabled,
        "run_record_redact_pii": a.run_record_redact_pii,
        "response_include_execution_summary": a.response_include_execution_summary,
        "llm_route_prefer_ollama": a.llm_route_prefer_ollama,
        "llm_route_high_stakes_cloud_escalation": a.llm_route_high_stakes_cloud_escalation,
        "uncensored_local_llm_mode": a.uncensored_local_llm_mode,
        "retriever_over_retrieve_factor": a.retriever_over_retrieve_factor,
        "retriever_second_pass_enabled": a.retriever_second_pass_enabled,
        "retriever_second_pass_keyword_threshold": a.retriever_second_pass_keyword_threshold,
        "retriever_trace_enabled": a.retriever_trace_enabled,
        "retriever_dedup_enabled": a.retriever_dedup_enabled,
        "retriever_default_profile": a.retriever_default_profile,
        "retriever_evidence_max_chars": a.retriever_evidence_max_chars,
        "retriever_bundle_pii_mode": a.retriever_bundle_pii_mode,
        "citation_aligner_enabled": a.citation_aligner_enabled,
        "crm_pipeline_cache_ttl_seconds": a.crm_pipeline_cache_ttl_seconds,
        "retriever_diversity_overlap_threshold": a.retriever_diversity_overlap_threshold,
        "deep_consolidation_interval_hours": a.deep_consolidation_interval_hours,
        "startup_qdrant_ensure_timeout_s": a.startup_qdrant_ensure_timeout_s,
        "startup_sql_schema_timeout_s": a.startup_sql_schema_timeout_s,
        "claude_code_delegate_enabled": a.claude_code_delegate_enabled,
        "git_ship_enabled": a.git_ship_enabled,
        "git_ship_allow_push": a.git_ship_allow_push,
        "operator_webhook_configured": bool(a.operator_webhook_url.get_secret_value().strip()),
    }
    from brain_os.brain.routing_metrics import routing_metrics_snapshot
    from brain_os.deployment_profile import deployment_public_snapshot
    from brain_os.provider_map import provider_map_public_snapshot

    snap.update(deployment_public_snapshot())
    snap.update(provider_map_public_snapshot())
    rm = routing_metrics_snapshot()
    snap["route_deterministic_ratio"] = rm.get("route_deterministic_ratio", 0.0)
    return snap


def get_email_ops_public_snapshot() -> dict[str, str | bool]:
    """Gmail / email workflow flags for health endpoints (no secrets, no token contents)."""
    g = get_settings().google
    creds = g.credentials_path.expanduser()
    tok = g.token_path.expanduser()
    return {
        "email_mode": g.email_mode.value,
        "poll_enabled": g.email_poll_enabled,
        "primary_mailbox_configured": bool(g.ira_email.strip()),
        "primary_oauth_files_present": creds.is_file() and tok.is_file(),
    }


def get_knowledge_store_public_snapshot() -> dict[str, str | bool]:
    """Active Qdrant collection target from settings (hybrid vs dense)."""
    from urllib.parse import urlparse

    s = get_settings()
    coll = s.qdrant.collection_hybrid if s.app.use_sparse_hybrid else s.qdrant.collection
    raw_url = s.qdrant.url.strip()
    parsed = urlparse(raw_url)
    netloc = parsed.netloc or raw_url[:48]
    fb = s.qdrant.fallback_url.strip()
    fb_parsed = urlparse(fb) if fb else None
    fb_netloc = (fb_parsed.netloc or fb[:48]) if fb else ""
    return {
        "qdrant_collection": coll,
        "use_sparse_hybrid": s.app.use_sparse_hybrid,
        "qdrant_netloc": netloc,
        "qdrant_cloud_sync_enabled": bool(s.qdrant.cloud_url.strip()),
        "qdrant_fallback_enabled": bool(fb),
        "qdrant_fallback_netloc": fb_netloc,
    }


def get_graph_store_public_snapshot() -> dict[str, str]:
    """Neo4j Bolt target from settings (no password)."""
    from urllib.parse import urlparse

    s = get_settings()
    uri = s.neo4j.uri.strip()
    parsed = urlparse(uri)
    netloc = parsed.netloc
    if not netloc and uri:
        netloc = uri.split("://", 1)[-1].split("/", 1)[0]
    user = (s.neo4j.user or "").strip() or "neo4j"
    return {"neo4j_netloc": netloc, "neo4j_user": user}


def get_crm_database_public_snapshot() -> dict[str, str]:
    """Postgres host and DB name from ``DATABASE_URL`` (no credentials)."""
    hostport, db = _public_postgres_host_db(get_settings().database.url)
    return {"postgres_netloc": hostport, "postgres_database": db}


def get_mem0_public_snapshot(immune_mem0: dict[str, Any] | None = None) -> dict[str, Any]:
    """Mem0 config and optional connectivity from :meth:`ImmuneSystem.run_startup_validation`.

    ``immune_mem0`` is the ``report["mem0"]`` dict when present (status, latency_ms, error).
    API key value is never exposed.
    """
    key = get_settings().memory.api_key.get_secret_value().strip()
    out: dict[str, Any] = {"api_key_configured": bool(key)}
    if not immune_mem0:
        return out
    st = immune_mem0.get("status")
    if st == "healthy":
        out["api_reachable"] = True
        out["latency_ms"] = immune_mem0.get("latency_ms")
    elif st == "unhealthy":
        out["api_reachable"] = False
        out["error"] = str(immune_mem0.get("error") or "unknown")
    elif st == "skipped":
        out["skipped"] = True
    return out


def get_redis_cache_public_snapshot() -> dict[str, str | bool]:
    """Redis URL presence and host:port (no password)."""
    raw = get_settings().redis.url.strip()
    if not raw:
        return {"configured": False, "redis_netloc": ""}
    return {"configured": True, "redis_netloc": _public_redis_netloc(raw)}


def get_llm_stack_public_snapshot() -> dict[str, str | bool]:
    """Configured LLM providers and model names (no API keys)."""
    s = get_settings().llm
    return {
        "openai_api_configured": bool(s.openai_api_key.get_secret_value().strip()),
        "anthropic_api_configured": bool(s.anthropic_api_key.get_secret_value().strip()),
        "ollama_configured": bool((s.ollama_base_url or "").strip()),
        "default_llm_provider": s.default_llm_provider,
        "openai_model": s.openai_model,
        "anthropic_model": s.anthropic_model,
        "ollama_model": s.ollama_model,
        "ira_model_fast": s.ira_model_fast,
        "ira_model_reasoning": s.ira_model_reasoning,
        "ira_model_writing": s.ira_model_writing,
        "ira_model_verifier": s.ira_model_verifier,
        "ira_profile_fast_provider": s.ira_profile_fast_provider or "(baseline)",
        "ira_profile_reasoning_provider": s.ira_profile_reasoning_provider or "(baseline)",
        "ira_profile_writing_provider": s.ira_profile_writing_provider or "(baseline)",
        "ira_profile_verifier_provider": s.ira_profile_verifier_provider or "(baseline)",
        "ira_anthropic_model_fast_set": bool((s.ira_anthropic_model_fast or "").strip()),
        "ira_anthropic_model_reasoning_set": bool((s.ira_anthropic_model_reasoning or "").strip()),
        "ira_anthropic_model_writing_set": bool((s.ira_anthropic_model_writing or "").strip()),
        "ira_anthropic_model_verifier_set": bool((s.ira_anthropic_model_verifier or "").strip()),
    }


def get_tracing_public_snapshot() -> dict[str, str | bool]:
    """Observability / tracing clients from settings (no secrets)."""
    from urllib.parse import urlparse

    s = get_settings()
    lf = s.langfuse
    lf_ok = bool(str(lf.public_key).strip() and lf.secret_key.get_secret_value().strip())
    parsed = urlparse(str(lf.base_url).strip())
    host = (parsed.netloc or str(lf.base_url).strip())[:64] if lf_ok else ""
    return {
        "langfuse_credentials_configured": lf_ok,
        "langfuse_host": host,
        "helicone_api_configured": bool(s.helicone.api_key.get_secret_value().strip()),
        "helicone_property_environment_set": bool(str(s.helicone.property_environment).strip()),
        "helicone_property_app_set": bool(str(s.helicone.property_app).strip()),
        "sentry_dsn_configured": bool(str(s.sentry.dsn).strip()),
    }


def get_vendor_apis_public_snapshot() -> dict[str, bool | str]:
    """Optional third-party APIs and Document AI (flags and non-secret fields only)."""
    s = get_settings()
    sch = s.search
    doc = s.document_ai
    tavily_ok = bool(sch.tavily_api_key.get_secret_value().strip())
    serper_ok = bool(sch.serper_api_key.get_secret_value().strip())
    searchapi_ok = bool(sch.searchapi_api_key.get_secret_value().strip())
    return {
        "tavily_configured": tavily_ok,
        "serper_configured": serper_ok,
        "searchapi_configured": searchapi_ok,
        "iris_web_search_configured": bool(tavily_ok or serper_ok or searchapi_ok),
        "searchapi_engine": (sch.searchapi_engine or "google").strip() or "google",
        "apollo_configured": bool(s.apollo.api_key.get_secret_value().strip()),
        "jina_configured": bool(s.jina.api_key.get_secret_value().strip()),
        "firecrawl_configured": bool(s.firecrawl.api_key.get_secret_value().strip()),
        "unstructured_configured": bool(s.unstructured.api_key.get_secret_value().strip()),
        "pdfco_configured": bool(s.pdfco.api_key.get_secret_value().strip()),
        "document_ai_project_configured": bool(str(doc.project_id).strip()),
        "document_ai_location": str(doc.location or "").strip(),
        "newsdata_configured": bool(s.external_apis.api_key.get_secret_value().strip()),
        "google_maps_configured": bool(s.google.maps_api_key.get_secret_value().strip()),
        "neverbounce_configured": bool(s.neverbounce.api_key.get_secret_value().strip()),
        "wolfram_configured": bool(
            s.wolfram.enabled and s.wolfram.app_id.get_secret_value().strip()
        ),
    }


def _file_size_or_zero(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def get_http_api_public_snapshot() -> dict[str, str | bool]:
    """HTTP server–related settings for ops (no secret values)."""
    a = get_settings().app
    cors = str(a.cors_origins or "").strip()
    preview = cors[:96] + ("…" if len(cors) > 96 else "")
    ical_key = bool(a.scheduling_ical_api_key.get_secret_value().strip())
    busy_ics = bool(str(a.scheduling_busy_ics_url or "").strip())
    ical_base = bool(str(a.scheduling_ical_api_base_url or "").strip())
    return {
        "api_secret_configured": bool(a.api_secret_key.get_secret_value().strip()),
        "cors_origins_preview": preview,
        "scheduling_public_base_url_set": bool(str(a.scheduling_public_base_url or "").strip()),
        "scheduling_token_secret_set": bool(a.scheduling_token_secret.get_secret_value().strip()),
        "scheduling_busy_ics_url_set": busy_ics,
        "scheduling_ical_api_key_set": ical_key,
        "scheduling_ical_api_base_url_set": ical_base,
    }


def get_ingestion_brain_public_snapshot() -> dict[str, str | bool | int]:
    """On-disk ingestion artifacts under ``BRAIN_DATA_DIR`` / ``brain/`` plus ingest-related flags."""
    from brain_os.systems.data_dir_lock import get_data_dir

    root = get_data_dir()
    brain = root / "brain"
    imeta = brain / "imports_metadata.json"
    ilog = brain / "ingestion_log.json"
    iprog = brain / "imports_index_progress.json"
    graphe_db = brain / "cursor_sessions.db"
    a = get_settings().app
    return {
        "data_dir_label": str(root),
        "brain_dir_exists": brain.is_dir(),
        "imports_metadata_present": imeta.is_file(),
        "ingestion_log_present": ilog.is_file(),
        "imports_index_progress_present": iprog.is_file(),
        "cursor_sessions_db_present": graphe_db.is_file(),
        "imports_metadata_bytes": _file_size_or_zero(imeta),
        "ingestion_log_bytes": _file_size_or_zero(ilog),
        "redact_pii_at_ingest": a.redact_pii_at_ingest,
        "mnemon_semantic_check": a.mnemon_semantic_check,
    }


def get_runtime_build_public_snapshot() -> dict[str, str]:
    """Installed Brain OS version and interpreter (for support tickets)."""
    import importlib.metadata
    import sys

    try:
        pkg_ver = importlib.metadata.version("ira")
    except importlib.metadata.PackageNotFoundError:
        pkg_ver = "unknown"
    return {
        "ira_package_version": pkg_ver,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
    }


def get_secondary_mail_public_snapshot() -> dict[str, bool]:
    """Optional second Gmail OAuth paths (vendor/procurement mailbox); no secret values."""
    g = get_settings().google
    cred = g.secondary_credentials_path
    tok = g.secondary_token_path
    if cred is None or tok is None:
        return {
            "secondary_mailbox_configured": False,
            "secondary_oauth_files_present": False,
            "secondary_full_access": bool(getattr(g, "secondary_full_access", False)),
        }
    cp = cred.expanduser()
    tp = tok.expanduser()
    return {
        "secondary_mailbox_configured": True,
        "secondary_oauth_files_present": cp.is_file() and tp.is_file(),
        "secondary_full_access": bool(getattr(g, "secondary_full_access", False)),
    }


def get_governance_brain_public_snapshot() -> dict[str, str | bool | int]:
    """Corrections, journal, and truth-hint artifacts under ``BRAIN_DATA_DIR`` / ``brain/``."""
    from brain_os.systems.data_dir_lock import get_data_dir

    root = get_data_dir() / "brain"
    a = get_settings().app
    corr_db = root / "corrections.db"
    ledger = root / "correction_ledger.json"
    aj = root / "agent_journals.db"
    known = root / "known_entities.json"
    th = root / "truth_hints.json"
    lth = root / "learned_truth_hints.json"
    return {
        "corrections_db_present": corr_db.is_file(),
        "corrections_db_bytes": _file_size_or_zero(corr_db),
        "correction_ledger_present": ledger.is_file(),
        "correction_ledger_bytes": _file_size_or_zero(ledger),
        "agent_journals_db_present": aj.is_file(),
        "known_entities_present": known.is_file(),
        "truth_hints_present": th.is_file(),
        "learned_truth_hints_present": lth.is_file(),
        "legacy_quarantine_strict": a.legacy_quarantine_strict,
    }


def get_storage_clients_public_snapshot() -> dict[str, float | int | bool]:
    """Qdrant/Neo4j client knobs from settings (no secret values)."""
    s = get_settings()
    q = s.qdrant
    return {
        "qdrant_timeout_s": float(q.timeout),
        "qdrant_api_key_configured": bool(q.api_key.get_secret_value().strip()),
        "qdrant_cloud_url_configured": bool(q.cloud_url.strip()),
        "qdrant_cloud_api_key_configured": bool(q.cloud_api_key.get_secret_value().strip()),
        "neo4j_max_pool_size": int(s.app.neo4j_max_pool_size),
    }


def get_workspace_content_public_snapshot() -> dict[str, str | bool]:
    """Repo-relative ``data/imports`` path (Alexandros index source); exists flag only for fast health."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    imports_dir = repo_root / "data" / "imports"
    return {
        "repo_imports_dir_exists": imports_dir.is_dir(),
        "repo_imports_path_label": str(imports_dir),
    }


def get_email_training_public_snapshot() -> dict[str, bool]:
    """TRAINING mailbox address from settings (presence only, not the address text)."""
    g = get_settings().google
    return {"training_mailbox_configured": bool(g.training_email.strip())}


def get_pipeline_observability_public_snapshot() -> dict[str, str | float | bool]:
    """Logging, cache overrides, Sentry sampling, OAuth app id presence, Doc AI processor flags, Unstructured host."""
    from urllib.parse import urlparse

    s = get_settings()
    a = s.app
    g = s.google
    doc = s.document_ai
    sen = s.sentry
    unst = s.unstructured
    parsed = urlparse(str(unst.api_url).strip())
    uhost = (parsed.netloc or str(unst.api_url).strip())[:64]
    return {
        "log_format": str(a.log_format or "").strip() or "text",
        "hf_cache_override_configured": bool(str(a.hf_cache_dir).strip()),
        "sentry_traces_sample_rate": float(sen.traces_sample_rate),
        "google_oauth_client_id_configured": bool(str(g.oauth_client_id).strip()),
        "document_ai_processor_configured": bool(str(doc.processor_id).strip()),
        "document_ai_invoice_processor_configured": bool(str(doc.invoice_processor_id).strip()),
        "document_ai_form_processor_configured": bool(str(doc.form_processor_id).strip()),
        "unstructured_api_host": uhost,
    }


def get_repo_brain_aux_public_snapshot() -> dict[str, str | bool | int]:
    """Auxiliary repo ``data/brain`` files (embedding cache, retrieval log, quality DB, scores)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    brain = repo_root / "data" / "brain"
    emb = brain / "embedding_cache.db"
    ret = brain / "retrieval_log.jsonl"
    asc = brain / "asclepius.db"
    scores = brain / "agent_scores.json"
    return {
        "repo_brain_path_label": str(brain),
        "embedding_cache_db_present": emb.is_file(),
        "embedding_cache_db_bytes": _file_size_or_zero(emb),
        "retrieval_log_present": ret.is_file(),
        "retrieval_log_bytes": _file_size_or_zero(ret),
        "asclepius_db_present": asc.is_file(),
        "asclepius_db_bytes": _file_size_or_zero(asc),
        "agent_scores_json_present": scores.is_file(),
        "agent_scores_json_bytes": _file_size_or_zero(scores),
    }


def get_process_context_public_snapshot() -> dict[str, str | bool]:
    """Process-level context for support (cwd, env flag, interpreter basename only)."""
    import os
    import sys

    raw = os.environ.get("BRAIN_DATA_DIR", "").strip()
    exe = Path(sys.executable)
    return {
        "ira_data_dir_env_set": bool(raw),
        "working_directory_label": str(Path.cwd().resolve())[:112],
        "python_executable_basename": exe.name,
    }


def get_sales_coaching_brain_public_snapshot() -> dict[str, str | bool | int]:
    """Chiron sales coaching JSON under repo ``data/brain``."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    brain = repo_root / "data" / "brain"
    st = brain / "sales_training.json"
    return {
        "repo_brain_path_label": str(brain),
        "sales_training_json_present": st.is_file(),
        "sales_training_json_bytes": _file_size_or_zero(st),
    }


def get_secondary_google_oauth_public_snapshot() -> dict[str, bool]:
    """Secondary Gmail OAuth *client id* configured (not secrets)."""
    g = get_settings().google
    return {"secondary_oauth_client_id_configured": bool(str(g.secondary_oauth_client_id).strip())}


def get_neo4j_connection_flags_public_snapshot() -> dict[str, bool]:
    """How Neo4j credentials are configured (no password or auth string values)."""
    n = get_settings().neo4j
    uri = n.uri.strip()
    cloud = n.cloud_uri.strip()
    cloud_cred = n.resolved_cloud_auth()
    return {
        "neo4j_password_field_set": bool(n.password.get_secret_value().strip()),
        "neo4j_auth_env_string_set": bool(n.auth.strip()),
        "neo4j_bolt_localhost": "localhost" in uri or "127.0.0.1" in uri,
        "neo4j_cloud_uri_set": bool(cloud),
        "neo4j_cloud_mirror_configured": bool(cloud and cloud_cred is not None),
    }


def get_repo_truth_hints_public_snapshot() -> dict[str, str | bool | int]:
    """Truth-hint JSON files under repo ``data/brain`` (fast-path / Mnemon seeds)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    brain = repo_root / "data" / "brain"
    th = brain / "truth_hints.json"
    lth = brain / "learned_truth_hints.json"
    return {
        "repo_brain_path_label": str(brain),
        "truth_hints_present": th.is_file(),
        "truth_hints_bytes": _file_size_or_zero(th),
        "learned_truth_hints_present": lth.is_file(),
        "learned_truth_hints_bytes": _file_size_or_zero(lth),
    }


@lru_cache(maxsize=128)
def _python_module_available(module: str) -> bool:
    """True if ``module`` imports cleanly in this interpreter (any failure → False)."""
    try:
        __import__(module)
    except Exception:
        return False
    return True


def get_runtime_capabilities_public_snapshot() -> dict[str, bool]:
    """Critical and optional Python modules (import probes in this interpreter)."""
    return {
        "filelock_available": _python_module_available("filelock"),
        "apscheduler_available": _python_module_available("apscheduler"),
        "httpx_available": _python_module_available("httpx"),
        "fastapi_available": _python_module_available("fastapi"),
        "typer_available": _python_module_available("typer"),
        "qdrant_client_available": _python_module_available("qdrant_client"),
        "neo4j_available": _python_module_available("neo4j"),
        "sqlalchemy_available": _python_module_available("sqlalchemy"),
    }


def get_repo_framework_public_snapshot() -> dict[str, bool | str]:
    """Framework / devops files at repo root (checkout layout), not secrets."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    return {
        "repo_root_label": str(repo_root)[:112],
        "web_ui_package_json_present": (repo_root / "web-ui" / "package.json").is_file(),
        "docker_compose_local_yml_present": (repo_root / "docker-compose.local.yml").is_file(),
        "env_example_present": (repo_root / ".env.example").is_file(),
        "alembic_ini_present": (repo_root / "alembic.ini").is_file(),
        "pyproject_toml_present": (repo_root / "pyproject.toml").is_file(),
    }


def get_repo_sources_public_snapshot() -> dict[str, bool | int]:
    """Main source directories and prompt inventory (checkout completeness)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    prompts = repo_root / "prompts"
    prompt_txt_count = sum(1 for _ in prompts.glob("*.txt")) if prompts.is_dir() else 0
    return {
        "src_ira_package_present": (repo_root / "src" / "ira" / "__init__.py").is_file(),
        "prompts_dir_present": prompts.is_dir(),
        "prompt_txt_files_count": prompt_txt_count,
        "tests_dir_present": (repo_root / "tests").is_dir(),
        "alembic_dir_present": (repo_root / "alembic").is_dir(),
        "scripts_dir_present": (repo_root / "scripts").is_dir(),
    }


def get_app_runtime_imports_public_snapshot() -> dict[str, bool]:
    """LLM / server / cache client imports beyond phase-17 core HTTP & DB stack."""
    return {
        "openai_available": _python_module_available("openai"),
        "anthropic_available": _python_module_available("anthropic"),
        "langfuse_available": _python_module_available("langfuse"),
        "redis_available": _python_module_available("redis"),
        "uvicorn_available": _python_module_available("uvicorn"),
        "voyageai_available": _python_module_available("voyageai"),
        "mem0_available": _python_module_available("mem0"),
        "mcp_available": _python_module_available("mcp"),
        "tiktoken_available": _python_module_available("tiktoken"),
        "jinja2_available": _python_module_available("jinja2"),
        "pydantic_settings_available": _python_module_available("pydantic_settings"),
    }


def get_optional_tooling_imports_public_snapshot() -> dict[str, bool]:
    """Heavier optional imports: ingestion, drivers, Google clients, tracing (beyond phase 18)."""
    return {
        "sentry_sdk_available": _python_module_available("sentry_sdk"),
        "alembic_available": _python_module_available("alembic"),
        "asyncpg_available": _python_module_available("asyncpg"),
        "aiosqlite_available": _python_module_available("aiosqlite"),
        "google_auth_available": _python_module_available("google.auth"),
        "googleapiclient_available": _python_module_available("googleapiclient"),
        "sse_starlette_available": _python_module_available("sse_starlette"),
        "crawl4ai_available": _python_module_available("crawl4ai"),
        "docling_available": _python_module_available("docling"),
        "neo4j_graphrag_available": _python_module_available("neo4j_graphrag"),
        "gliner_available": _python_module_available("gliner"),
        "instructor_available": _python_module_available("instructor"),
        "chonkie_available": _python_module_available("chonkie"),
        "pypdf_available": _python_module_available("pypdf"),
    }


def get_repo_vcs_public_snapshot() -> dict[str, bool | str]:
    """Git metadata at repo root: present, branch label, dirty flag (no hashes or file names)."""

    repo_root = Path(__file__).resolve().parent.parent.parent
    if not (repo_root / ".git").exists():
        return {
            "git_repo_present": False,
            "git_branch_label": "",
            "git_working_tree_dirty": False,
        }
    branch = ""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0:
            branch = (proc.stdout or "").strip()[:64]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        branch = ""
    dirty = False
    try:
        dproc = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=repo_root,
            capture_output=True,
            timeout=3,
            check=False,
        )
        dirty = dproc.returncode != 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        dirty = False
    return {
        "git_repo_present": True,
        "git_branch_label": branch,
        "git_working_tree_dirty": dirty,
    }


def get_app_feature_flags_public_snapshot() -> dict[str, bool]:
    """Boolean feature toggles from AppConfig (ingest/governance/cache; no secrets)."""
    a = get_settings().app
    return {
        "mnemon_semantic_check": a.mnemon_semantic_check,
        "legacy_quarantine_strict": a.legacy_quarantine_strict,
        "redact_pii_at_ingest": a.redact_pii_at_ingest,
        "flashrank_cache_override_configured": bool(str(a.flashrank_cache_dir).strip()),
        "progressive_tool_discovery": a.progressive_tool_discovery,
        "progressive_tool_discovery_expand_on_low_confidence": a.progressive_tool_discovery_expand_on_low_confidence,
        "tool_selection_telemetry": a.tool_selection_telemetry,
        "mcp_fastlane_enabled": a.mcp_fastlane_enabled,
        "mcp_route_prefer_cloud_fast": a.mcp_route_prefer_cloud_fast,
        "mcp_fast_fail_to_openai": a.mcp_fast_fail_to_openai,
        "request_prompt_snapshot": a.request_prompt_snapshot,
        "slack_listen_enabled": a.slack_listen_enabled,
    }


def get_quality_stack_imports_public_snapshot() -> dict[str, bool]:
    """Reranker + eval / guardrails packages (import probes; optional for production)."""
    return {
        "flashrank_available": _python_module_available("flashrank"),
        "deepeval_available": _python_module_available("deepeval"),
        "ragas_available": _python_module_available("ragas"),
        "guardrails_available": _python_module_available("guardrails"),
    }


def get_skills_registry_public_snapshot() -> dict[str, int | str]:
    """Registered Brain OS skills from :data:`SKILL_MATRIX` (name preview, no handler code)."""
    from brain_os.skills import SKILL_MATRIX

    names = sorted(SKILL_MATRIX)
    joined = ",".join(names)
    preview = joined if len(joined) <= 192 else f"{joined[:189]}…"
    return {
        "registered_skill_count": len(SKILL_MATRIX),
        "skill_names_preview": preview,
    }


def get_event_driven_ingestion_public_snapshot(
    services: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Event bus wiring and optional reactor / push / heartbeat flags (no secrets)."""
    from brain_os.service_keys import ServiceKey as SK

    settings = get_settings()
    app = settings.app
    google = settings.google
    bus_present = False
    if services is not None:
        bus = services.get(SK.DATA_EVENT_BUS)
        bus_present = bus is not None
    return {
        "data_event_bus": bus_present,
        "event_task_reactor_enabled": app.event_task_reactor_enabled,
        "event_task_rules_path": app.event_task_rules_path,
        "event_task_max_inflight": app.event_task_max_inflight,
        "gmail_push_enabled": google.gmail_push_enabled,
        "email_poll_enabled": google.email_poll_enabled,
        "heartbeat_server_enabled": app.heartbeat_server_enabled,
        "heartbeat_server_interval_minutes": app.heartbeat_server_interval_minutes,
        "heartbeat_jobs_path": app.heartbeat_jobs_path,
    }
