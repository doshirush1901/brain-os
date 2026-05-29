"""Mirror operator context and pipeline run records into Neo4j (context graph P1).

Fail-open: SQLite remains canonical; graph sync is optional via
``APP__OPERATOR_CONTEXT_NEO4J_SYNC=true``.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from brain_os.brain.knowledge_graph import normalize_entity_name
from brain_os.config import get_settings
from brain_os.exceptions import DatabaseError
from brain_os.schemas.operator_context import OperatorContextRun
from brain_os.schemas.run_record import RunRecord

logger = logging.getLogger(__name__)


def context_graph_neo4j_sync_enabled() -> bool:
    """Whether operator/pipeline reasoning nodes are mirrored to Neo4j."""
    return bool(get_settings().app.operator_context_neo4j_sync)


def company_name_from_run_record(record: RunRecord) -> str | None:
    """Best-effort company anchor from artifact links (no full-text parse)."""
    for link in record.artifacts.links or []:
        if not isinstance(link, dict):
            continue
        if str(link.get("kind") or "") != "operator_context":
            continue
        company = link.get("company")
        if isinstance(company, str) and company.strip():
            return _normalize_company(company)
        detail = link.get("detail")
        if isinstance(detail, dict):
            nested = detail.get("company")
            if isinstance(nested, str) and nested.strip():
                return _normalize_company(nested)
    return None


def _normalize_company(name: str) -> str:
    return normalize_entity_name((name or "").strip())


def _evidence_id(run_id: str, kind: str, ref: str) -> str:
    digest = hashlib.sha256(f"{run_id}:{kind}:{ref}".encode()).hexdigest()[:16]
    return f"{run_id[:48]}:{digest}"


async def sync_operator_context_run(
    record: OperatorContextRun,
    *,
    kg: Any | None = None,
) -> bool:
    """MERGE OperatorContextRun + Company link (+ optional PipelineRun link)."""
    if not context_graph_neo4j_sync_enabled():
        return False
    name = (record.company_name or "").strip()
    if not name:
        return False
    close_kg = False
    graph = kg
    try:
        if graph is None:
            from brain_os.brain.knowledge_graph import KnowledgeGraph

            graph = KnowledgeGraph()
            close_kg = True
        return await graph.merge_operator_context_run(
            run_id=record.run_id,
            kind=record.kind,
            outcome=record.outcome,
            ts=record.ts,
            company_name=name,
            summary=record.summary,
            domain=record.domain,
            machine_model=record.machine_model,
            crm_stage=record.crm_stage,
            pipeline_run_id=record.pipeline_run_id,
            success=record.success,
            source_id=f"operator_context:{record.run_id}",
        )
    except (DatabaseError, OSError, ValueError, TypeError):
        logger.debug("sync_operator_context_run failed run_id=%s", record.run_id, exc_info=True)
        return False
    finally:
        if close_kg and graph is not None:
            await graph.close()


async def sync_pipeline_run(
    record: RunRecord,
    *,
    kg: Any | None = None,
) -> bool:
    """MERGE PipelineRun, evidence nodes, optional Company link."""
    if not context_graph_neo4j_sync_enabled():
        return False
    rid = (record.run_id or "").strip()
    if not rid:
        return False
    company = company_name_from_run_record(record)
    evidence: list[dict[str, Any]] = []
    for idx, ev in enumerate(record.evidence or []):
        ref = (ev.ref or "").strip()
        if not ref:
            continue
        evidence.append(
            {
                "evidence_id": _evidence_id(rid, ev.kind, ref),
                "kind": ev.kind,
                "ref": ref[:512],
                "score": ev.score,
                "ordinal": idx,
            }
        )
    close_kg = False
    graph = kg
    try:
        if graph is None:
            from brain_os.brain.knowledge_graph import KnowledgeGraph

            graph = KnowledgeGraph()
            close_kg = True
        ok = await graph.merge_pipeline_run(
            run_id=rid,
            outcome=record.outcome,
            channel=record.channel,
            ts_end=record.ts_end,
            route_method=record.route.method,
            agents_count=len(record.agents),
            input_summary=record.input_summary,
            response_summary=record.response_summary,
            company_name=company,
            evidence=evidence,
            source_id=f"pipeline_run:{rid}",
        )
        for link in record.artifacts.links or []:
            if not isinstance(link, dict):
                continue
            op_id = link.get("operator_context_run_id")
            if isinstance(op_id, str) and op_id.strip():
                await graph.link_operator_context_to_pipeline(
                    op_id.strip(),
                    rid,
                    source_id=f"pipeline_run:{rid}",
                )
        return ok
    except (DatabaseError, OSError, ValueError, TypeError):
        logger.debug("sync_pipeline_run failed run_id=%s", rid, exc_info=True)
        return False
    finally:
        if close_kg and graph is not None:
            await graph.close()
