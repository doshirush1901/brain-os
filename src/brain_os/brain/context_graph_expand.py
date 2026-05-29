"""P2 — 1-hop Neo4j expansion for account briefs and company-scoped retrieval."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain_os.brain.knowledge_graph import KnowledgeGraph, normalize_entity_name
from brain_os.config import get_settings
from brain_os.exceptions import DatabaseError
from brain_os.schemas.account_brief import AccountBrief, SourceRef

logger = logging.getLogger(__name__)


def graph_expand_enabled() -> bool:
    return bool(get_settings().app.operator_context_graph_expand_enabled)


def format_expansion_lines(expansion: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Turn structured 1-hop graph rows into operator-readable lines."""
    lines: list[str] = []

    for row in expansion.get("operator_runs") or []:
        kind = str(row.get("kind") or "run").replace("_", " ")
        outcome = str(row.get("outcome") or "")
        summary = str(row.get("summary") or "").strip()
        ok = row.get("success")
        flag = ""
        if ok is True:
            flag = " [ok]"
        elif ok is False:
            flag = " [failed]"
        extra = []
        if row.get("crm_stage"):
            extra.append(f"stage={row.get('crm_stage')}")
        if row.get("machine_model"):
            extra.append(f"machine={row.get('machine_model')}")
        tail = f" — {'; '.join(extra)}" if extra else ""
        body = summary or row.get("run_id") or "operator run"
        lines.append(f"Operator {kind}{flag} ({outcome}): {body}{tail}")

    for row in expansion.get("pipeline_runs") or []:
        outcome = str(row.get("outcome") or "unknown")
        channel = str(row.get("channel") or "")
        route = str(row.get("route_method") or "")
        ev = row.get("evidence_count")
        ev_txt = f", {ev} evidence ref(s)" if ev is not None else ""
        agents = row.get("agents_count")
        ag_txt = f", {agents} agent(s)" if agents else ""
        lines.append(
            f"Pipeline {outcome} ({channel or 'n/a'}, route={route or 'n/a'}{ag_txt}{ev_txt})"
        )

    for row in expansion.get("contacts") or []:
        name = str(row.get("name") or "").strip()
        email = str(row.get("email") or "").strip()
        role = str(row.get("role") or "").strip()
        who = name or email or "contact"
        if email and name:
            who = f"{name} <{email}>"
        elif email:
            who = email
        role_txt = f", {role}" if role else ""
        lines.append(f"Contact: {who}{role_txt}")

    for row in expansion.get("quotes") or []:
        qid = str(row.get("quote_id") or "quote")
        machine = str(row.get("machine") or "").strip()
        status = str(row.get("status") or "").strip()
        company = str(row.get("company") or "").strip()
        value = row.get("value")
        value_txt = f", value={value}" if value is not None else ""
        machine_txt = f", machine={machine}" if machine else ""
        status_txt = f", {status}" if status else ""
        co_txt = f" ({company})" if company else ""
        lines.append(f"Quote {qid}{co_txt}{machine_txt}{status_txt}{value_txt}")

    related = expansion.get("related_companies") or []
    if len(related) > 1:
        names = [
            str(r.get("name") if isinstance(r, dict) else r).strip()
            for r in related
            if str(r.get("name") if isinstance(r, dict) else r).strip()
        ]
        if names:
            preview = ", ".join(names[:6])
            if len(names) > 6:
                preview += f" (+{len(names) - 6} more)"
            lines.append(f"Related company nodes: {preview}")

    for row in expansion.get("quote_documents") or []:
        company = str(row.get("company") or "").strip()
        source = str(row.get("source") or "").strip()
        if not source:
            continue
        label = Path(source).name or source
        prefix = f" ({company})" if company else ""
        lines.append(f"Quote document{prefix}: {label}")

    return lines[:24]


async def fetch_company_graph_context(
    company_name: str,
    *,
    graph: KnowledgeGraph | None = None,
) -> tuple[list[str], list[SourceRef]]:
    """1-hop Neo4j context lines for *company_name* (fail-open empty)."""
    if not graph_expand_enabled():
        return [], []
    name = normalize_entity_name((company_name or "").strip())
    if not name:
        return [], []
    kg = graph
    close = False
    if kg is None:
        kg = KnowledgeGraph()
        close = True
    try:
        if not await kg.company_node_exists(name):
            return [], []
        expansion = await kg.expand_company_one_hop(name)
        lines = format_expansion_lines(expansion)
        if not lines:
            return [], []
        sources = [
            SourceRef(
                channel="neo4j",
                label="context_graph_1hop",
                freshness=datetime.now(UTC).isoformat()[:19],
            )
        ]
        return lines, sources
    except (DatabaseError, OSError, ValueError, TypeError):
        logger.debug("fetch_company_graph_context failed name=%s", name[:80], exc_info=True)
        return [], []
    finally:
        if close:
            await kg.close()


async def attach_graph_context_to_brief(
    brief: AccountBrief,
    *,
    graph: KnowledgeGraph | None = None,
) -> AccountBrief:
    """Merge 1-hop graph lines into the brief (does not replace SQLite precedents)."""
    lines, sources = await fetch_company_graph_context(brief.company_name, graph=graph)
    if not lines:
        return brief
    merged_sources = list(brief.sources)
    for src in sources:
        if not any(s.channel == src.channel and s.label == src.label for s in merged_sources):
            merged_sources.append(src)
    return brief.model_copy(
        update={
            "graph_context_lines": lines,
            "sources": merged_sources,
        }
    )


def graph_context_to_retriever_hits(
    company_name: str,
    expansion: dict[str, list[dict[str, Any]]],
    *,
    base_score: float = 0.82,
) -> list[dict[str, Any]]:
    """Neo4j-shaped retriever rows from a 1-hop company expansion."""
    lines = format_expansion_lines(expansion)
    name = normalize_entity_name(company_name.strip())
    out: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        out.append(
            {
                "content": line,
                "score": max(0.5, base_score - idx * 0.02),
                "source": "neo4j",
                "source_type": "neo4j",
                "metadata": {
                    "type": "context_graph_1hop",
                    "company": name,
                    "line_index": idx,
                },
            }
        )
    if lines:
        summary = f"Graph 1-hop context for {name}: " + "; ".join(lines[:6])
        if len(lines) > 6:
            summary += f" (+{len(lines) - 6} more)"
        out.insert(
            0,
            {
                "content": summary,
                "score": base_score,
                "source": "neo4j",
                "source_type": "neo4j",
                "metadata": {"type": "context_graph_1hop_summary", "company": name},
            },
        )
    return out
