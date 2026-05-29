"""P5 — Read-only company context-graph bundle (Neo4j + optional SQLite precedents)."""

from __future__ import annotations

import logging
from typing import Any

from brain_os.brain.company_similarity import find_similar_companies
from brain_os.brain.context_graph_expand import format_expansion_lines
from brain_os.brain.knowledge_graph import KnowledgeGraph, normalize_entity_name
from brain_os.exceptions import DatabaseError
from brain_os.schemas.account_brief import AccountBrief
from brain_os.schemas.company_context_graph import CompanyContextGraphBundle
from brain_os.services.operator_context import fetch_context_precedents

logger = logging.getLogger(__name__)


def _jsonable_rows(rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        cleaned: dict[str, object] = {}
        for k, v in row.items():
            if v is None:
                cleaned[k] = None
            elif isinstance(v, (bool, int, float, str)):
                cleaned[k] = v
            else:
                cleaned[k] = str(v)
        out.append(cleaned)
    return out


async def fetch_company_context_graph(
    company_name: str,
    *,
    include_similar: bool = True,
    include_sqlite_precedents: bool = True,
    similar_limit: int = 5,
    graph: KnowledgeGraph | None = None,
) -> CompanyContextGraphBundle:
    """Assemble P2 1-hop + P3 similar + SQLite precedents for one account (fail-open)."""
    name = normalize_entity_name((company_name or "").strip())
    if not name:
        return CompanyContextGraphBundle(company_name=company_name or "", company_exists=False)

    close_kg = graph is None
    kg = graph or KnowledgeGraph()
    bundle = CompanyContextGraphBundle(
        company_name=(company_name or "").strip(),
        company_normalized=name,
    )
    try:
        bundle.company_exists = await kg.company_node_exists(name)
        if not bundle.company_exists:
            return bundle

        props_rows = await kg._read(
            """
            MATCH (c:Company {name: $name})
            RETURN c.region AS region, c.industry AS industry, c.website AS website
            LIMIT 1
            """,
            name=name,
        )
        if props_rows:
            row = props_rows[0]
            bundle.region = str(row["region"]) if row.get("region") else None
            bundle.industry = str(row["industry"]) if row.get("industry") else None
            bundle.website = str(row["website"]) if row.get("website") else None

        expansion = await kg.expand_company_one_hop(name)
        bundle.operator_runs = _jsonable_rows(expansion.get("operator_runs") or [])
        bundle.pipeline_runs = _jsonable_rows(expansion.get("pipeline_runs") or [])
        bundle.contacts = _jsonable_rows(expansion.get("contacts") or [])
        bundle.quotes = _jsonable_rows(expansion.get("quotes") or [])
        bundle.quote_documents = _jsonable_rows(expansion.get("quote_documents") or [])
        related = expansion.get("related_companies") or []
        bundle.related_company_names = [
            str(r.get("name") if isinstance(r, dict) else r).strip()
            for r in related
            if str(r.get("name") if isinstance(r, dict) else r).strip()
        ]
        bundle.graph_context_lines = format_expansion_lines(expansion)

        if include_similar:
            bundle.similar_companies = await find_similar_companies(
                name,
                limit=similar_limit,
                graph=kg,
            )

        if include_sqlite_precedents:
            brief = AccountBrief(company_name=name, domain=None)
            bundle.sqlite_precedents = await fetch_context_precedents(brief, limit=similar_limit)

    except (DatabaseError, OSError, ValueError, TypeError):
        logger.debug("fetch_company_context_graph failed name=%s", name[:80], exc_info=True)
    finally:
        if close_kg:
            await kg.close()

    return bundle


def format_company_context_graph_plain(bundle: CompanyContextGraphBundle) -> str:
    """Gmail-safe summary for MCP default text mode."""
    lines: list[str] = [
        f"CONTEXT GRAPH — {bundle.company_name}",
        f"Normalized: {bundle.company_normalized or 'n/a'}",
        f"Company node: {'yes' if bundle.company_exists else 'no'}",
    ]
    if bundle.region or bundle.industry:
        lines.append(
            f"Profile: region={bundle.region or 'n/a'} industry={bundle.industry or 'n/a'}"
        )
    if bundle.related_company_names and len(bundle.related_company_names) > 1:
        lines.append(
            f"Aliases: {', '.join(bundle.related_company_names[:8])}"
            + (
                f" (+{len(bundle.related_company_names) - 8} more)"
                if len(bundle.related_company_names) > 8
                else ""
            )
        )
    if bundle.quote_documents:
        lines.append("")
        lines.append("Quote documents (KB graph)")
        for doc in bundle.quote_documents[:8]:
            src = str(doc.get("source") or "")
            co = str(doc.get("company") or "")
            label = src.rsplit("/", 1)[-1] if src else "n/a"
            lines.append(f"- {label}" + (f" [{co}]" if co else ""))
    if bundle.graph_context_lines:
        lines.append("")
        lines.append("Graph (1-hop)")
        for ln in bundle.graph_context_lines[:12]:
            lines.append(f"- {ln}")
    if bundle.similar_companies:
        lines.append("")
        lines.append("Similar accounts")
        for s in bundle.similar_companies[:5]:
            lines.append(f"- {s.company_name} ({s.score:.2f})")
    if bundle.sqlite_precedents:
        lines.append("")
        lines.append("SQLite precedents")
        for p in bundle.sqlite_precedents[:5]:
            lines.append(f"- {p.kind}: {p.summary or 'n/a'}")
    if not bundle.company_exists:
        lines.append("")
        lines.append("UNVERIFIED: no :Company node — run brief or backfill graph.")
    return "\n".join(lines).rstrip() + "\n"
