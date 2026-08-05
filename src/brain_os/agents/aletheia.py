"""Aletheia — Compliance / Provenance agent.

Traces claims in agent responses back to sources (Qdrant, Neo4j, CRM).
Runs in the pipeline between execution and shaping to flag unverifiable
claims before they reach the user.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.knowledge.teacher_provenance import (
    format_teacher_citation,
    hit_has_teacher_authority,
    response_has_wolfram_teacher_attribution,
)
from brain_os.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_RETRIEVER_ERRORS = (RuntimeError, ValueError, KeyError, httpx.HTTPError, OSError)

_SYSTEM_PROMPT = load_prompt("aletheia_system")

_CLAIM_PATTERN = re.compile(
    r"(\d[\d,.]+\s*(?:INR|USD|EUR|Rs|lakh|crore|weeks?|days?|months?|kg|tons?|units?|%)|"
    r"\b(?:ready|dispatched|delivered|shipped|completed|scheduled)\s+(?:by|on|for)\s+\S+|"
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)",
    re.IGNORECASE,
)

_COMMITMENT_SNIPPET_RE = re.compile(
    r"\b(?:MOQ|minimum\s+order|money[- ]back|guaranteed|guarantee|warranty|warrantied|"
    r"liable|indemnify|liquidated\s+damages|binding\s+(?:offer|commitment)|"
    r"(?:we|I|our\s+(?:company|team))\s+(?:will|would|shall)\b)"
    r"[^.\n!?]{10,520}[.\n!?]",
    re.IGNORECASE | re.DOTALL,
)


class Aletheia(BaseAgent):
    name = "aletheia"
    role = "Compliance / Provenance"
    description = "Traces claims to sources and flags unverifiable assertions"
    knowledge_categories = [
        "company_internal",
        "sales_and_crm",
        "project_case_studies",
    ]
    timeout = 45

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="trace_claim",
                description=(
                    "Search Qdrant and Neo4j for source evidence matching a specific claim. "
                    "Returns source reference or 'unverifiable'."
                ),
                parameters={
                    "claim": "The specific claim to trace (e.g. a number, date, or assertion)"
                },
                handler=self._tool_trace_claim,
            )
        )

        self.register_tool(
            AgentTool(
                name="verify_sources",
                description="Extract key claims from a response and trace each to a source. Returns a provenance report.",
                parameters={"response_text": "The full response text to verify"},
                handler=self._tool_verify_sources,
            )
        )

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)

    async def check_provenance(self, response_text: str) -> dict[str, Any]:
        """Quick provenance check without the full ReAct loop."""
        extracted: list[str] = []
        for hit in _CLAIM_PATTERN.findall(response_text):
            piece = hit.strip()
            if piece:
                extracted.append(piece)
        for m in _COMMITMENT_SNIPPET_RE.finditer(response_text):
            piece = (m.group(0) or "").strip()
            piece = piece[:400]
            if len(piece) > 22 and piece not in extracted:
                extracted.append(piece)

        dedup = list(dict.fromkeys(extracted))
        claims = dedup[:14]
        if not claims:
            return {"verdict": "VERIFIED", "claims_checked": 0, "unverifiable": []}

        verified: list[str] = []
        unverifiable: list[str] = []

        wolfram_sourced = response_has_wolfram_teacher_attribution(response_text)

        for claim in claims[:10]:
            claim_str = claim.strip()
            if wolfram_sourced and _claim_likely_computational(claim_str):
                verified.append(claim_str)
                continue
            try:
                results = await self._retriever.search(claim_str, limit=3)
                if results:
                    best = max(results, key=lambda r: r.get("score", 0))
                    is_teacher, teacher_id = hit_has_teacher_authority(best)
                    if is_teacher and best.get("score", 0) > 0.35:
                        verified.append(claim_str)
                        continue
                    if any(r.get("score", 0) > 0.5 for r in results):
                        verified.append(claim_str)
                        continue
                unverifiable.append(claim_str)
            except _RETRIEVER_ERRORS:
                unverifiable.append(claim_str)

        total = len(verified) + len(unverifiable)
        if not unverifiable:
            verdict = "VERIFIED"
        elif len(unverifiable) < total / 2:
            verdict = "PARTIAL"
        else:
            verdict = "UNVERIFIED"

        return {
            "verdict": verdict,
            "claims_checked": total,
            "verified": verified,
            "unverifiable": unverifiable,
        }

    async def check_provenance_outbound(self, response_text: str) -> dict[str, Any]:
        """Stricter provenance for outbound drafts (Phase 2 — channel=outbound)."""
        from brain_os.config import get_settings

        base = await self.check_provenance(response_text)
        if not get_settings().app.aletheia_outbound_mode_enabled:
            return {**base, "outbound_mode": False}

        flagged_metrics: list[str] = []
        for claim in base.get("unverifiable") or []:
            if _CLAIM_PATTERN.search(claim):
                flagged_metrics.append(claim[:200])
        strict_unverifiable = list(
            dict.fromkeys(flagged_metrics + (base.get("unverifiable") or []))
        )
        verdict = base.get("verdict", "UNVERIFIED")
        if strict_unverifiable and verdict == "VERIFIED":
            verdict = "PARTIAL"
        if len(strict_unverifiable) >= 2:
            verdict = "UNVERIFIED"

        return {
            **base,
            "outbound_mode": True,
            "verdict": verdict,
            "unverifiable": strict_unverifiable[:12],
            "flagged_metrics": flagged_metrics[:8],
            "operator_note": (
                "Outbound mode: numeric/date claims need KB or proof registry citation."
                if strict_unverifiable
                else ""
            ),
        }

    async def _tool_trace_claim(self, claim: str) -> str:
        try:
            results = await self._retriever.search(claim, limit=5)
        except _RETRIEVER_ERRORS as exc:
            return f"Search error: {exc}"

        if not results:
            return f"UNVERIFIABLE: No source found for '{claim}'"

        best = max(results, key=lambda r: r.get("score", 0))
        if best.get("score", 0) < 0.4:
            return f"UNVERIFIABLE: Best match score {best.get('score', 0):.2f} is too low for '{claim}'"

        is_teacher, teacher_id = hit_has_teacher_authority(best)
        if is_teacher:
            cite = format_teacher_citation(teacher_id)
            content_preview = best.get("content", "")[:200]
            return f"TEACHER_SOURCED: '{claim}' → {cite}\n  Evidence: {content_preview}"

        source = best.get("metadata", {}).get("source", best.get("source_type", "unknown"))
        content_preview = best.get("content", "")[:200]
        return f"VERIFIED: '{claim}' → source: {source} (score: {best.get('score', 0):.2f})\n  Evidence: {content_preview}"

    async def _tool_verify_sources(self, response_text: str) -> str:
        result = await self.check_provenance(response_text)
        lines = [
            f"Provenance verdict: {result['verdict']} ({result['claims_checked']} claims checked)"
        ]
        if result.get("verified"):
            lines.append(f"Verified: {', '.join(result['verified'][:5])}")
        if result.get("unverifiable"):
            lines.append(f"Unverifiable: {', '.join(result['unverifiable'][:5])}")
        return "\n".join(lines)


def _claim_likely_computational(claim: str) -> bool:
    """Heuristic: numeric/unit claims often satisfied by Wolfram teacher attribution."""
    low = claim.lower()
    if any(tok in low for tok in ("inr", "usd", "rs", "lakh", "crore", "quote", "deal")):
        return False
    return bool(
        re.search(
            r"\d|psi|bar|°c|°f|w/m|kg|pa\b|convert|heat\s+transfer",
            low,
        )
    )
