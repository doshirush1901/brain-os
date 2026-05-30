"""Pricing and quote intelligence for Acme Corp.

Provides price estimation from historical quotes, aggregate pipeline
analytics, and formal quote-content generation.  All knowledge retrieval
goes through :class:`~brain_os.brain.retriever.UnifiedRetriever`; CRM data is
accessed via a thin repository interface defined here (implemented in
Phase 4).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from langfuse.decorators import observe

from brain_os.brain.retriever import UnifiedRetriever
from brain_os.brain.untrusted_input import sanitize_for_llm_embedding
from brain_os.data.models import Contact
from brain_os.prompt_loader import load_prompt
from brain_os.services.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)

_QUOTES_CATEGORY = "quotes_and_proposals"


# ── CRM repository interface (implemented in Phase 4) ───────────────────────


class CRMRepository(Protocol):
    """Minimal interface the pricing engine needs from the CRM layer."""

    async def get_deals_by_filter(self, filters: dict[str, Any]) -> list[dict[str, Any]]: ...
    async def get_contact(self, contact_id: str) -> dict[str, Any] | None: ...


# ── LLM prompts ─────────────────────────────────────────────────────────────

_ESTIMATE_SYSTEM_PROMPT = load_prompt("estimate_price")

_QUOTE_CONTENT_SYSTEM_PROMPT = load_prompt("quote_content")


class PricingEngine:
    """Pricing estimation, quote analytics, and quote-content generation."""

    def __init__(
        self,
        retriever: UnifiedRetriever,
        crm: CRMRepository | None = None,
        *,
        llm: LLMClient | None = None,
    ) -> None:
        self._retriever = retriever
        self._crm = crm
        self._llm = llm or get_llm_client()

    # ── price estimation ─────────────────────────────────────────────────

    @observe()
    async def estimate_price(
        self,
        machine_model: str,
        configuration: dict[str, Any],
    ) -> dict[str, Any]:
        """Estimate a price range based on historical quotes.

        Returns ``estimated_price`` (low/mid/high), ``confidence``,
        ``similar_quotes``, and ``reasoning``.
        """
        config_text = ", ".join(f"{k}: {v}" for k, v in configuration.items() if v)
        query = f"{machine_model} quote price {config_text}"

        similar = await self._retriever.search_by_category(
            query=query,
            category=_QUOTES_CATEGORY,
            limit=8,
        )

        context_lines = [
            f"Machine: {machine_model}",
            f"Configuration: {config_text}",
            "",
            "SIMILAR HISTORICAL QUOTES:",
        ]
        for i, r in enumerate(similar, 1):
            context_lines.append(f"{i}. {r.get('content', '')[:600]}")

        raw = await self._llm_call(_ESTIMATE_SYSTEM_PROMPT, "\n".join(context_lines))

        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            result = {
                "estimated_price": {"low": 0, "mid": 0, "high": 0, "currency": "USD"},
                "confidence": "low",
                "reasoning": raw,
            }

        result["similar_quotes"] = [
            {"content": r.get("content", "")[:300], "score": r.get("score", 0)} for r in similar[:5]
        ]
        return result

    # ── quote history analytics ──────────────────────────────────────────

    async def analyze_quote_history(
        self,
        filters: dict[str, Any],
    ) -> dict[str, Any]:
        """Aggregate statistics over historical deals.

        Supported filter keys: ``region``, ``machine_model``,
        ``date_from``, ``date_to``, ``stage``.

        Falls back to knowledge-base search when the CRM is not yet
        connected.
        """
        if self._crm is not None:
            return await self._analyze_from_crm(filters)

        return await self._analyze_from_kb(filters)

    async def _analyze_from_crm(self, filters: dict[str, Any]) -> dict[str, Any]:
        assert self._crm is not None
        deals = await self._crm.get_deals_by_filter(filters)
        if not deals:
            return self._empty_analysis()

        values = [d.get("value", 0) for d in deals if d.get("value")]
        won = [d for d in deals if d.get("stage") == "WON"]
        lost = [d for d in deals if d.get("stage") == "LOST"]
        total_decided = len(won) + len(lost)

        return {
            "total_deals": len(deals),
            "total_value": sum(values),
            "average_deal_size": sum(values) / len(values) if values else 0,
            "win_rate": len(won) / total_decided if total_decided else 0,
            "won_count": len(won),
            "lost_count": len(lost),
            "filters_applied": filters,
        }

    async def _analyze_from_kb(self, filters: dict[str, Any]) -> dict[str, Any]:
        filter_text = ", ".join(f"{k}: {v}" for k, v in filters.items() if v)
        query = f"quote history statistics {filter_text}"
        results = await self._retriever.search_by_category(
            query=query,
            category=_QUOTES_CATEGORY,
            limit=10,
        )
        return {
            "source": "knowledge_base",
            "note": "CRM not connected — results are approximate",
            "relevant_excerpts": [r.get("content", "")[:400] for r in results],
            "filters_applied": filters,
        }

    @staticmethod
    def _empty_analysis() -> dict[str, Any]:
        return {
            "total_deals": 0,
            "total_value": 0,
            "average_deal_size": 0,
            "win_rate": 0,
            "won_count": 0,
            "lost_count": 0,
        }

    # ── quote content generation ─────────────────────────────────────────

    @observe()
    async def generate_quote_content(
        self,
        contact: Contact,
        machine_model: str,
        configuration: dict[str, Any],
    ) -> dict[str, Any]:
        """Draft formal quote text for a contact and machine configuration."""
        raw_config = ", ".join(f"{k}: {v}" for k, v in configuration.items() if v)
        config_text = sanitize_for_llm_embedding(raw_config, max_chars=4_000)
        safe_model = sanitize_for_llm_embedding(machine_model, max_chars=500)

        context_lines = [
            f"Contact: {sanitize_for_llm_embedding(contact.name, max_chars=300)} "
            f"({sanitize_for_llm_embedding(contact.email, max_chars=320)})",
            f"Company: {sanitize_for_llm_embedding(contact.company or 'N/A', max_chars=500)}",
            f"Region: {sanitize_for_llm_embedding(contact.region or 'N/A', max_chars=200)}",
            f"Industry: {sanitize_for_llm_embedding(contact.industry or 'N/A', max_chars=200)}",
            "",
            f"Machine: {safe_model}",
            f"Configuration: {config_text}",
        ]

        price_est = await self.estimate_price(machine_model, configuration)
        if price_est.get("estimated_price"):
            ep = price_est["estimated_price"]
            context_lines.append(
                f"Price guidance: {ep.get('currency', 'USD')} "
                f"{ep.get('low', '?')} – {ep.get('high', '?')} "
                f"(mid {ep.get('mid', '?')})"
            )

        raw = await self._llm_call(
            _QUOTE_CONTENT_SYSTEM_PROMPT,
            "\n".join(context_lines),
        )

        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Quote content LLM returned non-JSON; wrapping")
            return {"raw_content": raw}

    # ── LLM helper ───────────────────────────────────────────────────────

    async def _llm_call(self, system: str, user: str) -> str:
        return await self._llm.generate_text(
            system,
            user,
            temperature=0.2,
            name="pricing_engine",
        )
