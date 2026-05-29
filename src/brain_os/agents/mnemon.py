"""Mnemon — The Memory Guardian.

Maintains a correction ledger as the single source of truth for all
user corrections.  Intercepts stale data at every retrieval point
(pipeline responses, retriever results, Alexandros file reads) and
overrides contradictions with the corrected facts.

Also flags data that appears outdated based on source timestamps,
even when no explicit correction exists.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles
import httpx

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.exceptions import LLMError
from brain_os.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("mnemon_system")
_LEDGER_PATH = Path("data/brain/correction_ledger.json")
_STALENESS_MONTHS = 3
_DEFAULT_FACT_TYPE = "general"
_DEFAULT_SCOPE = "global"
_DEFAULT_PRECEDENCE = 100
_EMBEDDING_ERRORS = (
    LLMError,
    httpx.HTTPError,
    asyncio.TimeoutError,
    OSError,
    ImportError,
    RuntimeError,
    ValueError,
)


async def _load_ledger() -> dict[str, Any]:
    """Load the correction ledger from disk."""
    if not _LEDGER_PATH.exists():
        return {"entities": {}, "_metadata": {"last_updated": ""}}
    try:
        async with aiofiles.open(_LEDGER_PATH, encoding="utf-8") as f:
            raw = await f.read()
        return json.loads(raw)
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read correction ledger; starting fresh")
        return {"entities": {}, "_metadata": {"last_updated": ""}}


async def _save_ledger(ledger: dict[str, Any]) -> None:
    """Persist the correction ledger to disk."""
    ledger["_metadata"]["last_updated"] = datetime.now(UTC).strftime("%Y-%m-%d")
    _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(ledger, indent=2, ensure_ascii=False) + "\n"
    async with aiofiles.open(_LEDGER_PATH, mode="w", encoding="utf-8") as f:
        await f.write(payload)


def _normalize_ledger_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize old/new ledger formats to canonical correction schema."""
    current_status = str(entry.get("current_status") or entry.get("correct_value") or "").strip()
    stale_values = entry.get("stale_values", [])
    if isinstance(stale_values, str):
        stale_values = [stale_values]
    stale_values = [str(v).strip() for v in stale_values if str(v).strip()]

    return {
        "correct_value": current_status,
        "current_status": current_status,
        "stale_values": stale_values,
        "fact_type": str(entry.get("fact_type") or _DEFAULT_FACT_TYPE),
        "effective_from": entry.get("effective_from"),
        "scope": str(entry.get("scope") or _DEFAULT_SCOPE),
        "precedence": int(entry.get("precedence") or _DEFAULT_PRECEDENCE),
        "version": int(entry.get("version") or 1),
        "corrected_at": entry.get("corrected_at"),
        "source": str(entry.get("source") or "unknown"),
    }


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class Mnemon(BaseAgent):
    name = "mnemon"
    role = "Memory Guardian"
    description = (
        "Correction authority agent. Maintains the correction ledger and "
        "intercepts stale data at every retrieval point, overriding it "
        "with the corrected truth."
    )
    knowledge_categories = [
        "company_internal",
        "sales_and_crm",
        "project_case_studies",
    ]

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="lookup_correction",
                description="Check if an entity has a correction in the ledger.",
                parameters={"entity": "Company name or topic to look up"},
                handler=self._tool_lookup_correction,
            )
        )
        self.register_tool(
            AgentTool(
                name="record_correction",
                description="Record a new correction in the ledger.",
                parameters={
                    "entity": "Company name or topic",
                    "current_status": "The corrected, current truth",
                    "stale_values": "Comma-separated list of old/wrong values to watch for",
                },
                handler=self._tool_record_correction,
            )
        )
        self.register_tool(
            AgentTool(
                name="list_all_corrections",
                description="List all entities in the correction ledger.",
                parameters={},
                handler=self._tool_list_all,
            )
        )
        self.register_tool(
            AgentTool(
                name="validate_correction_consistency",
                description="Check whether a statement conflicts with known corrections.",
                parameters={
                    "statement": "Statement to validate against correction truth",
                    "ledger_context": "Optional correction context",
                },
                handler=self._tool_validate_correction_consistency,
            )
        )
        self.register_tool(
            AgentTool(
                name="audit_decision_log",
                description="Generate a decision trace for correction overrides and risk review.",
                parameters={
                    "decision": "Decision/correction output text",
                    "evidence": "Optional evidence used in correction",
                },
                handler=self._tool_audit_decision_log,
            )
        )

    async def _tool_lookup_correction(self, entity: str) -> str:
        ledger = await _load_ledger()
        entry = ledger["entities"].get(entity.lower())
        if not entry:
            for key, val in ledger["entities"].items():
                if entity.lower() in key:
                    entry = val
                    break
        if not entry:
            return f"No correction found for '{entity}'."
        return json.dumps(_normalize_ledger_entry(entry), default=str)

    async def _tool_record_correction(
        self,
        entity: str,
        current_status: str,
        stale_values: str = "",
    ) -> str:
        stale_list = [v.strip() for v in stale_values.split(",") if v.strip()]
        await self.record_correction(entity, current_status, stale_list)
        return f"Correction recorded for '{entity}'."

    async def _tool_list_all(self) -> str:
        ledger = await _load_ledger()
        entities = ledger.get("entities", {})
        if not entities:
            return "Correction ledger is empty."
        lines = [f"Correction ledger ({len(entities)} entities):"]
        for key, val in entities.items():
            lines.append(f"- **{key}**: {val.get('current_status', '')[:120]}")
        return "\n".join(lines)

    async def _tool_validate_correction_consistency(
        self,
        statement: str,
        ledger_context: str = "",
    ) -> str:
        if not ledger_context:
            ledger = await _load_ledger()
            ledger_context = json.dumps(ledger.get("entities", {}), default=str)[:4000]
        return await self.use_skill(
            "validate_correction_consistency",
            statement=statement,
            ledger_context=ledger_context,
        )

    async def _tool_audit_decision_log(self, decision: str, evidence: str = "") -> str:
        return await self.use_skill(
            "audit_decision_log",
            decision=decision,
            evidence=evidence,
        )

    # ── core methods (called from pipeline, retriever, alexandros) ────────

    _entity_embeddings_cache: dict[str, list[float]] = {}

    async def check_and_correct(self, text: str) -> str:
        """Scan text for entities with known corrections and override stale values.

        Primary path is fast string matching.  When ``mnemon_semantic_check``
        is enabled in config and string matching finds nothing, a secondary
        embedding-based pass catches paraphrased references to corrected
        entities.
        """
        if not text:
            return text

        ledger = await _load_ledger()
        entities = ledger.get("entities", {})
        if not entities:
            return text

        corrections_applied: list[str] = []
        text_lower = text.lower()
        corrected_text = text

        for entity_key, entry in entities.items():
            normalized = _normalize_ledger_entry(entry)
            if entity_key not in text_lower:
                continue

            for stale in normalized.get("stale_values", []):
                if stale.lower() in text_lower:
                    # Hard override stale values in-place so stale truth is not left as-is.
                    corrected_text = corrected_text.replace(stale, normalized["correct_value"])
                    corrections_applied.append(
                        f"[CORRECTION] {entity_key}: "
                        f"'{stale}' is outdated. "
                        f"Current status: {normalized['correct_value']}"
                    )

        if not corrections_applied:
            corrections_applied = await self._semantic_check(text, entities)

        if not corrections_applied:
            return corrected_text

        correction_block = (
            "\n\n--- MNEMON CORRECTIONS (override stale data above) ---\n"
            + "\n".join(corrections_applied)
            + "\n--- END CORRECTIONS ---"
        )
        logger.info(
            "Mnemon applied %d corrections to response",
            len(corrections_applied),
        )
        return corrected_text + correction_block

    async def _semantic_check(
        self,
        text: str,
        entities: dict[str, Any],
    ) -> list[str]:
        """Embedding-based fallback when string matching finds nothing.

        Only runs when ``mnemon_semantic_check`` is enabled in config.
        Compares text segments against entity descriptions using cosine
        similarity.
        """
        from brain_os.config import get_settings

        if not get_settings().app.mnemon_semantic_check:
            return []

        try:
            from brain_os.brain.embeddings import EmbeddingService
        except ImportError:
            return []

        try:
            embedder = EmbeddingService()
        except _EMBEDDING_ERRORS:
            logger.debug("EmbeddingService unavailable for semantic check")
            return []

        _SIMILARITY_THRESHOLD = 0.75

        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        if not sentences:
            return []

        try:
            text_vecs = await embedder.embed_texts(sentences[:20])
        except _EMBEDDING_ERRORS:
            logger.debug("Embedding failed in Mnemon semantic check", exc_info=True)
            return []

        entity_keys = list(entities.keys())
        entity_descs = [f"{k}: {entities[k].get('current_status', '')}" for k in entity_keys]

        uncached_keys: list[int] = []
        entity_vecs: list[list[float] | None] = [None] * len(entity_keys)
        for i, k in enumerate(entity_keys):
            cached = self._entity_embeddings_cache.get(k)
            if cached is not None:
                entity_vecs[i] = cached
            else:
                uncached_keys.append(i)

        if uncached_keys:
            try:
                new_vecs = await embedder.embed_texts(
                    [entity_descs[i] for i in uncached_keys],
                )
                for idx, vec in zip(uncached_keys, new_vecs):
                    entity_vecs[idx] = vec
                    self._entity_embeddings_cache[entity_keys[idx]] = vec
            except _EMBEDDING_ERRORS:
                logger.debug("Entity embedding failed", exc_info=True)
                return []

        corrections: list[str] = []
        for ek_idx, ek in enumerate(entity_keys):
            ev = entity_vecs[ek_idx]
            if ev is None:
                continue
            for tv in text_vecs:
                sim = _cosine_similarity(ev, tv)
                if sim >= _SIMILARITY_THRESHOLD:
                    entry = _normalize_ledger_entry(entities[ek])
                    corrections.append(
                        f"[CORRECTION-SEMANTIC] {ek}: Current status: {entry['correct_value']}"
                    )
                    break

        if corrections:
            logger.info(
                "Mnemon semantic check found %d corrections",
                len(corrections),
            )
        return corrections

    async def flag_staleness(self, text: str, source_date: str | None = None) -> str:
        """Append a staleness warning if the source data is old."""
        if not source_date:
            return text
        try:
            src = datetime.fromisoformat(source_date.replace("Z", "+00:00"))
            if src.tzinfo is None:
                src = src.replace(tzinfo=UTC)
            age_days = (datetime.now(UTC) - src).days
            if age_days > _STALENESS_MONTHS * 30:
                months = age_days // 30
                text += (
                    f"\n\n[STALENESS WARNING] This data is from {source_date} "
                    f"({months} months old). It may be outdated. "
                    f"Cross-reference with recent emails or corrections."
                )
        except (ValueError, TypeError):
            pass
        return text

    async def record_correction(
        self,
        entity: str,
        current_status: str,
        stale_values: list[str] | None = None,
        source: str = "user_correction",
    ) -> None:
        """Add or update an entry in the correction ledger."""
        ledger = await _load_ledger()
        key = entity.lower().strip()

        existing = ledger["entities"].get(key, {})
        old_stale = _normalize_ledger_entry(existing).get("stale_values", [])
        merged_stale = list(set(old_stale + (stale_values or [])))

        ledger["entities"][key] = {
            "correct_value": current_status,
            "current_status": current_status,
            "stale_values": merged_stale,
            "fact_type": _DEFAULT_FACT_TYPE,
            "effective_from": datetime.now(UTC).date().isoformat(),
            "scope": _DEFAULT_SCOPE,
            "precedence": _DEFAULT_PRECEDENCE,
            "version": int(existing.get("version", 0)) + 1,
            "corrected_at": datetime.now(UTC).strftime("%Y-%m-%d"),
            "source": source,
        }

        await _save_ledger(ledger)
        logger.info("Mnemon: recorded correction for '%s'", key)

    async def record_correction_from_feedback(
        self,
        previous_query: str,
        correction_text: str,
    ) -> None:
        """Extract entity and status from a feedback correction and record it.

        Uses simple heuristics to parse the correction text.  For complex
        corrections, the LLM-based handle() method is more appropriate.
        """
        entity = previous_query[:100].strip()
        words = entity.split()
        if len(words) > 5:
            entity = " ".join(words[:5])

        await self.record_correction(
            entity=entity,
            current_status=correction_text[:500],
            source="feedback",
        )

    # ── agent handle (for delegation from Athena) ─────────────────────────

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        return await self.run(query, ctx, system_prompt=_SYSTEM_PROMPT)
