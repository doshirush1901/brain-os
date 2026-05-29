"""Optional LLM pass: map response snippets to evidence chunk indices."""

from __future__ import annotations

import logging
from typing import Any

from brain_os.exceptions import LLMError
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import CitationAlignmentResult
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

_SYSTEM = load_prompt("citation_aligner")


def _enumerate_evidence(rows: list[dict[str, Any]], *, max_items: int, preview_chars: int) -> str:
    lines: list[str] = []
    for i, r in enumerate((rows or [])[:max_items]):
        if not isinstance(r, dict):
            continue
        src = str(r.get("source") or r.get("id") or "")[:120]
        body = str(r.get("content") or "")[:preview_chars].replace("\n", " ")
        lines.append(f"[{i}] source={src!r} :: {body}")
    return "\n".join(lines) if lines else "(no evidence rows)"


async def align_response_to_evidence(
    response: str,
    evidence_rows: list[dict[str, Any]],
    *,
    max_evidence: int = 16,
    preview_chars: int = 320,
) -> dict[str, Any]:
    """Return JSON-serializable alignment (``spans``) or ``error`` on failure."""
    if not (response or "").strip() or not evidence_rows:
        return {"spans": [], "skipped": True}
    enum_block = _enumerate_evidence(
        evidence_rows, max_items=max_evidence, preview_chars=preview_chars
    )
    user = (
        f"RESPONSE (truncate ok):\n{(response or '')[:12000]}\n\n"
        f"EVIDENCE (use indices 0..n only):\n{enum_block}\n"
    )
    try:
        llm = get_llm_client()
        result = await llm.generate_structured(
            _SYSTEM,
            user,
            CitationAlignmentResult,
            name="citation_aligner.map",
            model_profile="verifier",
        )
        out = []
        n = len(evidence_rows)
        for s in result.spans:
            idx = int(s.evidence_index)
            if 0 <= idx < min(n, max_evidence):
                out.append(
                    {
                        "claim_snippet": (s.claim_snippet or "")[:500],
                        "evidence_index": idx,
                        "confidence": float(s.confidence),
                    }
                )
        return {"spans": out[:12]}
    except (LLMError, ValueError, TypeError, RuntimeError):
        logger.warning("Citation aligner failed", exc_info=True)
        return {"spans": [], "error": "aligner_exception"}
