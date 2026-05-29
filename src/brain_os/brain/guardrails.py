"""Guardrails AI integration for semantic validation.

Provides validators that Vera (fact-checker) and Sphinx (gatekeeper) can
use to programmatically check LLM outputs for common issues:

- PII/sensitive data leakage
- Hallucinated or unsupported claims
- Toxic or unprofessional language
- Competitor praise in external communications
- Confidential data leakage (pricing, margins, HR)

These complement the LLM-based reasoning in the ReAct loop with
deterministic, fast validation checks.

Faithfulness checking uses a four-tier strategy:
0. **Google Check Grounding** — Discovery Engine API, <700ms, per-claim citations
1. **Dual-model LLM** — OpenAI + Anthropic in parallel, averaged (~$0.006/check)
2. **Single-model LLM** — OpenAI only fallback (~$0.003/check)
3. **Keyword heuristic** — word-overlap fallback when all else fails
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from brain_os.exceptions import LLMError
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import ConfidentialityResult, FaithfulnessResult

logger = logging.getLogger(__name__)

_guard_instance = None
_hhem_model: Any | None = None
_hhem_load_attempted = False

_FAITHFULNESS_SYSTEM_PROMPT = load_prompt("faithfulness_check")


def _get_hhem() -> Any | None:
    """Compatibility shim for optional HHEM faithfulness model.

    We keep this API because tests and older call paths still patch/check it.
    """
    global _hhem_model, _hhem_load_attempted
    if _hhem_load_attempted:
        return _hhem_model
    _hhem_load_attempted = True
    _hhem_model = None
    return _hhem_model


async def _hhem_faithfulness(
    response: str,
    context_docs: list[str],
) -> dict[str, Any] | None:
    """Optional HHEM scorer. Returns None when unavailable."""
    _ = _get_hhem()
    if _ is None:
        return None
    # Reserved for environments where HHEM is installed/wired.
    return None


KNOWN_COMPETITORS: list[str] = [
    "ILLIG",
    "Kiefel",
    "GN Thermoforming",
    "WM Thermoforming",
    "Gabler",
    "Multivac",
    "AMUT",
    "Cannon",
]

_CONFIDENTIAL_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:margin|markup)\s*(?:is|of|at|:)?\s*\d+", "internal_margin"),
    (
        r"\b(?:cost\s*price|COGS|cost\s*of\s*goods)\s*(?:is|of|at|:)?\s*(?:EUR|USD|\$|€)\s*[\d,]+",
        "cost_price",
    ),
    (
        r"\b(?:salary|compensation|CTC|take[\s-]?home)\s*(?:is|of|at|:)?\s*(?:EUR|USD|INR|\$|€|₹)\s*[\d,]+",
        "hr_salary",
    ),
    (r"\b(?:employee\s*ID|emp[\s-]?ID)\s*(?:is|of|:)?\s*\w+", "hr_employee_id"),
    (
        r"\b(?:vendor\s*price|supplier\s*cost|purchase\s*price)\s*(?:is|of|at|:)?\s*(?:EUR|USD|\$|€)\s*[\d,]+",
        "vendor_pricing",
    ),
]


def _get_guard():
    """Lazy-load the Guardrails validator chain."""
    global _guard_instance
    if _guard_instance is not None:
        return _guard_instance

    try:
        from guardrails import Guard

        _guard_instance = Guard(name="ira_output_guard").use_many(_load_validators())
        logger.info("Guardrails AI loaded with validators")
        return _guard_instance
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
        logger.debug("Guardrails AI not available", exc_info=True)
        return None


def _load_validators() -> list:
    """Load available Guardrails validators."""
    validators = []

    try:
        from guardrails.hub import DetectPII

        validators.append(
            DetectPII(
                pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN"],
                on_fail="fix",
            )
        )
    except (ImportError, OSError, RuntimeError, AttributeError, ValueError, TypeError):
        logger.debug("DetectPII validator not available")

    try:
        from guardrails.hub import ToxicLanguage

        validators.append(
            ToxicLanguage(
                threshold=0.7,
                on_fail="noop",
            )
        )
    except (ImportError, OSError, RuntimeError, AttributeError, ValueError, TypeError):
        logger.debug("ToxicLanguage validator not available")

    return validators


async def validate_output(text: str) -> dict[str, Any]:
    """Run Guardrails validators on a text output.

    Returns a dict with:
    - ``valid``: bool — whether all checks passed
    - ``issues``: list[str] — descriptions of any issues found
    - ``sanitized``: str — the text with PII redacted (if applicable)
    """
    guard = _get_guard()
    if guard is None:
        return {"valid": True, "issues": [], "sanitized": text}

    try:
        result = await asyncio.to_thread(guard.validate, text)
        issues = []

        if result.validation_passed is False:
            for log in result.error_spans or []:
                issues.append(f"{log.reason}: '{log.text[:100]}'")

        return {
            "valid": result.validation_passed,
            "issues": issues,
            "sanitized": str(result.validated_output) if result.validated_output else text,
        }
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Guardrails validation failed: %s", exc)
        return {"valid": True, "issues": [], "sanitized": text}


async def _google_check_grounding(
    response: str,
    context_docs: list[str],
) -> dict[str, Any] | None:
    """Tier 0: Google Discovery Engine Check Grounding API.

    Returns a faithfulness dict or None if unavailable.  Sub-700ms,
    per-claim citations, purpose-built for RAG grounding.
    """
    from brain_os.config import get_settings

    project_id = get_settings().document_ai.project_id
    if not project_id:
        return None

    try:
        from pathlib import Path

        import httpx
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        token_path = Path("token_dlp.json")
        if not token_path.exists():
            return None

        creds = Credentials.from_authorized_user_file(
            str(token_path),
            ["https://www.googleapis.com/auth/cloud-platform"],
        )
        if creds.expired and creds.refresh_token:
            await asyncio.to_thread(creds.refresh, Request())
        if not creds.valid:
            return None

        url = (
            f"https://discoveryengine.googleapis.com/v1/projects/{project_id}"
            "/locations/global/groundingConfigs/default_grounding_config:check"
        )
        facts = [{"factText": doc[:10000]} for doc in context_docs[:200]]
        payload = {
            "answerCandidate": response[:16000],
            "facts": facts,
            "groundingSpec": {
                "citationThreshold": 0.5,
                "enableClaimLevelScore": True,
            },
        }

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {creds.token}",
                    "Content-Type": "application/json",
                    "X-Goog-User-Project": project_id,
                },
            )

        if resp.status_code != 200:
            logger.debug(
                "Google Check Grounding returned %d: %s", resp.status_code, resp.text[:200]
            )
            return None

        data = resp.json()
        support_score = float(data.get("supportScore", 0))
        claims = data.get("claims", [])

        unsupported: list[dict[str, str]] = []
        for claim in claims:
            claim_score = float(claim.get("score", 0))
            if claim.get("groundingCheckRequired", True) and claim_score < 0.5:
                unsupported.append(
                    {
                        "claim": claim.get("claimText", ""),
                        "reason": f"Google grounding score {claim_score:.2f} < 0.5",
                    }
                )

        faithful = support_score >= 0.5 and len(unsupported) == 0
        logger.debug(
            "Google Check Grounding: score=%.2f claims=%d unsupported=%d",
            support_score,
            len(claims),
            len(unsupported),
        )
        return {
            "faithful": faithful,
            "unsupported_claims": unsupported[:5],
            "score": round(support_score, 2),
        }

    except (
        TimeoutError,
        httpx.HTTPError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        KeyError,
        ImportError,
        AttributeError,
    ) as exc:
        logger.debug("Google Check Grounding failed", exc_info=True)
        return None


async def check_faithfulness(
    response: str,
    context_docs: list[str],
) -> dict[str, Any]:
    """Check whether a response is faithful to the provided context documents.

    Four-tier strategy:
    0. **Google Check Grounding** — Discovery Engine API, <700ms, per-claim citations
    1. **Dual-model LLM** — OpenAI + Anthropic in parallel, averaged (~$0.006/check)
    2. **Single-model LLM** — OpenAI only fallback (~$0.003/check)
    3. **Keyword heuristic** — word-overlap fallback when all else fails

    Returns:
    - ``faithful``: bool
    - ``unsupported_claims``: list[dict] — claims not found in context
    - ``score``: float — 0.0 to 1.0 faithfulness score
    """
    if not response.strip():
        return {"faithful": False, "unsupported_claims": [], "score": 0.0}
    if not context_docs:
        return {
            "faithful": False,
            "unsupported_claims": [
                {
                    "claim": "No source context was available to verify this response.",
                    "reason": "missing_context_docs",
                }
            ],
            "score": 0.0,
        }

    from brain_os.deployment_profile import faithfulness_heuristic_only

    if faithfulness_heuristic_only():
        return _heuristic_faithfulness(response, context_docs)

    # Optional HHEM pass (if available) takes precedence.
    hhem_result = await _hhem_faithfulness(response, context_docs)
    if hhem_result is not None:
        return hhem_result

    # Tier 0: Google Check Grounding (fastest, purpose-built)
    google_result = await _google_check_grounding(response, context_docs)
    if google_result is not None:
        logger.debug("Faithfulness scored by Google Check Grounding: %.2f", google_result["score"])
        return google_result

    context_text = "\n---\n".join(context_docs)
    user_msg = f"Response:\n{response}\n\nSource Context:\n{context_text}"

    # Tier 1: Dual-model verification (OpenAI + Anthropic in parallel)
    try:
        from brain_os.services.llm_client import get_llm_client

        llm = get_llm_client()

        openai_task = llm.generate_structured(
            _FAITHFULNESS_SYSTEM_PROMPT,
            user_msg,
            FaithfulnessResult,
            name="guardrails.faithfulness.openai",
            provider="openai",
            model_profile="verifier",
        )
        anthropic_task = llm.generate_structured(
            _FAITHFULNESS_SYSTEM_PROMPT,
            user_msg,
            FaithfulnessResult,
            name="guardrails.faithfulness.anthropic",
            provider="anthropic",
            model_profile="verifier",
        )

        results = await asyncio.gather(openai_task, anthropic_task, return_exceptions=True)

        valid_results: list[FaithfulnessResult] = [
            r for r in results if isinstance(r, FaithfulnessResult)
        ]

        if len(valid_results) == 2:
            avg_score = round((valid_results[0].score + valid_results[1].score) / 2, 2)
            all_unsupported = (
                valid_results[0].unsupported_claims + valid_results[1].unsupported_claims
            )
            seen_claims: set[str] = set()
            deduped: list[dict[str, str]] = []
            for c in all_unsupported:
                if c.claim not in seen_claims:
                    seen_claims.add(c.claim)
                    deduped.append(c.model_dump())
            faithful = avg_score >= 0.7 and all(r.faithful for r in valid_results)
            logger.debug(
                "Faithfulness dual-model: openai=%.2f anthropic=%.2f avg=%.2f",
                valid_results[0].score,
                valid_results[1].score,
                avg_score,
            )
            return {"faithful": faithful, "unsupported_claims": deduped[:5], "score": avg_score}

        if valid_results:
            r = valid_results[0]
            logger.debug("Faithfulness single-model fallback: %.2f", r.score)
            return {
                "faithful": r.faithful,
                "unsupported_claims": [c.model_dump() for c in r.unsupported_claims],
                "score": round(r.score, 2),
            }

        for r in results:
            if isinstance(r, Exception):
                logger.debug("Faithfulness model failed: %s", r)

    except (
        TimeoutError,
        LLMError,
        httpx.HTTPError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
    ) as exc:
        logger.debug("Dual-model faithfulness check failed, trying single", exc_info=True)

    # Tier 2: Single-model LLM fallback (OpenAI only)
    try:
        from brain_os.services.llm_client import get_llm_client

        llm = get_llm_client()
        result = await llm.generate_structured(
            _FAITHFULNESS_SYSTEM_PROMPT,
            user_msg,
            FaithfulnessResult,
            name="guardrails.faithfulness",
            provider="openai",
            model_profile="verifier",
        )
        score = round(result.score, 2)
        logger.debug("Faithfulness scored by single LLM: %.2f", score)
        return {
            "faithful": result.faithful,
            "unsupported_claims": [c.model_dump() for c in result.unsupported_claims],
            "score": score,
        }
    except (
        TimeoutError,
        LLMError,
        httpx.HTTPError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
    ) as exc:
        logger.debug("Single-model faithfulness also failed, using heuristic", exc_info=True)

    # Tier 3: keyword-overlap heuristic (last resort)
    return _heuristic_faithfulness(response, context_docs)


def _heuristic_faithfulness(
    response: str,
    context_docs: list[str],
) -> dict[str, Any]:
    """Keyword-overlap fallback for faithfulness checking."""
    context_text = " ".join(context_docs).lower()
    sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 20]

    if not sentences:
        return {"faithful": True, "unsupported_claims": [], "score": 1.0}

    supported = 0
    unsupported: list[dict[str, str]] = []

    for sentence in sentences:
        words = set(sentence.lower().split())
        significant_words = {w for w in words if len(w) > 3}
        if not significant_words:
            supported += 1
            continue

        overlap = sum(1 for w in significant_words if w in context_text)
        ratio = overlap / len(significant_words) if significant_words else 0

        if ratio >= 0.3:
            supported += 1
        else:
            unsupported.append({"claim": sentence, "reason": "Low keyword overlap with context"})

    score = supported / len(sentences) if sentences else 1.0

    return {
        "faithful": score >= 0.7,
        "unsupported_claims": unsupported[:5],
        "score": round(score, 2),
    }


async def check_competitor_mentions(
    text: str,
    competitors: list[str] | None = None,
) -> dict[str, Any]:
    """Flag responses that mention or praise competitors.

    Returns:
    - ``clean``: bool — True if no competitor issues found
    - ``mentions``: list[dict] — competitor mentions with context
    """
    competitor_list = competitors or KNOWN_COMPETITORS
    text_lower = text.lower()
    mentions: list[dict[str, str]] = []

    for competitor in competitor_list:
        comp_lower = competitor.lower()
        if comp_lower in text_lower:
            idx = text_lower.index(comp_lower)
            start = max(0, idx - 50)
            end = min(len(text), idx + len(competitor) + 50)
            snippet = text[start:end].strip()
            mentions.append(
                {
                    "competitor": competitor,
                    "context": snippet,
                }
            )

    return {
        "clean": len(mentions) == 0,
        "mentions": mentions,
    }


async def check_confidentiality(
    text: str,
    direction: str = "external",
) -> dict[str, Any]:
    """Detect internal pricing, margin, or HR data in external-facing responses.

    Uses regex patterns for fast detection and optionally an LLM for
    nuanced classification.

    Args:
        text: The response text to check.
        direction: ``"external"`` (strict) or ``"internal"`` (lenient).

    Returns:
    - ``safe``: bool — True if no confidential data detected
    - ``leaked_categories``: list[str] — types of data found
    - ``flagged_snippets``: list[str] — the offending text fragments
    """
    if direction == "internal":
        return {"safe": True, "leaked_categories": [], "flagged_snippets": []}

    leaked_categories: list[str] = []
    flagged_snippets: list[str] = []

    for pattern, category in _CONFIDENTIAL_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            if category not in leaked_categories:
                leaked_categories.append(category)
            for match in matches[:3]:
                flagged_snippets.append(match)

    if not leaked_categories:
        return {"safe": True, "leaked_categories": [], "flagged_snippets": []}

    try:
        from brain_os.services.llm_client import get_llm_client

        llm = get_llm_client()
        result = await llm.generate_structured(
            "You are a data loss prevention checker for Acme Corp. "
            "Determine if the flagged snippets contain genuinely confidential "
            "internal data (margins, cost prices, salaries, vendor pricing) "
            "or if they are publicly available information (list prices, "
            "published specs). Return safe=true if all snippets are public.",
            f"Text: {text[:4000]}\n\nFlagged snippets: {flagged_snippets}",
            ConfidentialityResult,
            name="guardrails.confidentiality",
        )
        return {
            "safe": result.safe,
            "leaked_categories": result.leaked_categories or leaked_categories,
            "flagged_snippets": result.flagged_snippets or flagged_snippets,
        }
    except (
        TimeoutError,
        LLMError,
        httpx.HTTPError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
    ) as exc:
        logger.debug("LLM confidentiality check failed, using regex results", exc_info=True)

    return {
        "safe": False,
        "leaked_categories": leaked_categories,
        "flagged_snippets": flagged_snippets[:5],
    }
