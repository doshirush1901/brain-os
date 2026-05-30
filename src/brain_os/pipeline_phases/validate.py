"""Phase: validate. Extracted from pipeline.py 2026-05-15.

Phase modules MUST NOT import from brain_os.pipeline (circular). Shared helpers
live in ira.pipeline_runtime; import from there when this slice needs them.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from brain_os.data.models import KnowledgeState
from brain_os.pipeline_phases.execute import (
    run_aegis_dlp_check,
    run_aletheia_compliance_check,
    run_gapper_resolution,
    run_mnemon_correction_check,
)
from brain_os.pipeline_transitions import (
    VALIDATE_SAFETY_STEP_ORDER,
    attach_transition_to_trace,
    decide_post_gapper_aegis,
    record_validate_safety_path,
)
from brain_os.services.degradation import merge_retriever_health_into_trace, record_degradation_event

FAITHFULNESS_SINGLE_AGENT_SKIP = frozenset(
    {"athena", "timeout", "truth_hints", "fast_path", "operator_deterministic", "quick_pipeline"}
)
ASSESS_SINGLE_AGENT_SKIP = frozenset({"athena", "timeout"})


def should_skip_faithfulness_for_gold_eval(resolved_input: str) -> bool:
    """Offline gold-eval threads ship their own evidence; skip KB faithfulness hard-gates."""
    return "[BRAIN_EMAIL_GOLD_EVAL_v1]" in resolved_input


def faithfulness_gate_eligible(
    route_method: str,
    *,
    skip_gold_eval: bool,
    uncensored_turn: bool,
) -> bool:
    """Whether the 6.4 faithfulness gate (retrieval + verifier) should run."""
    if route_method in ("truth_hint", "fast_path", "operator_deterministic", "quick_pipeline"):
        return False
    if skip_gold_eval:
        return False
    if uncensored_turn:
        return False
    return True


def agents_did_non_trivial_work(agents_used: list[str], skip_if_only: frozenset[str]) -> bool:
    """True when more than a trivial single-agent route produced the draft."""
    return len(agents_used) > 1 or (bool(agents_used) and agents_used[0] not in skip_if_only)


def compute_faithfulness_substantive_bypass(
    agents_used: list[str],
    raw_response: str,
    *,
    faith_strict_subject: bool,
) -> tuple[bool, bool, bool]:
    """``(agents_did_real_work, response_is_substantive, substantive_bypass)`` for 6.4."""
    agents_did = agents_did_non_trivial_work(agents_used, FAITHFULNESS_SINGLE_AGENT_SKIP)
    response_substantive = len(raw_response) > 200
    substantive_bypass = agents_did and response_substantive and not faith_strict_subject
    return agents_did, response_substantive, substantive_bypass


CONFIDENCE_FLOOR_FALLBACK_MESSAGE = (
    "I don't have reliable information to answer this "
    "question accurately. I'd recommend checking with "
    "the relevant team or documentation directly."
)


class ConfidenceFloorAction(str, Enum):
    """How the 7.1 confidence floor should adjust the draft."""

    NONE = "none"
    REPLACE_WITH_IDK = "replace"
    PREFIX = "prefix"
    APPEND_CONFLICT_NOTE = "conflict"


def classify_confidence_floor_action(
    *,
    state: KnowledgeState,
    confidence: float,
    confidence_floor: float,
    uncensored_local_llm: bool,
    agents_did_work: bool,
    response_has_substance: bool,
) -> ConfidenceFloorAction:
    """Classify metacognition confidence-floor handling (7.1)."""
    if uncensored_local_llm:
        return ConfidenceFloorAction.NONE
    is_unknown = state == KnowledgeState.UNKNOWN
    is_low_uncertain = state == KnowledgeState.UNCERTAIN and confidence < confidence_floor
    if (is_unknown or is_low_uncertain) and not (agents_did_work and response_has_substance):
        return ConfidenceFloorAction.REPLACE_WITH_IDK
    if (is_unknown or is_low_uncertain) and agents_did_work:
        return ConfidenceFloorAction.PREFIX
    if state == KnowledgeState.CONFLICTING:
        return ConfidenceFloorAction.APPEND_CONFLICT_NOTE
    return ConfidenceFloorAction.NONE


def format_conflict_notes_for_confidence_prefix(
    conflicts: list[Any] | None,
    *,
    max_items: int = 5,
) -> str:
    """Extra prefix text when metacognition reports conflicting evidence."""
    if not conflicts:
        return ""
    lines = "\n".join(f"- {c}" for c in conflicts[:max_items])
    return f"Specific conflicts found:\n{lines}\n\n"


async def run_guardrails_checks(
    *,
    route_method: str,
    raw_response: str,
    trace: dict[str, Any],
    uncensored_turn: bool,
    logger: logging.Logger,
) -> str:
    """Run 6.4b confidentiality/competitor guardrails and return response text."""
    if route_method in ("fast_path",):
        return raw_response

    try:
        from brain_os.brain.guardrails import check_competitor_mentions, check_confidentiality
        from brain_os.config import get_settings as _get_settings_guardrails

        guardrails_fail_closed = _get_settings_guardrails().app.guardrails_fail_closed

        guard_conf, guard_comp = await asyncio.gather(
            check_confidentiality(raw_response, "external"),
            check_competitor_mentions(raw_response),
            return_exceptions=True,
        )
        guardrail_results: dict[str, Any] = {}
        guardrail_violation = False

        if isinstance(guard_conf, Exception):
            logger.warning("GUARDRAILS | confidentiality check failed", exc_info=True)
            if guardrails_fail_closed:
                guardrail_results["confidentiality_check_error"] = str(guard_conf)
                guardrail_violation = True
        elif isinstance(guard_conf, dict) and not guard_conf.get("safe", True):
            guardrail_results["confidentiality"] = guard_conf
            logger.warning(
                "GUARDRAILS | confidential data detected: %s",
                guard_conf.get("leaked_categories"),
            )
            guardrail_violation = True

        if isinstance(guard_comp, Exception):
            logger.warning("GUARDRAILS | competitor check failed", exc_info=True)
            if guardrails_fail_closed:
                guardrail_results["competitor_check_error"] = str(guard_comp)
                guardrail_violation = True
        elif isinstance(guard_comp, dict) and not guard_comp.get("clean", True):
            guardrail_results["competitors"] = guard_comp
            logger.warning(
                "GUARDRAILS | competitor mentions: %s",
                [m["competitor"] for m in guard_comp.get("mentions", [])],
            )
            guardrail_violation = True

        if guardrail_results:
            trace["guardrails"] = guardrail_results

        if guardrails_fail_closed and guardrail_violation and not uncensored_turn:
            raw_response = (
                "I can't provide that response safely. "
                "Please rephrase your request for a policy-compliant summary."
            )
            logger.warning(
                "GUARDRAILS | fail-closed triggered; response replaced with safe fallback",
            )
    except Exception:
        logger.warning("Guardrails checks failed (non-critical)", exc_info=True)

    return raw_response


async def run_assessment_step(
    *,
    metacognition: Any,
    shared_kb_evidence: list[dict[str, Any]],
    run_id: str,
    meta: dict[str, Any],
    retriever: Any,
    resolved_input: str,
    agents_used: list[str],
    raw_response: str,
    uncensored_turn: bool,
    logger: logging.Logger,
) -> tuple[str, str, float | None]:
    """Run step 7 assess + 7.1 confidence floor; return response/prefix/confidence."""
    confidence_prefix = ""
    confidence_value: float | None = None
    if metacognition is None:
        return raw_response, confidence_prefix, confidence_value
    try:
        if shared_kb_evidence:
            kb_results = list(shared_kb_evidence)
        else:
            from brain_os.brain.retrieval_context import with_retrieval_context

            profile = None
            if isinstance(meta, dict) and meta.get("retrieval_profile"):
                profile = str(meta.get("retrieval_profile")).strip() or None
            kb_results = await with_retrieval_context(
                run_id,
                profile,
                retriever.search,
                resolved_input,
                limit=5,
            )
        assessment = await metacognition.assess_knowledge(resolved_input, kb_results)
        confidence_value = float(assessment["confidence"])
        confidence_prefix = metacognition.generate_confidence_prefix(
            assessment["state"],
            assessment["confidence"],
        )

        if assessment.get("gaps"):
            await metacognition.log_knowledge_gap(
                resolved_input,
                assessment["state"],
                assessment["gaps"],
            )

        from brain_os.config import get_settings as _get_settings_assess

        assess_cfg = _get_settings_assess().app
        state = assessment["state"]
        conf = assessment["confidence"]
        agents_did_work = agents_did_non_trivial_work(agents_used, ASSESS_SINGLE_AGENT_SKIP)
        response_has_substance = len(raw_response) > 200

        if uncensored_turn:
            confidence_prefix = ""
        else:
            cf_action = classify_confidence_floor_action(
                state=state,
                confidence=conf,
                confidence_floor=assess_cfg.confidence_floor,
                uncensored_local_llm=False,
                agents_did_work=agents_did_work,
                response_has_substance=response_has_substance,
            )
            if cf_action == ConfidenceFloorAction.REPLACE_WITH_IDK:
                raw_response = CONFIDENCE_FLOOR_FALLBACK_MESSAGE
                logger.info(
                    "CONFIDENCE FLOOR | state=%s confidence=%.2f — replaced response with honest 'I don't know'",
                    state,
                    conf,
                )
            elif cf_action == ConfidenceFloorAction.PREFIX:
                raw_response = confidence_prefix + raw_response
                logger.info(
                    "CONFIDENCE FLOOR | state=%s confidence=%.2f — agents provided substantive response, prefixing instead of replacing",
                    state,
                    conf,
                )
            elif cf_action == ConfidenceFloorAction.APPEND_CONFLICT_NOTE:
                conflicts = assessment.get("conflicts", [])
                extra = format_conflict_notes_for_confidence_prefix(conflicts)
                if extra:
                    confidence_prefix += extra
    except Exception:
        logger.exception("Metacognition assessment failed")
    return raw_response, confidence_prefix, confidence_value


async def run_faithfulness_gate(
    *,
    route_method: str,
    resolved_input: str,
    raw_response: str,
    agents_used: list[str],
    channel: str,
    run_id: str,
    retriever: Any,
    trace: dict[str, Any],
    on_progress: Any | None,
    uncensored_turn: bool,
    faithfulness_strict_intent_fn: Any,
    logger: logging.Logger,
) -> tuple[str, list[dict[str, Any]]]:
    """Run faithfulness retrieval + verification gate and return updated response/evidence."""
    skip_gold_eval = should_skip_faithfulness_for_gold_eval(resolved_input)
    shared_kb_evidence: list[dict[str, Any]] = []

    if uncensored_turn:
        trace["faithfulness_skipped_uncensored_local_llm"] = True

    if not faithfulness_gate_eligible(
        route_method,
        skip_gold_eval=skip_gold_eval,
        uncensored_turn=uncensored_turn,
    ):
        return raw_response, shared_kb_evidence

    from brain_os.brain.pipeline_uncertainty import (
        PARTIAL_VERIFICATION_NOTICE,
        STRICT_VERIFICATION_BLOCKED,
    )
    from brain_os.brain.retrieval_context import with_retrieval_context

    try:
        profile = trace.get("retrieval_profile")
        shared_kb_evidence = await with_retrieval_context(
            run_id,
            profile if isinstance(profile, str) else None,
            retriever.search,
            resolved_input,
            limit=6,
        )
        trace["shared_kb_hits"] = len(shared_kb_evidence)
        merge_retriever_health_into_trace(
            trace,
            getattr(retriever, "_last_search_health", None),
        )
    except Exception:
        logger.warning(
            "Shared KB retrieval for faithfulness/metacognition failed",
            exc_info=True,
        )
        shared_kb_evidence = []
        trace["kb_retrieval_degraded"] = True
        record_degradation_event(trace, layer="kb", code="shared_retrieval_exception")

    try:
        from brain_os.brain.evidence_bundle import (
            apply_bundle_pii,
            bundle_from_rows,
            trim_bundle_by_chars,
        )
        from brain_os.brain.guardrails import check_faithfulness
        from brain_os.config import get_settings as _get_settings

        app_cfg = _get_settings().app
        bundle = bundle_from_rows(resolved_input, shared_kb_evidence)
        bundle, trim_meta = trim_bundle_by_chars(
            bundle,
            max_chars=int(app_cfg.retriever_evidence_max_chars),
        )
        bundle, pii_meta = apply_bundle_pii(
            bundle,
            mode=str(app_cfg.retriever_bundle_pii_mode or "off"),
        )
        trace["evidence_bundle"] = {
            "trim": trim_meta,
            "pii": pii_meta,
            "n_chunks": len(bundle.chunks),
        }
        faith_docs = bundle.texts_for_faithfulness()
        if faith_docs:
            if on_progress:
                await on_progress({"type": "faithfulness_check"})
            faith_result = await check_faithfulness(raw_response, faith_docs)
            trace["faithfulness"] = faith_result.get("score", 1.0)
            faith_score = faith_result.get("score", 1.0)

            faith_strict_subject = getattr(
                app_cfg, "faithfulness_mode", "best_effort"
            ) == "strict" and faithfulness_strict_intent_fn(
                agents_used, resolved_input, channel=channel
            )
            agents_did_real_work, response_is_substantive, substantive_bypass = (
                compute_faithfulness_substantive_bypass(
                    agents_used,
                    raw_response,
                    faith_strict_subject=faith_strict_subject,
                )
            )

            if faith_score < app_cfg.faithfulness_hard_threshold and not substantive_bypass:
                raw_response = STRICT_VERIFICATION_BLOCKED
                trace["faithfulness_blocked"] = True
                trace["uncertainty_contract"] = "strict_block"
                logger.warning(
                    "FAITHFULNESS | score=%.2f < hard threshold %.2f — response blocked",
                    faith_score,
                    app_cfg.faithfulness_hard_threshold,
                )
                try:
                    from brain_os.memory.dream_triggers import bump_event

                    await asyncio.to_thread(bump_event, "faithfulness_block")
                except Exception:
                    logger.debug("dream_triggers bump failed", exc_info=True)
            elif faith_score < app_cfg.faithfulness_threshold or (
                faith_score < app_cfg.faithfulness_hard_threshold and agents_did_real_work
            ):
                raw_response += PARTIAL_VERIFICATION_NOTICE
                trace["partial_verification"] = True
                trace["uncertainty_contract"] = "partial_evidence"
                logger.info(
                    "FAITHFULNESS | score=%.2f < %.2f — caveat appended",
                    faith_score,
                    app_cfg.faithfulness_threshold,
                )
            if getattr(app_cfg, "citation_aligner_enabled", False):
                response_strip = (raw_response or "").strip()
                if response_strip and raw_response != STRICT_VERIFICATION_BLOCKED:
                    try:
                        from brain_os.brain.citation_aligner import align_response_to_evidence

                        ev_rows = [
                            {
                                "content": c.content,
                                "source": c.source,
                                "id": c.metadata.get("id", ""),
                            }
                            for c in bundle.chunks
                        ]
                        trace["citation_alignment"] = await align_response_to_evidence(
                            raw_response,
                            ev_rows,
                        )
                    except Exception:
                        logger.debug("citation_alignment failed", exc_info=True)
        else:
            trace["faithfulness"] = 0.0
            trace["faithfulness_no_docs"] = True
            raw_response += PARTIAL_VERIFICATION_NOTICE
            trace["partial_verification"] = True
            trace["uncertainty_contract"] = "no_evidence"
            logger.warning("FAITHFULNESS | no evidence docs available — caveat appended")
    except Exception:
        try:
            from brain_os.config import get_settings as _gf_err

            fca = _gf_err().app
            strict_on_error = getattr(
                fca, "faithfulness_mode", "best_effort"
            ) == "strict" and faithfulness_strict_intent_fn(
                agents_used, resolved_input, channel=channel
            )
        except Exception:
            strict_on_error = False

        if strict_on_error and not uncensored_turn:
            raw_response = STRICT_VERIFICATION_BLOCKED
            trace["faithfulness_blocked"] = True
            trace["faithfulness_strict_error"] = True
            trace["uncertainty_contract"] = "strict_block"
            logger.warning(
                "FAITHFULNESS | strict intent + verifier error — refusing unverified answer",
                exc_info=True,
            )
        else:
            logger.warning(
                "Faithfulness gate failed — continuing with unverified response",
                exc_info=True,
            )

    return raw_response, shared_kb_evidence


@dataclass
class ValidateSafetyResult:
    """Outputs from the post-EXECUTE compliance → DLP → Mnemon → Gapper chain."""

    raw_response: str
    agents_used: list[str]
    had_provenance: bool
    had_dlp: bool


async def run_validate_safety_chain(
    *,
    raw_response: str,
    agents_used: list[str],
    resolved_input: str,
    channel: str = "",
    strict_full_workflow: bool = False,
    trace: dict[str, Any],
    pantheon: Any,
    on_progress: Any | None,
    logger: logging.Logger,
    record_compliance_stage_fn: Any | None = None,
    record_dlp_stage_fn: Any | None = None,
) -> ValidateSafetyResult:
    """Run Aletheia → Aegis → Mnemon → Gapper (+ conditional post-Gapper Aegis)."""
    steps: list[str] = []
    had_provenance = False
    had_dlp = False

    aletheia = pantheon.get_agent("aletheia")
    raw_response, agents_used, step_prov = await run_aletheia_compliance_check(
        raw_response,
        agents_used,
        aletheia=aletheia,
        logger=logger,
    )
    had_provenance = had_provenance or step_prov
    steps.append("aletheia")
    if record_compliance_stage_fn is not None:
        record_compliance_stage_fn()

    aegis = pantheon.get_agent("aegis")
    raw_response, agents_used, step_dlp = await run_aegis_dlp_check(
        raw_response,
        agents_used,
        aegis=aegis,
        logger=logger,
    )
    had_dlp = had_dlp or step_dlp
    steps.append("aegis")
    if record_dlp_stage_fn is not None:
        record_dlp_stage_fn()

    mnemon = pantheon.get_agent("mnemon")
    raw_response, agents_used = await run_mnemon_correction_check(
        raw_response,
        agents_used,
        mnemon=mnemon,
        logger=logger,
    )
    steps.append("mnemon")

    gapper = pantheon.get_agent("gapper")
    text_before_gapper = raw_response
    raw_response, agents_used = await run_gapper_resolution(
        raw_response,
        agents_used,
        resolved_input=resolved_input,
        gapper=gapper,
        on_progress=on_progress,
        logger=logger,
        channel=channel,
        strict_full_workflow=strict_full_workflow,
    )
    steps.append("gapper")
    gapper_mutated = raw_response != text_before_gapper

    post_gapper = decide_post_gapper_aegis(
        text_mutated=gapper_mutated,
        aegis_available=aegis is not None,
    )
    post_gapper_ran = False
    if post_gapper is not None:
        attach_transition_to_trace(trace, post_gapper, phase="post_gapper_aegis")
        raw_response, agents_used, step_dlp_post = await run_aegis_dlp_check(
            raw_response,
            agents_used,
            aegis=aegis,
            logger=logger,
            post_gapper=True,
            append_agent=False,
        )
        had_dlp = had_dlp or step_dlp_post
        steps.append("aegis_post_gapper")
        post_gapper_ran = True

    record_validate_safety_path(
        trace,
        steps=steps,
        gapper_mutated=gapper_mutated,
        post_gapper_aegis=post_gapper_ran,
    )
    base = list(VALIDATE_SAFETY_STEP_ORDER)
    if steps[: len(base)] != base:
        msg = f"unexpected validate safety order: {steps}"
        raise RuntimeError(msg)
    if post_gapper_ran and steps[len(base) : len(base) + 1] != ["aegis_post_gapper"]:
        msg = f"post-Gapper Aegis step missing after order check: {steps}"
        raise RuntimeError(msg)

    return ValidateSafetyResult(
        raw_response=raw_response,
        agents_used=agents_used,
        had_provenance=had_provenance,
        had_dlp=had_dlp,
    )


__all__ = [
    "ASSESS_SINGLE_AGENT_SKIP",
    "CONFIDENCE_FLOOR_FALLBACK_MESSAGE",
    "FAITHFULNESS_SINGLE_AGENT_SKIP",
    "ConfidenceFloorAction",
    "ValidateSafetyResult",
    "agents_did_non_trivial_work",
    "classify_confidence_floor_action",
    "compute_faithfulness_substantive_bypass",
    "faithfulness_gate_eligible",
    "format_conflict_notes_for_confidence_prefix",
    "run_assessment_step",
    "run_faithfulness_gate",
    "run_guardrails_checks",
    "run_validate_safety_chain",
    "should_skip_faithfulness_for_gold_eval",
]
