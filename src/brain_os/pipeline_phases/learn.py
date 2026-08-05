"""Phase: learn (pipeline step 10). Extracted from pipeline.py 2026-05-17.

Records interactions across conversation memory, Mem0, Qdrant, Neo4j, CRM,
goals, procedural memory, and background reflection. Must not import from
``brain_os.pipeline``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from brain_os.brain.write_contract import (
    build_receipt,
    enqueue_retry,
    record_receipt,
)
from brain_os.data.models import Channel, Direction, KnowledgeItem
from brain_os.exceptions import DatabaseError, BrainOSError, LLMError, ToolExecutionError
from brain_os.pipeline_runtime import _extract_teaching_facts

logger = logging.getLogger(__name__)


@dataclass
class PipelineLearnDeps:
    """Subsystem handles required for LEARN (step 10)."""

    conversation: Any
    relationship: Any | None
    episodic: Any | None
    memory_blocks: Any | None
    power_level_tracker: Any | None
    long_term: Any | None
    qdrant: Any | None
    graph: Any | None
    crm: Any | None
    musculoskeletal: Any | None
    goals: Any | None
    procedural: Any | None
    pantheon: Any
    realtime_observer: Any | None
    endocrine: Any | None
    unified_ctx: Any | None
    learn_metrics: dict[str, Any]


async def learn_pipeline_turn(
    deps: PipelineLearnDeps,
    *,
    contact_email: str,
    channel: str,
    raw_input: str,
    raw_response: str,
    route_method: str,
    agents_used: list[str],
    active_goal: Any | None,
    resolved_input: str,
    run_id: str,
) -> None:
    """Step 10: record the interaction across all memory and tracking systems."""
    learn_t0 = time.monotonic()
    error_count = 0

    # Conversation memory
    _honcho_conv_ok = False
    try:
        await deps.conversation.add_message(contact_email, channel, "user", raw_input)
        await deps.conversation.add_message(contact_email, channel, "assistant", raw_response)
        _honcho_conv_ok = True
    except (DatabaseError, Exception):  # noqa: BLE001 — LEARN step isolation: recorded in error_count, pipeline continues
        error_count += 1
        logger.exception("ConversationMemory recording failed")

    if _honcho_conv_ok:
        try:
            from brain_os.brain.honcho_user_model import schedule_sync_turn

            schedule_sync_turn(contact_email, raw_input, raw_response)
        except Exception:  # noqa: BLE001 — Honcho transcript scheduling is best-effort
            logger.debug("Honcho transcript scheduling failed", exc_info=True)

    # Power/trust progression: successful co-work should strengthen bonds.
    if deps.power_level_tracker is not None:
        try:
            failed_agents: list[str] = []
            for agent_name in agents_used:
                if (
                    f"Agent '{agent_name}' timed out" in raw_response
                    or f"Agent '{agent_name}' encountered an error" in raw_response
                ):
                    failed_agents.append(agent_name)
            failed_set = set(failed_agents)
            successful_agents = [
                agent_name for agent_name in agents_used if agent_name not in failed_set
            ]
            await deps.power_level_tracker.record_turn_outcome(
                successful_agents=successful_agents,
                failed_agents=failed_agents,
            )
        except Exception:  # noqa: BLE001 — LEARN step isolation: recorded in error_count, pipeline continues
            error_count += 1
            logger.warning("PowerLevelTracker turn-outcome update failed", exc_info=True)

    # Durable fact extraction into long-term semantic memory
    mem0_attempted = False
    mem0_skipped_no_facts = False
    deduped_facts: list[str] = []
    if deps.long_term is not None:
        try:
            from brain_os.memory.store_policy import is_query_echo

            fact_candidates: list[str] = _extract_teaching_facts(raw_input)
            # Explicit teachings are verbatim by design — exempt from echo check.
            teaching_count = len(fact_candidates)
            from brain_os.services.llm_client import get_llm_client

            _fact_llm = get_llm_client()
            try:
                _facts_raw = await asyncio.wait_for(
                    _fact_llm.generate_text(
                        "Extract durable facts from this exchange — preferences, "
                        "decisions, business context about the contact. Return one "
                        "fact per line. Return NONE if no durable facts exist.",
                        f"Contact: {contact_email}\nUser: {raw_input[:500]}\n"
                        f"Assistant: {raw_response[:500]}",
                        name="pipeline.extract_facts",
                        model_profile="fast",
                        session_id=run_id,
                        user_id=contact_email,
                    ),
                    timeout=8.0,
                )
                if _facts_raw and _facts_raw.strip().upper() != "NONE":
                    for fact_line in _facts_raw.strip().splitlines():
                        fact = fact_line.strip().lstrip("- ")
                        if fact and len(fact) > 10:
                            fact_candidates.append(fact)
            except TimeoutError:
                logger.warning("Fact extraction timed out after 8s; using deterministic facts only")

            # Quality gates: salience + echo filter for LLM-extracted facts.
            from brain_os.memory.store_policy import evaluate_mem_store

            gated: list[str] = []
            for idx, fact in enumerate(fact_candidates):
                decision = evaluate_mem_store(
                    fact,
                    source=f"conversation:{contact_email}",
                    metadata={
                        "type": "fact",
                        "confidence": 0.7,
                        "memory_category": "fact",
                    },
                    category="fact",
                )
                if not decision.allow:
                    continue
                if idx >= teaching_count and is_query_echo(fact, raw_input):
                    continue
                gated.append(fact)

            # Preserve order while deduplicating small fact sets.
            deduped: list[str] = []
            seen: set[str] = set()
            for fact in gated:
                normalized = re.sub(r"\s+", " ", fact).strip().lower()
                if normalized and normalized not in seen:
                    deduped.append(fact.strip())
                    seen.add(normalized)

            deduped_facts = deduped
            for fact in deduped[:8]:
                mem0_attempted = True
                await deps.long_term.store_fact(
                    fact,
                    source=f"conversation:{contact_email}",
                    confidence=0.7,
                    mem0_user_id=contact_email,
                    run_id=run_id,
                    category="fact",
                )
            if deduped:
                logger.debug(
                    "LEARN | stored %d durable facts for %s", len(deduped[:8]), contact_email
                )
            else:
                # No durable facts is a deliberate, contract-compliant skip —
                # the turn content still lands in Qdrant below. Storing a
                # generic "Conversation turn summary" memory polluted Mem0.
                mem0_skipped_no_facts = True
                record_receipt(
                    build_receipt(
                        store="mem0",
                        attempted=False,
                        success=False,
                        operation="pipeline.fact_extraction",
                        run_id=run_id,
                        reason="no_durable_facts",
                        metadata={"user_id": contact_email},
                    )
                )
        except (LLMError, Exception):  # noqa: BLE001 — LEARN step isolation: recorded in error_count, pipeline continues
            error_count += 1
            logger.debug("Fact extraction failed (non-critical)", exc_info=True)

    eligible_turn = (
        route_method in {"llm", "deterministic", "quick_pipeline", "fast_path"}
        and len((raw_input or "").strip()) >= 20
    )

    # Episode capture audit (Phase 3 biology upgrade): every turn is counted;
    # ineligible turns log a drop reason so "dreams of nothing" is diagnosable.
    try:
        from brain_os.memory.episode_capture_metrics import record_drop, record_seen

        record_seen()
        if not eligible_turn:
            reason = (
                "input_too_short"
                if len((raw_input or "").strip()) < 20
                else f"route_{route_method}"
            )
            record_drop(f"not_eligible_turn:{reason}")
    except Exception:  # noqa: BLE001 — episode capture audit is best-effort telemetry
        logger.debug("Episode capture audit failed (non-critical)", exc_info=True)

    if eligible_turn and deps.relationship is not None and contact_email and "@" in contact_email:
        try:
            from datetime import UTC, datetime

            turn_interaction = type(
                "_TurnInteraction",
                (),
                {"content": None, "created_at": datetime.now(UTC)},
            )()
            await deps.relationship.update_relationship(contact_email, turn_interaction)
            if deduped_facts:
                from brain_os.memory.block_sync import merge_preferences_from_facts

                await merge_preferences_from_facts(deps.relationship, contact_email, deduped_facts)
            if deps.memory_blocks is not None:
                from brain_os.memory.block_sync import refresh_contact_blocks

                await refresh_contact_blocks(
                    deps.memory_blocks,
                    contact_email,
                    relationship=deps.relationship,
                    episodic=deps.episodic,
                    crm=deps.crm,
                )
        except Exception:  # noqa: BLE001 — relationship/memory-block refresh is best-effort
            logger.debug("Relationship / memory block LEARN refresh failed", exc_info=True)

    if deps.qdrant is None:
        qdrant_attempted = False
        q_receipt = build_receipt(
            store="qdrant",
            attempted=False,
            success=False,
            operation="pipeline.turn_summary",
            run_id=run_id,
            reason="qdrant_unavailable",
            metadata={"route_method": route_method, "eligible_turn": eligible_turn},
        )
        record_receipt(q_receipt)
        enqueue_retry(q_receipt)
    elif not eligible_turn:
        qdrant_attempted = False
        record_receipt(
            build_receipt(
                store="qdrant",
                attempted=False,
                success=False,
                operation="pipeline.turn_summary",
                run_id=run_id,
                reason="not_eligible_turn",
                metadata={"route_method": route_method},
            )
        )
    else:
        qdrant_attempted = True
        t0_q = time.monotonic()
        try:
            point_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"conversation_turn::{contact_email}::{run_id}",
            )
            turn_item = KnowledgeItem(
                id=point_id,
                source=f"conversation:turn:{contact_email}",
                source_category="conversation_turn",
                content=f"User: {raw_input[:1200]}\nAssistant: {raw_response[:1200]}",
                metadata={
                    "contact_email": contact_email,
                    "run_id": run_id,
                    "route_method": route_method,
                    "agents_used": agents_used[:8],
                },
            )
            await deps.qdrant.upsert_items([turn_item])
            record_receipt(
                build_receipt(
                    store="qdrant",
                    attempted=True,
                    success=True,
                    operation="pipeline.turn_summary",
                    run_id=run_id,
                    latency_ms=round((time.monotonic() - t0_q) * 1000, 2),
                    ids=[turn_item.id.hex],
                    metadata={"route_method": route_method},
                )
            )
        except Exception:  # noqa: BLE001 — store write failure is recorded as a receipt and retried
            q_receipt = build_receipt(
                store="qdrant",
                attempted=True,
                success=False,
                operation="pipeline.turn_summary",
                run_id=run_id,
                reason="qdrant_write_failed",
                latency_ms=round((time.monotonic() - t0_q) * 1000, 2),
                metadata={
                    "route_method": route_method,
                    "replay_turn_summary": True,
                    "contact_email": contact_email,
                    "turn_user_preview": raw_input[:1200],
                    "turn_assistant_preview": raw_response[:1200],
                    "turn_content_full": (
                        f"User: {raw_input[:1200]}\nAssistant: {raw_response[:1200]}"
                    ),
                },
            )
            record_receipt(q_receipt)
            enqueue_retry(q_receipt)

    if deps.graph is None:
        neo4j_attempted = False
        g_receipt = build_receipt(
            store="neo4j",
            attempted=False,
            success=False,
            operation="pipeline.turn_relationship",
            run_id=run_id,
            reason="neo4j_unavailable",
            metadata={
                "route_method": route_method,
                "eligible_turn": eligible_turn,
                "replay_turn_relationship": bool(eligible_turn),
                "contact_email": contact_email,
            },
        )
        record_receipt(g_receipt)
        enqueue_retry(g_receipt)
    elif not eligible_turn:
        neo4j_attempted = False
        record_receipt(
            build_receipt(
                store="neo4j",
                attempted=False,
                success=False,
                operation="pipeline.turn_relationship",
                run_id=run_id,
                reason="not_eligible_turn",
                metadata={"route_method": route_method},
            )
        )
    else:
        neo4j_attempted = True
        t0_g = time.monotonic()
        try:
            await deps.graph.add_person(
                name=contact_email.split("@")[0] if "@" in contact_email else contact_email,
                email=contact_email,
                company_name="",
                role="contact",
                source_id=f"pipeline::{run_id}",
            )
            await deps.graph.add_application(
                name="ConversationTurn",
                description="General conversation turns for write contract",
            )
            rel_ok = await deps.graph.add_relationship(
                from_type="Person",
                from_key=contact_email,
                rel_type="REFERS_TO",
                to_type="Application",
                to_key="ConversationTurn",
                properties={"run_id": run_id, "route_method": route_method},
                source_id=f"pipeline::{run_id}",
            )
            if rel_ok:
                record_receipt(
                    build_receipt(
                        store="neo4j",
                        attempted=True,
                        success=True,
                        operation="pipeline.turn_relationship",
                        run_id=run_id,
                        latency_ms=round((time.monotonic() - t0_g) * 1000, 2),
                        metadata={"route_method": route_method},
                    )
                )
            else:
                g_receipt = build_receipt(
                    store="neo4j",
                    attempted=True,
                    success=False,
                    operation="pipeline.turn_relationship",
                    run_id=run_id,
                    reason="relationship_not_written",
                    latency_ms=round((time.monotonic() - t0_g) * 1000, 2),
                    metadata={
                        "route_method": route_method,
                        "replay_turn_relationship": True,
                        "contact_email": contact_email,
                    },
                )
                record_receipt(g_receipt)
                enqueue_retry(g_receipt)
        except Exception:  # noqa: BLE001 — store write failure is recorded as a receipt and retried
            g_receipt = build_receipt(
                store="neo4j",
                attempted=True,
                success=False,
                operation="pipeline.turn_relationship",
                run_id=run_id,
                reason="neo4j_write_failed",
                latency_ms=round((time.monotonic() - t0_g) * 1000, 2),
                metadata={
                    "route_method": route_method,
                    "replay_turn_relationship": True,
                    "contact_email": contact_email,
                },
            )
            record_receipt(g_receipt)
            enqueue_retry(g_receipt)

    # CRM interaction log
    if deps.crm is not None:
        try:
            contact_record = await deps.crm.get_contact_by_email(contact_email)
            if contact_record is not None:
                await deps.crm.create_interaction(
                    contact_id=str(contact_record.id),
                    channel=Channel(channel),
                    direction=Direction.INBOUND,
                    subject=raw_input[:200],
                    content=json.dumps(
                        {
                            "query": raw_input,
                            "response_preview": raw_response[:500],
                            "route": route_method,
                            "agents": agents_used,
                        },
                        default=str,
                    ),
                )
        except (DatabaseError, Exception):  # noqa: BLE001 — LEARN step isolation: recorded in error_count, pipeline continues
            error_count += 1
            logger.exception("CRM interaction logging failed")

    # Musculoskeletal action tracking
    if deps.musculoskeletal is not None:
        try:
            from brain_os.systems.musculoskeletal import ActionRecord, ActionType

            await deps.musculoskeletal.record_action(
                ActionRecord(
                    action_type=ActionType.RESEARCH_COMPLETED,
                    target=contact_email,
                    details={
                        "channel": channel,
                        "route": route_method,
                        "agents": agents_used,
                        "query_preview": raw_input[:200],
                    },
                )
            )
        except (BrainOSError, Exception):  # noqa: BLE001 — LEARN step isolation: recorded in error_count, pipeline continues
            error_count += 1
            logger.exception("MusculoskeletalSystem recording failed")

    # Goal slot extraction
    if active_goal is not None and deps.goals is not None:
        try:
            extracted = await deps.goals.extract_slots(active_goal, raw_input)
            if extracted:
                await deps.goals.update_goal(active_goal.id, extracted)
        except (DatabaseError, Exception):  # noqa: BLE001 — LEARN step isolation: recorded in error_count, pipeline continues
            error_count += 1
            logger.exception("GoalManager slot update failed")

    # Goal detection for new goals
    if active_goal is None and deps.goals is not None:
        try:
            await deps.goals.detect_goal(
                resolved_input,
                {"contact_id": contact_email, "channel": channel},
            )
        except (DatabaseError, Exception):  # noqa: BLE001 — LEARN step isolation: recorded in error_count, pipeline continues
            error_count += 1
            logger.exception("GoalManager detection failed")

    # Procedural learning
    if deps.procedural is not None and route_method in ("deterministic", "llm"):
        try:
            await deps.procedural.learn_procedure(
                resolved_input,
                agents_used,
            )
        except (DatabaseError, Exception):  # noqa: BLE001 — LEARN step isolation: recorded in error_count, pipeline continues
            error_count += 1
            logger.exception("ProceduralMemory learning failed")

    # Trigger Sophia for background reflection (fire-and-forget)
    sophia = deps.pantheon.get_agent("sophia")
    if sophia is not None:
        try:
            await sophia.handle(
                f"Reflect on this interaction: {raw_input[:300]}",
                {"response": raw_response[:300], "route": route_method},
            )
        except (ToolExecutionError, Exception):  # noqa: BLE001 — LEARN step isolation: recorded in error_count, pipeline continues
            error_count += 1
            logger.warning("Sophia reflection failed", exc_info=True)

    try:
        if deps.realtime_observer is not None:
            observer = deps.realtime_observer
        else:
            from brain_os.brain.realtime_observer import RealTimeObserver

            observer = RealTimeObserver()
            await observer._load()
        await asyncio.wait_for(
            observer.observe_turn(raw_input, raw_response, contact_email),
            timeout=10.0,
        )
    except TimeoutError:
        error_count += 1
        logger.debug("RealTimeObserver timed out (10s) — skipping")
    except (BrainOSError, Exception):  # noqa: BLE001 — LEARN step isolation: recorded in error_count, pipeline continues
        error_count += 1
        logger.warning("RealTimeObserver not available", exc_info=True)

    # Endocrine feedback
    if deps.endocrine is not None:
        try:
            deps.endocrine.boost("growth_signal", 0.02)
            if route_method == "deterministic":
                deps.endocrine.boost("confidence", 0.01)
        except (BrainOSError, Exception):  # noqa: BLE001 — LEARN step isolation: recorded in error_count, pipeline continues
            error_count += 1
            logger.warning("Endocrine update failed", exc_info=True)

    if deps.unified_ctx is not None:
        try:
            deps.unified_ctx.record_turn(
                contact_email,
                channel,
                raw_input,
                raw_response,
            )
        except (DatabaseError, Exception):  # noqa: BLE001 — LEARN step isolation: recorded in error_count, pipeline continues
            error_count += 1
            logger.exception("UnifiedContextManager recording failed")

    duration_ms = round((time.monotonic() - learn_t0) * 1000, 1)
    mem0_ok = mem0_attempted or mem0_skipped_no_facts
    contract_pass = (not eligible_turn) or (mem0_ok and qdrant_attempted and neo4j_attempted)
    deps.learn_metrics["runs"] += 1
    if mem0_skipped_no_facts:
        deps.learn_metrics["mem0_skipped_no_facts"] = (
            int(deps.learn_metrics.get("mem0_skipped_no_facts", 0)) + 1
        )
    deps.learn_metrics["last_duration_ms"] = duration_ms
    deps.learn_metrics["last_error_count"] = error_count
    deps.learn_metrics["last_contract_pass"] = contract_pass
    deps.learn_metrics["contract_pass"] = int(deps.learn_metrics.get("contract_pass", 0)) + (
        1 if contract_pass else 0
    )
    deps.learn_metrics["contract_fail"] = int(deps.learn_metrics.get("contract_fail", 0)) + (
        0 if contract_pass else 1
    )
    if error_count == 0:
        deps.learn_metrics["success"] += 1
    else:
        deps.learn_metrics["failed"] += 1

    logger.info("LEARN | recorded for %s", contact_email)
