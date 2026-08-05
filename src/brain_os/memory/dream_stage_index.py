"""Canonical dream-cycle stage order (audit P3 — single source for docs/tests).

Authoritative runner: ``dream_mode_cycle.execute_dream_cycle``.
Stage bodies: ``dream_mode.DreamMode`` (+ ``DreamBrainHooks`` for 12* brain work).

The legacy label "11-stage dream" referred to integer stages 0–10 before the
stage-12 learning cluster (12, 12b–12f) and substages (0.5, 0.55, 3.6, 11b)
were added. Prefer ``DREAM_STAGE_EXECUTION_ORDER`` or ``dream_stage_count()``.
"""

from __future__ import annotations

from typing import NamedTuple


class DreamStageRef(NamedTuple):
    """One logged stage in ``data/dream_log.json`` → ``stages``."""

    log_key: str
    label: str
    summary: str


# Order matches ``execute_dream_cycle`` (journal → sleep → 0..12 → 11b).
DREAM_STAGE_EXECUTION_ORDER: tuple[DreamStageRef, ...] = (
    DreamStageRef("11_agent_journaling", "11", "Agent journaling (runs first)"),
    DreamStageRef("sleep_phantom_limb", "Sleep", "Phantom-limb / health maintenance"),
    DreamStageRef("sleep_trust", "Sleep", "Trust maintenance"),
    DreamStageRef("sleep_curiosity", "Sleep", "Curiosity / exploration maintenance"),
    DreamStageRef("0_deferred_ingestion", "0", "Deferred Alexandros ingestion"),
    DreamStageRef("0_5_sleep_training", "0.5", "Nemesis sleep training"),
    DreamStageRef("0_55_pending_memory", "0.55", "Drain pending memory queue → Mem0"),
    DreamStageRef("1_memory_ingestion", "1", "CRM / interaction ingestion"),
    DreamStageRef("2_episodic_consolidation", "2", "Episodic consolidation"),
    DreamStageRef(
        "2_1_episode_hygiene",
        "2.1",
        "Episode hygiene (empty narrative delete + near-duplicate dedupe)",
    ),
    DreamStageRef(
        "2_2_sleep_curation",
        "2.2",
        "Sleep curation — 180d zero-access decay (jsonl archive) + episode→fact promotion queue",
    ),
    DreamStageRef(
        "2_5_mem0_replay",
        "2.5",
        "Mem0 sleep replay (reinforce frequently recalled assemblies)",
    ),
    DreamStageRef("3a_cross_episode_insights", "3a", "Cross-episode insights"),
    DreamStageRef("3b_gap_detection", "3b", "Knowledge gap detection"),
    DreamStageRef("3c_creative_synthesis", "3c", "Creative synthesis"),
    DreamStageRef("3d_campaign_reflection", "3d", "Campaign / myokine reflection"),
    DreamStageRef("3e_gap_resolution", "3e", "Active gap resolution"),
    DreamStageRef("3f_revenue_reflection", "3f", "Daily revenue funnel reflection"),
    DreamStageRef("3g_forced_collisions", "3g", "Serendipity forced collision ritual"),
    DreamStageRef("3h_aftermarket_reflection", "3h", "Installed-base trigger sweep"),
    DreamStageRef("3_6_prediction_reconciliation", "3.6", "Sophia prediction reconciliation"),
    DreamStageRef(
        "3i_memory_reconciliation",
        "3i",
        "Cross-memory contradiction reconciliation (ledger/episodic/Mem0/relationship)",
    ),
    DreamStageRef("4_procedural_learning", "4", "Procedural learning"),
    DreamStageRef("skills_index_refresh", "4+", "Refresh ``data/brain/skills_index.json``"),
    DreamStageRef(
        "5_memory_pruning",
        "5",
        "Episode archive (two-stage) + summarise + procedural/instinct decay; "
        "Mem0 deletion owned by 5b",
    ),
    DreamStageRef(
        "5b_mem0_forgetting",
        "5b",
        "Mem0 forgetting (archive → hard-delete after 30d; exclusions in forgetting_exclusions); "
        "sole Mem0 deletion owner",
    ),
    DreamStageRef(
        "5c_qdrant_vector_hygiene",
        "5c",
        "Qdrant vector hygiene (KB collection only; ignores mem0_archive.db)",
    ),
    DreamStageRef("6_price_conflict", "6", "Price conflict scan"),
    DreamStageRef("7_quality_review", "7", "Conversation quality / co-access review"),
    DreamStageRef("8_graph_consolidation", "8", "Neo4j graph consolidation"),
    DreamStageRef("9_follow_up", "9", "Stale quote follow-up detection"),
    DreamStageRef("10_morning_summary", "10", "Morning summary log"),
    DreamStageRef("12_cursor_sessions", "12", "Graphe cursor session learning"),
    DreamStageRef("12c_learning_compiler", "12c", "Learning compiler"),
    DreamStageRef("12e_gepa_overlay_compile", "12e", "GEPA overlay compile"),
    DreamStageRef("12d_learning_promotion", "12d", "Procedure + overlay promotion"),
    DreamStageRef("12f_skill_candidates", "12f", "Skill candidate harvest"),
    DreamStageRef("12g_instinct_evolve", "12g", "Instinct evolve (cluster → skill candidates)"),
    DreamStageRef("12b_curator_lite", "12b", "Curator-lite"),
    DreamStageRef("11b_operator_reflection", "11b", "Operator nightly reflection (runs last)"),
)

DREAM_STAGE_LOG_KEYS: frozenset[str] = frozenset(s.log_key for s in DREAM_STAGE_EXECUTION_ORDER)

# Stage-12 learning cluster order inside the cycle (GEPA / moat docs).
DREAM_STAGE_12_LEARNING_ORDER: tuple[str, ...] = (
    "12_cursor_sessions",
    "12c_learning_compiler",
    "12e_gepa_overlay_compile",
    "12d_learning_promotion",
    "12f_skill_candidates",
    "12g_instinct_evolve",
    "12b_curator_lite",
)


def dream_stage_count() -> int:
    """Number of distinct logged stages in a full cycle."""
    return len(DREAM_STAGE_EXECUTION_ORDER)


def format_dream_stage_index_markdown() -> str:
    """Markdown table for operator docs (Diataxis reference)."""
    lines = [
        "| Log key | Stage | Summary |",
        "|:--------|:------|:--------|",
    ]
    for ref in DREAM_STAGE_EXECUTION_ORDER:
        lines.append(f"| `{ref.log_key}` | {ref.label} | {ref.summary} |")
    return "\n".join(lines)
