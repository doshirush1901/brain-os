"""Dream mode paths, prompts, and LLM limits (Phase 7 split)."""

from __future__ import annotations

from pathlib import Path

from brain_os.prompt_loader import load_prompt

DREAM_STRUCTURED_MAX_USER_CHARS = 280_000

_DREAM_DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data"
DREAM_LOG_PATH = _DREAM_DATA_ROOT / "dream_log.json"
DREAM_CHECKPOINT_PATH = _DREAM_DATA_ROOT / "dream_checkpoint.json"

GAP_SYSTEM_PROMPT = load_prompt("dream_gap_analysis")
CREATIVE_SYSTEM_PROMPT = load_prompt("dream_creative")
CAMPAIGN_SYSTEM_PROMPT = load_prompt("dream_campaign")
INSIGHT_SYSTEM_PROMPT = load_prompt("dream_insight")
PROCEDURAL_SYSTEM_PROMPT = load_prompt("dream_procedural")
PRUNE_SYSTEM_PROMPT = load_prompt("dream_prune")
JOURNAL_SYSTEM_PROMPT = load_prompt("dream_agent_journal")
