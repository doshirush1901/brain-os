"""Dream cycle checkpoint persistence (Phase 9 split)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_dream_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {}
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.debug("Dream checkpoint read failed", exc_info=True)
        return {}


def reset_dream_checkpoint(checkpoint_path: Path) -> None:
    """Remove the prior checkpoint so each ``brain dream`` runs a full cycle."""
    prior = load_dream_checkpoint(checkpoint_path)
    if prior:
        logger.info(
            "DREAM CYCLE resetting checkpoint (was %s for %s)",
            prior.get("status"),
            prior.get("cycle_date"),
        )
    try:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    except OSError as exc:
        logger.warning("Dream checkpoint reset failed: %s", exc)


def save_dream_checkpoint(
    checkpoint_path: Path,
    *,
    cycle_date: str,
    status: str,
    stage_log: dict[str, Any],
) -> None:
    payload = {
        "cycle_date": cycle_date,
        "status": status,
        "updated_at": datetime.now(UTC).isoformat(),
        "stages": stage_log.get("stages", {}),
        "metrics": stage_log.get("metrics", {}),
    }
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Dream checkpoint write failed", exc_info=True)
