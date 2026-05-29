"""License tiers for Brain OS Community vs Pro."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from brain_os.licensing.store import load_license


class Tier(str, Enum):
    COMMUNITY = "community"
    TRIAL = "trial"
    PRO = "pro"


def effective_tier() -> Tier:
    """Return active tier; expired trial/pro falls back to Community."""
    record = load_license()
    if record is None:
        return Tier.COMMUNITY
    if record.expires_at is not None:
        exp = record.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=UTC)
        if datetime.now(UTC) >= exp:
            return Tier.COMMUNITY
    try:
        return Tier(record.tier)
    except ValueError:
        return Tier.COMMUNITY


def is_pro() -> bool:
    return effective_tier() in {Tier.PRO, Tier.TRIAL}


def tier_label() -> str:
    return effective_tier().value
