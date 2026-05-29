"""Shared correction-store types (no backend imports)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class CorrectionCategory(str, Enum):
    PRICING = "PRICING"
    SPECS = "SPECS"
    CUSTOMER = "CUSTOMER"
    COMPETITOR = "COMPETITOR"
    GENERAL = "GENERAL"


class CorrectionSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


def row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "entity": row[1],
        "category": row[2],
        "severity": row[3],
        "old_value": row[4],
        "new_value": row[5],
        "source": row[6],
        "created_at": row[7],
        "status": row[8],
    }
