"""Persist Brain OS license on disk (gitignored data dir)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

TierName = Literal["community", "trial", "pro"]


class LicenseRecord(BaseModel):
    tier: TierName = "community"
    org_id: str | None = None
    org_name: str | None = None
    license_key_prefix: str | None = Field(
        default=None,
        description="First 12 chars of key for support (never store full secret)",
    )
    activated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    jwt: str | None = None


def _license_path() -> Path:
    try:
        from brain_os.systems.data_dir_lock import get_data_dir

        base = get_data_dir()
    except Exception:
        base = Path("data")
    return Path(base) / "brain" / "license.json"


def load_license() -> LicenseRecord | None:
    path = _license_path()
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return LicenseRecord.model_validate(raw)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Could not read license file %s: %s", path, exc)
        return None


def save_license(record: LicenseRecord) -> Path:
    path = _license_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def clear_license() -> None:
    path = _license_path()
    if path.is_file():
        path.unlink()
