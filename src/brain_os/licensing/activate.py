"""Activate Pro / trial licenses (HTTP or local trial)."""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime, timedelta

import httpx
from brain_os.licensing.store import LicenseRecord, load_license, save_license

logger = logging.getLogger(__name__)

_KEY_PREFIX = re.compile(r"^bos_(live|trial|test)_[A-Za-z0-9_-]{8,}$")
_TRIAL_DAYS = 14


def activate_with_key(license_key: str) -> LicenseRecord:
    key = license_key.strip()
    if not _KEY_PREFIX.match(key):
        msg = "Invalid key format. Expected bos_live_... or bos_trial_..."
        raise ValueError(msg)

    server = os.environ.get("BRAIN_LICENSE_SERVER_URL", "").strip()
    if server:
        return _activate_remote(server, key)

    if os.environ.get("BRAIN_LICENSE_STUB_ACCEPT", "").lower() in {"1", "true", "yes"}:
        return _activate_stub(key)

    raise ValueError(
        "Set BRAIN_LICENSE_SERVER_URL for online activation, "
        "or use brain activate --trial for a local 14-day trial."
    )


def activate_trial(*, org_name: str = "Trial workspace") -> LicenseRecord:
    expires = datetime.now(UTC) + timedelta(days=_TRIAL_DAYS)
    record = LicenseRecord(
        tier="trial",
        org_name=org_name,
        license_key_prefix="bos_trial_local",
        expires_at=expires,
    )
    save_license(record)
    return record


def _activate_stub(key: str) -> LicenseRecord:
    tier: str = "trial" if key.startswith("bos_trial_") else "pro"
    expires = datetime.now(UTC) + timedelta(days=365 if tier == "pro" else _TRIAL_DAYS)
    record = LicenseRecord(
        tier=tier,  # type: ignore[arg-type]
        license_key_prefix=key[:16],
        expires_at=expires,
    )
    save_license(record)
    return record


def _activate_remote(server: str, key: str) -> LicenseRecord:
    url = server.rstrip("/") + "/v1/licenses/activate"
    payload = {
        "license_key": key,
        "product": "brain-os",
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    expires_raw = data.get("expires_at")
    expires_at = None
    if expires_raw:
        expires_at = datetime.fromisoformat(str(expires_raw).replace("Z", "+00:00"))

    record = LicenseRecord(
        tier=data.get("tier", "pro"),
        org_id=data.get("org_id"),
        org_name=data.get("org_name"),
        license_key_prefix=key[:16],
        expires_at=expires_at,
        jwt=data.get("jwt"),
    )
    save_license(record)
    return record


def license_status_dict() -> dict[str, object]:
    record = load_license()
    if record is None:
        return {
            "tier": "community",
            "pro_features": False,
            "document_limit": 500,
            "message": "Community tier — brain activate --trial for 14-day Pro",
        }
    exp = record.expires_at
    expired = False
    if exp is not None:
        exp_cmp = exp if exp.tzinfo else exp.replace(tzinfo=UTC)
        expired = datetime.now(UTC) >= exp_cmp
    return {
        "tier": "community" if expired else record.tier,
        "pro_features": not expired and record.tier in {"pro", "trial"},
        "org_name": record.org_name,
        "expires_at": exp.isoformat() if exp else None,
        "expired": expired,
        "document_limit": None if record.tier in {"pro", "trial"} and not expired else 500,
    }
