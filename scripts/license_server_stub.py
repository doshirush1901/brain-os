#!/usr/bin/env python3
"""Dev-only Brain OS license activation stub (FastAPI).

Run from an exported brain-os tree or ira-v3 skeleton:

    poetry run python scripts/license_server_stub.py
    # → http://127.0.0.1:8765/v1/licenses/activate

Then:

    export BRAIN_LICENSE_SERVER_URL=http://127.0.0.1:8765
    poetry run brain activate --key bos_test_dev_workspace_01
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_KEY_RE = re.compile(r"^bos_(live|trial|test)_[A-Za-z0-9_-]{8,}$")

app = FastAPI(
    title="Brain OS License Stub",
    description="Local activation server for development — not for production.",
    version="0.1.0-stub",
)


class ActivateRequest(BaseModel):
    license_key: str
    product: str = Field(default="brain-os")


class ActivateResponse(BaseModel):
    tier: str
    org_id: str | None = None
    org_name: str | None = None
    expires_at: str | None = None
    jwt: str | None = None


def _tier_for_key(key: str) -> str:
    if key.startswith("bos_trial_"):
        return "trial"
    return "pro"


def _expiry_days(tier: str) -> int:
    return 14 if tier == "trial" else int(os.environ.get("BRAIN_LICENSE_STUB_PRO_DAYS", "365"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "brain-os-license-stub"}


@app.post("/v1/licenses/activate", response_model=ActivateResponse)
def activate_license(body: ActivateRequest) -> ActivateResponse:
    key = body.license_key.strip()
    if body.product != "brain-os":
        raise HTTPException(status_code=400, detail="Unsupported product")
    if not _KEY_RE.match(key):
        raise HTTPException(status_code=400, detail="Invalid key format")

    tier = _tier_for_key(key)
    expires = datetime.now(UTC) + timedelta(days=_expiry_days(tier))
    slug = key.split("_", 2)[-1][:32]
    return ActivateResponse(
        tier=tier,
        org_id=f"stub-{slug}",
        org_name=f"Stub Org ({slug})",
        expires_at=expires.isoformat(),
        jwt=f"stub.{tier}.{slug}",
    )


def main() -> None:
    import uvicorn

    host = os.environ.get("BRAIN_LICENSE_STUB_HOST", "127.0.0.1")
    port = int(os.environ.get("BRAIN_LICENSE_STUB_PORT", "8765"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
