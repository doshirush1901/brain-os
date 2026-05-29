"""Daily JSON snapshots of the operator hot-leads board for prediction reconciliation."""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from brain_os.contracts.learning_paths import board_snapshots_dir
from brain_os.services.hot_leads_board import (
    LeadBoardRow,
    clear_lead_board_cache,
    load_lead_board,
)

logger = logging.getLogger(__name__)


def _snapshot_path(for_date: date) -> Path:
    return board_snapshots_dir() / f"{for_date.isoformat()}.json"


def board_rows_to_payload(rows: tuple[LeadBoardRow, ...]) -> list[dict[str, Any]]:
    return [
        {
            "account": r.account,
            "bucket": r.bucket.value,
            "owner_contact": r.owner_contact,
            "last_real_touch": r.last_real_touch,
            "machine": r.machine,
            "next_action": r.next_action,
            "do_not_email_until": r.do_not_email_until,
        }
        for r in rows
    ]


def payload_to_board_map(payload: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in payload:
        account = str(row.get("account") or "").strip()
        if not account:
            continue
        out[_normalize_account_key(account)] = row
    return out


def _normalize_account_key(account: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (account or "").lower())


def write_board_snapshot(
    *,
    for_date: date | None = None,
    rows: tuple[LeadBoardRow, ...] | None = None,
) -> Path:
    snap_date = for_date or date.today()
    clear_lead_board_cache()
    board_rows = rows if rows is not None else load_lead_board()
    payload = {
        "snapshot_date": snap_date.isoformat(),
        "row_count": len(board_rows),
        "rows": board_rows_to_payload(board_rows),
    }
    path = _snapshot_path(snap_date)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    logger.info("Lead board snapshot written: %s (%d rows)", path, len(board_rows))
    return path


def read_board_snapshot(for_date: date) -> dict[str, Any] | None:
    path = _snapshot_path(for_date)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        logger.warning("Lead board snapshot read failed: %s", path, exc_info=True)
        return None


def latest_snapshot_date() -> str | None:
    """Re-export from contracts (brain metrics must not import this memory module)."""

    from brain_os.contracts.lead_board_snapshots import latest_snapshot_date as _latest

    return _latest()


def bucket_for_account(board_map: dict[str, dict[str, Any]], account: str) -> str | None:
    row = board_map.get(_normalize_account_key(account))
    if not row:
        return None
    return str(row.get("bucket") or "").strip() or None
