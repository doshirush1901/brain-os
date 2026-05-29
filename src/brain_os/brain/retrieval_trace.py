"""Structured DEBUG logs for UnifiedRetriever (optional; no secrets in payloads)."""

from __future__ import annotations

import logging
import time
from typing import Any

from brain_os.brain.retrieval_context import retrieval_run_id_var
from brain_os.config import get_settings

logger = logging.getLogger(__name__)


def emit_retrieval_trace(stage: str, **fields: Any) -> None:
    """Emit one JSON-serializable retrieval row when tracing is enabled."""
    try:
        if not get_settings().app.retriever_trace_enabled:
            return
    except (AttributeError, RuntimeError, ValueError):
        return
    row: dict[str, Any] = {"stage": stage, "ts_ms": int(time.time() * 1000)}
    rid = retrieval_run_id_var.get()
    if rid:
        row["run_id"] = rid
    for k, v in fields.items():
        if v is not None:
            row[k] = v
    logger.debug("retrieval_trace %s", row)
