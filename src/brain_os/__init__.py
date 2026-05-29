"""Ira — Brain OS Pantheon."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

# Suppress requests' urllib3/chardet version warning (transitive deps; requests 2.32.x works)
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")


def _install_config_compat_shims() -> None:
    """Backfill optional config exports expected by newer CLI/routes/tests."""
    cfg = importlib.import_module("brain_os.config")
    if hasattr(cfg, "get_event_driven_ingestion_public_snapshot"):
        return

    def _event_snapshot_fallback(
        *,
        services: dict[str, Any] | None = None,
    ) -> dict[str, str | bool]:
        s = cfg.get_settings()
        app = s.app
        google = s.google
        out: dict[str, str | bool] = {
            "event_task_reactor_enabled": bool(getattr(app, "event_task_reactor_enabled", False)),
            "event_task_rules_path": str(
                getattr(app, "event_task_rules_path", "data/automation/event_task_rules.yaml")
            ).strip()
            or "data/automation/event_task_rules.yaml",
            "gmail_push_enabled": bool(getattr(google, "gmail_push_enabled", False)),
        }
        if services is not None:
            out["event_bus_available"] = services.get("event_bus") is not None
            out["heartbeat_runner_available"] = services.get("heartbeat_runner") is not None
        return out

    cfg.get_event_driven_ingestion_public_snapshot = _event_snapshot_fallback


_install_config_compat_shims()


def _install_graph_compat_shims() -> None:
    """Backfill optional knowledge_graph exports expected by sync/tests."""
    kg = importlib.import_module("brain_os.brain.knowledge_graph")
    if hasattr(kg, "normalize_relationship_type"):
        return

    def _normalize_relationship_type(raw: str) -> str:
        rel = (raw or "").strip().upper()
        mapping = {
            "BUYS_COMPONENTS_FROM": "SUPPLIES",
            "SELLS": "CUSTOMER_OF",
            "PARTNERS_WITH": "DISTRIBUTES_FOR",
        }
        return mapping.get(rel, rel)

    kg.normalize_relationship_type = _normalize_relationship_type


_install_graph_compat_shims()
