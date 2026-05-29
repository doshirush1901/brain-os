"""Startup audit: sensitive paths must declare ``require_sensitive_api_key``."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute

from brain_os.exceptions import ConfigurationError
from brain_os.middleware.auth import SENSITIVE_PATHS_BLOCKED_WITHOUT_SECRET

logger = logging.getLogger(__name__)


def _dependant_uses_call(dependant: Any, target: Callable[..., Any]) -> bool:
    if dependant.call is target:
        return True
    for sub in dependant.dependencies:
        if _dependant_uses_call(sub, target):
            return True
    return False


def route_has_sensitive_api_key(
    route: APIRoute, *, require_sensitive_api_key: Callable[..., Any]
) -> bool:
    """Return True if the route's dependency tree includes *require_sensitive_api_key*."""
    if _dependant_uses_call(route.dependant, require_sensitive_api_key):
        return True
    for dep in route.dependencies:
        if getattr(dep, "dependency", None) is require_sensitive_api_key:
            return True
    return False


def assert_sensitive_routes_protected(app: FastAPI) -> None:
    """Fail closed when a listed sensitive path lacks ``require_sensitive_api_key``."""
    from brain_os.middleware.auth import require_sensitive_api_key

    missing: list[str] = []
    registered: set[str] = set()

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        registered.add(route.path)
        if route.path not in SENSITIVE_PATHS_BLOCKED_WITHOUT_SECRET:
            continue
        if not route_has_sensitive_api_key(
            route, require_sensitive_api_key=require_sensitive_api_key
        ):
            methods = ",".join(sorted(route.methods or ()))
            missing.append(f"{methods} {route.path}")

    unregistered = SENSITIVE_PATHS_BLOCKED_WITHOUT_SECRET - registered
    if unregistered:
        logger.warning(
            "Sensitive path list includes routes not registered on app: %s",
            sorted(unregistered),
        )

    if missing:
        raise ConfigurationError(
            "Sensitive routes missing require_sensitive_api_key dependency: " + "; ".join(missing)
        )
