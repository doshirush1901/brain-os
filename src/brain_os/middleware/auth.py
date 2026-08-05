"""API key authentication for Brain OS's FastAPI endpoints."""

from __future__ import annotations

import logging
import os

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from brain_os.config import get_settings

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)

# Health checks stay anonymous everywhere (load balancers / probes).
_HEALTH_PUBLIC_PATHS = frozenset({"/api/health", "/api/live"})
# Webhooks authenticated by their own signature scheme (not Bearer):
# WhatsApp uses Meta's X-Hub-Signature-256 HMAC + hub.verify_token handshake,
# both enforced inside the route (fail-closed when secrets are configured).
_SIGNED_WEBHOOK_PATHS = frozenset({"/api/whatsapp/webhook"})
# Public one-click unsubscribe (RFC 8058): the path token IS the credential —
# HMAC-signed + expiring, verified inside the route. Mail clients cannot send
# Bearer headers, so the prefix stays anonymous.
_SIGNED_TOKEN_PATH_PREFIXES = ("/api/optout/",)
# Public inbound inquiry capture (example-company.org contact widget): browsers
# cannot hold the Bearer secret. The route enforces its own optional widget
# key (APP__INBOUND_WIDGET_API_KEY), per-IP rate limit, and honeypot field.
_PUBLIC_FORM_PATHS = frozenset({"/api/inbound/inquiry"})
# OpenAPI UI + schema: only anonymous on trusted development hosts (see require_api_key).
_DOCS_PATHS = frozenset({"/docs", "/openapi.json", "/redoc"})

# Reference list for ops / startup warning when APP__API_SECRET_KEY is empty.
SENSITIVE_PATHS_BLOCKED_WITHOUT_SECRET = frozenset(
    sorted(
        {
            "/api/feedback",
            "/api/llm/reset-breakers",
            "/api/crm/sync-apollo",
            "/api/crm/seed-demo-programme",
            "/api/crm/enrich-programme",
            "/api/write-contract/retry",
            "/api/ingest",
            "/api/document-ai/parse",
            "/api/reingest-scanned",
            "/api/board-meeting",
            "/api/memory/store",
            "/api/email/rescan",
            "/api/email/send",
            "/api/email/trash",
            "/api/email/create-draft",
            "/api/email/interconnections/outcome",
            "/api/scheduling/propose",
            "/api/scheduling/book",
            "/api/outbound/campaigns",
            "/api/outbound/campaigns/draft",
            "/api/outbound/campaigns/approve",
            "/api/outbound/campaigns/reject",
            "/api/operator/inbox",
            "/api/operator/inbox/decide",
            "/api/operator/release",
            "/api/operator/session",
            "/api/operator/activity/today",
            "/api/operator/dashboard/today",
            "/api/operator/dashboard/history",
            "/api/operator/dashboard/stats",
            "/api/operator/dashboard/build",
            "/api/observability/snapshot",
            "/api/observability/pipeline-recent",
            "/api/observability/agent-stats",
            "/api/observability/memory-health",
            "/api/revenue/desk",
            "/api/recruitment/candidates",
            "/api/recruitment/candidates/by-email/events",
            "/api/recruitment/candidates/by-email/score",
            "/api/anu/draft-recruitment-stage2",
            "/api/anu/parse-resume",
            "/api/anu/parse-resume-text",
            "/api/anu/score",
            "/api/anu/chat",
            "/api/anu/export",
            "/api/anu/candidates/by-email",
            "/api/recruitment/candidates/by-email",
            "/api/task/stream",
            "/api/task/clarify",
            "/api/task/abort",
            "/api/task/retry/stream",
            "/api/vendors",
            "/api/vendors/payables",
        }
    )
)


def _is_relaxed_http_auth_environment() -> bool:
    """True only on trusted dev shells — missing API secrets stay permitted."""
    brain_env = os.getenv("BRAIN_ENV", "").strip().lower()
    if brain_env:
        return brain_env in ("development", "dev", "local", "test")
    return get_settings().app.environment.strip().lower() in ("development", "dev", "local", "test")


def _bearer_ok(secret: str, credentials: HTTPAuthorizationCredentials | None) -> bool:
    return credentials is not None and credentials.credentials == secret


def log_sensitive_api_policy_if_keyless() -> None:
    """Log API-key posture when APP__API_SECRET_KEY is unset."""
    secret = get_settings().app.api_secret_key.get_secret_value().strip()
    if secret:
        return

    if _is_relaxed_http_auth_environment():
        preview = ", ".join(sorted(SENSITIVE_PATHS_BLOCKED_WITHOUT_SECRET)[:12])
        logger.warning(
            "IRA API: APP__API_SECRET_KEY unset — relaxed development auth active "
            "(expose only on localhost). Sensitive writes still require Bearer where "
            "require_sensitive_api_key is enforced. Examples of gated paths: %s, ...",
            preview,
        )
        return

    logger.error(
        "IRA API: APP__API_SECRET_KEY unset while BRAIN_ENV / APP__ENVIRONMENT is not "
        "development-like — authenticated routes reject anonymous callers."
    )


def _docs_disabled_response() -> HTTPException:
    """Hide OpenAPI surfaces in staging/production (uniform 404)."""
    return HTTPException(status_code=404, detail="Not found")


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> None:
    """Validate Bearer token against APP__API_SECRET_KEY.

    ``/api/health`` stays open everywhere.

    ``/docs``, ``/openapi.json``, ``/redoc`` are anonymous **only** in relaxed
    development environments. In staging/production they are **disabled** unless a
    valid Bearer matches ``APP__API_SECRET_KEY`` (otherwise **404**, not 401, to avoid
    fingerprinting).

    Other routes: missing secret outside relaxed environments → **401** fail-closed.
    """
    path = request.url.path
    if path in _HEALTH_PUBLIC_PATHS:
        return
    if path in _SIGNED_WEBHOOK_PATHS:
        return
    if path.startswith(_SIGNED_TOKEN_PATH_PREFIXES):
        return
    if path in _PUBLIC_FORM_PATHS:
        return

    secret_stripped = get_settings().app.api_secret_key.get_secret_value().strip()

    if path in _DOCS_PATHS:
        if _is_relaxed_http_auth_environment():
            return
        if not secret_stripped or not _bearer_ok(secret_stripped, credentials):
            raise _docs_disabled_response()
        return

    if not secret_stripped:
        if _is_relaxed_http_auth_environment():
            return
        logger.warning(
            "IRA API: rejecting request without APP__API_SECRET_KEY "
            "(configure BRAIN_ENV=development only on trusted workstations)."
        )
        raise HTTPException(
            status_code=401,
            detail=(
                "APP__API_SECRET_KEY must be configured outside development environments "
                "(set BRAIN_ENV=development for local open access)."
            ),
        )

    if not _bearer_ok(secret_stripped, credentials):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def require_sensitive_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> None:
    """Reject requests when APP__API_SECRET_KEY is unset, or Bearer token is invalid.

    Used on high-impact routes (writes, sends, ingestion, CRM seed, tasks) so that
    an open-local configuration cannot mutate external systems or ingest garbage.
    """
    path = request.url.path
    if path in _HEALTH_PUBLIC_PATHS:
        return
    secret = get_settings().app.api_secret_key.get_secret_value().strip()
    if path in _DOCS_PATHS:
        if _is_relaxed_http_auth_environment():
            return
        if not secret or not _bearer_ok(secret, credentials):
            raise _docs_disabled_response()
        return

    if not secret:
        raise HTTPException(
            status_code=401,
            detail=(
                "This endpoint requires APP__API_SECRET_KEY to be set and a Bearer token. "
                "Configure the same value in Authorization: Bearer <key>."
            ),
        )

    if not _bearer_ok(secret, credentials):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
