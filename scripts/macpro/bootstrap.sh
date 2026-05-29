#!/usr/bin/env bash
# Brain OS — Mac Pro / Mac Studio local bootstrap (Docker + Poetry + health).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.local.yml}"
MAX_WAIT="${MAX_WAIT:-120}"

log() { printf '==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "Docker Desktop is required."
command -v poetry >/dev/null 2>&1 || die "Poetry is required (https://python-poetry.org/)."

if [[ ! -f "$COMPOSE_FILE" ]]; then
  die "Missing $COMPOSE_FILE — run from Brain OS repo root."
fi

if [[ -f .env.example && ! -f .env ]]; then
  cp .env.example .env
  log "Created .env from .env.example — add LLM and optional Gmail keys."
fi

log "Starting Docker services ($COMPOSE_FILE)"
docker compose -f "$COMPOSE_FILE" up -d

wait_healthy() {
  local service="$1"
  local elapsed=0
  while (( elapsed < MAX_WAIT )); do
    if docker compose -f "$COMPOSE_FILE" ps "$service" 2>/dev/null | grep -q "(healthy)"; then
      log "$service is healthy"
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  die "$service did not become healthy within ${MAX_WAIT}s"
}

for svc in qdrant postgres redis neo4j; do
  if docker compose -f "$COMPOSE_FILE" config --services 2>/dev/null | grep -qx "$svc"; then
    wait_healthy "$svc" || true
  fi
done

log "Poetry install"
poetry install --no-interaction

if [[ -d alembic ]]; then
  log "Alembic migrations"
  poetry run alembic upgrade head || log "Alembic skipped (no DATABASE_URL or first-run)"
fi

log "Acme demo pack (proof registry; CRM when Postgres is ready)"
poetry run brain seed-acme || log "Acme seed skipped — rerun after alembic + DATABASE_URL"

log "Brain OS health check"
poetry run brain health

log "Bootstrap complete."
log "Next: cp .cursor/mcp.json.example .cursor/mcp.json — see docs/MCP_SETUP.md"
log "Demo data: examples/acme/ — see examples/acme/journey.md"
