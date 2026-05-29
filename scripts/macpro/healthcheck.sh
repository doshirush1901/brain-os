#!/usr/bin/env bash
# Quick infra probe — no Poetry required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.local.yml}"

command -v docker >/dev/null 2>&1 || { echo "docker: missing"; exit 1; }

docker compose -f "$COMPOSE_FILE" ps
curl -sf "http://127.0.0.1:6333/healthz" >/dev/null && echo "qdrant: ok" || echo "qdrant: down"
curl -sf "http://127.0.0.1:7475" >/dev/null && echo "neo4j: ok" || echo "neo4j: down (port 7475)"
redis-cli -h 127.0.0.1 ping 2>/dev/null | grep -q PONG && echo "redis: ok" || echo "redis: down"
