# Quickstart — Brain OS on Mac

~15 minutes from clone to first `brain ask` on the **Acme** demo.

## Prerequisites

- macOS on Apple Silicon (Mac Studio / MacBook Pro recommended)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- [Poetry](https://python-poetry.org/docs/#installation) 1.8+
- `OPENAI_API_KEY` in `.env`

## 1. Clone and configure

```bash
git clone https://github.com/doshirush1901/brain-os.git
cd brain-os
cp .env.example .env
# Edit .env — set OPENAI_API_KEY=sk-...
```

## 2. Bootstrap infrastructure

```bash
./scripts/macpro/bootstrap.sh
```

Starts Postgres, Qdrant, Neo4j, and Redis via `docker-compose.local.yml`.

## 3. Install and verify

```bash
poetry install
poetry run brain health
```

Expect green checks for Docker services you enabled.

## 4. License tier (demo)

```bash
poetry run brain activate --trial   # 14-day Pro trial for MCP smoke tests
poetry run brain seed-acme
```

## 5. First query

```bash
poetry run brain ask "Summarize the Acme demo CRM pipeline" --json
```

## 5b. Acme “wow path” (recommended)

Load demo docs into Qdrant and run an account brief on **Northwind Components**. Step-by-step: [ACME_DEMO_PATH.md](ACME_DEMO_PATH.md).

## 6. Cursor MCP (optional)

```bash
cp .cursor/mcp.json.example .cursor/mcp.json
poetry run brain mcp
```

Enable **brain-os** in Cursor **Settings → MCP**. Full guide: [MCP_SETUP.md](MCP_SETUP.md).

## Next steps

| Goal | Doc |
|:-----|:----|
| Peer / friend path | [FRIEND_FORK.md](FRIEND_FORK.md) |
| Build your KB | [BUILD_YOUR_KB.md](BUILD_YOUR_KB.md) |
| Operator governance | [OPERATOR_GOVERNANCE.md](OPERATOR_GOVERNANCE.md) |
| Private company fork | [FORK_GUIDE.md](FORK_GUIDE.md) |
| Licensing / Pro | [LICENSING.md](LICENSING.md) |
| Acme operator story | [examples/acme/journey.md](../examples/acme/journey.md) |
| Acme wow path | [ACME_DEMO_PATH.md](ACME_DEMO_PATH.md) |
| Triangulation | [TRIANGULATION.md](TRIANGULATION.md) |

## Troubleshooting

| Issue | Fix |
|:------|:----|
| Port **6379** in use (Redis) | Another stack often owns 6379. In `docker-compose.local.yml`, map Redis to host **6380** (`"6380:6379"`), set `REDIS_URL=redis://localhost:6380/0` in `.env`, then re-run bootstrap. |
| Port 5432 / Neo4j clash | Stop the other compose stack or change host ports in `docker-compose.local.yml` and matching `.env` URIs. Neo4j in this compose is published on host **7688** (`NEO4J_URI=bolt://localhost:7688`). |
| `brain health` DB errors | `docker compose -f docker-compose.local.yml up -d` and retry; confirm `DATABASE_URL` database name matches compose (`brain_crm` by default). |
| MCP not listed in Cursor | Restart Cursor after editing `.cursor/mcp.json` |
| Lock file busy | Only one `brain` process per data dir |
| Poetry version error | `pyproject.toml` must use PEP 440 (e.g. `0.1.0`, not `0.1.0-skeleton`) |

See [SECURITY.md](../SECURITY.md) to report issues.
