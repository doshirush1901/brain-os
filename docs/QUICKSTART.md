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

## 6. Cursor MCP (optional)

```bash
cp .cursor/mcp.json.example .cursor/mcp.json
poetry run brain mcp
```

Enable **brain-os** in Cursor **Settings → MCP**. Full guide: [MCP_SETUP.md](MCP_SETUP.md).

## Next steps

| Goal | Doc |
|:-----|:----|
| Private company fork | [FORK_GUIDE.md](FORK_GUIDE.md) |
| Licensing / Pro | [LICENSING.md](LICENSING.md) |
| Acme operator story | [examples/acme/journey.md](../examples/acme/journey.md) |
| Triangulation | [IRA_TRIANGULATION.md](IRA_TRIANGULATION.md) |

## Troubleshooting

| Issue | Fix |
|:------|:----|
| Port 6379 in use | Stop other Redis or change port in `docker-compose.local.yml` |
| `brain health` DB errors | `docker compose -f docker-compose.local.yml up -d` and retry |
| MCP not listed in Cursor | Restart Cursor after editing `.cursor/mcp.json` |
| Lock file busy | Only one `brain` / `ira` process per data dir |

See [SECURITY.md](../SECURITY.md) to report issues.
