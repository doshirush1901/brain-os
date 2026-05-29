# Brain OS — MCP setup (Cursor & Claude Code)

Brain OS exposes a **local stdio MCP server** so Cursor, Claude Code, and Claude Desktop can call tools (`search_knowledge`, `get_account_brief`, `draft_email`, …) against **your** fork on **your** machine.

**Not hosted:** there is no public `mcp.brain-os.com`. You clone the repo, configure MCP locally, and run `poetry run brain-mcp`.

---

## Prerequisites

1. Clone and install:

   ```bash
   git clone https://github.com/doshirush1901/brain-os.git
   cd brain-os
   cp .env.example .env
   poetry install
   ```

2. Start infrastructure (first time):

   ```bash
   ./scripts/macpro/bootstrap.sh
   # or: docker compose -f docker-compose.local.yml up -d
   ```

3. Add API keys in `.env` (`OPENAI_API_KEY`, optional `VOYAGE_API_KEY`, `DATABASE_URL`).

4. Optional Pro/trial (unlocks extra MCP tools):

   ```bash
   poetry run brain activate --trial
   ```

---

## Cursor

1. Copy the example config:

   ```bash
   mkdir -p .cursor
   cp .cursor/mcp.json.example .cursor/mcp.json
   ```

   Or merge the `brain-os` block from `.mcp.json.example` into your existing `.cursor/mcp.json`.

2. Open the **brain-os repo root** as the Cursor workspace folder (`cwd` must be that root so `poetry` finds `pyproject.toml`).

3. **Cursor Settings → MCP** — enable the `brain-os` server. Restart MCP after code or `.env` changes.

4. Smoke test in chat: ask the model to call `get_system_status` or `search_knowledge` (Community tier).

### Troubleshooting (Cursor)

| Symptom | Fix |
|:--------|:----|
| Server won’t start | Run `poetry run brain-mcp` manually in the repo; read stderr |
| No tools listed | `poetry install`; check MCP panel for spawn errors |
| Qdrant/Neo4j errors | `docker compose -f docker-compose.local.yml up -d` |
| Pro tool blocked | `brain activate --trial` or Pro license |

---

## Claude Code (CLI)

Claude Code uses MCP config in the project or user settings. From the **brain-os repo root**:

1. Ensure `poetry run brain-mcp` runs without error (stdio server; logs on stderr).

2. Add an MCP server entry equivalent to:

   ```json
   {
     "command": "poetry",
     "args": ["run", "brain-mcp"],
     "cwd": "/absolute/path/to/brain-os",
     "env": { "BRAIN_DATA_DIR": "/absolute/path/to/brain-os/data" }
   }
   ```

   Use Claude Code’s documented config path for your version (`claude mcp` / project `.mcp.json` — see Anthropic docs).

3. Restart the Claude Code session after editing config.

---

## Claude Desktop

1. Open **Settings → Developer → Edit Config** (`claude_desktop_config.json` on macOS).

2. Merge the `brain-os` block from `claude_desktop_config.example.json` (replace `/ABSOLUTE/PATH/TO/brain-os`).

3. Restart Claude Desktop.

---

## Commands

| Command | Purpose |
|:--------|:--------|
| `poetry run brain-mcp` | Start MCP server (what Cursor/Claude spawn) |
| `poetry run brain mcp` | Same (Typer alias) |
| `poetry run python -m brain_os.interfaces.mcp_server` | Equivalent module entry |

---

## Tool tiers

| Tier | MCP |
|:-----|:----|
| **Community** | ~26 tools — search, brief, CRM read, draft email, ingest (500 doc cap), memory |
| **Pro / trial** | + operator inbox, CRM write, send email, extra graph tools |

Runtime guard: `hardened_mcp_tool` checks `mcp_tool_allowed()` before each call.

Full list: `src/brain_os/licensing/caps.py` (`COMMUNITY_MCP_TOOL_NAMES`, `PRO_MCP_TOOL_NAMES`).

---

## Security

- MCP runs with **your** `.env` and **your** `BRAIN_DATA_DIR` — it can read/write Gmail (if configured), CRM, and memory on that machine.
- Use only on **trusted devices**. Do not expose stdio MCP to the public internet.
- Default email mode is **TRAINING** (draft-first). Sending requires Pro tools **and** operational send policy; see `docs/LICENSING.md`.

---

## Vs private Ira

| | **brain-os** (this repo) | **ira-v3** (private) |
|:--|:--------------------------|:---------------------|
| MCP tools | Tiered starter set | Full operator surface (~197 tools) |
| Data | Acme demo + your fork | Private operator production |

---

## Related

- [docs/LICENSING.md](LICENSING.md) — Community vs Pro
- [docs/BRAND.md](BRAND.md) — assets
- [.mcp.json.example](../.mcp.json.example) — generic MCP clients
