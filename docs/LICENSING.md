# Brain OS — Licensing (Community vs Pro)

**Licensor (BSL 1.1):** see root `LICENSE` (legal entity parameters).

## Community (default)

- No key required
- **500 indexed documents** cap (`data/brain/license_usage.json`)
- **~26 MCP tools** (search, brief, CRM read, draft email, ingest with cap)
- No operator inbox, send email, or graph/CRM write tools

## Pro / Trial

```bash
poetry run brain activate --trial          # 14-day local trial
poetry run brain activate --key bos_live_xxx # online (needs BRAIN_LICENSE_SERVER_URL)
poetry run brain license status
```

Trial/Pro unlocks:

- Unlimited document ingest (fair use)
- Operator inbox MCP tools
- Pro graph + CRM tools
- Send mail MCP tools (still subject to operational confirm gates)

## Environment

```bash
# Data directory (preferred; BRAIN_DATA_DIR still accepted for migration)
BRAIN_DATA_DIR=/path/to/data

# Optional online activation server
BRAIN_LICENSE_SERVER_URL=http://127.0.0.1:8765

# Dev/stub only — accept bos_test_* keys locally without HTTP
BRAIN_LICENSE_STUB_ACCEPT=1
```

License file: `data/brain/license.json` (gitignored).

## Local license server stub

For development, run the bundled stub (also copied into exports under `scripts/`):

```bash
# Terminal A — from brain-os export root
poetry run python scripts/license_server_stub.py

# Terminal B
export BRAIN_LICENSE_SERVER_URL=http://127.0.0.1:8765
poetry run brain activate --key bos_test_dev_workspace_01
poetry run brain license status
```

Stub endpoints:

- `GET /health`
- `POST /v1/licenses/activate` — body `{"license_key":"bos_test_...","product":"brain-os"}`

Optional: `BRAIN_LICENSE_STUB_HOST`, `BRAIN_LICENSE_STUB_PORT`, `BRAIN_LICENSE_STUB_PRO_DAYS`.

## Runtime MCP guard

Every MCP tool wrapped with `hardened_mcp_tool` checks `mcp_tool_allowed()` before execution
(defense in depth if tier expires or a Pro tool is invoked on Community).
