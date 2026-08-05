# Friend fork — your company brain

Peer founders: shortest path from “I want a digital company brain” to a **private** Brain OS on your Mac.

| Layer | Role |
|:------|:-----|
| **[forkmybrain.org](https://forkmybrain.org)** | Story, SOUL preview, clinic |
| **This repo ([brain-os](https://github.com/doshirush1901/brain-os))** | Forkable code + Acme demo |
| **Your private fork** | Your `SOUL.md`, docs, CRM, mail — never push customer data upstream |

The maintainer’s private operator stack stays private. **Brain OS** is the public engine + governance patterns so you build **your** brain.

---

## Hard rules

1. Fork to a **private** GitHub repo (or keep production data gitignored).
2. No real customer names, mail, quotes, or tokens in commits to the public upstream.
3. Outbound stays **draft-only** until a human explicitly says **send** for that exact message.
4. Do **not** wire the full Brain MCP (send / CRM write) into Cursor **Cloud Agents** — desktop only for mutation. See [OPERATOR_GOVERNANCE.md](OPERATOR_GOVERNANCE.md).

---

## 90-minute first win

Smoke checklist (print and tick):

```bash
git clone https://github.com/doshirush1901/brain-os.git   # or: fork → clone your private remote
cd brain-os
cp .env.example .env
# Set OPENAI_API_KEY=… (optional VOYAGE_API_KEY)

./scripts/macpro/bootstrap.sh
poetry install
poetry check
poetry run brain health
poetry run brain activate --trial
poetry run brain seed-acme
poetry run brain ask "Summarize the Acme demo CRM" --json
```

| Step | Pass when |
|:-----|:----------|
| `poetry check` | exits 0 |
| `brain health` | Docker services you enabled look OK |
| `brain activate --trial` | trial / Pro status shown |
| `brain seed-acme` | Acme CRM seed loaded |
| `brain ask … --json` | JSON with a CRM-style summary (not empty error) |

**Acme wow path** (docs into KB + Northwind brief): [ACME_DEMO_PATH.md](ACME_DEMO_PATH.md).

**Cursor MCP smoke:**

```bash
cp .cursor/mcp.json.example .cursor/mcp.json
# Cursor → Settings → MCP → enable brain-os
poetry run brain mcp   # optional local smoke
```

Ask in chat: `get_system_status` or `search_knowledge` on Acme. Full guide: [MCP_SETUP.md](MCP_SETUP.md).

If Redis port **6379** is already taken (another stack on the machine), see [QUICKSTART.md](QUICKSTART.md) § Troubleshooting — map host **6380** and set `REDIS_URL`.

---

## Week-1 path (your company)

1. **Voice** — `cp SOUL.template.md SOUL.md` and edit identity, values, outbound boundaries. Commit only in **your** private fork.
2. **Knowledge** — put company docs under `examples/acme/docs/your_company/` and follow [BUILD_YOUR_KB.md](BUILD_YOUR_KB.md) (index → ingest order).
3. **One account** — `poetry run brain brief "Your Prospect" --json` (or MCP `get_account_brief`). Gaps → mark **UNVERIFIED**; do not invent mail history.
4. **Outbound** — draft only; triangulation before any send ([TRIANGULATION.md](TRIANGULATION.md), [OPERATOR_GOVERNANCE.md](OPERATOR_GOVERNANCE.md)).
5. **Detail checklist** — [FORK_GUIDE.md](FORK_GUIDE.md) (Day 0–7).

Community tier caps: 500 indexed docs · ~26 MCP tools · draft-only email. Pro/trial: [LICENSING.md](LICENSING.md).

---

## Get help

- Setup mirror: [forkmybrain.org/setup](https://forkmybrain.org/setup)
- Free clinic: email [operator@example.com](mailto:operator@example.com?subject=Brain%20OS%20clinic) or book via [forkmybrain.org/setup](https://forkmybrain.org/setup) (Calendly / clinic links on the site)

---

## Related

| Doc | Use |
|:----|:----|
| [QUICKSTART.md](QUICKSTART.md) | Clone → first ask |
| [BUILD_YOUR_KB.md](BUILD_YOUR_KB.md) | Index + ingest your docs |
| [OPERATOR_GOVERNANCE.md](OPERATOR_GOVERNANCE.md) | Autonomy ladder, Cloud vs desktop |
| [FORK_GUIDE.md](FORK_GUIDE.md) | Full private-fork checklist |
| [PUBLIC_PRIVATE_BOUNDARY.md](PUBLIC_PRIVATE_BOUNDARY.md) | What never goes public |
