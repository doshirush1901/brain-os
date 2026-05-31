# Acme fork journey — 7 days (PUBLIC DEMO)

PUBLIC DEMO DATA — fully synthetic.

## Day 0 — Fork

```bash
git clone https://github.com/{org}/brain-os.git acme-brain
cd acme-brain
git remote add private git@github.com:your-org/acme-brain.git
```

## Day 1 — Bootstrap Mac

```bash
cp examples/acme/SOUL.md ./SOUL.md   # or merge into your own soul
./scripts/macpro/bootstrap.sh
poetry run brain activate --trial
poetry run brain seed-acme
```

## Day 2 — Ingest demo docs

Use MCP `ingest_document` on each file under `examples/acme/docs/` (see [docs/ACME_DEMO_PATH.md](../../docs/ACME_DEMO_PATH.md)).

## Day 3 — CRM + pipeline

```bash
poetry run brain ask "List Acme demo deals by stage and USD value" --json
```

## Day 4 — Cursor MCP

Copy `.mcp.json.example` → `.cursor/mcp.json`, set `cwd` to this repo.

## Day 5 — First brief

Ask: *Brief me on Northwind Components — CRM, knowledge base, gaps only. Contact: jordan.lee@northwind-demo.example*

## Day 6 — Custom agent

Copy `src/brain_os/agents/_template_agent.py` → `compliance.py`, register in `pantheon.py`.

## Day 7 — Operator dry run

Draft-only email to a synthetic contact; approve via operator inbox when on Pro trial.
