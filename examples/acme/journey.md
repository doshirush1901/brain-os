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
```

## Day 2 — Ingest demo docs

```bash
poetry run brain ingest examples/acme/docs/   # when ingest CLI lands
# Until then: copy docs into examples/acme/docs/ (gitignored) and ingest via operator path
```

## Day 3 — CRM seed

Load `examples/acme/crm_seed.json` into Postgres (manual or seed command when wired).

## Day 4 — Cursor MCP

Copy `.mcp.json.example` → `.cursor/mcp.json`, set `cwd` to this repo.

## Day 5 — First brief

Ask: *Brief me on Northwind Components — CRM, mail, KB; gaps only.*

## Day 6 — Custom agent

Copy `src/brain_os/agents/_template_agent.py` → `compliance.py`, register in `pantheon.py`.

## Day 7 — Operator dry run

Draft-only email to a synthetic contact; approve via operator inbox when wired.
