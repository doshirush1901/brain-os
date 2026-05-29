# Fork guide — your private Brain OS

Use this checklist after you fork [brain-os](https://github.com/doshirush1901/brain-os) to a **private** repository. Customer data, Gmail tokens, and production CRM never belong in the public upstream.

## Day 0 — Fork and remote

```bash
# GitHub: Fork → create private repo your-org/brain
git clone git@github.com:your-org/brain.git
cd brain
git remote add upstream https://github.com/doshirush1901/brain-os.git
```

- [ ] Repo is **private**
- [ ] Default branch protected (optional)
- [ ] Team access via SSO / org permissions

## Day 1 — Company voice

```bash
cp SOUL.template.md SOUL.md
# Edit SOUL.md: identity, values, outbound boundaries (no private OEM copy-paste)
```

- [ ] `SOUL.md` committed only in **your** fork
- [ ] Review `prompts/calliope_system.txt` for your proof protocol

## Day 2 — Mac bootstrap and health

```bash
cp .env.example .env   # OPENAI_API_KEY required; VOYAGE optional
./scripts/macpro/bootstrap.sh
poetry run brain health
```

- [ ] Docker stack up (Postgres, Qdrant, Neo4j, Redis)
- [ ] `brain health` reports OK for services you enabled

## Day 3 — Your documents and CRM

```bash
poetry run brain ingest ./your-docs/    # when ingest CLI is wired in your fork
poetry run brain seed-acme              # optional: keep Acme for tests
# Replace examples/acme/ or add your_company/ with synthetic-safe seeds first
```

- [ ] No customer PII in commits
- [ ] `public_repo_guard.py` passes if you open-source a subtree later

## Day 4 — Gmail and CRM (optional)

- [ ] Google OAuth token path in `.env` (never commit tokens)
- [ ] CRM seed or Apollo sync per your process
- [ ] Email mode: **training** until operators trust drafts

## Day 5 — Cursor MCP

```bash
cp .cursor/mcp.json.example .cursor/mcp.json
poetry run brain mcp   # smoke test
```

See [MCP_SETUP.md](MCP_SETUP.md). Enable the server in **Cursor → Settings → MCP**.

- [ ] `get_account_brief` or `brain ask` works on a test company
- [ ] Community tier tool cap understood ([LICENSING.md](LICENSING.md))

## Day 6 — Proof registry and outbound

- [ ] Copy `examples/acme/proof_registry.json` → your approved URLs only
- [ ] Draft-only until explicit human **send**
- [ ] Run triangulation before outbound ([IRA_TRIANGULATION.md](IRA_TRIANGULATION.md))

## Day 7 — Operator inbox dry run

- [ ] Approval queue configured (training mode)
- [ ] No auto-send in production CRM mail

## Week 2 — Custom agents

- [ ] Duplicate a starter agent under `src/brain_os/agents/`
- [ ] Register in `pantheon.py` and add `prompts/your_agent_system.txt`
- [ ] Track gaps in [PANTHEON_SLIM_TODO.md](PANTHEON_SLIM_TODO.md)

## Month 1 — Memory and corrections

- [ ] Corrections ledger reviewed (Mnemon)
- [ ] Optional dream / consolidation cycle if enabled in your fork

## Upstream merges

```bash
git fetch upstream
git merge upstream/main   # resolve conflicts in your private overlays
```

Prefer keeping **your** `SOUL.md`, `.env`, and `data/` out of merge commits to upstream.

## Design partners

If you joined as a design partner, label issues `design-partner` and avoid posting real customer names in public forks of upstream.
