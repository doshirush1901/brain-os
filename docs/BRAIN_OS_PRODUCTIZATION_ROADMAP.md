# Brain OS — Productization Roadmap & Public Skeleton Spec

**Purpose:** Blueprint for extracting a vertical-agnostic **Brain OS** SKU from Ira v3 into a new public Git repository that companies fork, run on **Mac Pro (Apple Silicon)**, and extend with their own agents, data, and workflows — using Cursor / Claude Code as the operator surface.

**Audience:** Machinecraft platform team (build) · design partners · future fork customers.

**Status:** Export pipeline landed (`scripts/export_brain_os.py`); public repo [brain-os](https://github.com/doshirush1901/brain-os). GTM/legal: `docs/BRAIN_OS_GTM.md`.

---

## 1. Product definition (one sentence)

> **Brain OS** is a local-first, multi-agent operating system for a company’s knowledge, relationships, and approved actions — vector + graph + memory RAG, human-in-the-loop outbound, and a forkable agent pantheon — packaged for Mac Pro deployment and operated from Cursor via MCP.

**Not:** a chatbot, not a generic LangChain starter, not Machinecraft-in-a-box.

**Is:** the **engine + governance patterns** Ira proved (pipeline, pantheon, triangulation, corrections, operator inbox) with **Acme Corp** synthetic demos and empty slots for vertical customization.

---

## 2. Repo strategy: three repos, one journey

| Repo | Role | Visibility |
|:-----|:-----|:-----------|
| **`ira-v3`** (this tree) | Canonical operator system for Machinecraft; full vertical depth | Private operator remote |
| **`brain-os`** (new) | Forkable skeleton: engine, generic agents, demo data, Mac bootstrap | **Public** (Apache-2.0 or BSL 1.1 — see §12) |
| **`{customer}/brain`** (fork) | Customer’s private fork: prompts, CRM extensions, imports, `.env` | Customer private |

**Customer journey:**

```text
git clone https://github.com/{org}/brain-os
cp .env.example .env          # LLM keys, optional Gmail
./scripts/macpro/bootstrap.sh   # Docker infra + health
poetry run brain ingest ./examples/acme_docs/
poetry run brain ask "What do we know about Acme?" --json
# Cursor: enable brain-os MCP → start customizing agents/prompts/
```

---

## 3. What to copy, strip, and rewrite

### 3.1 Copy wholesale (~60% of engine value)

These modules are **vertical-agnostic** or need only rename (`ira` → `brain`):

| Source (ira-v3) | Destination (brain-os) | Notes |
|:----------------|:-----------------------|:------|
| `src/brain_os/pipeline.py`, `pipeline_phases/`, `pipeline_loop.py` | `src/brain_os/pipeline/` | Core 17-step flow; rename config keys |
| `src/brain_os/agents/base_agent.py` | `src/brain_os/agents/base.py` | ReAct loop, default tools |
| `src/brain_os/agents/athena.py`, `sphinx.py`, `clio.py`, `mnemon.py`, `mnemosyne.py`, `vera.py`, `aletheia.py`, `aegis.py`, `graphe.py`, `metis.py`, `sophia.py`, `nemesis.py`, `gapper.py` | Same names under `agents/` | Orchestration, KB, memory, safety |
| `src/brain_os/pantheon.py` | `src/brain_os/pantheon.py` | Slim default roster (§4) |
| `src/brain_os/brain/retriever.py`, `qdrant_manager.py`, `knowledge_graph.py`, `document_ingestor.py`, `embeddings*.py`, `semantic_chunker*.py` | `src/brain_os/brain/` | RAG stack |
| `src/brain_os/memory/*` (10 subsystems) | `src/brain_os/memory/` | Dream mode optional flag |
| `src/brain_os/services/llm_client.py`, `tool_runner.py` | `src/brain_os/services/` | LLM + tool reliability |
| `src/brain_os/middleware/`, `message_bus.py` | Same | |
| `src/brain_os/interfaces/server.py` (slim), CLI core, MCP runtime | `src/brain_os/interfaces/` | Fewer routes in v1 |
| `docker-compose.local.yml` | Root | Postgres, Qdrant, Neo4j, Redis |
| `scripts/public_repo_guard.py` | Adapt for `brain-os` | Already have public-safe patterns |
| `prompts/*_system.txt` for copied agents | `prompts/` | **De-Machinecraft** (§3.3) |
| `tests/` patterns | `tests/` | Target 60% coverage floor at launch |

### 3.2 Copy as **patterns**, reimplement slimmer (~25%)

| Ira pattern | Brain OS v1 | Defer to v1.1+ |
|:------------|:------------|:-----------------|
| `systems/operator_inbox.py` | Generic approval queue (email, CRM note, doc publish) | Quote-specific types |
| `services/account_brief.py` | `company_brief` — triangle only | Hex legs as plugins |
| `services/triangulation_gaps.py` | Same API, generic gap labels | Industry formulas |
| `interfaces/mcp_server.py` | **~40 core MCP tools** (see §6) | 200-tool surface |
| `data/crm.py` | Generic: Company, Contact, Deal, Activity | Quote line items, machine SKUs |
| `web-ui/` | Optional: `/chat`, `/operator`, `/crm` only | Board portal, recruitment |
| Faithfulness / guardrails | Tier 2–3 only at launch | Google Check Grounding |

### 3.3 **Do not copy** — Machinecraft vertical (~15% of repo bulk, 40% of complexity)

Delete or replace with stub interfaces in brain-os:

| Ira module / area | Reason |
|:------------------|:-------|
| `agents/na_sales.py`, `pf1_react_tools.py`, `hephaestus.py` (machine specs), `quotebuilder.py`, `maestro.py` (Wolfram thermoforming) | Industrial OEM domain |
| `systems/thermoformer_*`, `na_*`, `tinder_*`, `bfp_*`, `vapi_*`, `onshoring_*` | Sales motion specific to Machinecraft |
| `workflows/presets/na_*`, `services/machinecraft_quote_prep.py` | Vertical workflows |
| `knowledge/formpack_bd.py`, `active_customer_application_enricher.py` | Customer-specific |
| `interfaces/commands/leads/*` (most), thermoformer MCP tools | Lead-gen vertical |
| `data/knowledge/demo_*` thermoforming copy | Replace with Acme synthetic pack |

### 3.4 Rewrite for vertical agnosticism

| Artifact | Brain OS replacement |
|:---------|:---------------------|
| `SOUL.md` | `SOUL.template.md` + `examples/acme/SOUL.md` (synthetic Acme Services Inc.) |
| `VISION.md` | `VISION.md` — explicitly fork-friendly platform |
| `AGENTS.md` | **12 starter agents** + “add your own” guide |
| `data/knowledge/outbound_proof_artifacts.json` | `examples/acme/proof_registry.json` |
| CRM seed data | `examples/acme/crm_seed.sql` or JSON |
| `.env.example` | Mac-local defaults; Gmail optional |

---

## 4. Starter pantheon (Brain OS v1 — 12 agents)

Fork customers extend via `agents/{name}.py` + `prompts/{name}_system.txt` + pantheon registration.

| Agent | Role | Generic tools |
|:------|:-----|:--------------|
| **Athena** | Orchestrator | `ask_agent`, routing |
| **Clio** | Librarian / KB search | `search_knowledge`, graph |
| **Alexandros** | Document archive | imports fallback |
| **Prometheus** | CRM / pipeline | `search_crm`, deals |
| **Calliope** | External comms drafts | `draft_email` (draft-only default) |
| **Artemis** | Mailbox intelligence | `search_emails` |
| **Iris** | Web / external research | `web_search`, scrape |
| **Plutus** | Finance summaries | read-only ledgers in v1 |
| **Atlas** | Projects / milestones | generic project log |
| **Mnemon** | Corrections authority | correction ledger |
| **Vera** | Fact check | faithfulness |
| **Sphinx** | Clarifying questions | gate vague queries |

**Optional plug-in slots (empty stubs in repo):**

- `agents/industry.py` — customer implements
- `agents/compliance.py` — regulated verticals
- `workflows/custom/` — YAML preset loader already in Ira pattern

---

## 5. Target repository layout (`brain-os`)

```text
brain-os/
├── README.md                    # Fork me · Mac Pro · 15-min quickstart
├── SOUL.template.md
├── VISION.md
├── AGENTS.md
├── LICENSE
├── pyproject.toml               # package name: brain-os
├── docker-compose.local.yml
├── .env.example
├── scripts/
│   ├── macpro/
│   │   ├── bootstrap.sh         # Docker + poetry + alembic + health
│   │   ├── healthcheck.sh
│   │   └── backup.sh            # Qdrant + Neo4j + Postgres snapshot
│   ├── export_brain_os.py       # Maintainer-only extraction (private ira-v3)
│   └── public_repo_guard.py
├── src/brain_os/
│   ├── pipeline/
│   ├── agents/                  # 12 starters + _template_agent.py
│   ├── brain/                   # RAG
│   ├── memory/
│   ├── systems/                 # operator_inbox, sensory, voice (slim)
│   ├── services/
│   ├── interfaces/
│   │   ├── cli/
│   │   ├── mcp_server.py        # ~40 tools
│   │   └── server.py            # optional API
│   ├── data/                    # generic CRM models
│   └── config.py
├── prompts/                     # de-verticalized system prompts
├── examples/
│   └── acme/                    # synthetic company pack
│       ├── SOUL.md
│       ├── docs/                # PDF/markdown to ingest
│       ├── crm_seed.json
│       ├── proof_registry.json
│       └── journey.md             # 7-day fork tutorial
├── docs/
│   ├── FORK_GUIDE.md
│   ├── MACPRO_DEPLOY.md
│   ├── MCP_OPERATOR_GUIDE.md
│   ├── TRIANGULATION.md
│   └── ADDING_AN_AGENT.md
├── alembic/
└── tests/
```

---

## 6. MCP tool surface (v1 — 40 tools)

Tier A for Cursor operators (fork customers enable in `.cursor/mcp.json`):

| Category | Tools |
|:---------|:------|
| Query | `query_brain`, `search_knowledge`, `ask_agent` |
| Account | `get_company_brief`, `search_crm`, `get_deal`, `list_deals` |
| Mail | `search_emails`, `read_email_thread`, `draft_email` |
| Memory | `recall_memory`, `store_memory`, `submit_correction` |
| Graph | `find_related_entities`, `find_company_contacts` |
| Ingest | `ingest_document`, `parse_document_ai` (optional) |
| Operator | `operator_inbox_summary`, `operator_inbox_decide` |
| System | `get_system_status`, `discover_tools_for_query` |

**No send tool in public default** — `draft_email` only; send behind `BRAIN_OPERATIONAL_MODE=true` + explicit CLI/MCP confirm gate (same pattern as Ira).

---

## 7. Mac Pro appliance packaging

### 7.1 Reference hardware

| Spec | Minimum | Recommended |
|:-----|:--------|:------------|
| Machine | Mac Studio M2 Max | **Mac Pro / Mac Studio M4 Max–Ultra** |
| RAM | 64 GB | **128 GB** (local embedding + Docker) |
| Storage | 1 TB SSD | 2 TB (imports + backups) |
| Network | Ethernet | Static IP or Tailscale for team |

### 7.2 Software stack (all local)

```text
┌─────────────────────────────────────────┐
│  Cursor / Claude Code (operator UI)      │
│  MCP → localhost:brain-mcp               │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  brain-os CLI / optional FastAPI :8000   │
└──────────────────┬──────────────────────┘
                   │
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
  Qdrant       Neo4j         Postgres
  Redis        Mem0 (opt)    Gmail OAuth (opt)
```

### 7.3 Bootstrap contract

`scripts/macpro/bootstrap.sh` must:

1. Verify Docker Desktop + Poetry + Python 3.11+
2. `docker compose up -d` — wait for healthy
3. `alembic upgrade head`
4. `poetry run brain seed --demo acme` (optional)
5. Print MCP snippet for Cursor + `brain health --json`

**Update channel:** `brain upgrade --check` pulls tagged releases; customer forks merge upstream quarterly.

---

## 8. Extraction workflow (maintainer, from ira-v3)

Phase 0 before first public tag:

1. Run `python scripts/export_brain_os.py` (to be written — mirror `export_ira_universe_repo.py`)
2. Path remap `ira` → `brain_os`, strip forbidden modules (§3.3)
3. Prompt pass: LLM-assisted de-verticalize + human review
4. `public_repo_guard.py` green + gitleaks on archive
5. Tag `v0.1.0-skeleton` — **no Machinecraft strings** in tracked tree

**Cadence:** Monthly upstream sync from ira-v3 engine fixes; quarterly semver for brain-os.

---

## 9. Twenty-four month engineering roadmap

### Year 1 — Skeleton → design partners

| Quarter | Engineering milestone | Exit criteria |
|:--------|:----------------------|:--------------|
| **Q1** | Export script + brain-os repo + Acme demo pack | Fork → `brain ask` works in <30 min on Mac |
| **Q2** | Mac bootstrap, 40 MCP tools, operator inbox | 3 design partners on private forks |
| **Q3** | Triangulation brief, correction ledger, ingest hardening | Partner ingests 1k+ docs without manual fixes |
| **Q4** | Upgrade channel, backup/restore, 60% test coverage | v1.0.0 tagged; public docs complete |

### Year 2 — Product → repeatable SKU

| Quarter | Engineering milestone | Exit criteria |
|:--------|:----------------------|:--------------|
| **Q5** | Multi-tenant-ready config (single Mac, multi workspace) | Agency can host 3 clients on 1 box |
| **Q6** | Plugin SDK: `agents/_template`, workflow YAML loader | Customer adds agent without core PR |
| **Q7** | Observability dashboard (pipeline timings, memory growth) | Support can diagnose remotely (read-only) |
| **Q8** | Optional cloud relay (no data — telemetry only) + SOC2 prep | Enterprise procurement unblock |

---

## 10. Twenty-four month GTM roadmap

### Positioning

**Category:** Private AI operating system / “second brain with hands (draft-only by default)”  
**ICP v1:** 20–200 employee B2B companies with dense documents + email + CRM, data-sensitive (manufacturing, professional services, healthcare admin, family holding companies)  
**Beachhead:** Same profile as Machinecraft — **founder-led industrial B2B** — but **Acme** in all public materials.

### GTM phases

| Quarter | Motion | Target |
|:--------|:-------|:-------|
| **Q1** | Stealth design partners (3–5) | Free fork + hands-on setup; learn vertical gaps |
| **Q2** | **Founder-led outbound** + case study (“Acme deployed in 2 weeks”) | 10 paid pilots |
| **Q3** | **Mac Pro appliance offer** + install SOW | 5 appliances shipped |
| **Q4** | Channel: Cursor/AI consultancies as implementers | 2 partner certifications |
| **Q5–Q6** | Self-serve fork + paid **Brain OS Pro** support tier | 30 paying orgs |
| **Q7–Q8** | Enterprise tier + compliance pack | 10 enterprises @ higher ACV |

### Sales collateral (public repo)

- `examples/acme/journey.md` — 7-day fork path
- `docs/MACPRO_DEPLOY.md` — IT checklist
- Video: Cursor + MCP + `get_company_brief` demo (synthetic only)

---

## 11. Pricing model (recommended)

### 11.1 Tiers

| Tier | Year 1 price (USD) | Renewal | Includes |
|:-----|:-------------------|:--------|:---------|
| **Community** | $0 | — | Public repo, no SLA, Discord |
| **Pro (software)** | **$36k** | $18k/yr | Priority releases, 20h onboarding remote, email support |
| **Appliance** | **$95k** | $24k/yr | Mac Pro configured, shipped, 40h onsite/virtual install, backup playbook |
| **Enterprise** | **$150k–$250k** | 18% of Y1 | Multi-workspace, custom agent pack (3 agents), compliance doc pack, 99.5% SLA |
| **Implementer cert** | **$15k** per seat | $5k/yr | Train consultancies to deploy; lead referral |

### 11.2 Services (high margin, optional)

| SKU | Price |
|:----|:------|
| Vertical agent pack (5 agents + prompts) | $40k–$80k one-time |
| Historical email + doc ingest (≤500GB) | $25k–$50k |
| Quarterly “dream tuning” retainer | $8k/quarter |

### 11.3 Unit economics target (Year 2)

| Metric | Target |
|:-------|:-------|
| Blended ACV | **$85k** |
| Gross margin (software) | **75%** |
| Gross margin (appliance) | **55%** |
| CAC payback | **<14 months** |
| Net revenue retention | **>115%** |

### 11.4 Path to $100M narrative (Year 3–4)

| ARR | Mix | Valuation @ 10× |
|:----|:----|:----------------|
| **$10M** | ~80 Pro + 25 Appliance + 5 Enterprise | **$100M** |
| **$12.5M** | @ 8× multiple (hardware-heavy) | **$100M** |

**Proof points for investors/acquirers:**

- 15+ reference logos with measurable time saved (brief prep, draft quality)
- <5% logo churn on Pro+
- Repeatable install runbook (<10 business days to first `company_brief`)
- Upstream engine commits from ira-v3 → brain-os visible in changelog

---

## 12. Licensing recommendation

| Option | Pros | Cons |
|:-------|:-----|:-----|
| **Apache 2.0** | Maximum fork adoption | Hard to monetize engine |
| **BSL 1.1 → Apache after 3yr** | Adopt now, convert competitors later | Legal review needed |
| **Dual: MIT core + commercial appliance** | Community + appliance margin | Two codebases drift |

**Recommendation:** **BSL 1.1** on `src/brain_os/` with change date + Apache conversion; **Appliance image** and **vertical packs** sold under commercial license. Customer forks stay private; contributions back via PR optional.

---

## 13. Fork customer checklist (ship in `docs/FORK_GUIDE.md`)

```text
Day 0  — Fork brain-os → private remote
Day 1  — Copy SOUL.template.md → SOUL.md (your company voice)
Day 2  — Mac bootstrap + ingest your /docs
Day 3  — Wire Gmail OAuth (optional) + CRM seed
Day 4  — Cursor MCP + first get_company_brief
Day 5  — Duplicate agents/_template_agent.py → your industry agent
Day 6  — proof_registry.json (your approved URLs only)
Day 7  — Operator inbox dry run; TRAINING mode only
Week 2 — First custom workflow YAML
Month 1 — Mnemon corrections + dream cycle review
```

---

## 14. Immediate next steps (Machinecraft team)

1. **Name + GitHub org** for `brain-os` public repo
2. **Write `scripts/export_brain_os.py`** — module allowlist from §3.1–3.3
3. **Author Acme synthetic pack** — replace thermoforming demos
4. **Legal:** BSL review + trademark “Brain OS” / “Ira” separation
5. **Design partner LOI** — 3 companies, free Q1 in exchange for case study rights
6. **Do not** generalize ira-v3 in place — extract to new repo (per VISION.md)

---

## 15. Relationship to existing repos

| Repo | Relationship |
|:-----|:-------------|
| **ira-v3** | Upstream engine R&D; private operator |
| **ira-universe** | Visitor-safe **explain Ira** corpus (~12 MCP tools); stays separate, links to brain-os docs |
| **brain-os** | Forkable **build your own** skeleton |

---

*Generated from Ira v3 productization planning — 2026-05-30.*
