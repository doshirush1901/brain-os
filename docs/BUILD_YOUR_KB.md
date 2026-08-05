# Build your knowledge base

After the Acme demo works, load **your** company documents into the brain. Pattern: **index → ingest** (metadata catalog, then vector + graph). No customer PII belongs in the public upstream — keep production imports in your **private** fork.

Acme-only path (no private data yet): [ACME_DEMO_PATH.md](ACME_DEMO_PATH.md).

---

## Recommended ingest order (“colostrum”)

1. **SOUL.md** — not ingested; system preamble (`cp SOUL.template.md SOUL.md`)
2. **Company truth** — about, positioning, website source-of-truth markdown
3. **Product & specs** — catalogues, manuals, FAQs
4. **Commercial** — sanitized playbooks / process docs (quotes only if you accept the risk in a private repo)
5. **Curated gold** — small set of grounded “must never be wrong” docs (optional)

```bash
# Example prefixes under examples/acme/docs/
poetry run brain index-imports --include-prefix your_company/website_source_of_truth
poetry run brain ingest --include-prefix your_company/website_source_of_truth --workers 5

poetry run brain index-imports --include-prefix your_company/product
poetry run brain ingest --include-prefix your_company/product --workers 5

# Full tree when ready:
poetry run brain index-imports --force
poetry run brain ingest --force --workers 5
poetry run brain reingest-scanned   # OCR path for scanned PDFs
```

Tiny packs: MCP `ingest_document` from Cursor (see [MCP_SETUP.md](MCP_SETUP.md)).

---

## Tools that build the KB

| Tool / command | Phase | Role |
|:---------------|:------|:-----|
| `brain index-imports` | A | LLM metadata per file → imports catalog |
| `brain ingest` | B | Gatekeeper + digestion → Qdrant + Neo4j |
| `brain reingest-scanned` | C | Document AI OCR → re-digest |
| MCP `ingest_document` | B | Single-file ingest from Cursor |
| MCP `parse_document_ai` | — | OCR/invoice/form JSON (no KB write by itself) |
| `brain seed-acme` | — | **CRM demo only**, not your KB |
| `brain ask` / `search_knowledge` | — | Verify retrieval |

---

## Storage

| Component | Purpose |
|:----------|:--------|
| **Qdrant** | Chunk embeddings (Voyage when `VOYAGE_API_KEY` is set) |
| **Neo4j** | Entities / relationships from ingest |
| **Postgres** | CRM (Acme seed or your sync) — parallel to KB |
| **imports metadata / ingest log** | Gatekeeper catalog + dedup |

Community tier: **500 indexed documents** cap. Pro/trial: see [LICENSING.md](LICENSING.md).

---

## Cost and time (expectations)

| Item | Guidance |
|:-----|:---------|
| **API cost** | You pay OpenAI/Anthropic + Voyage; not bundled in the license |
| **Rough API $** | Often tens–low hundreds USD for hundreds–low thousands of typical office files — **not** a guarantee |
| **Wall clock** | Scales with **file count** (and OCR for scan-heavy PDFs), not GB alone |

---

## Verify

```bash
poetry run brain health
poetry run brain ask "Summarize our company from the knowledge base" --json
```

In Cursor (MCP):

```text
Brief me on {Company} — knowledge base + CRM, gaps only; no draft.
```

Before outbound: [TRIANGULATION.md](TRIANGULATION.md) · [OPERATOR_GOVERNANCE.md](OPERATOR_GOVERNANCE.md).

---

## Related

- [FRIEND_FORK.md](FRIEND_FORK.md) — 90-minute + week-1 path
- [FORK_GUIDE.md](FORK_GUIDE.md) — private fork checklist
- [QUICKSTART.md](QUICKSTART.md) — bootstrap
