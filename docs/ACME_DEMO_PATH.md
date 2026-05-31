# Acme demo — “wow path” (~20 minutes)

PUBLIC DEMO DATA — fully synthetic. This walkthrough uses **Northwind Components** as the anchor account.

Goal: feel how an Brain OS-class operator combines **CRM + knowledge base + triangulation** before you fork for your company.

## Prerequisites

Complete [QUICKSTART.md](QUICKSTART.md) through **step 4** (`brain activate --trial`, `brain seed-acme`).

## Step A — Seed CRM (if not done)

```bash
poetry run brain seed-acme
```

Expect Postgres deals for Northwind, Summit, Harbor, Redwood, and Lakeview (see `examples/acme/crm_seed.json`).

## Step B — Ingest demo knowledge (MCP)

In **Cursor** with Brain OS MCP enabled (`docs/MCP_SETUP.md`):

```text
Ingest these files into the knowledge base (one call per file):
  examples/acme/docs/product_overview.md
  examples/acme/docs/sales_playbook.md
  examples/acme/docs/northwind_account_context.md
  examples/acme/docs/demo_kb_faq.md
  examples/acme/docs/imported_threads/northwind_thread.txt
```

Or from a shell with MCP wired, use the `ingest_document` tool with absolute paths under your clone.

Community tier: **500 document cap** — this pack is five small files.

## Step C — CLI smoke test

```bash
poetry run brain ask "Summarize the Acme demo CRM pipeline by stage and value" --json
poetry run brain ask "What do we know about Northwind Components from CRM and docs?" --json
```

## Step D — Account brief (Cursor MCP)

Paste into Cursor chat (MCP **get_account_brief** or **brain ask**):

```text
Brief me on Northwind Components — CRM, knowledge base, gaps only; no draft email.
Contact: jordan.lee@northwind-demo.example
```

You should see **intent** (docs/KB), **relationship** (CRM deal + synthetic thread), and explicit **gaps** (no live Gmail unless you configured OAuth).

## Step E — Triangulation mindset

Read [TRIANGULATION.md](TRIANGULATION.md). In your fork you replace Acme with your SOUL, proof registry, and real connectors.

## What Pro unlocks (after trial)

| Capability | Community | Pro / trial |
|:-----------|:----------|:------------|
| `get_account_brief`, `search_knowledge`, `draft_email` | Yes | Yes |
| `send_email`, operator inbox, CRM/graph writes | No | Yes |

See [LICENSING.md](LICENSING.md).

## Next

- [examples/acme/journey.md](../examples/acme/journey.md) — 7-day operator story
- [FORK_GUIDE.md](FORK_GUIDE.md) — private company fork checklist
