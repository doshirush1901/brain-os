PUBLIC DEMO DATA — fully synthetic. No real customers, pricing, quotes, CRM, Gmail, or email data.

# Acme Platform — demo FAQ (knowledge base)

## What is Brain OS in this repo?

A **local-first multi-agent runtime** (vectors + graph + CRM) operated from Cursor MCP or `brain ask`. The **Acme** pack is synthetic data for evaluation.

## How is outbound email handled?

Draft-only by default. Human must say **send** explicitly. Pro tier unlocks send MCP tools with confirm gates.

## What is triangulation?

Cross-check **intent** (KB/docs), **relationship** (CRM/mail), and **identity** (company/domain) before acting. See `docs/TRIANGULATION.md`.

## Typical implementation timeline (demo)

| Phase | Duration | Outcome |
|:------|:---------|:--------|
| Week 1 | 5 days | SOUL + ingest + CRM seed |
| Week 2 | 5 days | MCP in Cursor, first account brief |
| Week 3–4 | 10 days | Custom agent + proof registry |
| Month 2+ | Ongoing | Private fork, optional Gmail |

## Support posture (fictional)

Acme Services Inc. demo support: support@acme-corp.example (not monitored — demo only).
