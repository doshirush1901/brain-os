# Data privacy — Ira deployments

## What Ira may process

Depending on enabled integrations, Ira-class stacks may touch:

- Mailbox contents (headers, bodies, attachments metadata)
- CRM rows (contacts, deals, interactions)
- Vector embeddings derived from uploaded PDFs/Markdown
- Knowledge-graph entities (companies, people, quotes — **your models**)
- Long-term conversational memory (Mem0 / local episodic stores)

**This public repository does not ship customer datasets.** Anything resembling leads, threads, or quotes under `examples/` is synthetic.

## Keep private data private

1. Store secrets only in `.env`, secret managers, or encrypted vaults — never in tracked files.
2. Prefer **local** Postgres/Qdrant/Neo4j for customer payloads; isolate networks from the public Internet.
3. Use **`IRA_EMAIL_MODE=TRAINING`** until governance allows operational sends.
4. Enable optional ingest-time redaction (`APP__REDACT_PII_AT_INGEST`) when experimenting with realistic dumps in dev sandboxes.

## Synthetic-first workflow for forks

- Copy `examples/` into a scratch workspace when demonstrating pipelines.
- Replace synthetic domains (`example.com`) with **consenting** test tenants before enabling Gmail scopes.

## Customer rights & retention

Operators remain responsible for GDPR/CCPA-style obligations (access, deletion, retention). Ira stores embeddings and CRM mirrors — deletion must span DB + vector stores + caches.

## Incident mindset

If CRM exports or mailbox archives leak from **any** clone/fork:

1. Rotate OAuth tokens & API keys tied to that environment.
2. Invalidate webhook secrets.
3. Follow `SECURITY.md` for coordinated disclosure.
