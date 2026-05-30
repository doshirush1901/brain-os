# Public vs private boundary (Brain OS)

This GitHub repository is a **synthetic demo / reference architecture** for a local-first multi-agent operator.

## Safe in the public tree

- Application source under `src/brain_os/`.
- **Synthetic** JSON under `examples/acme/` (CRM seed, proof registry).
- Architecture and security documentation without customer identifiers.
- CI: `scripts/public_repo_guard.py`.

## Never commit to a public remote

- Real customer or supplier names, emails, domains, or addresses.
- Quotes, invoices, proformas, PO numbers, or commercial amounts tied to real parties.
- CRM exports, mailbox dumps, Gmail thread text, or production memory corpora.
- Lead lists, recruitment CVs, or internal HR data on real people.
- OAuth tokens, API keys, database passwords, or `credentials/` trees.
- Operational send scripts or anything that could trigger accidental outbound mail.

## Operator workflow

1. Clone this public repo (or fork it).
2. Copy `.env.example` → `.env` and configure **your** connectors.
3. Keep proprietary knowledge and production data in a **private fork** or gitignored paths.
4. Run `python scripts/public_repo_guard.py` before every push.

## Your private fork

Production company data, SOUL.md, imports, and Gmail/CRM live in **your** private clone — not in this public skeleton.
