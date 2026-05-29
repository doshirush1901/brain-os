# Public vs private boundary (Ira reference repo)

This GitHub repository is intended as a **synthetic demo / reference architecture** for an AI-assisted manufacturing sales intelligence stack.

## Safe in the public tree

- Application **source code** under `src/brain_os/`.
- **Synthetic** JSON under `examples/public_demo/` (outbound proof registry, recruitment scoring demo, etc.).
- **Synthetic** markdown under `data/knowledge/` (`demo_*.md` only in public clones).
- Architecture and security **documentation** that contains no customer identifiers.
- CI checks: `scripts/public_repo_guard.py`, `gitleaks detect --no-git`.

## Never commit to a public remote

- Real **customer or supplier** names, emails, domains, or addresses.
- **Quotes, invoices, proformas**, PO numbers, payment terms, or commercial amounts tied to real parties.
- **CRM exports**, mailbox dumps, Gmail thread text, Mem0 episodes, Nemesis training corpora from real use.
- **Lead lists**, Apollo/LinkedIn scrapes, recruitment CVs, or internal HR scoring on real people.
- **OAuth tokens**, API keys, database passwords, or `credentials/` trees.
- Operational **send** scripts, invoice HTML, or anything that could cause an accidental outbound.

## Operator workflow

1. Clone the public repo.
2. Copy `.env.example` → `.env` and fill **private** connectors (Postgres, Gmail, LLM keys).
3. Add proprietary knowledge under **gitignored** paths (see `.gitignore` and this doc).
4. Keep a **private git remote** (or local-only branch) for full operational snapshots.

Rotating credentials and rewriting git history after any leak is covered in `docs/GIT_HISTORY_REWRITE_GUIDE.md`.

## Ira Universe (public slim repo)

The separate **[ira-universe](https://github.com/doshirush1901/ira-universe)** repository is **not** a copy of full `ira-v3`. It ships only:

- `src/ira_universe/` — visitor MCP (~12 tools, allowlisted corpus)
- Public docs (including `IRA_BIRTH_CERTIFICATE.md`, `CURSOR_AGENTIC_LOOP.md`, `TIMEOUT_MODEL.md`, `PIPELINE_AUDIT.md` excerpt, `CURSOR_WORKFLOWS.md` excerpt), synthetic demos, and `examples/marketing/case_study_workflow.md`
- Minimal `pyproject.toml` (`mcp` + `pydantic` only)

Regenerate and publish: `bash scripts/republish_ira_universe.sh` (incremental export, test, commit, push). One-time history reset: `bash scripts/republish_ira_universe_clean.sh`. Export only: `python scripts/export_ira_universe_repo.py`.

**Archive repo:** Keep `ira-universe-archived-monolith` **private** (old monolith + PSG history). Public clone/share only `https://github.com/doshirush1901/ira-universe`.

## Named case studies (e.g. PSG)

Customer-specific case-study build trees (LinkedIn/PDF scripts, real logos, installation photos) stay in the **private operator** clone only. Public posture: describe the workflow in `examples/marketing/case_study_workflow.md` and use synthetic deck briefs under `examples/public_demo/deck_briefs/`.
