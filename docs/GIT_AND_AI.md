# Git and AI — practices for Brain OS

Short checklist for **humans and AI assistants** working on this repo or a **private fork**. Maintainer engine work happens in private **ira-v3** — see `docs/GIT_AND_AI.md` in that tree for the full operator runbook (pre-commit, export, gitleaks).

## Top 10 (2026)

| # | Practice | This repo |
|:--|:---------|:----------|
| 1 | **No secrets in Git** | `.env` gitignored; CI runs `public_repo_guard.py` |
| 2 | **CI green on `main`** | [.github/workflows/ci.yml](../.github/workflows/ci.yml) |
| 3 | **`poetry.lock` committed** | Run `poetry check` before push |
| 4 | **Small, intentional commits** | One logical change per commit |
| 5 | **Synthetic data only** here | Real company data stays in **your private fork** |
| 6 | **Pre-commit** (optional) | Copy hooks from ira-v3 or use `gitleaks protect --staged` |
| 7 | **Use templates** | Bug / feature / design-partner issues; PR template |
| 8 | **No force-push to `main`** | Branch protection recommended on GitHub |
| 9 | **Deps need human approval** | No drive-by `poetry add` from agents |
| 10 | **Fork for production** | [FORK_GUIDE.md](FORK_GUIDE.md) — not this public tree |

## Before you open a PR (this repo)

```bash
python scripts/public_repo_guard.py
poetry check
poetry install
poetry run python -c "import brain_os"
```

## Using Brain OS (not Git)

New users: [QUICKSTART.md](QUICKSTART.md) → [MCP_SETUP.md](MCP_SETUP.md).

Private company fork: [FORK_GUIDE.md](FORK_GUIDE.md).

## What AI assistants must not do

- Commit `.env`, license keys, Gmail tokens, or customer PII.
- Add dependencies without maintainer approval.
- Push or tag releases without explicit instruction.
- Put private-operator-only paths or imports in the public tree.

## Maintainer note

Changes to **engine** code should be exported from ira-v3 (`brain-os/scripts/export_brain_os.py`), not edited only here — see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Related

| Doc | Topic |
|:----|:------|
| [PUBLIC_PRIVATE_BOUNDARY.md](PUBLIC_PRIVATE_BOUNDARY.md) | What may appear in Git |
| [DATA_PRIVACY.md](DATA_PRIVACY.md) | Privacy posture |
| [LICENSING.md](LICENSING.md) | Community / Pro |
