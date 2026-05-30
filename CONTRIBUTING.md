# Contributing to Brain OS

Thanks for helping improve the public skeleton.

## Where to contribute

- **Engine / export:** Changes land in the private maintainer export tree (`brain-os/scripts/export_brain_os.py`), then republish to this repo.
- **Docs & Acme demo:** PRs welcome here for `examples/acme/`, `docs/`, and `brain-os/skeleton/` in the maintainer upstream.

## Before you PR

1. No customer PII, real mailbox content, or OEM-specific imports.
2. Run `python scripts/public_repo_guard.py` from repo root.
3. Keep Community/Pro licensing behavior documented in `docs/LICENSING.md`.
4. Read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
5. Read [docs/GIT_AND_AI.md](docs/GIT_AND_AI.md) — Git + AI practices for this repo and forks.

## Issue labels

| Label | Use |
|:------|:----|
| `bug` | Something broken in the public repo |
| `enhancement` | Feature request |
| `design-partner` | Pilot feedback (no PII) |
| `good first issue` | Small doc or guard fixes |

## Design partners

Design partners: use the **Design partner feedback** issue template (`.github/ISSUE_TEMPLATE/design_partner.md`).

## Proprietary operator

Production OEM operator work happens in the private maintainer tree, not in this fork.
