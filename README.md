# Brain OS

> **Ira** is Machinecraft's private operator system (CRM, Gmail, production truth).
> **Brain OS** is the public forkable skeleton — same architecture patterns, synthetic Acme demo only.
> Run your business in a **private fork** of this repo; do not expect Machinecraft customer data here.

Forkable multi-agent operating system extracted from [Ira v3](https://github.com/doshirush1901/ira-v3) (private operator tree).

## Quickstart (Mac Pro / Mac Studio)

```bash
cp .env.example .env   # optional — bootstrap creates if missing
./scripts/macpro/bootstrap.sh
poetry run brain health
```

Demo company pack: ``examples/acme/`` (see ``examples/acme/journey.md``).

Licensing: Community tier (500 doc cap, tiered MCP). ``poetry run brain activate --trial`` for 14-day Pro.
See ``docs/LICENSING.md`` and ``LICENSE`` (BSL 1.1 — counsel review before production use; see ``docs/BRAIN_OS_GTM.md``).

See ``docs/BRAIN_OS_PRODUCTIZATION_ROADMAP.md`` and ``export_manifest.json``.

## Related repos

| Repo | Role |
|:-----|:-----|
| [brain-os](https://github.com/doshirush1901/brain-os) | **This repo** — fork and customize |
| [ira-v3](https://github.com/doshirush1901/ira-v3) | Private Machinecraft operator (reference only) |
| [ira-universe](https://github.com/doshirush1901/ira-universe) | Visitor-safe architecture docs |

Starter pantheon: ``src/brain_os/pantheon.py`` (20 agents).
See ``docs/PANTHEON_SLIM_TODO.md`` to add custom agents.
