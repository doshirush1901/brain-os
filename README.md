# Brain OS

Forkable multi-agent operating system extracted from [Ira v3](https://github.com/doshirush1901/ira-v3).

## Quickstart (Mac Pro / Mac Studio)

```bash
cp .env.example .env   # optional — bootstrap creates if missing
./scripts/macpro/bootstrap.sh
poetry run brain health
```

Demo company pack: ``examples/acme/`` (see ``examples/acme/journey.md``).

Licensing: Community tier (500 doc cap, tiered MCP). ``poetry run brain activate --trial`` for 14-day Pro.
See ``docs/LICENSING.md`` and ``LICENSE``.

See ``docs/BRAIN_OS_PRODUCTIZATION_ROADMAP.md`` and ``export_manifest.json``.

Starter pantheon: ``src/brain_os/pantheon.py`` (20 agents).
See ``docs/PANTHEON_SLIM_TODO.md`` to add custom agents.
