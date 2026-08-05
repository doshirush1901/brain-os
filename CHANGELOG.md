# Changelog

All notable changes to the public **brain-os** repository are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Friend-fork kit: [FRIEND_FORK.md](docs/FRIEND_FORK.md), [BUILD_YOUR_KB.md](docs/BUILD_YOUR_KB.md), [OPERATOR_GOVERNANCE.md](docs/OPERATOR_GOVERNANCE.md); README + FORK_GUIDE links; forkmybrain.org as story front door.

### Changed

- BSL Licensor placeholder resolved to **Machinecraft Technologies Pvt. Ltd.** (pending counsel review before v1.0 tag).
- `.env.example` / QUICKSTART: Redis 6380 clash note; Neo4j host **7688** and Postgres DB `brain_crm` aligned with `docker-compose.local.yml`.

## [0.1.0] - 2026-05-30

### Added

- Public export: 17-step pipeline, slim pantheon (~20 agents), Community / Pro licensing.
- **Acme** synthetic demo: CRM seed, journey, proof registry, product docs.
- Mac bootstrap: `scripts/macpro/bootstrap.sh`, Docker Compose stack (Postgres, Qdrant, Neo4j, Redis).
- CLI: `brain ask`, `brain health`, `brain seed-acme`, `brain activate --trial`, `brain mcp`.
- MCP setup for Cursor and Claude Desktop (`.cursor/mcp.json.example`, [docs/MCP_SETUP.md](docs/MCP_SETUP.md)).
- Brand assets: optimized README hero and social preview PNGs ([docs/BRAND.md](docs/BRAND.md)).
- Docs: [QUICKSTART](docs/QUICKSTART.md), [FORK_GUIDE](docs/FORK_GUIDE.md), triangulation, licensing, GTM roadmap.
- CI: `public_repo_guard.py`, Poetry check, import smoke test.
- GitHub: issue templates, Dependabot, Contributor Covenant.

### Notes

- **License:** BSL 1.1 parameters — Licensor **Machinecraft Technologies Pvt. Ltd.** Full counsel-signed text and **v1.0.0** tag still pending. Do not tag **v1.0.0** until approved.

[0.1.0]: https://github.com/doshirush1901/brain-os/releases/tag/v0.1.0
