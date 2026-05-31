# Brain OS — architecture (public skeleton)

Brain OS is a **local-first multi-agent runtime**. This public repo ships the **engine patterns** and an **Acme** synthetic demo — not a vertical OEM product.

## Request path

```
Cursor / Claude MCP          CLI (brain ask)
         \                         /
          v                       v
       +--------------------------------+
       |   Pantheon  +  17-step pipeline |
       +----------------+----------------+
                        |
          +-------------+-------------+
          v             v             v
      Qdrant        Neo4j         Postgres
      (vectors)     (graph)       (CRM)
```

## Layers

| Layer | Role |
|:------|:-----|
| **Interfaces** | Typer CLI, optional FastAPI, MCP server (`brain-mcp`) |
| **Pipeline** | Perceive → route → execute agents → ground → shape → learn |
| **Pantheon** | Starter specialists (Athena, Clio, Calliope, CRM, memory, safety) |
| **Brain** | Retrieval (Qdrant), graph (Neo4j), ingest, guardrails |
| **Memory** | Conversation, long-term (Mem0 optional), corrections |
| **Data** | CRM models (Postgres); demo seed under `examples/acme/` |

## Deployment (Mac)

- Docker Compose: Postgres, Qdrant, Neo4j, Redis
- `scripts/macpro/bootstrap.sh` — infra + health hints
- Secrets in `.env` only (never committed)

## Boundaries

- **Community tier:** document cap, tiered MCP tools — see [LICENSING.md](LICENSING.md)
- **Private operator stack:** full vertical depth, live Gmail/CRM — not in this tree
- **Your fork:** company `SOUL.md`, imports, and production data stay private

## Related

- [QUICKSTART.md](QUICKSTART.md) · [ACME_DEMO_PATH.md](ACME_DEMO_PATH.md) · [MCP_SETUP.md](MCP_SETUP.md) · [TRIANGULATION.md](TRIANGULATION.md)
