# `alembic/` — Database Migrations

PostgreSQL schema migrations for the CRM database, managed by Alembic.

## Current Migrations

| Revision | Description |
|:---------|:------------|
| `001` | Initial CRM schema (companies, contacts, deals, interactions, quotes) |
| `002` | Performance indexes for common query patterns |

## Usage

```bash
# Apply all pending migrations
alembic upgrade head

# Create a new migration after changing models
alembic revision --autogenerate -m "description of change"

# Check current revision
alembic current

# View migration history
alembic history
```

## Models

Migrations are auto-generated from ORM models in:
- `src/brain_os/data/crm.py` — Companies, contacts, deals, interactions
- `src/brain_os/data/quotes.py` — Quotes and line items
- `src/brain_os/data/vendors.py` — Vendors and payables
