"""Alembic async migration environment for Brain OS CRM.

Imports all ORM models so Base.metadata reflects the full schema,
then runs migrations using the async engine.
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

import brain_os.data.atlas_logbook
import brain_os.data.installed_base_models
import brain_os.data.receivables
import brain_os.data.service_tickets
import brain_os.data.spares_catalog
import brain_os.data.work_orders
from alembic import context
from brain_os.data.crm import Base
from brain_os.data.quotes import QuoteModel
from brain_os.data.recruitment import (
    RecruitmentCandidateModel,
    RecruitmentStageEventModel,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://brain:brain@localhost:5432/brain_crm",
    )


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL to stdout."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode using an async engine."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _get_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
