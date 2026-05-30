"""Minimal async helpers for Brain OS CLI (extracted from operator cli_runtime)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any


def _run(coro: Any) -> Any:
    """Run an async coroutine from synchronous CLI context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
