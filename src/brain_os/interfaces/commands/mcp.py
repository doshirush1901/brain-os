"""CLI: ``brain mcp`` — start stdio MCP server for Cursor / Claude."""

from __future__ import annotations

import logging
import sys

import typer

logger = logging.getLogger(__name__)


def mcp(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging on stderr."),
) -> None:
    """Start Brain OS MCP server (stdio JSON-RPC on stdout; logs on stderr)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("brain_os.interfaces.cli").setLevel(logging.WARNING)
    logger.info(
        "Starting Brain OS MCP (stdio). Restart MCP in Cursor/Claude after code or .env changes."
    )
    from brain_os.interfaces.mcp_server import main as mcp_main

    mcp_main()
