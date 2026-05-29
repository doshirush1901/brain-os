"""Root Typer app — minimal Brain OS CLI (``brain health`` v1)."""

from __future__ import annotations

import typer

from brain_os.interfaces.commands.health import health, version
from brain_os.interfaces.commands.activate import activate, register_license_commands
from brain_os.interfaces.commands.ask import ask, seed_acme

app = typer.Typer(
    name="brain",
    help="Brain OS — local-first multi-agent operating system",
    no_args_is_help=True,
)

register_license_commands(app)
app.command("activate", help="Activate Pro or start 14-day trial")(activate)
app.command("ask", help="Run one pipeline query")(ask)
app.command("seed-acme", help="Load Acme demo pack into data/ and CRM")(seed_acme)
app.command("health", help=health.__doc__ or "")(health)
app.command("version", help=version.__doc__ or "")(version)
