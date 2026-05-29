"""CLI: ``brain activate`` / ``brain license`` — Pro and trial licensing."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from brain_os.licensing.activate import activate_trial, activate_with_key, license_status_dict
from brain_os.licensing.caps import community_document_limit, get_indexed_document_count
from brain_os.licensing.store import clear_license
from brain_os.licensing.tiers import effective_tier, tier_label

console = Console()
license_app = typer.Typer(help="License activation and status")


@license_app.command("status")
def license_status() -> None:
    """Show Community vs Pro tier, expiry, and document usage."""
    status = license_status_dict()
    table = Table(title="Brain OS License")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key, val in status.items():
        table.add_row(str(key), str(val))
    table.add_row("documents_indexed", str(get_indexed_document_count()))
    table.add_row("community_cap", str(community_document_limit()))
    table.add_row("effective_tier", tier_label())
    console.print(table)


@license_app.command("clear")
def license_clear(
    force: bool = typer.Option(False, "--force", help="Clear Pro/trial license (revert to Community)."),
) -> None:
    """Remove local license file (revert to Community tier)."""
    if not force:
        console.print("Pass --force to clear the license file.")
        raise typer.Exit(code=1)
    clear_license()
    console.print("[green]License cleared — Community tier.[/green]")


def register_license_commands(app: typer.Typer) -> None:
    app.add_typer(license_app, name="license")


def activate(
    key: str | None = typer.Option(None, "--key", "-k", help="License key (bos_live_... or bos_trial_...)."),
    trial: bool = typer.Option(False, "--trial", help="Start a local 14-day Pro trial (no key required)."),
    org_name: str = typer.Option("Trial workspace", "--org", help="Organization label for trial."),
    json_out: bool = typer.Option(False, "--json", help="Print license record as JSON."),
) -> None:
    """Activate Brain OS Pro or start a 14-day trial."""
    if trial:
        record = activate_trial(org_name=org_name)
    elif key:
        record = activate_with_key(key)
    else:
        console.print("Provide --trial or --key bos_live_...")
        raise typer.Exit(code=1)

    if json_out:
        console.print(record.model_dump_json(indent=2))
        return

    console.print(f"[green]Activated[/green] tier={record.tier} expires={record.expires_at}")
    console.print("Run: brain license status")
