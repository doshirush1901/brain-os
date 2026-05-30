"""CLI: ``brain ask`` — single query through the Brain OS pipeline."""

from __future__ import annotations

import json

import typer
from brain_os.interfaces.cli.runtime import _configure_logging, _run
from brain_os.systems.data_dir_lock import data_dir_lock
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()
err_console = Console(stderr=True)


def ask(
    query: str = typer.Argument(..., help="Question for Brain OS."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit JSON: response, agents_consulted, run_id.",
    ),
) -> None:
    """Run one pipeline query (requires LLM keys and Docker services)."""
    _configure_logging(verbose)

    async def _ask() -> None:
        from brain_os.config import get_settings
        from brain_os.interfaces.cli_runtime import _build_pantheon, _build_pipeline

        pantheon, shared_services = _build_pantheon()
        user_id = get_settings().app.default_user_id

        async with pantheon:
            pipeline, _feedback, _redis, _voice = await _build_pipeline(pantheon, shared_services)
            response, agents_used, run_id = await pipeline.process_request(
                raw_input=query,
                channel="cli",
                sender_id=user_id,
            )
            await pipeline.wait_for_background_tasks(timeout=45.0)

            if json_output:
                typer.echo(
                    json.dumps(
                        {
                            "response": response,
                            "agents_consulted": agents_used or [],
                            "run_id": run_id,
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                console.print(Panel(Markdown(response), title="Brain OS", border_style="green"))

    try:
        with data_dir_lock():
            _run(_ask())
    except RuntimeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        err_console.print(f"[red]brain ask failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc


def seed_acme(
    force: bool = typer.Option(False, "--force", help="Re-copy proof registry and re-seed CRM."),
    skip_crm: bool = typer.Option(False, "--skip-crm", help="Only copy proof registry."),
) -> None:
    """Load examples/acme demo pack into data/knowledge and CRM."""
    from brain_os.systems.acme_demo import seed_acme_demo

    report = seed_acme_demo(force=force, crm=not skip_crm)
    if not report.get("ok"):
        err_console.print(f"[red]{report.get('error', 'seed failed')}[/red]")
        raise typer.Exit(code=1)
    for action in report.get("actions", []):
        console.print(f"[green]✓[/green] {action}")
