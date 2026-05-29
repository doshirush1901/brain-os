"""CLI: ``brain health`` — immune-system startup validation."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from brain_os.exceptions import IraError
from brain_os.interfaces.cli.runtime import _configure_logging, _run

console = Console()
err_console = Console(stderr=True)


def version() -> None:
    """Print the Brain OS skeleton version."""
    console.print("brain-os 0.1.0-skeleton")


def health(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run startup validation against Qdrant, Neo4j, Postgres, and LLM providers."""
    _configure_logging(verbose)

    async def _health() -> None:
        from brain_os.brain.embeddings import EmbeddingService
        from brain_os.brain.knowledge_graph import KnowledgeGraph
        from brain_os.brain.qdrant_manager import QdrantManager
        from brain_os.systems.data_dir_lock import get_data_dir
        from brain_os.systems.immune import ImmuneSystem

        embedding = EmbeddingService()
        qdrant = QdrantManager(embedding_service=embedding)
        graph = KnowledgeGraph()
        immune = ImmuneSystem(
            qdrant=qdrant,
            knowledge_graph=graph,
            embedding_service=embedding,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Running health checks..."),
            console=err_console,
            transient=True,
        ) as progress:
            progress.add_task("health", total=None)
            try:
                report = await immune.run_startup_validation()
            except (IraError, Exception) as exc:
                report = getattr(exc, "health_report", {})

        table = Table(title="Brain OS Health")
        table.add_column("Service", style="cyan", width=18)
        table.add_column("Status", width=12)
        table.add_column("Details")

        for service, info in sorted(report.items()):
            status = info.get("status", "unknown")
            style = (
                "green"
                if status == "healthy"
                else "red"
                if status == "unhealthy"
                else "yellow"
            )
            details = info.get("error", info.get("latency_ms", ""))
            table.add_row(service, f"[{style}]{status}[/{style}]", str(details))

        table.add_row("data_directory", "[cyan]info[/cyan]", str(get_data_dir()))
        console.print(table)

    _run(_health())
