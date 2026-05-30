"""CLI: ``brain operator`` — unified approval inbox."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from brain_os.interfaces.cli_runtime import _configure_logging
from brain_os.service_keys import ServiceKey as SK
from brain_os.services.brain_daily_dashboard import (
    build_and_persist_daily_snapshot,
    compute_dashboard_stats,
    get_dashboard_today,
    list_snapshot_history,
)
from brain_os.services.operator_activity_today import (
    format_operator_activity_today_text,
    gather_operator_activity_today,
)
from brain_os.systems.data_dir_lock import get_data_dir
from brain_os.systems.operator_inbox import OperatorInboxService
from brain_os.systems.tinder_email_mode import TinderEmailModeService

console = Console()
err_console = Console(stderr=True)

operator_app = typer.Typer(help="Unified operator approval inbox.")
dashboard_app = typer.Typer(help="Daily control room snapshots (facts + narrative).")
operator_app.add_typer(dashboard_app, name="dashboard")


def _shared_services() -> tuple[Any, Any, Any]:
    from brain_os.interfaces.cli_runtime import _build_pantheon

    _, shared = _build_pantheon()
    outbound = shared.get(SK.OUTBOUND_APPROVALS)
    if outbound is None:
        from brain_os.systems.outbound_approvals import OutboundApprovalService

        outbound = OutboundApprovalService(
            storage_path=get_data_dir() / "operations" / "outbound_approvals.json",
        )
    email = shared.get(SK.EMAIL_PROCESSOR)
    quotes = shared.get(SK.QUOTES)
    return outbound, email, quotes


@operator_app.command("inbox")
def operator_inbox_cmd(
    json_output: bool = typer.Option(False, "--json", help="Print JSON payload."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List pending outbound emails, lead reviews, and quote drafts."""
    _configure_logging(verbose)
    outbound, email, quotes = _shared_services()
    tinder = TinderEmailModeService(data_root=get_data_dir())
    inbox = OperatorInboxService()

    async def _run() -> dict[str, Any]:
        payload = await inbox.collect_inbox(
            outbound_approvals=outbound,
            quotes=quotes,
            tinder_service=tinder,
        )
        return payload.model_dump()

    payload = asyncio.run(_run())
    if json_output:
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return

    summary = payload.get("summary") or {}
    console.print(
        f"[bold]Operator inbox[/bold] — "
        f"{summary.get('outbound_email', 0)} outbound · "
        f"{summary.get('lead_review', 0)} leads · "
        f"{summary.get('quote_draft', 0)} quotes · "
        f"{summary.get('external_pending', 0)} external pending"
    )
    table = Table()
    table.add_column("Id", style="cyan", max_width=28)
    table.add_column("Kind")
    table.add_column("Title", max_width=48)
    table.add_column("Risk")
    for row in payload.get("items") or []:
        table.add_row(
            str(row.get("id") or ""),
            str(row.get("kind") or ""),
            str(row.get("title") or "")[:48],
            str(row.get("risk") or ""),
        )
    console.print(table)
    checklist = payload.get("math_promote_checklist") or {}
    if checklist:
        console.print("\n[bold]Math promote checklist[/bold] (phase 4)")
        console.print(f"  ready_to_promote={checklist.get('ready_to_promote')}")
        for line in checklist.get("checklist") or []:
            console.print(f"  · {line}")
    hints = payload.get("math_priority_hints") or []
    if hints:
        console.print("\n[bold]Math priority hints[/bold] (top shadow/advisory)")
        for h in hints[:5]:
            console.print(
                f"  {h.get('company_name', '')} — priority={h.get('action_priority')} "
                f"({h.get('next_action', '')})"
            )


@operator_app.command("decide")
def operator_decide_cmd(
    item_id: str = typer.Argument(..., help="Inbox item id (or prefix)."),
    decision: str = typer.Argument(..., help="approve | reject | snooze"),
    actor: str = typer.Option("cli_operator", "--actor", help="Audit actor name."),
    snooze_days: int = typer.Option(7, "--snooze-days", min=1, max=90),
    to_address: str | None = typer.Option(
        None, "--to", help="Recipient for revenue draft approve."
    ),
    json_output: bool = typer.Option(False, "--json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Approve (Gmail draft), reject, or snooze one item."""
    _configure_logging(verbose)
    dec = decision.strip().lower()
    if dec not in ("approve", "reject", "snooze"):
        err_console.print("[red]decision must be approve, reject, or snooze[/red]")
        raise typer.Exit(1)

    outbound, email, quotes = _shared_services()
    tinder = TinderEmailModeService(data_root=get_data_dir())
    inbox = OperatorInboxService()

    async def _run() -> dict[str, Any]:
        return await inbox.decide(
            item_id=item_id,
            decision=dec,  # type: ignore[arg-type]
            actor=actor,
            snooze_days=snooze_days,
            to_address=to_address,
            outbound_approvals=outbound,
            email_processor=email,
            tinder_service=tinder,
            quotes=quotes,
        )

    result = asyncio.run(_run())
    if json_output:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    elif result.get("ok"):
        console.print(f"[green]OK[/green] {result.get('action')} — {item_id}")
    else:
        err_console.print(f"[red]{result.get('error')}[/red]")
        raise typer.Exit(1)


async def _inbox_payload_for_cli() -> dict[str, Any]:
    outbound, _, quotes = _shared_services()
    tinder = TinderEmailModeService(data_root=get_data_dir())
    inbox = OperatorInboxService()
    payload = await inbox.collect_inbox(
        outbound_approvals=outbound,
        quotes=quotes,
        tinder_service=tinder,
    )
    return payload.model_dump()


@dashboard_app.command("today")
def operator_dashboard_today_cmd(
    json_output: bool = typer.Option(False, "--json", help="Print JSON payload."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Daily dashboard for operator local today (snapshot or live facts)."""
    _configure_logging(verbose)

    async def _run():
        return await get_dashboard_today(inbox_payload=await _inbox_payload_for_cli())

    payload = asyncio.run(_run())
    if json_output:
        print(json.dumps(payload.model_dump(), ensure_ascii=False), flush=True)
        return
    m = payload.metrics
    console.print(
        f"[bold]{payload.local_date}[/bold] ({payload.timezone}) "
        f"— snapshot={'yes' if payload.snapshot_persisted else 'live'}"
    )
    if payload.narrative and payload.narrative.big_win:
        console.print(f"Big win: {payload.narrative.big_win}")
    console.print(
        f"Emails {m.emails_sent_count} · agents {m.agents_active_count} · "
        f"runs {m.pipeline_runs_count} · Metis {m.brain_score}"
    )


@dashboard_app.command("history")
def operator_dashboard_history_cmd(
    days: int = typer.Option(30, "--days"),
    offset: int = typer.Option(0, "--offset"),
    json_output: bool = typer.Option(False, "--json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List stored daily dashboard snapshots."""
    _configure_logging(verbose)
    payload = list_snapshot_history(days=days, offset=offset)
    if json_output:
        print(json.dumps(payload.model_dump(), ensure_ascii=False), flush=True)
        return
    table = Table(title="Dashboard history")
    table.add_column("Date")
    table.add_column("Emails")
    table.add_column("Agents")
    table.add_column("Metis")
    table.add_column("Headline")
    for item in payload.items:
        table.add_row(
            item.local_date,
            str(item.metrics.emails_sent_count),
            str(item.metrics.agents_active_count),
            str(item.metrics.brain_score or "—"),
            (item.big_win_preview or "")[:60],
        )
    console.print(table)


@dashboard_app.command("stats")
def operator_dashboard_stats_cmd(
    days: int = typer.Option(30, "--days"),
    json_output: bool = typer.Option(False, "--json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Rolling totals and daily averages over stored snapshots."""
    _configure_logging(verbose)
    payload = compute_dashboard_stats(days=days)
    if json_output:
        print(json.dumps(payload.model_dump(), ensure_ascii=False), flush=True)
        return
    a = payload.averages
    console.print(
        f"{payload.days_with_data}/{payload.days_requested} days — "
        f"avg emails {a.emails_sent_count} · avg Metis {a.brain_score}"
    )


@dashboard_app.command("build")
def operator_dashboard_build_cmd(
    local_date: str | None = typer.Option(None, "--date", help="YYYY-MM-DD (default: today)."),
    skip_llm: bool = typer.Option(False, "--skip-llm", help="Facts only, no narrative."),
    json_output: bool = typer.Option(False, "--json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Gather facts and persist snapshot (optional LLM narrative)."""
    _configure_logging(verbose)

    async def _run():
        return await build_and_persist_daily_snapshot(
            local_date=local_date,
            include_llm=not skip_llm,
            inbox_payload=await _inbox_payload_for_cli(),
        )

    snap = asyncio.run(_run())
    if json_output:
        print(json.dumps(snap.model_dump(), ensure_ascii=False), flush=True)
        return
    console.print(
        f"[green]Saved[/green] {snap.local_date} narrative={'yes' if snap.narrative else 'no'}"
    )


@operator_app.command("activity")
def operator_activity_cmd(
    limit: int = typer.Option(40, "--limit", help="Max timeline events."),
    json_output: bool = typer.Option(False, "--json", help="Print JSON payload."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """What Brain OS did today (operator timezone): runs, dream, heartbeat, journal, inbox audit."""
    _configure_logging(verbose)
    outbound, _, quotes = _shared_services()
    tinder = TinderEmailModeService(data_root=get_data_dir())
    inbox = OperatorInboxService()

    async def _run():
        inbox_payload = await inbox.collect_inbox(
            outbound_approvals=outbound,
            quotes=quotes,
            tinder_service=tinder,
        )
        return await gather_operator_activity_today(
            limit=limit,
            inbox_payload=inbox_payload.model_dump(),
        )

    payload = asyncio.run(_run())
    if json_output:
        print(json.dumps(payload.model_dump(), ensure_ascii=False), flush=True)
        return
    console.print(format_operator_activity_today_text(payload))


@operator_app.command("release")
def operator_release_cmd(
    actor: str = typer.Option("cli_operator", "--actor"),
    hours: float | None = typer.Option(None, "--hours", help="Release TTL hours."),
    force: bool = typer.Option(False, "--force", help="Release even if external items pending."),
    json_output: bool = typer.Option(False, "--json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Release autonomous mode — allow heartbeat query jobs when APP__OPERATOR_RELEASE_REQUIRED=true."""
    _configure_logging(verbose)
    from brain_os.config import get_settings

    outbound, _, quotes, _ = _shared_services()
    tinder = TinderEmailModeService(data_root=get_data_dir())
    inbox = OperatorInboxService()

    async def _run() -> dict[str, Any]:
        payload = await inbox.collect_inbox(
            outbound_approvals=outbound,
            quotes=quotes,
            tinder_service=tinder,
        )
        ext = payload.summary.external_pending
        if ext > 0 and not force:
            return {
                "ok": False,
                "error": f"{ext} external_visible item(s) pending",
                "summary": payload.summary.model_dump(),
            }
        app = get_settings().app
        ttl = float(hours) if hours is not None else float(app.operator_release_ttl_hours)
        session = inbox.save_release(actor=actor, ttl_hours=ttl, pending_at_release=ext)
        return {"ok": True, "session": session.model_dump()}

    result = asyncio.run(_run())
    if json_output:
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    if not result.get("ok"):
        err_console.print(f"[red]{result.get('error')}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Brain OS released[/green] until {result['session'].get('released_until')}")
