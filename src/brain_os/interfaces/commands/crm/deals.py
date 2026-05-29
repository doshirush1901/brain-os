"""CRM CLI commands — see module docstring in package ``__init__``."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import typer
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from brain_os.interfaces.cli_runtime import (
    _build_digestive,
    _build_email_processor,
    _build_pantheon,
    _coerce_strings_for_json,
    _configure_logging,
    _run,
)
from brain_os.service_keys import ServiceKey as SK

from ._helpers import (
    _OPEN_DEAL_STAGES,
)
from .app import _progress_console, console, crm_app, err_console


@crm_app.command("deals")
def crm_deals(
    months: int = typer.Option(
        6, "--months", "-m", help="Look back this many months (~30 days each)."
    ),
    window: str = typer.Option(
        "created",
        "--window",
        help="Apply date filter to deal 'created' time or 'updated' (last touched).",
    ),
    open_only: bool = typer.Option(
        True,
        "--open-only/--all-stages",
        help="Only open pipeline stages (default) or include WON and LOST.",
    ),
    sort: str = typer.Option(
        "both",
        "--sort",
        help="Output: both (recency table + value table) | recency | value.",
    ),
    recency: str = typer.Option(
        "updated",
        "--recency",
        help="When sorting by recency: use 'updated' (default) or 'created' timestamp.",
    ),
    limit: int = typer.Option(
        5000,
        "--limit",
        help="Maximum deals to load from the database (safety cap).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of Rich tables."),
) -> None:
    """Sales deals in a date window: newest-first and highest-value-first lists.

    Uses the same Postgres CRM as ``ira ask`` / Prometheus. Date window is
    approximate (``months`` × 30 days). Default: deals **created** in the window,
    **open stages** only, two tables (latest activity first, then by value).
    """
    win = window.strip().lower()
    if win not in ("created", "updated"):
        console.print("[red]--window must be 'created' or 'updated'.[/red]")
        raise typer.Exit(1)
    sort_key = sort.strip().lower()
    if sort_key not in ("both", "recency", "value"):
        console.print("[red]--sort must be both, recency, or value.[/red]")
        raise typer.Exit(1)
    rec = recency.strip().lower()
    if rec not in ("updated", "created"):
        console.print("[red]--recency must be updated or created.[/red]")
        raise typer.Exit(1)

    async def _deals() -> None:
        from brain_os.data.crm import CRMDatabase

        crm = CRMDatabase()
        await crm.create_tables()

        now_naive = datetime.now(UTC).replace(tzinfo=None)
        days = max(1, int(months * 30))
        date_from = now_naive - timedelta(days=days)
        date_to = now_naive
        filters: dict[str, Any] = {"date_from": date_from, "date_to": date_to}
        date_range_on = "updated_at" if win == "updated" else "created_at"

        rows = await crm.list_deals_with_details(
            limit=limit,
            filters=filters,
            date_range_on=date_range_on,
        )
        if open_only:
            rows = [r for r in rows if r.get("stage") in _OPEN_DEAL_STAGES]

        recency_key = "updated_at" if rec == "updated" else "created_at"

        def _rk(r: dict[str, Any]) -> str:
            return (r.get(recency_key) or "")[:26]

        def _vk(r: dict[str, Any]) -> float:
            return float(r.get("value") or 0)

        by_recency = sorted(rows, key=_rk, reverse=True)
        by_value = sorted(rows, key=_vk, reverse=True)

        if as_json:
            out: dict[str, Any] = {
                "filters": {
                    "months": months,
                    "window": win,
                    "open_only": open_only,
                    "date_from": date_from.isoformat(),
                    "date_to": date_to.isoformat(),
                    "date_range_on": date_range_on,
                },
                "count": len(rows),
            }
            if sort_key in ("both", "recency"):
                out["by_recency"] = _coerce_strings_for_json(by_recency)
            if sort_key in ("both", "value"):
                out["by_value"] = _coerce_strings_for_json(by_value)
            console.print(json.dumps(out, default=str, indent=2))
            return

        hdr = (
            f"[bold]Deals[/bold] — last ~{months} month(s), "
            f"window=[cyan]{win}[/cyan], open_only={open_only}, "
            f"loaded={len(rows)} (cap {limit})"
        )
        console.print(Panel(hdr, border_style="cyan"))

        def _emit_table(title: str, data: list[dict[str, Any]]) -> None:
            t = Table(title=title, show_header=True, header_style="bold")
            t.add_column("Updated", style="dim", width=10)
            t.add_column("Created", style="dim", width=10)
            t.add_column("Value", justify="right", width=12)
            t.add_column("Cur", width=4)
            t.add_column("Stage", width=11)
            t.add_column("Company", max_width=22, no_wrap=True)
            t.add_column("Title", max_width=36, no_wrap=True)
            for r in data:
                ua = (r.get("updated_at") or "")[:10]
                ca = (r.get("created_at") or "")[:10]
                val = float(r.get("value") or 0)
                cur = (r.get("currency") or "USD")[:4]
                st = (r.get("stage") or "")[:11]
                co = (r.get("company_name") or "")[:22]
                ti = (r.get("title") or "")[:80]
                t.add_row(ua, ca, f"{val:,.0f}", cur, st, co, ti)
            console.print(t)

        if sort_key in ("both", "recency"):
            label = f"Newest → oldest (by {recency_key})"
            _emit_table(label, by_recency)
        if sort_key in ("both", "value"):
            if sort_key == "both":
                console.print()
            _emit_table("Highest value → lowest", by_value)

    _run(_deals())


@crm_app.command("segment-classify")
def crm_segment_classify(
    dry_run: bool = typer.Option(
        True, "--dry-run/--apply", help="Dry-run snapshot only vs write Postgres."
    ),
    output: Path = typer.Option(
        Path("data/reports/segment_snapshot.json"),
        "--output",
        "-o",
        help="Snapshot JSON path (dry-run or post-apply).",
    ),
    top100: Path = typer.Option(
        Path("data/reports/top100_leads_excl_customers.csv"),
        "--top100",
        help="Quoted-lead CSV.",
    ),
    digest: Path = typer.Option(
        Path("data/reports/mailbox_digest_4m_dry_2026-05-18.json"),
        "--digest",
        help="Mailbox digest JSON for mail evidence.",
    ),
    search_inbound: bool = typer.Option(
        False,
        "--search-inbound",
        help="Gmail search per candidate (slow; digest usually enough).",
    ),
    json_output: bool = typer.Option(False, "--json", help="JSON summary to stdout."),
) -> None:
    """Classify contacts into five Ira relationship segments (imports + Atlas + mail)."""

    async def _classify() -> dict[str, Any]:
        from brain_os.data.crm import CRMDatabase
        from brain_os.services.relationship_segment import (
            apply_segment_decisions,
            run_segment_classification,
            snapshot_payload,
        )

        email_processor = None
        if search_inbound:
            from brain_os.interfaces.email_processor import EmailProcessor

            email_processor = EmailProcessor()

        decisions = await run_segment_classification(
            email_processor=email_processor,
            top100_path=top100,
            digest_path=digest,
            search_inbound=search_inbound,
        )
        payload = snapshot_payload(decisions)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        result: dict[str, Any] = {
            "dry_run": dry_run,
            "output": str(output),
            "counts": payload.get("counts"),
            "rows": len(payload.get("rows") or []),
        }
        if not dry_run:
            crm = CRMDatabase()
            await crm.create_tables()
            stats = await apply_segment_decisions(crm, decisions)
            result["apply"] = stats
        return result

    result = _run(_classify())
    if json_output:
        typer.echo(json.dumps(result, indent=2, default=str))
        return
    counts = result.get("counts") or {}
    lines = [
        "[bold]Segment classification[/bold]",
        f"Rows: {result.get('rows', 0)} → {output}",
        "",
    ]
    for key, n in sorted(counts.items()):
        lines.append(f"  • {key}: {n}")
    if result.get("apply"):
        lines.append("")
        lines.append(f"[green]Applied to Postgres:[/green] {result['apply']}")
    elif result.get("dry_run"):
        lines.append("")
        lines.append("[dim]Dry-run only — re-run with --apply to write Postgres[/dim]")
    console.print(Panel("\n".join(lines), title="CRM segments", border_style="cyan"))


@crm_app.command("twenty-sync-segments")
def crm_twenty_sync_segments(
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview vs push to Twenty."),
    limit: int = typer.Option(500, "--limit", help="Max contacts to scan."),
    json_output: bool = typer.Option(False, "--json", help="JSON stats."),
) -> None:
    """Push ira_segment labels from Postgres to Twenty People (jobTitle prefix)."""

    async def _sync() -> dict[str, Any]:
        from brain_os.data.twenty_settings import get_twenty_config
        from brain_os.systems.twenty_client import TwentyGraphQLClient
        from brain_os.systems.twenty_segment_sync import sync_ira_segments_to_twenty

        cfg = get_twenty_config()
        if not (cfg.api_url or "").strip() or not cfg.api_key.get_secret_value().strip():
            return {"error": "Twenty not configured (TWENTY_API_URL / TWENTY_API_KEY)."}
        client = TwentyGraphQLClient.from_config(cfg)
        return await sync_ira_segments_to_twenty(client, dry_run=dry_run, limit=limit)

    result = _run(_sync())
    if json_output:
        typer.echo(json.dumps(result, indent=2))
        return
    if result.get("error"):
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(1)
    console.print(
        Panel(json.dumps(result, indent=2), title="Twenty segment sync", border_style="cyan")
    )


@crm_app.command("mailbox-digest")
def crm_mailbox_digest(
    months: int = typer.Option(
        4,
        "--months",
        help="Lookback window in months (approximate). Use 4, 12, or 36; overridden by --after.",
    ),
    after: str = typer.Option(
        "", "--after", help="Start date YYYY/MM/DD (inclusive). Overrides --months."
    ),
    before: str = typer.Option(
        "", "--before", help="End date YYYY/MM/DD (exclusive). Default: tomorrow."
    ),
    mailbox: str = typer.Option(
        "",
        "--mailbox",
        "-m",
        help="Gmail account to scan: empty = primary token; or secondary profile email (e.g. sales@…).",
    ),
    query_extra: str = typer.Option(
        "",
        "--query",
        "-q",
        help="Extra Gmail operators appended (e.g. subject:PF1).",
    ),
    max_contacts: int = typer.Option(
        80,
        "--max-contacts",
        min=1,
        max=2000,
        help="Max external recipients to run LLM summarisation on (cost control).",
    ),
    max_threads: int = typer.Option(
        25,
        "--max-threads",
        help="Max distinct threads merged per recipient.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only list external recipients and thread counts; no LLM.",
    ),
    exclude_domains: str = typer.Option(
        "",
        "--exclude-domains",
        help="Comma-separated domains to skip (e.g. informa.com,stripe.com). Merged with IRA_MAILBOX_DIGEST_EXCLUDE_DOMAINS.",
    ),
    exclude_emails: str = typer.Option(
        "",
        "--exclude-emails",
        help="Comma-separated full addresses to drop (e.g. personal Gmail). Merged with IRA_MAILBOX_DIGEST_EXCLUDE_EMAILS.",
    ),
    no_crm: bool = typer.Option(
        False,
        "--no-crm",
        help="Do not join Postgres CRM (no lead/customer columns).",
    ),
    all_sent: bool = typer.Option(
        False,
        "--all-sent",
        help="Search all sent mail (disable sales-keyword filter). Noisier; use if real proposals/quotes seem missing.",
    ),
    output: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Write full JSON report to this path (e.g. data/reports/mailbox_digest.json).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    """Summarise outbound sent mail per external contact (CRM-from-mailbox view).

    Lists ``in:sent`` in the date window (default: sales-keyword filter to cut
    irrelevant mail), groups To/Cc/Bcc/Reply-To by external address, loads full
    threads, and produces an LLM structured summary per contact. Use ``--all-sent``
    if real proposals are missing from the list. Start with ``--months 4``;
    widen to 12 or 36 later.

    Internal domains default to example.com / example.net in fresh clones — override via
    env ``IRA_MAILBOX_DIGEST_INTERNAL_DOMAINS`` (comma-separated).

    CRM columns come from Postgres ``contacts.contact_type`` (customer vs lead
    enums) when the email matches; otherwise ``NOT_IN_CRM``. Use
    ``IRA_MAILBOX_DIGEST_EXCLUDE_EMAILS`` or ``--exclude-emails`` for addresses
    that are not sales leads.
    """
    _configure_logging(verbose)

    async def _digest() -> None:
        from brain_os.systems.mailbox_outbound_digest import run_mailbox_outbound_digest, write_report

        pantheon, shared = _build_pantheon()
        digestive, _ingestor, _qdrant = _build_digestive()
        email_proc = _build_email_processor(pantheon, digestive, shared)
        crm = shared.get(SK.CRM) if not no_crm else None

        _meta_prog = re.compile(r"^Metadata (\d+)/(\d+)$")
        _recv_prog = re.compile(
            r"^Recipient (\d+)/(\d+) \(summarised (\d+)/(\d+)\): (.+)$",
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=_progress_console,
            transient=False,
        ) as progress:
            list_tid: TaskID | None = None
            meta_tid: TaskID | None = None
            llm_tid: TaskID | None = None

            def _drop_list_task() -> None:
                nonlocal list_tid
                if list_tid is None:
                    return
                try:
                    progress.remove_task(list_tid)
                except KeyError:
                    pass
                list_tid = None

            def _complete_meta() -> None:
                if meta_tid is None:
                    return
                for task in progress.tasks:
                    if task.id != meta_tid:
                        continue
                    if task.total is None:
                        return
                    if task.completed < task.total:
                        progress.update(meta_tid, completed=task.total)
                    return

            def _mailbox_digest_progress(msg: str) -> None:
                nonlocal list_tid, meta_tid, llm_tid
                t = msg.strip()
                mm = _meta_prog.match(t)
                if mm:
                    cur, tot = int(mm.group(1)), int(mm.group(2))
                    _drop_list_task()
                    if meta_tid is None:
                        meta_tid = progress.add_task("[cyan]Gmail metadata[/cyan]", total=tot)
                    progress.update(meta_tid, completed=cur, total=tot)
                    return
                rm = _recv_prog.match(t)
                if rm:
                    idx, scan_cap = int(rm.group(1)), int(rm.group(2))
                    summ, max_rows = int(rm.group(3)), int(rm.group(4))
                    em = rm.group(5).strip()
                    _complete_meta()
                    if llm_tid is None:
                        llm_tid = progress.add_task(
                            "[green]Contacts + LLM[/green]",
                            total=scan_cap,
                        )
                    short = em if len(em) <= 44 else em[:41] + "…"
                    progress.update(
                        llm_tid,
                        completed=idx,
                        total=scan_cap,
                        description=(
                            f"[green]{summ}/{max_rows} summaries[/green] · scan {idx}/{scan_cap} · {short}"
                        ),
                    )
                    return
                if t.startswith("Listing sent mail"):
                    if list_tid is None:
                        list_tid = progress.add_task("[yellow]Listing Gmail[/yellow]", total=None)
                    return
                err_console.print(f"[dim]{msg}[/dim]")

            result = await run_mailbox_outbound_digest(
                email_proc,
                months=months,
                after=after,
                before=before,
                mailbox=mailbox.strip() or None,
                query_extra=query_extra,
                max_contacts=max_contacts,
                max_threads_per_contact=max_threads,
                dry_run=dry_run,
                progress_sink=_mailbox_digest_progress,
                exclude_domains_extra=exclude_domains.strip() or None,
                exclude_emails_extra=exclude_emails.strip() or None,
                sales_focus=not all_sent,
                crm=crm,
                enrich_crm=not no_crm,
            )

            _drop_list_task()
            _complete_meta()

        meta = result.get("meta") or {}
        console.print(
            Panel(
                f"[bold]Mailbox:[/bold] {meta.get('mailbox')}\n"
                f"[bold]Sales-keyword filter:[/bold] "
                f"{'on (default)' if meta.get('sales_focus') else 'off (--all-sent)'}\n"
                f"[bold]Query:[/bold] {meta.get('gmail_query')}\n"
                f"[bold]Sent messages listed:[/bold] {meta.get('sent_messages_listed')}\n"
                f"[bold]External recipients:[/bold] {meta.get('external_recipients_found')}\n"
                f"[bold]Summarise cap:[/bold] {meta.get('recipients_summarised_cap')}",
                title="Mailbox digest",
                border_style="cyan",
            )
        )

        if dry_run and result.get("dry_run_recipients"):
            t = Table(title="Recipients (dry run)")
            t.add_column("Email", style="cyan", no_wrap=True)
            t.add_column("Thr", justify="right")
            if not no_crm:
                t.add_column("CRM bucket", style="magenta")
                t.add_column("Contact type")
                t.add_column("Company")
                t.add_column("Deal stage")
            for item in result["dry_run_recipients"]:
                row_cells = [
                    item.get("email", ""),
                    str(item.get("thread_count", 0)),
                ]
                if not no_crm:
                    row_cells.extend(
                        [
                            str(item.get("crm_bucket") or "—"),
                            str(item.get("contact_type") or "—"),
                            str(item.get("company_name") or "—"),
                            str(item.get("deal_stage") or "—"),
                        ]
                    )
                t.add_row(*row_cells)
            console.print(t)
            if output.strip():
                write_report(Path(output.strip()), result)
                console.print(f"[green]Wrote[/green] {output.strip()}")
            from brain_os.services.operator_webhook import fire_operator_webhook

            m = result.get("meta") or {}
            fire_operator_webhook(
                "mailbox_digest",
                {
                    "dry_run": True,
                    "months": int(months),
                    "all_sent": bool(all_sent),
                    "no_crm": bool(no_crm),
                    "sent_messages_listed": m.get("sent_messages_listed"),
                    "external_recipients_found": m.get("external_recipients_found"),
                    "output_path_set": bool(output.strip()),
                },
            )
            return

        for row in result.get("rows") or []:
            email = row.get("contact_email", "")
            summary = row.get("conversation_summary") or ""
            crm_md = ""
            if not no_crm:
                crm_md = (
                    f"\n\n**CRM:** {row.get('crm_bucket') or '—'} · "
                    f"type {row.get('contact_type') or '—'} · "
                    f"company {row.get('company_name') or '—'} · "
                    f"deal {row.get('deal_stage') or '—'}"
                )
            console.print(
                Panel(
                    Markdown(
                        f"**{email}** — {row.get('company_inferred') or 'company unknown'}"
                        f"{crm_md}\n\n"
                        f"{summary}\n\n"
                        f"**Machines:** {', '.join(row.get('machines_discussed') or []) or '—'}\n"
                        f"**Commercial:** {row.get('price_or_commercial_notes') or '—'}\n"
                        f"**Next steps:**\n"
                        + "\n".join(
                            f"- {s}" for s in (row.get("suggested_next_steps") or []) or ["—"]
                        ),
                    ),
                    title=email[:40],
                    border_style="green",
                )
            )

        if output.strip():
            write_report(Path(output.strip()), result)
            console.print(f"[green]Wrote[/green] {output.strip()}")

        from brain_os.services.operator_webhook import fire_operator_webhook

        m2 = result.get("meta") or {}
        fire_operator_webhook(
            "mailbox_digest",
            {
                "dry_run": False,
                "months": int(months),
                "all_sent": bool(all_sent),
                "no_crm": bool(no_crm),
                "sent_messages_listed": m2.get("sent_messages_listed"),
                "external_recipients_found": m2.get("external_recipients_found"),
                "rows_summarised": len(result.get("rows") or []),
                "output_path_set": bool(output.strip()),
            },
        )

    _run(_digest())


@crm_app.command("screen-thermoforming")
def crm_screen_thermoforming(
    input_file: str = typer.Argument(..., help="Path to company list file (.csv/.xlsx/.xls)."),
    sheet: str = typer.Option("", "--sheet", help="Excel sheet name (default: all sheets)."),
    min_tier: str = typer.Option(
        "medium",
        "--min-tier",
        help="Minimum tier to keep: irrelevant, low, medium, high.",
    ),
    include_competitors: bool = typer.Option(
        False,
        "--include-competitors",
        help="Keep rows flagged as machinery competitors (default: excluded).",
    ),
    top: int = typer.Option(
        200,
        "--top",
        help="Show top N rows in terminal (0 = all). CSV output keeps all matches.",
    ),
    output: str = typer.Option(
        "data/conversation/k_trade_fair_thermoforming_targets.csv",
        "--output",
        "-o",
        help="Write screened shortlist CSV to this path.",
    ),
) -> None:
    """Screen exhibitor/company lists for Machinecraft thermoforming-sales fit.

    Uses deterministic keyword scoring for process fit (thermoforming, vacuum
    forming, pressure forming), gauge fit (heavy/light), and thermoplastic
    material signals (ABS, HDPE, PC, PP, PET, PLA, PS, etc.).
    """
    console.print(
        "[red]Disabled:[/red] `ira crm screen-thermoforming` depended on "
        "`brain_os.systems.thermoforming_target_screener`, which is not shipped in this repository."
    )
    raise typer.Exit(1)
