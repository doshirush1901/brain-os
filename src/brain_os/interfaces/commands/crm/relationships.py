"""CRM CLI commands — see module docstring in package ``__init__``."""

from __future__ import annotations

import re
from email.utils import parseaddr
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from brain_os.interfaces.cli_runtime import (
    _run,
)

from ._helpers import (
    _subject_thread_key,
    _type_style,
)
from .app import console, crm_app


@crm_app.command("show")
def crm_show(
    email: str = typer.Argument(..., help="Email address of the contact to inspect."),
) -> None:
    """Show full detail for a single contact: info, company, deals, interactions."""

    async def _show() -> None:
        from brain_os.data.crm import CRMDatabase

        crm = CRMDatabase()
        await crm.create_tables()

        contact = await crm.get_contact_by_email(email)
        if not contact:
            console.print(f"[red]Contact not found: {email}[/red]")
            raise typer.Exit(1)

        ct = contact.contact_type.value if contact.contact_type else "UNCLASSIFIED"
        style = _type_style(ct)

        console.print(
            Panel(
                f"[bold]{contact.name}[/bold]\n"
                f"Email: [cyan]{contact.email}[/cyan]\n"
                f"Type: [{style}]{ct}[/{style}]\n"
                f"Score: {contact.lead_score:.0f}\n"
                f"Role: {contact.role or '—'}\n"
                f"Source: {contact.source or '—'}\n"
                f"Created: {contact.created_at.strftime('%Y-%m-%d') if contact.created_at else '—'}",
                title="Contact",
                border_style="cyan",
            )
        )

        if contact.company_id:
            comp = await crm.get_company(str(contact.company_id))
            if comp:
                console.print(
                    Panel(
                        f"[bold]{comp.name}[/bold]\n"
                        f"Region: {comp.region or '—'}\n"
                        f"Industry: {comp.industry or '—'}\n"
                        f"Website: {comp.website or '—'}",
                        title="Company",
                        border_style="blue",
                    )
                )

        deals = await crm.get_deals_for_contact(str(contact.id))
        if deals:
            deal_table = Table(title=f"Deals ({len(deals)})", show_header=True)
            deal_table.add_column("Title", width=30)
            deal_table.add_column("Stage", width=14)
            deal_table.add_column("Value", justify="right", width=12)
            deal_table.add_column("Machine", width=16)

            for d in deals:
                deal_table.add_row(
                    (d.get("title") or "")[:30],
                    d.get("stage", "?"),
                    f"{d.get('currency', 'USD')} {d.get('value', 0):,.0f}",
                    (d.get("machine_model") or "—")[:16],
                )
            console.print(deal_table)
        else:
            console.print("[dim]No deals.[/dim]")

        interactions = await crm.get_interactions_for_contact(str(contact.id))
        if interactions:
            int_table = Table(title=f"Interactions ({len(interactions)})", show_header=True)
            int_table.add_column("Date", width=12)
            int_table.add_column("Channel", width=10)
            int_table.add_column("Dir", width=4)
            int_table.add_column("Subject", width=45)

            for ix in interactions[:15]:
                date_str = ix.get("created_at", "")
                if date_str and len(date_str) > 10:
                    date_str = date_str[:10]
                int_table.add_row(
                    date_str,
                    (ix.get("channel") or "")[:10],
                    (ix.get("direction") or "")[:4],
                    (ix.get("subject") or "")[:45],
                )
            console.print(int_table)
            if len(interactions) > 15:
                console.print(f"[dim]... and {len(interactions) - 15} more interactions[/dim]")
        else:
            console.print("[dim]No interactions.[/dim]")

    _run(_show())


@crm_app.command("card")
def crm_card(
    email: str = typer.Argument(..., help="Email address of the contact to render as a card."),
) -> None:
    """Render a Pokemon-style contact card with summary stats and interconnections."""

    async def _card() -> None:
        from brain_os.data.crm import CRMDatabase
        from brain_os.schemas.interconnections import BuildInterconnectionsRequest
        from brain_os.services.email_interconnections import EmailInterconnectionsService

        crm = CRMDatabase()
        await crm.create_tables()

        contact = await crm.get_contact_by_email(email)
        if not contact:
            console.print(f"[red]Contact not found: {email}[/red]")
            raise typer.Exit(1)

        ct = contact.contact_type.value if contact.contact_type else "UNCLASSIFIED"
        style = _type_style(ct)
        score = int(contact.lead_score or 0)
        rarity = "Common"
        if score >= 80:
            rarity = "Legendary"
        elif score >= 60:
            rarity = "Epic"
        elif score >= 40:
            rarity = "Rare"

        deals = await crm.get_deals_for_contact(str(contact.id))
        interactions = await crm.get_interactions_for_contact(str(contact.id))

        inbound = 0
        outbound = 0
        last_touch = "—"
        if interactions:
            for ix in interactions:
                d = (ix.get("direction") or "").upper()
                if d.startswith("IN"):
                    inbound += 1
                elif d.startswith("OUT"):
                    outbound += 1
            last_touch = (interactions[0].get("created_at") or "—")[:19]

        company_name = "—"
        if contact.company_id:
            comp = await crm.get_company(str(contact.company_id))
            if comp:
                company_name = comp.name or "—"

        header = (
            f"[bold]{contact.name}[/bold]  [{style}]{ct}[/{style}]  "
            f"[bold]Rarity:[/bold] {rarity}\n"
            f"Email: [cyan]{contact.email}[/cyan]\n"
            f"Company: {company_name}\n"
            f"Role: {contact.role or '—'}\n"
            f"Lead score: {score}/100"
        )
        console.print(Panel(header, title="Pokemon Contact Card", border_style="magenta"))

        stats = Table(title="Battle Stats", show_header=True)
        stats.add_column("Metric", width=28)
        stats.add_column("Value", width=20, justify="right")
        stats.add_row("Deals", str(len(deals)))
        stats.add_row("Interactions", str(len(interactions)))
        stats.add_row("Inbound", str(inbound))
        stats.add_row("Outbound", str(outbound))
        stats.add_row("Last touch", last_touch)
        console.print(stats)

        if interactions:
            recent_table = Table(title="Recent Interaction Timeline", show_header=True)
            recent_table.add_column("Date", width=19)
            recent_table.add_column("Dir", width=6)
            recent_table.add_column("Subject", width=58)
            for ix in interactions[:8]:
                recent_table.add_row(
                    (ix.get("created_at") or "—")[:19],
                    (ix.get("direction") or "")[:6],
                    (ix.get("subject") or "—")[:58],
                )
            console.print(recent_table)

        context_lines: list[str] = []
        if interactions:
            for ix in interactions[:10]:
                subj = (ix.get("subject") or "").strip()
                body = (ix.get("content") or "").strip()
                if subj:
                    context_lines.append(subj)
                if body:
                    context_lines.append(body[:220])
        context_blob = "\n".join(context_lines)

        inter_svc = EmailInterconnectionsService(crm=crm)
        brief = await inter_svc.build_brief(
            BuildInterconnectionsRequest(
                thread_id="",
                recipient_email=contact.email,
                intent="followup",
                target_outcome="next_best_email",
                context_text=context_blob,
                date_options=[],
                constraints={"tone": "warm_professional"},
            ),
            thread_messages=[],
        )

        if brief.interconnections:
            ic_table = Table(title="Top Interconnections (Next Email)", show_header=True)
            ic_table.add_column("#", width=3, style="dim")
            ic_table.add_column("Type", width=16)
            ic_table.add_column("Claim", width=66)
            ic_table.add_column("Score", width=7, justify="right")
            for i, item in enumerate(brief.interconnections[:6], 1):
                ic_table.add_row(
                    str(i),
                    item.type,
                    item.claim[:66],
                    f"{item.scores.total:.2f}",
                )
            console.print(ic_table)

            cta = brief.selection.cta
            console.print(
                Panel(
                    f"[bold]Primary CTA:[/bold] {cta.primary or '—'}\n"
                    f"[bold]Secondary CTA:[/bold] {cta.secondary or '—'}\n"
                    f"[bold]Must include IDs:[/bold] {', '.join(brief.selection.must_include) or '—'}",
                    title="Draft Guidance",
                    border_style="cyan",
                )
            )

    _run(_card())


@crm_app.command("relationship-scan")
def crm_relationship_scan(
    mailbox_email: str = typer.Argument(..., help="Your mailbox email (e.g. you@example.com)."),
    contact_email: str = typer.Argument(
        ..., help="Counterparty email (e.g. buyer@partner.example.com)."
    ),
    max_results: int = typer.Option(
        200, "--max-results", min=20, max=500, help="Max messages per direction to scan."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit structured JSON output."),
) -> None:
    """Scan complete mailbox relationship between two emails and summarize leads + actions."""

    async def _collect_relationship_scan_data() -> dict[str, Any]:
        from collections import Counter

        from brain_os.config import get_settings
        from brain_os.data.crm import CRMDatabase
        from brain_os.data.models import Channel, Direction
        from brain_os.interfaces.email_processor import EmailProcessor

        own = (mailbox_email or "").strip().lower()
        other = (contact_email or "").strip().lower()
        if "@" not in own or "@" not in other:
            raise ValueError("Both mailbox and contact must be valid emails.")

        settings = get_settings()
        crm = CRMDatabase()
        await crm.create_tables()
        ep = EmailProcessor(delphi=None, digestive=None, sensory=None, crm=crm, settings=settings)

        from_msgs = await ep.search_emails(
            from_address=other,
            query=f"(to:{own} OR cc:{own})",
            max_results=max_results,
        )
        to_msgs = await ep.search_emails(
            from_address=own,
            query=f"(to:{other} OR cc:{other})",
            max_results=max_results,
        )
        broad_msgs = await ep.search_emails(
            query=(
                f"(from:{own} AND (to:{other} OR cc:{other})) OR "
                f"(from:{other} AND (to:{own} OR cc:{own}))"
            ),
            max_results=max_results,
        )

        dedup: dict[str, Any] = {}
        for m in from_msgs + to_msgs + broad_msgs:
            dedup[m.id] = m
        messages = sorted(dedup.values(), key=lambda x: x.received_at)

        contact = await crm.get_contact_by_email(other)
        if contact is None:
            contact = await crm.create_contact(
                name=other.split("@", 1)[0].replace(".", " ").title(),
                email=other,
                source="relationship_scan",
                lead_score=25.0,
            )

        existing = await crm.get_interactions_for_contact(str(contact.id))
        existing_keys = {
            f"{(ix.get('created_at') or '')[:19]}|{(ix.get('direction') or '').upper()}|{(ix.get('subject') or '').strip().lower()}"
            for ix in existing
        }
        inserted = 0

        def _norm_addr(raw: str) -> str:
            return (parseaddr(raw or "")[1] or raw or "").strip().lower()

        for m in messages:
            sender = _norm_addr(m.from_address or "")
            direction = Direction.INBOUND if sender == other else Direction.OUTBOUND
            created_dt = (
                m.received_at.replace(tzinfo=None)
                if m.received_at.tzinfo is not None
                else m.received_at
            )
            key = f"{created_dt.strftime('%Y-%m-%dT%H:%M:%S')}|{direction.value}|{(m.subject or '').strip().lower()}"
            if key in existing_keys:
                continue
            await crm.create_interaction(
                contact_id=str(contact.id),
                channel=Channel.EMAIL,
                direction=direction,
                subject=(m.subject or "")[:500],
                content=(m.body or "")[:4000],
                source_mailbox=getattr(m, "source_mailbox", None),
                created_at=created_dt,
            )
            existing_keys.add(key)
            inserted += 1

        def _is_contact_sender(raw: str) -> bool:
            addr = _norm_addr(raw)
            if addr == other:
                return True
            text = (raw or "").lower()
            local = other.split("@", 1)[0]
            parts = [p for p in re.split(r"[._-]+", local) if p]
            return all(p in text for p in parts[:2]) if len(parts) >= 2 else (local in text)

        inbound_msgs = [m for m in messages if _is_contact_sender(m.from_address or "")]
        outbound_msgs = [m for m in messages if _norm_addr(m.from_address or "") == own]

        text_blob = "\n".join([f"{m.subject or ''}\n{(m.body or '')[:800]}" for m in messages])
        company_patterns = {
            "DemoCorpA": r"\bDemoCorpA\b",
            "DemoCorpB": r"\bDemoCorpB\b",
        }
        machine_patterns = {
            "IMG": r"\bIMG\b",
            "ATF": r"\bATF\b",
            "DEMO": r"\bPF1\b",
        }
        company_hits = Counter()
        machine_hits = Counter()
        for name, pat in company_patterns.items():
            n = len(re.findall(pat, text_blob, flags=re.IGNORECASE))
            if n:
                company_hits[name] = n
        for name, pat in machine_patterns.items():
            n = len(re.findall(pat, text_blob, flags=re.IGNORECASE))
            if n:
                machine_hits[name] = n

        intro_keywords = [
            "introduction",
            "intro",
            "connect",
            "looping in",
            "copied",
            "cc",
            "meet",
            "meeting",
            "visit",
            "roadshow",
            "partnership",
        ]
        intro_signals = []
        for m in inbound_msgs[:80]:
            txt = f"{m.subject or ''}\n{(m.body or '')[:1000]}".lower()
            if any(k in txt for k in intro_keywords):
                intro_signals.append(m)

        actions: list[str] = []
        if not inbound_msgs:
            actions.append(
                "No inbound from contact found: prioritize a direct prompt asking for decision/date."
            )
        if len(outbound_msgs) > 0 and len(inbound_msgs) == 0:
            actions.append(
                "High outbound-only pattern: next email should ask one binary decision (date/location)."
            )
        if company_hits:
            top = ", ".join(company_hits.keys())
            actions.append(
                f"Company keyword hits ({top}): confirm next step and one concrete date in reply."
            )
        if not actions:
            actions.append(
                "Continue thread with concise meeting-lock CTA and one proof interconnection."
            )

        thread_buckets: dict[str, dict[str, Any]] = {}
        for m in messages:
            key = _subject_thread_key(m.subject or "")
            row = thread_buckets.setdefault(
                key, {"thread_key": key, "count": 0, "inbound": 0, "outbound": 0, "last_at": ""}
            )
            row["count"] += 1
            sender = _norm_addr(m.from_address or "")
            if sender == own:
                row["outbound"] += 1
            elif _is_contact_sender(m.from_address or ""):
                row["inbound"] += 1
            row["last_at"] = max(row["last_at"], m.received_at.strftime("%Y-%m-%d %H:%M"))
        thread_scoreboard = sorted(
            thread_buckets.values(),
            key=lambda r: (r["count"], r["last_at"]),
            reverse=True,
        )[:12]

        return {
            "mailbox": own,
            "contact": other,
            "total_messages": len(messages),
            "inbound_count": len(inbound_msgs),
            "outbound_count": len(outbound_msgs),
            "synced_to_crm_now": inserted,
            "date_range": {
                "start": messages[0].received_at.strftime("%Y-%m-%d") if messages else "",
                "end": messages[-1].received_at.strftime("%Y-%m-%d") if messages else "",
            },
            "company_mentions": dict(company_hits),
            "machine_mentions": dict(machine_hits),
            "recent_timeline": [
                {
                    "date": m.received_at.strftime("%Y-%m-%d %H:%M"),
                    "direction": (
                        "IN"
                        if _is_contact_sender(m.from_address or "")
                        else ("OUT" if _norm_addr(m.from_address or "") == own else "OTHER")
                    ),
                    "subject": (m.subject or "—")[:120],
                }
                for m in messages[-12:]
            ],
            "intro_signals": [
                {
                    "date": m.received_at.strftime("%Y-%m-%d %H:%M"),
                    "subject": (m.subject or "—")[:120],
                }
                for m in intro_signals[:10]
            ],
            "action_points": actions[:6],
            "thread_scoreboard": thread_scoreboard,
        }

    async def _scan() -> None:
        try:
            data = await _collect_relationship_scan_data()
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

        if json_output:
            console.print_json(data=data)
            return

        panel = (
            f"[bold]Mailbox:[/bold] {data['mailbox']}\n"
            f"[bold]Contact:[/bold] [cyan]{data['contact']}[/cyan]\n"
            f"[bold]Total messages:[/bold] {data['total_messages']}\n"
            f"[bold]Inbound from contact:[/bold] {data['inbound_count']}\n"
            f"[bold]Outbound to contact:[/bold] {data['outbound_count']}\n"
            f"[bold]Synced to CRM now:[/bold] {data['synced_to_crm_now']}\n"
            f"[bold]Date range:[/bold] {data['date_range']['start'] or '—'} -> {data['date_range']['end'] or '—'}"
        )
        console.print(Panel(panel, title="Relationship Scan", border_style="green"))

        top = Table(title="Tracked Companies / Lead Context", show_header=True)
        top.add_column("Company", width=16)
        top.add_column("Mentions", justify="right", width=8)
        if data["company_mentions"]:
            for comp, n in sorted(
                data["company_mentions"].items(), key=lambda x: x[1], reverse=True
            )[:12]:
                top.add_row(comp, str(n))
        else:
            top.add_row("—", "0")
        console.print(top)

        machine_table = Table(title="Machine Context Signals", show_header=True)
        machine_table.add_column("Machine", width=16)
        machine_table.add_column("Mentions", justify="right", width=8)
        if data["machine_mentions"]:
            for mname, n in sorted(
                data["machine_mentions"].items(), key=lambda x: x[1], reverse=True
            )[:10]:
                machine_table.add_row(mname, str(n))
        else:
            machine_table.add_row("—", "0")
        console.print(machine_table)

        timeline = Table(title="Recent Timeline", show_header=True)
        timeline.add_column("Date", width=19)
        timeline.add_column("Dir", width=6)
        timeline.add_column("Subject", width=72)
        for row in data["recent_timeline"]:
            timeline.add_row(row["date"], row["direction"], row["subject"][:72])
        console.print(timeline)

        console.print(
            Panel(
                "\n".join(f"- {a}" for a in data["action_points"]),
                title="Action Points",
                border_style="cyan",
            )
        )

        if data["intro_signals"]:
            intro_table = Table(title="Potential Intro / Lead Signals (Inbound)", show_header=True)
            intro_table.add_column("Date", width=19)
            intro_table.add_column("Subject", width=72)
            for r in data["intro_signals"][:8]:
                intro_table.add_row(r["date"], r["subject"][:72])
            console.print(intro_table)

    _run(_scan())


@crm_app.command("relationship-story")
def crm_relationship_story(
    mailbox_email: str = typer.Argument(..., help="Your mailbox email."),
    contact_email: str = typer.Argument(..., help="Counterparty email."),
    max_results: int = typer.Option(
        200, "--max-results", min=20, max=500, help="Max messages per direction to scan."
    ),
) -> None:
    """Narrative relationship summary with thread scoreboard and top 3 actions."""

    async def _story() -> None:
        # Recompute quickly via subprocess-free call.
        from collections import Counter

        from brain_os.config import get_settings
        from brain_os.data.crm import CRMDatabase
        from brain_os.interfaces.email_processor import EmailProcessor

        own = (mailbox_email or "").strip().lower()
        other = (contact_email or "").strip().lower()
        if "@" not in own or "@" not in other:
            console.print("[red]Both mailbox and contact must be valid emails.[/red]")
            raise typer.Exit(1)

        settings = get_settings()
        crm = CRMDatabase()
        await crm.create_tables()
        ep = EmailProcessor(delphi=None, digestive=None, sensory=None, crm=crm, settings=settings)
        from_msgs = await ep.search_emails(
            from_address=other, query=f"(to:{own} OR cc:{own})", max_results=max_results
        )
        to_msgs = await ep.search_emails(
            from_address=own, query=f"(to:{other} OR cc:{other})", max_results=max_results
        )
        broad_msgs = await ep.search_emails(
            query=f"(from:{own} AND (to:{other} OR cc:{other})) OR (from:{other} AND (to:{own} OR cc:{own}))",
            max_results=max_results,
        )
        dedup: dict[str, Any] = {}
        for m in from_msgs + to_msgs + broad_msgs:
            dedup[m.id] = m
        messages = sorted(dedup.values(), key=lambda x: x.received_at)

        def _norm_addr(raw: str) -> str:
            return (parseaddr(raw or "")[1] or raw or "").strip().lower()

        def _is_contact_sender(raw: str) -> bool:
            addr = _norm_addr(raw)
            if addr == other:
                return True
            text = (raw or "").lower()
            local = other.split("@", 1)[0]
            parts = [p for p in re.split(r"[._-]+", local) if p]
            return all(p in text for p in parts[:2]) if len(parts) >= 2 else (local in text)

        inbound = [m for m in messages if _is_contact_sender(m.from_address or "")]
        outbound = [m for m in messages if _norm_addr(m.from_address or "") == own]

        text_blob = "\n".join([f"{m.subject or ''}\n{(m.body or '')[:800]}" for m in messages])
        company_patterns = {
            "DemoCorpA": r"\bDemoCorpA\b",
            "DemoCorpB": r"\bDemoCorpB\b",
        }
        company_hits = Counter()
        for name, pat in company_patterns.items():
            n = len(re.findall(pat, text_blob, flags=re.IGNORECASE))
            if n:
                company_hits[name] = n

        thread_buckets: dict[str, dict[str, Any]] = {}
        for m in messages:
            key = _subject_thread_key(m.subject or "")
            row = thread_buckets.setdefault(
                key, {"thread": key, "messages": 0, "inbound": 0, "outbound": 0, "last_at": ""}
            )
            row["messages"] += 1
            sender = _norm_addr(m.from_address or "")
            if sender == own:
                row["outbound"] += 1
            elif _is_contact_sender(m.from_address or ""):
                row["inbound"] += 1
            row["last_at"] = max(row["last_at"], m.received_at.strftime("%Y-%m-%d %H:%M"))
        scoreboard = sorted(
            thread_buckets.values(), key=lambda r: (r["messages"], r["last_at"]), reverse=True
        )[:10]

        actions: list[str] = []
        if company_hits:
            actions.append(
                "Named-entity signals in thread: reply with one decision request and explicit owners/dates."
            )
        if not actions:
            actions.append("Send one concise decision email: choose date window and meeting site.")
        if len(actions) < 3:
            actions.append(
                "Convert open threads into one priority tracker with explicit due dates."
            )
        if len(actions) < 3:
            actions.append(
                "Ask your internal champion for the top two accounts to prioritize next week."
            )
        actions = actions[:3]

        narrative = (
            f"Relationship intensity is [bold]{len(messages)} messages[/bold] "
            f"({len(inbound)} inbound / {len(outbound)} outbound) from "
            f"{messages[0].received_at.strftime('%Y-%m-%d') if messages else '—'} to "
            f"{messages[-1].received_at.strftime('%Y-%m-%d') if messages else '—'}. "
            f"Current active lanes are {', '.join([k for k, _ in company_hits.most_common(4)]) or 'general coordination'}."
        )
        console.print(Panel(narrative, title="Relationship Story", border_style="magenta"))

        score_table = Table(title="Thread Scoreboard", show_header=True)
        score_table.add_column("Thread", width=54)
        score_table.add_column("Msgs", justify="right", width=6)
        score_table.add_column("IN", justify="right", width=4)
        score_table.add_column("OUT", justify="right", width=5)
        score_table.add_column("Last", width=16)
        for row in scoreboard:
            score_table.add_row(
                row["thread"][:54],
                str(row["messages"]),
                str(row["inbound"]),
                str(row["outbound"]),
                row["last_at"][:16],
            )
        console.print(score_table)

        console.print(
            Panel(
                "\n".join(f"{i + 1}. {a}" for i, a in enumerate(actions)),
                title="Top 3 Next Actions",
                border_style="cyan",
            )
        )

    _run(_story())
