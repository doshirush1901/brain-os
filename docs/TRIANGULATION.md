# Triangulation (Brain OS)

Evidence-first account prep before outbound mail, quotes, or CRM stage changes.

## Triangle (minimum)

| Leg | Question | Typical tools |
|:----|:---------|:--------------|
| **Intent** | What do they need? | `search_knowledge`, CRM notes, web scrape |
| **Relationship** | What did we already do with them? | `search_crm`, `search_emails`, `read_email_thread` |
| **Identity** | Who are they (company + domain)? | `find_company_contacts`, `ask_agent` (research) |

Fast path:

```bash
poetry run brain brief "Acme Corp" --contact jane@acme.com --json
```

MCP: `get_account_brief` with the same company/contact anchors.

## Hexagon (before send / quote)

Add production truth, graph quotes, external intel, proof registry URLs, and corrections ledger checks. Run a persuasion or quote draft only when hex legs are filled or gaps are explicitly listed.

## Gaps

If any leg is empty, mark **UNVERIFIED** in the operator brief and state the next command to run. Do not invent mail history or deal stage.

## Related

- [QUICKSTART.md](QUICKSTART.md) · [MCP_SETUP.md](MCP_SETUP.md) · [FORK_GUIDE.md](FORK_GUIDE.md)
