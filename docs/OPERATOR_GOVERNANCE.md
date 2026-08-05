# Operator governance

How to run a company brain safely: powerful exploration, **gated mutation**. The human stays liable at send and merge — same idea whether you are on day 1 with Acme or month 3 with live CRM.

---

## Autonomy ladder

Each step does more with less sitting-time, and needs more evidence (tests, review, decision logs).

| Step | Hands over | Typical surface |
|:-----|:-----------|:----------------|
| **Sync chat** | breath | Cursor + MCP (`brief`, draft) |
| **Async work** | time | `brain ask` / long tasks, overnight scripts, dream if enabled |
| **PR babysit** | goal | CI / Cloud Agents on the **code** repo only |
| **Orchestrate** | plan | Multi-agent pantheon + operator inbox |

```text
sync chat → async work → PR babysit → orchestrate
              ↑ more autonomy → more evidence required ↑
                    human liable at send / merge
```

---

## Evidence before outbound

Do not draft (and never send) cold or account mail until the triangle is filled or gaps are explicit:

| Leg | Question |
|:----|:---------|
| **Intent** | What do they need? |
| **Relationship** | What did we already do with them? |
| **Identity** | Who are they (company + domain)? |

Details and hexagon (before send/quote): [TRIANGULATION.md](TRIANGULATION.md).

Fast path:

```bash
poetry run brain brief "Acme Corp" --contact jane@acme.com --json
```

MCP: `get_account_brief`.

---

## Draft-only until explicit send

- Default email mode is **TRAINING** (draft-first). See `.env.example`.
- Community tier: draft tools only ([LICENSING.md](LICENSING.md)).
- Pro/trial may unlock send MCP tools — still require an explicit human **send** / **publish** on the exact To/Subject/Body. “Looks good” / “continue” ≠ send.

---

## Desktop vs Cloud

| Surface | Owns |
|:--------|:-----|
| **Desktop Cursor + full Brain MCP** | Brief, draft, CRM, mail read/write (with gates), operator inbox |
| **Cursor Cloud Agents** | Code health, CI, docs PRs — **read/status only** |

**Do not** point Cloud Agents at a full `brain-mcp` / send / CRM-write profile. A dedicated read-only `brain cloud-mcp` profile may ship later; until then, keep mutation tools off Cloud.

---

## Cursor hooks (pattern)

Prefer project hooks that **fail closed** on irreversible tools (e.g. refuse `send_email` unless a confirm flag / typed approval is present). Rules (`.mdc`) are prompts — hooks (or CLI outside Auto-run) enforce gates. Exact Cursor hook JSON evolves; start from your IDE’s hooks docs and deny send paths until you trust the loop.

---

## Related

- [FRIEND_FORK.md](FRIEND_FORK.md) — peer onboarding
- [PUBLIC_PRIVATE_BOUNDARY.md](PUBLIC_PRIVATE_BOUNDARY.md) — what never goes public
- [MCP_SETUP.md](MCP_SETUP.md) — local MCP
- [FORK_GUIDE.md](FORK_GUIDE.md) — Day 0–7 checklist
