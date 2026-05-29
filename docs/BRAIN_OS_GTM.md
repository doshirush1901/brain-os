# Brain OS — GTM pack (legal, LOI, positioning)

**Not legal advice.** Have counsel review before relying on BSL parameters, LOI signatures, or customer-facing license copy.

---

## Positioning (one-liners)

| Repo | One-liner |
|:-----|:----------|
| **Ira** (`ira-v3`, private) | Machinecraft’s operator brain — full vertical depth, live CRM/mail, customer data; not forked as a product. |
| **Brain OS** (`brain-os`, public) | Forkable local-first multi-agent OS — same engine patterns as Ira, Acme demo only; your company data stays in **your** private fork. |
| **ira-universe** (public, separate) | Visitor-safe “what is Ira?” corpus — not the runnable fork. |

**GitHub About (brain-os):**  
`Forkable multi-agent Brain OS for Mac — RAG, CRM, MCP/Cursor. Ira = private operator; this repo = build your own.`

**GitHub About (ira-v3, if private description only):**  
`Private operator stack for Machinecraft — not the public fork. Public skeleton: github.com/doshirush1901/brain-os`

---

## README snippets (copy-paste)

### brain-os `README.md` (top, after title)

```markdown
> **Ira** is Machinecraft’s private operator system (CRM, Gmail, production truth). **Brain OS** is the public forkable skeleton — same architecture patterns, synthetic Acme demo only. Run your business in a **private fork** of this repo; do not expect Machinecraft customer data here.
```

### ira-v3 `README.md` (under “What this repository is”)

```markdown
**Public fork:** Customers and design partners should start from **[brain-os](https://github.com/doshirush1901/brain-os)** — not this tree. Ira v3 is the canonical private operator codebase for Machinecraft.
```

---

## BSL / LICENSE — counsel review checklist

Current public stub: `brain_os_skeleton/LICENSE` (parameters only). Full text: [BSL 1.1](https://mariadb.com/bsl11/).

| # | Question for counsel | Current draft answer |
|:--|:---------------------|:---------------------|
| 1 | Is **BSL 1.1** appropriate vs Apache-2.0-only for Community? | BSL + commercial Pro/Appliance |
| 2 | **Licensor** legal entity name and address | “Brain OS Maintainers” → replace with Machinecraft entity |
| 3 | **Change Date** (2029-01-01) — acceptable? | 4 years from first public tag; confirm |
| 4 | **Additional Use Grant** — does “non-production” cover design-partner pilots? | Explicit pilot clause if needed |
| 5 | **Production** definition — SaaS, internal team >N users, on-prem appliance? | Define in commercial agreement |
| 6 | Relationship to **Community caps** (500 docs, MCP tier) — enforceable or honor-system? | Technical + contract |
| 7 | **Trademark** — “Brain OS” vs “Ira” — clearance and co-existence | Register / disclaim |
| 8 | **Third-party** deps in export (OpenAI, Neo4j, Qdrant) — notice file | Add `NOTICE` if required |
| 9 | **EU / India** data residency — does license text need jurisdiction? | If selling abroad |
| 10 | **Design-partner LOI** below — NDA + license grant consistent with BSL? | Align LOI §4 with BSL grant |

**Before v1.0 tag on brain-os:**

- [ ] Replace parameter stub with signed-off BSL 1.1 full text in `LICENSE`
- [ ] Add `NOTICE` / third-party attributions if counsel requires
- [ ] Publish `docs/LICENSING.md` tier table matching contract SKUs
- [ ] Remove or narrow `BRAIN_LICENSE_STUB_ACCEPT` from any customer-facing docs

---

## Design-partner LOI (letter of intent) — template

Use for **2–3** pilots (Q1). Goal: free/discounted Pro access + case-study rights + feedback; not a binding MSA.

```text
LETTER OF INTENT — Brain OS Design Partner
Date: ___________
Parties:
  Provider: [Machinecraft Technologies / legal entity], ("Provider")
  Partner:  [Company legal name], ("Partner")

1. Purpose
   Partner will evaluate Brain OS (public repo: github.com/doshirush1901/brain-os)
   on Partner-owned infrastructure for [90] days starting ___________.

2. What Provider grants
   - Early access to Brain OS Pro features (trial/license key: bos_trial_… or bos_live_…)
   - [Optional] Up to [N] hours onboarding / office hours
   - Export updates via republish cadence (best-effort)

3. What Partner provides
   - Named technical lead: ___________
   - Written feedback biweekly (bugs, UX, missing agents)
   - Permission to publish a **case study** (name/logo approval required):
     title, quote, metrics (no confidential KPIs without written approval)

4. License & data
   - Partner runs a **private fork**; Partner data stays in Partner environment.
   - Evaluation under BSL Additional Use Grant + separate Pro terms TBD.
   - No production resale of Brain OS without commercial agreement.

5. No exclusivity; no minimum spend
   LOI is non-binding except §6–7.

6. Confidentiality
   Mutual NDA for non-public roadmap, pricing, and security findings.

7. Term / termination
   Either party may end with 14 days notice. Partner may keep private fork;
   Pro keys revoked on termination unless converted to paid.

8. Non-binding
   Except confidentiality, no obligation to sign MSA. Commercial terms
   negotiated separately.

Signed:
Provider: ___________________  Date: _______
Partner:  ___________________  Date: _______
```

**Three partner profiles to target:**

| # | Profile | Why |
|:--|:--------|:----|
| 1 | Industrial OEM / machinery (non-competitor) | Close to Machinecraft ICP; validates pantheon + CRM |
| 2 | B2B SaaS 50–200 FTE | Validates fork + Cursor MCP without factory jargon |
| 3 | Services firm (consulting / integration) | Validates “build agents for clients” resale narrative |

---

## Commercial SKUs (align with `docs/LICENSING.md`)

| SKU | Audience | Price (placeholder) |
|:----|:---------|:--------------------|
| **Community** | Fork + eval | $0 |
| **Pro** | Team, unlimited docs + MCP | $___ / seat / mo |
| **Appliance** | Mac Pro / on-prem image | $___ / year |
| **Enterprise** | SSO, SLA, custom agents | Quote |

---

## Next actions (human)

1. Send BSL checklist (§ above) to counsel; block **v1.0.0** tag until signed off.
2. Fill LOI template for 3 targets; send for signature.
3. After legal OK: `republish` brain-os with final `LICENSE` + README one-liners.
4. Add GitHub **About** + link brain-os from ira-universe / Machinecraft site when ready.

*Last updated: 2026-05-30*
