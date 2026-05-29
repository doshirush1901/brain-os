# Brain OS — BSL 1.1 review request (for counsel)

**To:** [Counsel name]  
**From:** Brain OS Operator / Acme Services Inc.  
**Re:** Public open-source skeleton **brain-os** — license parameters before **v1.0.0** tag  
**Repo:** https://github.com/doshirush1901/brain-os  
**Date:** 2026-05-30  

---

## Request

Please review whether the **Business Source License 1.1 (BSL)** parameters below are appropriate for our public forkable product (“Brain OS”), and what to change before we tag **v1.0.0** and onboard **2–3 design partners** under LOIs.

Full parameter stub in repo: `LICENSE`  
Tier/caps (technical): `docs/LICENSING.md`  
GTM context: `docs/BRAIN_OS_GTM.md`

**Not asking you to implement software** — only license text, entity name, production definition, trademark, and alignment with design-partner LOI template in `docs/BRAIN_OS_GTM.md`.

---

## Proposed BSL 1.1 parameters (current draft)

| Parameter | Draft value |
|:----------|:--------------|
| **Licensed Work** | Brain OS (brain-os GitHub repository) |
| **Licensor** | [TBD — Acme Corp legal entity name + address] |
| **Additional Use Grant** | Use, copy, modify, create derivative works, and redistribute for **non-production** purposes (evaluation, research, internal development), subject to Change License |
| **Change Date** | **2029-01-01** (≈4 years from first public release) |
| **Change License** | Apache License, Version 2.0 |
| **Commercial** | Production deployment, team use beyond Community caps, appliance resale, and Pro tier (unlimited docs, expanded MCP) require **separate commercial agreement** |

**Community tier (technical, not in BSL text today):** 500 indexed documents; ~26 MCP tools; no send-email / operator-inbox tools without Pro/trial key.

---

## Questions (please advise)

1. Is **BSL 1.1** the right choice vs **Apache-2.0-only** for the public Community tier, with Pro/Enterprise only in contract?
2. What **Licensor** line should appear (entity name, jurisdiction, address)?
3. Is **Change Date 2029-01-01** acceptable, or should we use “four years from first commercial availability”?
4. Does **“non-production”** in the Additional Use Grant clearly cover **design-partner pilots** (90-day eval on partner infra)?
5. How should we define **“Production Use”** in the commercial agreement (SaaS, internal headcount threshold, on-prem appliance, MCP connected to live customer mail)?
6. Are **Community caps** (document count, MCP allowlist) best enforced only technically, or also spelled out in license + MSA?
7. **Trademark:** “Brain OS” vs existing **“Ira”** — clearance, registration, and README disclaimers?
8. Do we need a **NOTICE** file for third-party components (Neo4j, Qdrant, OpenAI SDK, etc.)?
9. **India / EU** customers — any license or data-residency wording required in LICENSE or MSA?
10. Does the **Design Partner LOI** template in `docs/BRAIN_OS_GTM.md` (§4 License & data) conflict with BSL, and what edits do you require?

---

## Deliverables we need back

- [ ] Approved **full BSL 1.1 text** (or alternative license) ready to paste into `LICENSE`
- [ ] Approved **Licensor** string
- [ ] Approved **production / commercial** definition for MSA schedule
- [ ] Any **LOI** redlines (1 page)
- [ ] Go / no-go on **v1.0.0** public tag

---

## Relationship to private repo

- **ira-v3** (private): Acme Corp operator — **not** licensed under this public BSL for customer forks.
- **brain-os** (public): Forkable skeleton with synthetic Acme demo only.

---

*Internal reference only until counsel marks otherwise. Do not forward customer PII or Ira customer data with this brief.*
