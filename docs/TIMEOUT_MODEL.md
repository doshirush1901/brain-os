# Timeout Model — Total, Sub-Agents, and Athena Synthesis

Brain OS uses a layered timeout model so that within a **total** time budget, **sub-agents** run in parallel (each with a slot), and **Athena** has her own time to package the final answer for Cursor/API.

---

## 1. Total timeout (pipeline)

**What it is:** The maximum time for the full request (perceive → route → explore/think/memory → sub-agents → synthesis → shape). After this, the user gets a short “request timed out” message.

**Config:** `APP__PIPELINE_TIMEOUT` (seconds). Default: `600` (10 minutes).

**Typical presets:**

| Preset | Seconds | Use case |
|--------|---------|----------|
| 30s    | 30      | Quick factual lookup |
| 2 min  | 120     | Single-topic question |
| 5 min  | 300     | Multi-agent research |
| 10 min | 600     | Complex / Cursor tasks (default) |
| 20 min | 1200    | Deep analysis / task loop |

**Optional shorter cap for simple prompts:** When `APP__DYNAMIC_PIPELINE_TIMEOUT=true`, the outer `asyncio.wait_for` budget may use `APP__PIPELINE_TIMEOUT_QUICK` (capped by `APP__PIPELINE_TIMEOUT`) for short single-line queries without “deep” substrings — see `_effective_pipeline_timeout_seconds` in `src/brain_os/pipeline.py`.

**Future: Brain OS/Athena chooses total by request type.**
Further tuning (e.g. intent-classified presets) can build on the same hook.

---

## 2. Sub-process timeouts (inside the pipeline)

Before and after sub-agents, the pipeline runs other steps (perceive, remember, route, enrich, execute, gap resolve, faithfulness, shape). Each of these can have its own timeouts (e.g. Sphinx 15s, retriever backends 35s/12s). They all run within the **total** pipeline timeout above.

---

## 3. Parallel sub-agents and per-agent “slot” timeout

**What it is:** Athena (or the deterministic router) selects which agents to run. Up to **N** of them run **in parallel** (e.g. N=5). Each sub-agent gets a **slot**: a maximum time to return the best possible answer. If an agent hits its slot timeout, its response is replaced with a timeout message and synthesis still runs with the rest.

**Config:**

- `APP__MAX_PARALLEL_AGENTS` — Max number of sub-agents running at once (default: `5`).
- `APP__AGENT_TIMEOUT` — Per-agent slot in seconds (default: `90`). Each agent has this long to finish.

**Behavior:**

- Pantheon runs selected agents with `asyncio.gather` and a **semaphore** of size `max_parallel_agents`. So at most 5 (or whatever you set) run concurrently; the rest wait and run as slots free up.
- Each agent is wrapped in `asyncio.wait_for(..., timeout=agent_timeout)` (or the agent’s own `timeout` if set). So each sub-agent has a clear “slot” and must give Athena the best answer within that time.

**Deterministic / procedural route:** When the pipeline runs specialists **sequentially** via `_execute_routed`, each `agent.handle` uses the **same** `APP__AGENT_TIMEOUT` (not a separate hardcoded value). See `docs/PIPELINE_AUDIT.md`.

---

## 3.5 Agent-loop execution policy

The `plan_task` / `execute_phase` loop in `src/brain_os/pipeline_loop.py` follows a stricter concurrency model than broad pipeline routing:

- **Single mutating phase at a time:** mutating phase execution is serialized.
- **Serial default within phase:** agent calls are serial unless explicitly marked read-only.
- **Parallel read-only fan-out:** only phases with `delegation_type` set to `readonly` or `read_only` are allowed to fan out with parallel `agent.handle` calls.

Operator guidance:

- If you need throughput, mark pure research/verification phases as read-only.
- Do not mark writing/modifying phases as read-only just for speed; this can reintroduce race conditions and inconsistent state.

---

## 4. Athena synthesis timeout (package answer for Cursor)

**What it is:** After Athena has all inputs (exploration, thinking, memory, sub-agent responses), she has a **dedicated timeout** to call the LLM and package the final answer so it can be displayed in the Cursor tab (or API).

**Config:** `APP__ATHENA_SYNTHESIS_TIMEOUT` (seconds). Default: `90`.

**Behavior:**

- Athena’s synthesis step is wrapped in `asyncio.wait_for(..., timeout=athena_synthesis_timeout)`.
- If it times out, the pipeline returns the concatenated sub-agent responses (no LLM-shaped answer) so the user still sees content.

---

## Summary

| Level              | Config                         | Default | Meaning |
|--------------------|--------------------------------|--------|---------|
| **Total**          | `APP__PIPELINE_TIMEOUT`        | 600s   | Full request; 30s / 2m / 5m / 10m / 20m presets; future: Athena/learning by request type. |
| **Sub-agent slot** | `APP__AGENT_TIMEOUT`           | 90s    | Per parallel sub-agent; best answer within this. |
| **Parallel cap**   | `APP__MAX_PARALLEL_AGENTS`     | 5      | Up to this many sub-agents at once. |
| **Athena synthesis** | `APP__ATHENA_SYNTHESIS_TIMEOUT` | 90s  | Time to package final answer for Cursor/API. |

## 5. Max rounds (ReAct loop depth)

**What it is:** Each sub-agent's ReAct loop (think → act → observe) runs for at most N iterations before forcing a final answer.

**Config:** `APP__REACT_MAX_ITERATIONS` (default: `8`).

**Stability tracking (Metis):** The Metis agent scores each response (0-100) and tracks a rolling average. When the average crosses 75 for 10 consecutive requests, Metis announces "I think we are stable." If the user rejects, Metis increases max_rounds by 20%. See `src/brain_os/agents/metis.py`.

---

All timeouts are enforced so that within the total budget, exploration, thinking, memory, and sub-agents run (with parallel slots), and Athena has her own window to produce the final answer for the Cursor tab.

---

## 7. Metrics: contract health snapshot

`GET /api/metrics` now includes a `contract_health` block derived from runtime counters:

- `assertions_total` — number of contract assertions evaluated.
- `assertions_passed` — assertions validated as passing.
- `assertions_failed` — computed as `assertions_total - assertions_passed`.
- `assertion_pass_rate_pct` — pass rate percentage (`null` when no assertions were evaluated yet).
- `replans_triggered` — how often execution shifted into corrective re-planning.
- `clarifications_triggered` — how often repeated assertion failures escalated to user clarification.

Use this section with `runtime_counters` and `pipeline_recent_stage_timings` to judge whether loop quality is improving as model/profile tuning changes.

---

## 6. Embedded timeouts (inside `pipeline.py`)

Beyond the four top-level knobs above, `pipeline.py` has several **embedded** timeouts on specific sub-steps. They all run inside the **total** pipeline budget but are not individually configurable today. Listed for operator awareness so unexplained cancels under load are easier to attribute.

| Sub-step | Default | Where | Notes |
|----------|---------|-------|-------|
| Sphinx clarify gate | 15s | `pipeline.py` (Sphinx call) | Vague queries get clarifying questions; if Sphinx itself stalls, the gate gives up so the rest of the pipeline can proceed. |
| Outreach thread prefetch | 10s | `pipeline.py` (outreach branch) | Reads referenced Gmail thread context before the LLM drafts — bounded so a slow Gmail call cannot eat the whole budget. |
| LEARN fact extraction (Mem0) | 8s | `_learn` (LLM-based fact extraction when enabled) | Background; failure here is logged and recorded in `_learn_metrics`. |
| RealTimeObserver (in `_learn`) | 10s | `_learn` (observer call) | Background; on timeout, the observer is skipped and the rest of `_learn` continues. |
| Faithfulness shared KB pull | implicit (`Qdrant` client default ~`QDRANT_TIMEOUT`) | `pipeline.py` (faithfulness gate) | One retrieval bundle reused for verifier + metacognition. Soft-fail on exception; sets `trace["kb_retrieval_degraded"] = True`. |
| Background LEARN wait on shutdown | 30s | `wait_for_background_tasks(timeout=30)` | Used by CLI before exiting so durable writes don't get cancelled mid-flight. |

**Tuning guidance:** these are intentional inner bounds. If you find yourself wanting to extend them, prefer raising `APP__PIPELINE_TIMEOUT` first — then reach for a per-step knob if you have evidence that one specific step is the bottleneck.

**Dynamic pipeline timeout:** when `APP__DYNAMIC_PIPELINE_TIMEOUT=true`, simple single-line queries get capped at `APP__PIPELINE_TIMEOUT_QUICK` rather than the full `APP__PIPELINE_TIMEOUT` (see `_effective_pipeline_timeout_seconds`).

---

## 8. Tool runner (ReAct + MCP + skills)

Individual tool invocations use `brain_os.services.tool_runner.run_tool` (retries, per-attempt timeout, structured errors).

| Knob | Env | Default | Applies to |
|------|-----|---------|------------|
| Transient retries | `APP__REACT_TOOL_TRANSIENT_RETRIES` | `1` | Extra attempts on 429/5xx, timeouts, connection errors. |
| ReAct per-tool ceiling | `APP__REACT_TOOL_TIMEOUT_SECONDS` | `45` | `BaseAgent._execute_tool`, `use_skill`. |
| MCP per-tool ceiling | `APP__MCP_TOOL_TIMEOUT_SECONDS` | `60` | `hardened_mcp_tool` on MCP registrations. |

ReAct observations include `attempts` in `IRA_TOOL_META`; MCP failures return `Error (tool_name): … [attempts=N]`. CI: `python scripts/ci/check_tool_runner_adoption.py`.
