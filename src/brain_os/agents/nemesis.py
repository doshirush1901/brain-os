"""Nemesis — Trainer / Adversarial agent.

Generates adversarial training scenarios, stress-tests specialist agents,
and feeds the results back through the :class:`~brain_os.systems.learning_hub.LearningHub`
to close the improvement loop.

The primary tool is :meth:`create_training_scenario`, which:

1. Queries the LearningHub for areas with low performance scores.
2. Synthesises a challenging hypothetical query for that area.
3. Routes the query to the appropriate specialist agent.
4. Generates an ideal reference answer via a high-quality LLM call.
5. Scores the agent's response against the ideal and logs the result.

Equipped with ReAct tools for correction ingestion, training cycle
execution, adversarial scenario creation, and training stats retrieval.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.brain.correction_store import CorrectionCategory, CorrectionSeverity, CorrectionStore
from brain_os.exceptions import DatabaseError, ToolExecutionError
from brain_os.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

# ── Agent ↔ domain mapping ───────────────────────────────────────────────
# Used to select which agent should handle a scenario in a given domain.

_DOMAIN_AGENTS: dict[str, str] = {
    "sales": "prometheus",
    "pricing": "plutus",
    "finance": "plutus",
    "marketing": "hermes",
    "production": "hephaestus",
    "hr": "themis",
    "research": "clio",
    "writing": "calliope",
    "forecasting": "tyche",
}

# ── LLM prompts ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = load_prompt("nemesis_system")

_SCENARIO_GEN_PROMPT = load_prompt("nemesis_scenario_gen")

_SCENARIO_GEN_FALLBACK_PROMPT = load_prompt("nemesis_scenario_gen_fallback")

_IDEAL_RESPONSE_PROMPT = load_prompt("nemesis_ideal_response")

_SCORING_PROMPT = load_prompt("nemesis_scoring")


@dataclass
class TrainingResult:
    """Outcome of a single training scenario."""

    scenario_id: str = field(default_factory=lambda: str(uuid4()))
    domain: str = ""
    target_agent: str = ""
    test_query: str = ""
    agent_response: str = ""
    ideal_response: str = ""
    scores: dict[str, Any] = field(default_factory=dict)
    overall_score: int = 0


class Nemesis(BaseAgent):
    name = "nemesis"
    role = "Trainer / Adversarial"
    description = "Generates training scenarios and stress-tests agent responses"
    knowledge_categories = [
        "company_internal",
        "sales_and_crm",
        "market_research_and_analysis",
        "project_case_studies",
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._learning_hub: Any | None = None
        self._peer_agents: dict[str, BaseAgent] = {}
        self._correction_store: CorrectionStore | None = None

    def configure(
        self,
        learning_hub: Any,
        peer_agents: dict[str, BaseAgent],
    ) -> None:
        """Wire Nemesis to the LearningHub and the agent roster.

        Called after Pantheon construction so Nemesis can reach its peers.
        """
        self._learning_hub = learning_hub
        self._peer_agents = peer_agents

    async def _ensure_correction_store(self) -> CorrectionStore:
        if self._correction_store is None:
            self._correction_store = CorrectionStore()
            await self._correction_store.initialize()
        return self._correction_store

    async def get_pending_corrections(self, limit: int = 20) -> list[dict[str, Any]]:
        """Public API for other agents (e.g. Sophia) to read correction history."""
        store = await self._ensure_correction_store()
        return await store.get_pending_corrections(limit=limit)

    # ── tool registration ────────────────────────────────────────────────

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="ingest_correction",
                description="Ingest a correction for an agent — records what was wrong and what the correct answer is.",
                parameters={
                    "agent_name": "Name of the agent that made the error",
                    "original": "The original (incorrect) response or value",
                    "corrected": "The correct response or value",
                    "reason": "Why this is a correction / what was wrong",
                },
                handler=self._tool_ingest_correction,
            )
        )
        self.register_tool(
            AgentTool(
                name="run_training",
                description="Run a training cycle: generate adversarial scenarios, test agents, and score responses.",
                parameters={
                    "agent_name": "Target agent name (empty string for all agents)",
                    "num_scenarios": "Number of scenarios to generate (default '3')",
                },
                handler=self._tool_run_training,
            )
        )
        self.register_tool(
            AgentTool(
                name="create_adversarial_scenario",
                description="Create a single adversarial training scenario for a specific agent.",
                parameters={
                    "agent_name": "Target agent name",
                    "difficulty": "Difficulty level: 'easy', 'medium', or 'hard' (default 'medium')",
                },
                handler=self._tool_create_adversarial_scenario,
            )
        )
        self.register_tool(
            AgentTool(
                name="get_training_stats",
                description="Get training statistics: correction counts, recent corrections, and store health.",
                parameters={},
                handler=self._tool_get_training_stats,
            )
        )
        self.register_tool(
            AgentTool(
                name="audit_decision_log",
                description="Generate an auditable decision trace for training outcomes.",
                parameters={
                    "decision": "Decision or training outcome text",
                    "evidence": "Optional evidence context",
                },
                handler=self._tool_audit_decision_log,
            )
        )
        self.register_tool(
            AgentTool(
                name="validate_correction_consistency",
                description="Validate statements against correction consistency constraints.",
                parameters={
                    "statement": "Statement to validate",
                    "ledger_context": "Optional correction context",
                },
                handler=self._tool_validate_correction_consistency,
            )
        )

    # ── tool handlers ────────────────────────────────────────────────────

    async def _tool_ingest_correction(
        self,
        agent_name: str,
        original: str,
        corrected: str,
        reason: str,
    ) -> str:
        context = {
            "agent_name": agent_name,
            "original_response": original,
            "reason": reason,
        }
        user_message = (
            f"Correction for agent '{agent_name}': "
            f"The original answer was: {original}. "
            f"The correct answer is: {corrected}. "
            f"Reason: {reason}"
        )
        return await self.ingest_correction(user_message, context)

    async def _tool_run_training(
        self,
        agent_name: str = "",
        num_scenarios: str = "3",
    ) -> str:
        count = int(num_scenarios) if num_scenarios.isdigit() else 3

        if self._learning_hub is None or not self._peer_agents:
            return "Training infrastructure not configured (no LearningHub or peer agents)."

        results = await self.run_training_cycle(num_scenarios=count)
        return self._format_training_report(results)

    async def _tool_create_adversarial_scenario(
        self,
        agent_name: str,
        difficulty: str = "medium",
    ) -> str:
        if not self._peer_agents:
            return "No peer agents configured for training."

        target_agent = agent_name.lower() if agent_name else "clio"
        domain = "research"
        for d, a in _DOMAIN_AGENTS.items():
            if a == target_agent:
                domain = d
                break

        scenario = {
            "test_query": (
                f"Generate a {difficulty}-difficulty test for the {target_agent} agent "
                f"in the {domain} domain."
            ),
            "domain": domain,
            "difficulty": difficulty,
        }

        raw = await self.call_llm(
            _SCENARIO_GEN_FALLBACK_PROMPT,
            f"Generate 1 {difficulty}-difficulty training scenario for the "
            f"{domain} domain targeting the {target_agent} agent.",
            temperature=0.7,
        )
        parsed = self._safe_parse(raw)
        if isinstance(parsed, list) and parsed:
            scenario = parsed[0] if isinstance(parsed[0], dict) else scenario
        elif isinstance(parsed, dict) and "test_query" in parsed:
            scenario = parsed

        scenario["domain"] = domain

        result = await self.create_training_scenario(scenario)
        return (
            f"**Scenario:** {result.test_query}\n"
            f"**Agent:** {result.target_agent}\n"
            f"**Score:** {result.overall_score}/10\n"
            f"**Response preview:** {result.agent_response[:500]}"
        )

    async def _tool_get_training_stats(self) -> str:
        store = await self._ensure_correction_store()
        try:
            all_corrections = await store.get_corrections()
            total = len(all_corrections)

            by_category: dict[str, int] = {}
            by_severity: dict[str, int] = {}
            recent = all_corrections[:5] if all_corrections else []

            for c in all_corrections:
                cat = c.get("category", "GENERAL")
                sev = c.get("severity", "MEDIUM")
                by_category[cat] = by_category.get(cat, 0) + 1
                by_severity[sev] = by_severity.get(sev, 0) + 1

            stats = {
                "total_corrections": total,
                "by_category": by_category,
                "by_severity": by_severity,
                "recent_corrections": [
                    {
                        "id": c.get("id", "?"),
                        "entity": c.get("entity", "?"),
                        "category": c.get("category", "?"),
                        "source": c.get("source", "?"),
                    }
                    for c in recent
                ],
            }
            return json.dumps(stats, default=str)
        except (DatabaseError, Exception) as exc:
            return f"Could not retrieve training stats: {exc}"

    async def _tool_audit_decision_log(self, decision: str, evidence: str = "") -> str:
        return await self.use_skill(
            "audit_decision_log",
            decision=decision,
            evidence=evidence,
        )

    async def _tool_validate_correction_consistency(
        self,
        statement: str,
        ledger_context: str = "",
    ) -> str:
        return await self.use_skill(
            "validate_correction_consistency",
            statement=statement,
            ledger_context=ledger_context,
        )

    # ── correction ingestion ──────────────────────────────────────────────

    async def ingest_correction(self, user_message: str, context: dict[str, Any]) -> str:
        """Extract a correction from natural language and persist it.

        Uses the LLM to parse the user's message into structured correction
        fields, stores it in the CorrectionStore, and immediately reinforces
        it in Mem0 via long-term memory so subsequent queries benefit.
        """
        extraction_prompt = (
            "You are a correction parser. Extract the correction from the user's message.\n"
            "Return JSON with keys: entity, category (one of PRICING/SPECS/CUSTOMER/COMPETITOR/GENERAL), "
            "severity (one of CRITICAL/HIGH/MEDIUM/LOW), old_value, new_value.\n"
            "If a field is unknown, use an empty string for old_value and MEDIUM for severity."
        )
        context_str = json.dumps(context, default=str)[:4000] if context else ""
        raw = await self.call_llm(
            extraction_prompt,
            f"User message: {user_message}\n\nContext: {context_str}",
            temperature=0.0,
        )
        parsed = self._safe_parse(raw)
        if not isinstance(parsed, dict) or "entity" not in parsed:
            return "Could not extract a structured correction from that message."

        store = await self._ensure_correction_store()

        try:
            category = CorrectionCategory(parsed.get("category", "GENERAL"))
        except ValueError:
            category = CorrectionCategory.GENERAL
        try:
            severity = CorrectionSeverity(parsed.get("severity", "MEDIUM"))
        except ValueError:
            severity = CorrectionSeverity.MEDIUM

        correction_id = await store.add_correction(
            entity=parsed["entity"],
            new_value=parsed.get("new_value", ""),
            category=category,
            severity=severity,
            old_value=parsed.get("old_value", ""),
            source="user_correction",
        )

        long_term = self._services.get("long_term_memory")
        if long_term is not None:
            content = (
                f"CORRECTION for {parsed['entity']}: "
                f"The correct information is: {parsed.get('new_value', '')}"
            )
            if parsed.get("old_value"):
                content += f" (previously stated as: {parsed['old_value']})"
            await long_term.store(
                content,
                user_id="global",
                metadata={
                    "type": "correction",
                    "entity": parsed["entity"],
                    "category": category.value,
                    "severity": severity.value,
                    "correction_id": correction_id,
                },
            )

        return (
            f"Correction #{correction_id} recorded: {parsed['entity']} — "
            f"{category.value}/{severity.value}. "
            "It will be fully integrated during the next sleep-training cycle."
        )

    async def ingest_failure(self, query: str, bad_response: str, context: dict[str, Any]) -> str:
        """Log a failed response so sleep training can learn from it."""
        extraction_prompt = (
            "Analyse this failed AI response. Identify what entity or topic was wrong, "
            "what the bad answer stated, and what the correct answer likely is.\n"
            'Return JSON: {"entity": "...", "old_value": "...", "new_value": "...", '
            '"category": "GENERAL", "severity": "HIGH"}'
        )
        raw = await self.call_llm(
            extraction_prompt,
            f"Original query: {query}\n\nBad response: {bad_response[:4000]}",
            temperature=0.0,
        )
        parsed = self._safe_parse(raw)
        if not isinstance(parsed, dict) or "entity" not in parsed:
            logger.warning("Could not extract failure details from bad response")
            parsed = {
                "entity": query[:200],
                "old_value": bad_response[:500],
                "new_value": "(needs manual correction)",
                "category": "GENERAL",
                "severity": "HIGH",
            }

        store = await self._ensure_correction_store()

        try:
            category = CorrectionCategory(parsed.get("category", "GENERAL"))
        except ValueError:
            category = CorrectionCategory.GENERAL
        try:
            severity = CorrectionSeverity(parsed.get("severity", "HIGH"))
        except ValueError:
            severity = CorrectionSeverity.HIGH

        correction_id = await store.add_correction(
            entity=parsed["entity"],
            new_value=parsed.get("new_value", ""),
            category=category,
            severity=severity,
            old_value=parsed.get("old_value", ""),
            source="failure_report",
        )
        return f"Failure logged as correction #{correction_id}. Will be addressed in next training cycle."

    # ── BaseAgent interface ──────────────────────────────────────────────

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        num_scenarios = ctx.get("num_scenarios", 3)

        if self._learning_hub is None or not self._peer_agents:
            return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)

        results = await self.run_training_cycle(num_scenarios=num_scenarios)
        return self._format_training_report(results)

    # ── training cycle ───────────────────────────────────────────────────

    async def run_training_cycle(
        self,
        num_scenarios: int = 3,
    ) -> list[TrainingResult]:
        """Run a full training cycle: generate scenarios, test agents, score."""
        scenarios = await self._generate_scenarios(num_scenarios)
        results: list[TrainingResult] = []

        for scenario in scenarios:
            result = await self.create_training_scenario(scenario)
            results.append(result)

        return results

    async def create_training_scenario(
        self,
        scenario: dict[str, Any],
    ) -> TrainingResult:
        """Execute a single training scenario end-to-end.

        1. Pick the target agent for the scenario's domain.
        2. Send the test query to that agent.
        3. Generate an ideal reference response.
        4. Score the agent's response against the ideal.
        5. Log the result through the LearningHub.
        """
        test_query = scenario.get("test_query", "")
        domain = scenario.get("domain", "research")
        agent_name = _DOMAIN_AGENTS.get(domain, "clio")

        result = TrainingResult(
            domain=domain,
            target_agent=agent_name,
            test_query=test_query,
        )

        agent = self._peer_agents.get(agent_name)
        if agent is None:
            result.agent_response = f"(Agent '{agent_name}' not available)"
            result.overall_score = 0
            return result

        try:
            result.agent_response = await agent.handle(test_query)
        except (ToolExecutionError, Exception):
            logger.exception("Agent '%s' failed on training query", agent_name)
            result.agent_response = f"(Agent '{agent_name}' raised an error)"
            result.overall_score = 0
            return result

        result.ideal_response = await self.call_llm(
            _IDEAL_RESPONSE_PROMPT,
            test_query,
            temperature=0.2,
        )

        scoring_input = (
            f"QUERY:\n{test_query}\n\n"
            f"AGENT RESPONSE:\n{result.agent_response[:4000]}\n\n"
            f"IDEAL RESPONSE:\n{result.ideal_response[:4000]}"
        )
        raw_scores = await self.call_llm(
            _SCORING_PROMPT,
            scoring_input,
            temperature=0.0,
        )
        result.scores = self._safe_parse(raw_scores)
        result.overall_score = result.scores.get("overall", 5)

        if self._learning_hub is not None:
            try:
                interaction = await self._learning_hub._crm.create_interaction(
                    contact_id=str(uuid4()),
                    channel="CLI",
                    direction="INBOUND",
                    subject=test_query[:500],
                    content=result.agent_response,
                )
                await self._learning_hub.process_feedback(
                    interaction_id=str(interaction.id),
                    feedback_score=result.overall_score,
                    correction=(result.ideal_response if result.overall_score <= 5 else None),
                )
            except (DatabaseError, Exception):
                logger.exception("Failed to log training result to LearningHub")

        logger.info(
            "Training scenario: agent=%s domain=%s score=%d/10",
            agent_name,
            domain,
            result.overall_score,
        )
        return result

    # ── scenario generation ──────────────────────────────────────────────

    async def _generate_scenarios(
        self,
        count: int,
    ) -> list[dict[str, Any]]:
        """Build a list of training scenarios.

        Prefers targeting known weak areas from the LearningHub.  Falls
        back to LLM-generated scenarios when no feedback history exists.
        """
        scenarios: list[dict[str, Any]] = []

        if self._learning_hub is not None:
            weak_areas = self._learning_hub.get_weak_areas(limit=count)
            for area in weak_areas:
                gap = area.get("gap_analysis", {})
                description = gap.get("description", "general weakness")
                gap_type = gap.get("gap_type", "UNKNOWN")

                prompt = (
                    f"WEAK AREA:\n"
                    f"Type: {gap_type}\n"
                    f"Description: {description}\n"
                    f"Original score: {area.get('score', '?')}/10"
                )
                raw = await self.call_llm(
                    _SCENARIO_GEN_PROMPT,
                    prompt,
                    temperature=0.7,
                )
                parsed = self._safe_parse(raw)
                if isinstance(parsed, dict) and "test_query" in parsed:
                    scenarios.append(parsed)

        remaining = count - len(scenarios)
        if remaining > 0:
            raw = await self.call_llm(
                _SCENARIO_GEN_FALLBACK_PROMPT,
                f"Generate {remaining} diverse training scenarios.",
                temperature=0.8,
            )
            parsed = self._safe_parse(raw)
            if isinstance(parsed, list):
                for item in parsed[:remaining]:
                    if isinstance(item, dict) and "test_query" in item:
                        scenarios.append(item)

        if not scenarios:
            scenarios.append(
                {
                    "test_query": "What is the lead time for a DEMO1500 CNC machine order to Saudi Arabia?",
                    "domain": "sales",
                    "difficulty": "medium",
                    "rationale": "Fallback scenario covering sales + production + logistics.",
                }
            )

        return scenarios[:count]

    # ── formatting ───────────────────────────────────────────────────────

    def _format_training_report(self, results: list[TrainingResult]) -> str:
        if not results:
            return "No training scenarios were executed."

        lines = ["# Training Cycle Report\n"]
        total_score = 0

        for i, r in enumerate(results, 1):
            total_score += r.overall_score
            status = "PASS" if r.overall_score >= 7 else "WARN" if r.overall_score >= 4 else "FAIL"
            lines.append(f"## Scenario {i}: [{status}] {r.domain} → {r.target_agent}")
            lines.append(f"**Query:** {r.test_query}")
            lines.append(f"**Score:** {r.overall_score}/10\n")

            strengths = r.scores.get("strengths", [])
            if strengths:
                lines.append("**Strengths:**")
                for s in strengths:
                    lines.append(f"- {s}")

            weaknesses = r.scores.get("weaknesses", [])
            if weaknesses:
                lines.append("\n**Weaknesses:**")
                for w in weaknesses:
                    lines.append(f"- {w}")

            suggestion = r.scores.get("improvement_suggestion")
            if suggestion:
                lines.append(f"\n**Suggestion:** {suggestion}")

            lines.append("")

        avg = total_score / len(results) if results else 0
        lines.append(f"---\n**Average score:** {avg:.1f}/10 across {len(results)} scenarios")

        return "\n".join(lines)

    @staticmethod
    def _safe_parse(raw: str) -> Any:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return {"raw_response": raw}
