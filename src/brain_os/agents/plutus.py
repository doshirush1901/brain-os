"""Plutus — Finance / CFO agent.

Handles financial analysis, pricing review, margin calculations,
quote generation, and budget oversight.  When the PricingEngine and CRM
are available (injected via ``services``), Plutus can generate structured
price estimates and pull real deal history.  Otherwise falls back to
knowledge-base search.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.agents.formula_react_tools import register_formula_tools
from brain_os.agents.pf1_react_tools import register_pf1_icp_tools
from brain_os.brain import financial_engineering as fe
from brain_os.exceptions import DatabaseError, LLMError
from brain_os.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("plutus_system")

_FE_FIELD_PARAMS: dict[str, tuple[str, ...]] = {
    "run_price_waterfall": (
        "currency",
        "list_price",
        "discounts",
        "rebates",
        "freight",
        "insurance",
        "duties",
        "taxes",
        "material_cost",
        "conversion_cost",
        "overhead_alloc",
        "finance_cost",
    ),
    "working_capital_impact": (
        "deal_id",
        "order_value",
        "milestones",
        "expected_collection_days",
        "supplier_payment_days",
        "inventory_days",
    ),
    "optimize_payment_terms": (
        "deal_id",
        "customer_tier",
        "credit_risk_tier",
        "candidate_terms",
        "production_schedule",
        "procurement_profile",
        "historical_acceptance_profile",
    ),
    "evaluate_quote_npv_irr": (
        "cashflows",
        "discount_rate",
        "initial_outlay",
        "terminal_value",
    ),
    "scenario_engine": (
        "base_case",
        "scenarios",
    ),
    "customer_credit_risk_score": (
        "company_id",
        "domain",
        "payment_history",
        "crm_disputes_count",
        "email_signal_flags",
        "external_signal",
    ),
}


def _coerce_fe_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
        if stripped.replace(".", "", 1).replace("-", "", 1).isdigit():
            try:
                return float(stripped) if "." in stripped else int(stripped)
            except ValueError:
                return stripped
        return stripped
    return value


class Plutus(BaseAgent):
    name = "plutus"
    role = "Chief Financial Officer"
    description = "Financial analysis, pricing, quote generation, and budget oversight"
    knowledge_categories = [
        "quotes_and_proposals",
        "tally_exports",
        "acme-corp finance",
        "contracts_and_legal",
        "business plans",
    ]

    @property
    def _pricing_engine(self) -> Any | None:
        return self._services.get("pricing_engine")

    @property
    def _crm(self) -> Any | None:
        return self._services.get("crm")

    @property
    def _quotes(self) -> Any | None:
        return self._services.get("quotes")

    # ── tool registration ─────────────────────────────────────────────────

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        if self._pricing_engine:
            self.register_tool(
                AgentTool(
                    name="estimate_price",
                    description="Estimate the price for a machine model with optional features.",
                    parameters={
                        "machine_model": "Machine model identifier (e.g. DEMO-500)",
                        "features": "Comma-separated optional features (default empty)",
                    },
                    handler=self._tool_estimate_price,
                )
            )

        if self._quotes:
            self.register_tool(
                AgentTool(
                    name="get_quote",
                    description="Retrieve a specific quote by its ID.",
                    parameters={"quote_id": "The quote identifier"},
                    handler=self._tool_get_quote,
                )
            )

        self.register_tool(
            AgentTool(
                name="search_financial_docs",
                description="Search the financial knowledge base for documents, contracts, and reports.",
                parameters={"query": "Search query"},
                handler=self._tool_search_financial_docs,
            )
        )

        if self._crm:
            self.register_tool(
                AgentTool(
                    name="get_deal_financials",
                    description="Get financial details for a specific deal by ID.",
                    parameters={"deal_id": "The deal identifier"},
                    handler=self._tool_get_deal_financials,
                )
            )

        self.register_tool(
            AgentTool(
                name="generate_invoice",
                description="Generate an invoice for a customer, optionally from a quote.",
                parameters={
                    "customer": "Customer name or identifier",
                    "quote_id": "Optional quote ID to base the invoice on",
                    "items": "Optional comma-separated line items",
                },
                handler=self._tool_generate_invoice,
            )
        )
        self.register_tool(
            AgentTool(
                name="calculate_quote_skill",
                description="Calculate quote values via canonical pricing skill.",
                parameters={
                    "machine_model": "Machine model identifier",
                    "configuration": "Optional JSON configuration",
                },
                handler=self._tool_calculate_quote_skill,
            )
        )
        self.register_tool(
            AgentTool(
                name="analyze_revenue_skill",
                description="Analyze revenue and pipeline velocity via canonical skill.",
                parameters={"filters": "Optional JSON filters"},
                handler=self._tool_analyze_revenue_skill,
            )
        )

        if self._services.get("pantheon"):
            self.register_tool(
                AgentTool(
                    name="ask_prometheus",
                    description="Delegate a sales/CRM question to the Prometheus (CRO) agent.",
                    parameters={"query": "The sales or CRM question"},
                    handler=self._tool_ask_prometheus,
                )
            )

        self._register_financial_engineering_tools()
        register_pf1_icp_tools(
            self,
            include_read_icp=False,
            include_pf1_fit_brief=False,
            include_format_crm_note=False,
            include_physics=True,
            include_maestro=True,
        )
        register_formula_tools(self, include_registry=True, include_math_mode=False)

    def _register_financial_engineering_tools(self) -> None:
        """Deterministic deal economics (same logic as MCP financial_engineering tools)."""
        specs: tuple[tuple[str, str, Any, Any], ...] = (
            (
                "run_price_waterfall",
                "Gross-to-net price waterfall and contribution margin from list price, discounts, and costs.",
                fe.run_price_waterfall,
                self._tool_run_price_waterfall,
            ),
            (
                "working_capital_impact",
                "Estimate cash conversion cycle and peak cash need from milestones and cycle days.",
                fe.working_capital_impact,
                self._tool_working_capital_impact,
            ),
            (
                "optimize_payment_terms",
                "Rank candidate payment milestone structures by cash curve, win probability, and risk.",
                fe.optimize_payment_terms,
                self._tool_optimize_payment_terms,
            ),
            (
                "evaluate_quote_npv_irr",
                "NPV, IRR, payback, and value-creation status for a quote cashflow stream.",
                fe.evaluate_quote_npv_irr,
                self._tool_evaluate_quote_npv_irr,
            ),
            (
                "scenario_engine",
                "Deterministic best/base/worst-style scenario stress on price, volume, cost, FX.",
                fe.scenario_engine,
                self._tool_scenario_engine,
            ),
            (
                "customer_credit_risk_score",
                "Credit risk tier and suggested commercial controls from payment and CRM signals.",
                fe.customer_credit_risk_score,
                self._tool_customer_credit_risk_score,
            ),
        )
        for name, description, _fn, handler in specs:
            self.register_tool(
                AgentTool(
                    name=name,
                    description=description,
                    parameters=self._fe_tool_parameters(name),
                    handler=handler,
                )
            )

    @staticmethod
    def _fe_tool_parameters(tool_name: str) -> dict[str, str]:
        params = {
            "payload_json": (
                "Optional JSON string of all inputs. Use for nested arrays/objects; "
                "otherwise pass individual fields below."
            ),
        }
        for field in _FE_FIELD_PARAMS[tool_name]:
            params[field] = f"Optional {field.replace('_', ' ')}."
        return params

    def _build_fe_payload(
        self,
        payload_json: str,
        raw: dict[str, Any],
        field_names: tuple[str, ...],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if (payload_json or "").strip():
            try:
                parsed = json.loads(payload_json)
                if isinstance(parsed, dict):
                    payload = dict(parsed)
            except json.JSONDecodeError:
                pass
        for key in field_names:
            if key not in raw:
                continue
            coerced = _coerce_fe_value(raw[key])
            if coerced is None:
                continue
            payload[key] = coerced
        return payload

    def _fe_envelope_json(self, fn: Any, payload: dict[str, Any]) -> str:
        try:
            envelope = fn(payload)
            return json.dumps(envelope.model_dump(), default=str)
        except Exception:
            logger.exception("Financial engineering tool %s failed", getattr(fn, "__name__", fn))
            return json.dumps(
                {
                    "ok": False,
                    "result": {},
                    "gaps": ["computation_error"],
                    "verification_status": "unverified",
                    "confidence": 0.0,
                }
            )

    async def _tool_run_price_waterfall(
        self,
        payload_json: str = "",
        currency: str = "",
        list_price: str = "",
        discounts: str = "",
        rebates: str = "",
        freight: str = "",
        insurance: str = "",
        duties: str = "",
        taxes: str = "",
        material_cost: str = "",
        conversion_cost: str = "",
        overhead_alloc: str = "",
        finance_cost: str = "",
    ) -> str:
        payload = self._build_fe_payload(
            payload_json, locals(), _FE_FIELD_PARAMS["run_price_waterfall"]
        )
        return self._fe_envelope_json(fe.run_price_waterfall, payload)

    async def _tool_working_capital_impact(
        self,
        payload_json: str = "",
        deal_id: str = "",
        order_value: str = "",
        milestones: str = "",
        expected_collection_days: str = "",
        supplier_payment_days: str = "",
        inventory_days: str = "",
    ) -> str:
        payload = self._build_fe_payload(
            payload_json, locals(), _FE_FIELD_PARAMS["working_capital_impact"]
        )
        return self._fe_envelope_json(fe.working_capital_impact, payload)

    async def _tool_optimize_payment_terms(
        self,
        payload_json: str = "",
        deal_id: str = "",
        customer_tier: str = "",
        credit_risk_tier: str = "",
        candidate_terms: str = "",
        production_schedule: str = "",
        procurement_profile: str = "",
        historical_acceptance_profile: str = "",
    ) -> str:
        payload = self._build_fe_payload(
            payload_json, locals(), _FE_FIELD_PARAMS["optimize_payment_terms"]
        )
        return self._fe_envelope_json(fe.optimize_payment_terms, payload)

    async def _tool_evaluate_quote_npv_irr(
        self,
        payload_json: str = "",
        cashflows: str = "",
        discount_rate: str = "",
        initial_outlay: str = "",
        terminal_value: str = "",
    ) -> str:
        payload = self._build_fe_payload(
            payload_json, locals(), _FE_FIELD_PARAMS["evaluate_quote_npv_irr"]
        )
        return self._fe_envelope_json(fe.evaluate_quote_npv_irr, payload)

    async def _tool_scenario_engine(
        self,
        payload_json: str = "",
        base_case: str = "",
        scenarios: str = "",
    ) -> str:
        payload = self._build_fe_payload(
            payload_json, locals(), _FE_FIELD_PARAMS["scenario_engine"]
        )
        return self._fe_envelope_json(fe.scenario_engine, payload)

    async def _tool_customer_credit_risk_score(
        self,
        payload_json: str = "",
        company_id: str = "",
        domain: str = "",
        payment_history: str = "",
        crm_disputes_count: str = "",
        email_signal_flags: str = "",
        external_signal: str = "",
    ) -> str:
        payload = self._build_fe_payload(
            payload_json, locals(), _FE_FIELD_PARAMS["customer_credit_risk_score"]
        )
        return self._fe_envelope_json(fe.customer_credit_risk_score, payload)

    # ── tool handlers ─────────────────────────────────────────────────────

    async def _tool_estimate_price(self, machine_model: str, features: str = "") -> str:
        config = {"features": features} if features else {}
        estimate = await self._pricing_engine.estimate_price(machine_model, config)
        return json.dumps(estimate, default=str)

    async def _tool_get_quote(self, quote_id: str) -> str:
        quote = await self._quotes.get_quote(quote_id)
        return json.dumps(quote, default=str) if quote else f"Quote '{quote_id}' not found."

    async def _tool_search_financial_docs(self, query: str) -> str:
        results = await self.search_domain_knowledge(query)
        return self._format_context(results)

    async def _tool_get_deal_financials(self, deal_id: str) -> str:
        deal = await self._crm.get_deal(deal_id)
        return json.dumps(deal, default=str) if deal else f"Deal '{deal_id}' not found."

    async def _tool_generate_invoice(
        self, customer: str, quote_id: str = "", items: str = ""
    ) -> str:
        return await self.use_skill(
            "generate_invoice",
            customer=customer,
            quote_id=quote_id,
        )

    async def _tool_calculate_quote_skill(self, machine_model: str, configuration: str = "") -> str:
        parsed_config: Any = {}
        if configuration:
            try:
                parsed_config = json.loads(configuration)
            except json.JSONDecodeError:
                parsed_config = {"raw_configuration": configuration}
        return await self.use_skill(
            "calculate_quote",
            machine_model=machine_model,
            configuration=parsed_config,
        )

    async def _tool_analyze_revenue_skill(self, filters: str = "") -> str:
        parsed_filters: Any = None
        if filters:
            try:
                parsed_filters = json.loads(filters)
            except json.JSONDecodeError:
                parsed_filters = {"raw_filters": filters}
        return await self.use_skill("analyze_revenue", filters=parsed_filters)

    async def _tool_ask_prometheus(self, query: str) -> str:
        return await self._tool_ask_agent("prometheus", query)

    # ── main handler ──────────────────────────────────────────────────────

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        task = ctx.get("task", "")

        if task == "generate_invoice":
            return await self.use_skill(
                "generate_invoice",
                customer=ctx.get("customer", ""),
                quote_id=ctx.get("quote_id", ""),
                items=ctx.get("items", []),
            )

        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)

    # ── private helpers (kept for backward compat) ────────────────────────

    def _should_estimate_price(self, query: str, ctx: dict[str, Any]) -> bool:
        price_keywords = {"quote", "price", "cost", "estimate", "pricing", "budget", "value"}
        return bool(price_keywords & set(query.lower().split())) or "machine_model" in ctx

    def _should_pull_crm(self, query: str, ctx: dict[str, Any]) -> bool:
        crm_keywords = {"deal", "pipeline", "history", "revenue", "analytics", "forecast", "win"}
        return bool(crm_keywords & set(query.lower().split())) or "contact_id" in ctx

    async def _get_pricing_context(self, query: str, ctx: dict[str, Any]) -> str:
        try:
            machine_model = ctx.get("machine_model", "")
            configuration = ctx.get("configuration", {})

            if not machine_model:
                machine_model = self._extract_machine_from_query(query)

            if not machine_model:
                return "Error: No machine model identified in the query. Ask for a specific model (e.g. DEMO-500) or use the estimate_price tool with a machine_model parameter."

            estimate = await self._pricing_engine.estimate_price(
                machine_model,
                configuration,
            )

            lines = []
            ep = estimate.get("estimated_price", {})
            if ep:
                lines.append(
                    f"Estimated price: {ep.get('currency', 'USD')} "
                    f"{ep.get('low', '?')} – {ep.get('high', '?')} "
                    f"(mid: {ep.get('mid', '?')})"
                )
            lines.append(f"Confidence: {estimate.get('confidence', 'unknown')}")
            if estimate.get("reasoning"):
                lines.append(f"Reasoning: {estimate['reasoning']}")

            similar = estimate.get("similar_quotes", [])
            if similar:
                lines.append("Similar historical quotes:")
                for sq in similar[:3]:
                    lines.append(f"  - {sq.get('content', '')[:200]}")

            return "\n".join(lines)
        except (LLMError, Exception):
            logger.exception("PricingEngine call failed in Plutus")
            return "(Pricing engine unavailable)"

    async def _get_crm_context(self, query: str, ctx: dict[str, Any]) -> str:
        try:
            filters = {}
            if "machine_model" in ctx:
                filters["machine_model"] = ctx["machine_model"]
            if "contact_id" in ctx:
                filters["contact_id"] = ctx["contact_id"]

            deals = await self._crm.get_deals_by_filter(filters)
            if not deals:
                summary = await self._crm.get_pipeline_summary(filters or None)
                return f"Pipeline summary: {json.dumps(summary, default=str)}"

            lines = [f"Found {len(deals)} matching deals:"]
            for d in deals[:5]:
                lines.append(
                    f"  - {d.get('title', 'Untitled')} | "
                    f"{d.get('stage', '?')} | "
                    f"{d.get('currency', 'USD')} {d.get('value', 0):,.2f} | "
                    f"Machine: {d.get('machine_model', 'N/A')}"
                )
            return "\n".join(lines)
        except (DatabaseError, Exception):
            logger.exception("CRM query failed in Plutus")
            return "(CRM data unavailable)"

    @staticmethod
    def _extract_machine_from_query(query: str) -> str:
        tokens = query.upper().split()
        for token in tokens:
            if (
                token.startswith("PF")
                or token.startswith("AM-")
                or token.startswith("RF-")
                or token.startswith("SL-")
            ):
                return token
        for i, token in enumerate(tokens):
            if token in ("PF1", "DEMO2") and i + 1 < len(tokens):
                return f"{token}-{tokens[i + 1]}"
        return ""
