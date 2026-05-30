"""MCP (Model Context Protocol) server for Brain OS.

Exposes Brain OS's core capabilities as MCP tools, making them accessible
to Claude, Cursor, and any MCP-compatible client.  Uses the same
bootstrap logic as the CLI to avoid service duplication.

Start with::

    ira mcp
    # or
    poetry run python -m ira.interfaces.mcp_server

Bind locally (``127.0.0.1``) or behind an authenticated gateway. Publishing raw MCP endpoints
without network-level trust is unsafe: tools can read Gmail, mutate CRM rows, ingest data, or
write memory on the host's behalf.
"""

from __future__ import annotations

import logging
import sys

from brain_os.interfaces.mcp_runtime import mcp

logger = logging.getLogger(__name__)


# ── External tool registration ─────────────────────────────────────────────
# Submodule tools register via register(mcp); must run at import time so the
# registry is complete before tests or mcp.run().
from brain_os.interfaces.mcp_tools import register_all_tools

register_all_tools(mcp)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point for running the MCP server."""
    # Stdio MCP owns stdout for JSON-RPC; keep all diagnostics on stderr.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    # Keep stderr concise in stdio mode; excessive stderr can destabilize some MCP clients.
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.WARNING)
    logging.getLogger("brain_os.interfaces.cli").setLevel(logging.WARNING)
    mcp.run()


if __name__ == "__main__":
    main()


# ── Test / CLI compatibility re-exports ───────────────────────────────────
# Submodule tools; tests call mcp_mod.<name>. Runtime names re-bound for
# monkeypatch (same pattern as W2 server EOF). Use module-import + assignment
# so ruff does not strip unused imports; names used only by lazy-facade tools
# live here instead of the top import block.
import brain_os.interfaces.mcp_runtime as _mcp_rt
import brain_os.interfaces.mcp_tools.agent_loop as _mcp_agent_loop_tools
import brain_os.interfaces.mcp_tools.agents as _mcp_agents_tools
import brain_os.interfaces.mcp_tools.board as _mcp_board_tools
import brain_os.interfaces.mcp_tools.brand as _mcp_brand_tools
import brain_os.interfaces.mcp_tools.brief as _mcp_brief_tools
import brain_os.interfaces.mcp_tools.corrections as _mcp_corrections_tools
import brain_os.interfaces.mcp_tools.creative as _mcp_creative_tools
import brain_os.interfaces.mcp_tools.crm as _mcp_crm_tools
import brain_os.interfaces.mcp_tools.delegate as _mcp_delegate_tools
import brain_os.interfaces.mcp_tools.dream as _mcp_dream_tools
import brain_os.interfaces.mcp_tools.email as _mcp_email_tools
import brain_os.interfaces.mcp_tools.engineering as _mcp_engineering_tools
import brain_os.interfaces.mcp_tools.financial_engineering as _mcp_financial_engineering_tools
import brain_os.interfaces.mcp_tools.graph as _mcp_graph_tools
import brain_os.interfaces.mcp_tools.ingest as _mcp_ingest_tools
import brain_os.interfaces.mcp_tools.math_mode as _mcp_math_mode_tools
import brain_os.interfaces.mcp_tools.memory as _mcp_memory_tools
import brain_os.interfaces.mcp_tools.na_sales as _mcp_na_sales_tools
import brain_os.interfaces.mcp_tools.projects as _mcp_projects_tools
import brain_os.interfaces.mcp_tools.query as _mcp_query_tools
import brain_os.interfaces.mcp_tools.quotes as _mcp_quotes_tools
import brain_os.interfaces.mcp_tools.revenue as _mcp_revenue_tools
import brain_os.interfaces.mcp_tools.scheduling as _mcp_scheduling_tools
import brain_os.interfaces.mcp_tools.slack as _mcp_slack_tools
import brain_os.interfaces.mcp_tools.system as _mcp_system_tools
import brain_os.interfaces.mcp_tools.task as _mcp_task_tools
import brain_os.interfaces.mcp_tools.web_maps as _mcp_web_maps_tools

_all_mcp_tool_names = _mcp_rt._all_mcp_tool_names
_initialized = _mcp_rt._initialized
_knowledge_graph = _mcp_rt._knowledge_graph
_long_term_memory = _mcp_rt._long_term_memory
_conversation_memory = _mcp_rt._conversation_memory
_relationship_memory = _mcp_rt._relationship_memory
_goal_manager = _mcp_rt._goal_manager
_model_to_dict = _mcp_rt._model_to_dict
_crm = _mcp_rt._crm
_email_processor = _mcp_rt._email_processor
_tinder_email_mode = _mcp_rt._tinder_email_mode
_ingestor = _mcp_rt._ingestor
_MCP_DOCAI_MAX_BYTES = _mcp_rt._MCP_DOCAI_MAX_BYTES
# 2.11 — scheduling tools moved; last inline _shared_services reader gone
_shared_services = _mcp_rt._shared_services
# 2.12 — web_maps tools moved; last inline _pantheon reader gone
_pantheon = _mcp_rt._pantheon
# 2.13 — task orchestrator tools moved; last inline _task_orchestrator reader gone
_task_orchestrator = _mcp_rt._task_orchestrator
# 2.14 — agent-loop tools moved; last inline users for agent loop + helpers
_agent_loop = _mcp_rt._agent_loop
_AGENT_LOOP_RETRY_POLICY = _mcp_rt._AGENT_LOOP_RETRY_POLICY
_is_transient_agent_loop_error = _mcp_rt._is_transient_agent_loop_error
_loop_telemetry = _mcp_rt._loop_telemetry
_normalize_validator_mode = _mcp_rt._normalize_validator_mode
_utc_now_iso = _mcp_rt._utc_now_iso
# 2.16 — query_brain moved; last inline users for pipeline + discovery helpers
_pipeline = _mcp_rt._pipeline
_is_short_factual_query = _mcp_rt._is_short_factual_query
_quick_answer_core = _mcp_rt._quick_answer_core
_rank_tools_for_query = _mcp_rt._rank_tools_for_query
_mcp_query_sender_id = _mcp_rt._mcp_query_sender_id
_retriever = _mcp_rt._retriever
# 2.17 — get_system_status moved; _ensure_initialized last inline user gone.
# Required at EOF for srv._ensure_initialized() facade reads from all extracted tools.
_ensure_initialized = _mcp_rt._ensure_initialized

find_related_entities = _mcp_graph_tools.find_related_entities
find_company_contacts = _mcp_graph_tools.find_company_contacts
find_company_quotes = _mcp_graph_tools.find_company_quotes
get_project_status = _mcp_projects_tools.get_project_status
get_overdue_milestones = _mcp_projects_tools.get_overdue_milestones
submit_correction = _mcp_corrections_tools.submit_correction
create_contact = _mcp_crm_tools.create_contact
get_deal = _mcp_crm_tools.get_deal
get_stale_leads = _mcp_crm_tools.get_stale_leads
list_deals = _mcp_crm_tools.list_deals
update_deal = _mcp_crm_tools.update_deal
enrich_contact_apollo = _mcp_crm_tools.enrich_contact_apollo
get_pipeline_summary = _mcp_crm_tools.get_pipeline_summary
search_crm = _mcp_crm_tools.search_crm
search_people_apollo = _mcp_crm_tools.search_people_apollo
sync_crm_apollo = _mcp_crm_tools.sync_crm_apollo
draft_email = _mcp_email_tools.draft_email
get_account_mail_journey = _mcp_email_tools.get_account_mail_journey
read_email_thread = _mcp_email_tools.read_email_thread
search_emails = _mcp_email_tools.search_emails
tinder_mode_left = _mcp_email_tools.tinder_mode_left
tinder_mode_right_draft = _mcp_email_tools.tinder_mode_right_draft
tinder_mode_send = _mcp_email_tools.tinder_mode_send
tinder_mode_start = _mcp_email_tools.tinder_mode_start
tinder_mode_status = _mcp_email_tools.tinder_mode_status
ingest_document = _mcp_ingest_tools.ingest_document
parse_document_ai = _mcp_ingest_tools.parse_document_ai
get_agent_list = _mcp_agents_tools.get_agent_list
ask_agent = _mcp_agents_tools.ask_agent
search_na_sales_knowledge = _mcp_na_sales_tools.search_na_sales_knowledge
list_na_outbound_proof_artifacts = _mcp_na_sales_tools.list_na_outbound_proof_artifacts
discover_na_link_candidates = _mcp_na_sales_tools.discover_na_link_candidates
check_google_calendar_availability = _mcp_scheduling_tools.check_google_calendar_availability
create_google_meet_invite = _mcp_scheduling_tools.create_google_meet_invite
schedule_google_meet_best_slot = _mcp_scheduling_tools.schedule_google_meet_best_slot
web_search = _mcp_web_maps_tools.web_search
searchapi_search = _mcp_web_maps_tools.searchapi_search
fetch_youtube_transcript = _mcp_web_maps_tools.fetch_youtube_transcript
scrape_url = _mcp_web_maps_tools.scrape_url
maps_geocode = _mcp_web_maps_tools.maps_geocode
maps_driving_route = _mcp_web_maps_tools.maps_driving_route
maps_places_text_search = _mcp_web_maps_tools.maps_places_text_search
maps_places_nearby_search = _mcp_web_maps_tools.maps_places_nearby_search
maps_place_details = _mcp_web_maps_tools.maps_place_details
search_newsdata = _mcp_web_maps_tools.search_newsdata
get_task_status = _mcp_task_tools.get_task_status
abort_task = _mcp_task_tools.abort_task
list_tasks = _mcp_task_tools.list_tasks
get_task_events = _mcp_task_tools.get_task_events
retry_task = _mcp_task_tools.retry_task
plan_task = _mcp_agent_loop_tools.plan_task
execute_phase = _mcp_agent_loop_tools.execute_phase
generate_report = _mcp_agent_loop_tools.generate_report
pause_standing_goal = _mcp_agent_loop_tools.pause_standing_goal
resume_standing_goal = _mcp_agent_loop_tools.resume_standing_goal
clear_standing_goal = _mcp_agent_loop_tools.clear_standing_goal
abort_agent_loop_plan = _mcp_agent_loop_tools.abort_agent_loop_plan
recall_memory = _mcp_memory_tools.recall_memory
math_mode_analyze_account = _mcp_math_mode_tools.math_mode_analyze_account
math_mode_rank_operator_queue = _mcp_math_mode_tools.math_mode_rank_operator_queue
math_mode_explain_score = _mcp_math_mode_tools.math_mode_explain_score
math_mode_learn_weights = _mcp_math_mode_tools.math_mode_learn_weights
math_mode_training_report = _mcp_math_mode_tools.math_mode_training_report
math_mode_ops_cycle = _mcp_math_mode_tools.math_mode_ops_cycle
math_mode_bootstrap_training_dataset = _mcp_math_mode_tools.math_mode_bootstrap_training_dataset
score_prospect_fit = _mcp_math_mode_tools.score_prospect_fit
store_memory = _mcp_memory_tools.store_memory
get_conversation_history = _mcp_memory_tools.get_conversation_history
check_relationship = _mcp_memory_tools.check_relationship
check_goals = _mcp_memory_tools.check_goals
list_procedures = _mcp_memory_tools.list_procedures
match_procedure = _mcp_memory_tools.match_procedure
read_memory_block = _mcp_memory_tools.read_memory_block
update_memory_block = _mcp_memory_tools.update_memory_block
trigger_dream_mode = _mcp_dream_tools.trigger_dream_mode
get_knowledge_gaps = _mcp_dream_tools.get_knowledge_gaps
search_cursor_sessions = _mcp_dream_tools.search_cursor_sessions
get_system_topology = _mcp_engineering_tools.get_system_topology
resolve_runtime_config = _mcp_engineering_tools.resolve_runtime_config
query_traces = _mcp_engineering_tools.query_traces
get_slo_status = _mcp_engineering_tools.get_slo_status
analyze_change_impact = _mcp_engineering_tools.analyze_change_impact
run_test_matrix = _mcp_engineering_tools.run_test_matrix
run_quality_gates = _mcp_engineering_tools.run_quality_gates
evaluate_release_guard = _mcp_engineering_tools.evaluate_release_guard
simulate_db_migration = _mcp_engineering_tools.simulate_db_migration
assemble_evidence_bundle = _mcp_engineering_tools.assemble_evidence_bundle
convene_board_meeting = _mcp_board_tools.convene_board_meeting
invoke_claude_code = _mcp_delegate_tools.invoke_claude_code
git_ship_status = _mcp_delegate_tools.git_ship_status
persuasion_sprint = _mcp_revenue_tools.persuasion_sprint
get_account_brief = _mcp_brief_tools.get_account_brief
prepare_formal_quote = _mcp_quotes_tools.prepare_formal_quote
query_brain = _mcp_query_tools.query_brain
discover_tools_for_query = _mcp_query_tools.discover_tools_for_query
search_knowledge = _mcp_query_tools.search_knowledge
quick_answer = _mcp_query_tools.quick_answer
run_price_waterfall = _mcp_financial_engineering_tools.run_price_waterfall
working_capital_impact = _mcp_financial_engineering_tools.working_capital_impact
optimize_payment_terms = _mcp_financial_engineering_tools.optimize_payment_terms
evaluate_quote_npv_irr = _mcp_financial_engineering_tools.evaluate_quote_npv_irr
scenario_engine = _mcp_financial_engineering_tools.scenario_engine
customer_credit_risk_score = _mcp_financial_engineering_tools.customer_credit_risk_score
get_system_status = _mcp_system_tools.get_system_status
load_brand_design = _mcp_brand_tools.load_brand_design
list_deck_briefs = _mcp_brand_tools.list_deck_briefs
load_deck_brief = _mcp_brand_tools.load_deck_brief
draft_deck_brief_from_company = _mcp_brand_tools.draft_deck_brief_from_company
creative_brief_distill = _mcp_creative_tools.creative_brief_distill
concept_diverge = _mcp_creative_tools.concept_diverge
evidence_creative_map = _mcp_creative_tools.evidence_creative_map
hook_generate = _mcp_creative_tools.hook_generate
creative_critic_panel = _mcp_creative_tools.creative_critic_panel
idea_evolve = _mcp_creative_tools.idea_evolve
voice_palette_mixer = _mcp_creative_tools.voice_palette_mixer
objection_drama_simulator = _mcp_creative_tools.objection_drama_simulator
campaign_theme_architect = _mcp_creative_tools.campaign_theme_architect
post_slack_message = _mcp_slack_tools.post_slack_message
