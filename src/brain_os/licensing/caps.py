"""Community tier caps (document ingest count, MCP policy helpers)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from brain_os.licensing.tiers import Tier, effective_tier

logger = logging.getLogger(__name__)

COMMUNITY_MAX_DOCUMENTS = 500

# Tier A MCP tools (Community). Pro registers additional modules/tools.
COMMUNITY_MCP_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "query_brain",
        "discover_tools_for_query",
        "search_knowledge",
        "quick_answer",
        "ask_agent",
        "get_agent_list",
        "get_account_brief",
        "search_crm",
        "get_deal",
        "list_deals",
        "get_pipeline_summary",
        "find_related_entities",
        "find_company_contacts",
        "ingest_document",
        "parse_document_ai",
        "recall_memory",
        "store_memory",
        "get_conversation_history",
        "check_relationship",
        "check_goals",
        "submit_correction",
        "submit_praise",
        "search_emails",
        "read_email_thread",
        "draft_email",
        "get_system_status",
    }
)

PRO_MCP_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "operator_inbox_summary",
        "operator_inbox_decide",
        "operator_release",
        "operator_dashboard_today",
        "operator_activity_today",
        "find_company_quotes",
        "find_similar_companies_mcp",
        "get_company_context_graph",
        "get_company_icp_profile",
        "create_contact",
        "update_deal",
        "get_stale_leads",
        "send_email",
        "email_touch_audit",
        "get_account_mail_journey",
    }
)


class LicenseCapError(Exception):
    """Raised when a Community cap blocks an action."""


def community_document_limit() -> int:
    return COMMUNITY_MAX_DOCUMENTS


def _usage_path() -> Path:
    try:
        from brain_os.systems.data_dir_lock import get_data_dir

        base = get_data_dir()
    except (ImportError, OSError, RuntimeError):
        base = Path("data")
    return Path(base) / "brain" / "license_usage.json"


def get_indexed_document_count() -> int:
    path = _usage_path()
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("document_count", 0))
    except (OSError, ValueError, TypeError):
        return 0


def record_document_ingested(*, increment: int = 1) -> int:
    path = _usage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    count = get_indexed_document_count() + max(1, increment)
    path.write_text(json.dumps({"document_count": count}, indent=2) + "\n", encoding="utf-8")
    return count


def assert_ingest_allowed(*, increment: int = 1) -> None:
    """Raise LicenseCapError if Community document cap would be exceeded."""
    if effective_tier() in {Tier.PRO, Tier.TRIAL}:
        return
    current = get_indexed_document_count()
    if current + increment > COMMUNITY_MAX_DOCUMENTS:
        raise LicenseCapError(
            f"Community tier limit: {COMMUNITY_MAX_DOCUMENTS} indexed documents "
            f"({current} used). Upgrade: brain activate --trial or brain activate --key <key>"
        )


def mcp_tool_allowed(tool_name: str) -> bool:
    name = tool_name.strip()
    if name in COMMUNITY_MCP_TOOL_NAMES:
        return True
    if effective_tier() in {Tier.PRO, Tier.TRIAL} and name in PRO_MCP_TOOL_NAMES:
        return True
    return False


def mcp_upgrade_hint(tool_name: str) -> str:
    return (
        f"MCP tool `{tool_name}` requires Brain OS Pro or an active trial. "
        "Run: brain activate --trial  (14 days)  or  brain activate --key bos_live_..."
    )
