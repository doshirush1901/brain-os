"""MCP: single-account operator brief (read-only; never sends email)."""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from brain_os.interfaces.mcp_tool_hardening import hardened_mcp_tool
from brain_os.schemas.account_brief import format_account_brief_plain_text
from brain_os.services.account_brief import build_account_brief

logger = logging.getLogger(__name__)


def _mcp_shared_services(srv: Any) -> dict[str, Any]:
    from brain_os.service_keys import ServiceKey as SK

    shared: dict[str, Any] = {}
    if srv._crm is not None:
        shared[SK.CRM] = srv._crm
    if srv._retriever is not None:
        shared[SK.RETRIEVER] = srv._retriever
    return shared


async def get_account_brief(
    company: str,
    contact_email: str = "",
    deep: bool = False,
    as_json: bool = False,
    no_llm: bool = False,
    skip_kb: bool = False,
    skip_mail: bool = False,
    run_id: str = "",
) -> str:
    """Operator account brief: CRM, optional Gmail/KB, proof links, optional Argus (--deep).

    Plain text by default (Gmail-safe). Never sends email.
    """
    from brain_os.interfaces import mcp_server as srv

    await srv._ensure_initialized()

    email_processor = None
    if not skip_mail and srv._pantheon is not None:
        try:
            from brain_os.interfaces.cli_runtime import _build_digestive, _build_email_processor

            digestive, _ing, _qd = _build_digestive()
            email_processor = _build_email_processor(
                srv._pantheon, digestive, _mcp_shared_services(srv) or None
            )
        except Exception as exc:
            logger.warning("MCP brief: email processor unavailable: %s", exc)

    try:
        brief = await build_account_brief(
            company,
            contact_email=contact_email.strip() or None,
            deep=deep,
            use_llm=not no_llm,
            skip_kb=skip_kb,
            skip_mail=skip_mail,
            pantheon=srv._pantheon,
            shared_services=_mcp_shared_services(srv),
            email_processor=email_processor,
            pipeline_run_id=(run_id or "").strip() or None,
        )
    except TimeoutError:
        return "Account brief timed out (90s). Try skip_kb=true and skip_mail=true."
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        logger.exception("MCP get_account_brief failed")
        return f"Error: {exc}"

    if as_json:
        return json.dumps(brief.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str)
    return format_account_brief_plain_text(brief)


def register(mcp: FastMCP) -> None:
    mcp.tool()(hardened_mcp_tool(get_account_brief))
