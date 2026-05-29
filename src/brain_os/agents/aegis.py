"""Aegis — DLP / Content Safety agent.

Scans outbound content (drafts, final answers) for PII, confidential
business terms, and unverified commitments.  Runs in the pipeline
between agent execution and final answer shaping.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.prompt_loader import load_prompt
from brain_os.service_keys import ServiceKey as SK

logger = logging.getLogger(__name__)

_DLP_SERVICE_ERRORS = (
    httpx.HTTPError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    ImportError,
)

_SYSTEM_PROMPT = load_prompt("aegis_system")

_CONFIDENTIAL_PATTERNS = re.compile(
    r"(margin[s]?\s*[:=]?\s*\d|"
    r"internal\s+(cost|price|rate)|"
    r"vendor\s+(cost|price|rate|margin)|"
    r"discount\s*[:=]?\s*\d|"
    r"salary|compensation|CTC|take.?home|"
    r"HR\s+data|headcount\s+\d|"
    r"confidential|proprietary|NDA.?protected)",
    re.IGNORECASE,
)


class Aegis(BaseAgent):
    name = "aegis"
    role = "Content Safety / DLP"
    description = "Scans outbound content for PII, confidential terms, and data leakage risks"
    knowledge_categories = ["company_internal", "contracts_and_legal"]
    timeout = 30

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="dlp_inspect",
                description="Detect PII in text using Google Cloud DLP. Returns findings with types and locations.",
                parameters={"text": "The text to inspect for PII"},
                handler=self._tool_dlp_inspect,
            )
        )

        self.register_tool(
            AgentTool(
                name="dlp_redact",
                description="Redact PII from text by replacing with masking characters.",
                parameters={"text": "The text to redact PII from"},
                handler=self._tool_dlp_redact,
            )
        )

        self.register_tool(
            AgentTool(
                name="check_confidential_terms",
                description=(
                    "Pattern-based check for confidential business terms: "
                    "margins, vendor pricing, HR data, internal costs."
                ),
                parameters={"text": "The text to check for confidential terms"},
                handler=self._tool_check_confidential,
            )
        )

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)

    async def check_content(self, text: str) -> dict[str, Any]:
        """Quick content check without the full ReAct loop. Returns verdict + findings."""
        findings: list[str] = []

        confidential_matches = _CONFIDENTIAL_PATTERNS.findall(text)
        if confidential_matches:
            findings.append(f"Confidential terms detected: {len(confidential_matches)} matches")

        dlp_svc = self._services.get(SK.DLP) if self._services else None
        if dlp_svc is not None and dlp_svc.available:
            try:
                report = await dlp_svc.inspect_and_report(text)
                if report.get("has_pii"):
                    findings.append(
                        f"PII detected: {report['total_findings']} findings "
                        f"({', '.join(f'{k}:{v}' for k, v in report['findings_by_type'].items())})"
                    )
            except _DLP_SERVICE_ERRORS:
                logger.debug("DLP inspect failed in Aegis.check_content", exc_info=True)

        if not findings:
            verdict = "SAFE"
        elif any("Confidential" in f for f in findings):
            verdict = "BLOCK"
        else:
            verdict = "REVIEW_NEEDED"

        return {"verdict": verdict, "findings": findings}

    async def _tool_dlp_inspect(self, text: str) -> str:
        dlp_svc = self._services.get(SK.DLP)
        if dlp_svc is None or not dlp_svc.available:
            return "DLP service not available. Cannot inspect for PII."
        try:
            report = await dlp_svc.inspect_and_report(text)
            if not report.get("has_pii"):
                return "No PII detected."
            lines = [f"PII detected: {report['total_findings']} findings"]
            for info_type, count in report["findings_by_type"].items():
                lines.append(f"  - {info_type}: {count}")
            return "\n".join(lines)
        except _DLP_SERVICE_ERRORS as exc:
            return f"DLP inspect error: {exc}"

    async def _tool_dlp_redact(self, text: str) -> str:
        dlp_svc = self._services.get(SK.DLP)
        if dlp_svc is None or not dlp_svc.available:
            return "DLP service not available. Cannot redact."
        try:
            return await dlp_svc.deidentify_text(text)
        except _DLP_SERVICE_ERRORS as exc:
            return f"DLP redact error: {exc}"

    async def _tool_check_confidential(self, text: str) -> str:
        matches = _CONFIDENTIAL_PATTERNS.findall(text)
        if not matches:
            return "No confidential business terms detected."
        return f"Confidential terms detected ({len(matches)} matches): {', '.join(set(str(m) for m in matches[:10]))}"
