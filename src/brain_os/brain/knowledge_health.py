"""Proactive knowledge monitoring with domain-specific validation.

Runs health checks against the Qdrant knowledge base and
``data/machine_knowledge.json`` to catch stale documents, hallucinated
claims, and business-rule violations before they reach users.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain_os.exceptions import DatabaseError

logger = logging.getLogger(__name__)

_MACHINE_KNOWLEDGE_PATH = Path("data/machine_knowledge.json")
_HEALTH_ISSUES_PATH = Path("data/brain/health_issues.json")

_CRITICAL_DOCUMENTS = [
    "price list",
    "product catalogue",
    "machine specifications",
]

_HALLUCINATION_PHRASES = [
    "world's leading",
    "100% guarantee",
    "best in class",
    "industry leading",
    "unmatched quality",
    "number one",
]

_AM_MAX_THICKNESS_MM = 1.5
_STANDARD_LEAD_TIME_WEEKS = (12, 16)
_PRICE_TOLERANCE = 0.30


class KnowledgeHealthMonitor:
    """Validates knowledge-base integrity and flags domain-specific issues."""

    def __init__(
        self,
        qdrant_manager: Any,
        knowledge_graph: Any | None = None,
        machine_knowledge_path: str = "data/machine_knowledge.json",
    ) -> None:
        self._qdrant = qdrant_manager
        self._graph = knowledge_graph
        self._mk_path = Path(machine_knowledge_path)
        self._machine_knowledge: dict[str, Any] | None = None
        self._recurring_issues: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._recurring_issues_loaded: bool = False

    async def _ensure_recurring_issues_loaded(self) -> None:
        """Load persisted recurring issues without blocking the event loop."""
        if self._recurring_issues_loaded:
            return
        self._recurring_issues_loaded = True
        if not _HEALTH_ISSUES_PATH.exists():
            return

        def _read_issues() -> dict[str, Any]:
            raw = _HEALTH_ISSUES_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}

        try:
            data = await asyncio.to_thread(_read_issues)
            for k, v in data.items():
                if isinstance(v, list):
                    self._recurring_issues[k] = v
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not load health issues; starting fresh")

    def _load_recurring_issues(self) -> None:
        """Sync preload for tests or non-async callers; avoids disk I/O in ``__init__``."""
        if self._recurring_issues_loaded:
            return
        self._recurring_issues_loaded = True
        if not _HEALTH_ISSUES_PATH.exists():
            return
        try:
            data = json.loads(_HEALTH_ISSUES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        self._recurring_issues[k] = v
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not load health issues; starting fresh")

    def _persist_issues(self) -> None:
        try:
            _HEALTH_ISSUES_PATH.parent.mkdir(parents=True, exist_ok=True)
            _HEALTH_ISSUES_PATH.write_text(
                json.dumps(dict(self._recurring_issues), indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Failed to persist health issues", exc_info=True)

    async def _load_machine_knowledge(self) -> dict[str, Any]:
        if self._mk_path.exists():
            try:
                raw = await asyncio.to_thread(
                    self._mk_path.read_text,
                    encoding="utf-8",
                )
                return json.loads(raw)
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not load machine knowledge from %s", self._mk_path)
        return {}

    async def _get_machine_knowledge(self) -> dict[str, Any]:
        if self._machine_knowledge is None:
            self._machine_knowledge = await self._load_machine_knowledge()
        return self._machine_knowledge

    # ── public API ────────────────────────────────────────────────────────

    async def run_health_check(self) -> dict[str, Any]:
        """Run all checks and return a consolidated report."""
        await self._ensure_recurring_issues_loaded()

        report: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": {},
        }

        missing = await self.check_critical_documents()
        report["checks"]["critical_documents"] = {
            "missing": missing,
            "status": "ok" if not missing else "warning",
        }

        chronic = self.get_chronic_issues()
        report["checks"]["chronic_issues"] = {
            "count": len(chronic),
            "issues": chronic,
            "status": "ok" if not chronic else "warning",
        }

        report["overall"] = (
            "healthy" if all(c["status"] == "ok" for c in report["checks"].values()) else "degraded"
        )
        logger.info("Health check complete: %s", report["overall"])
        return report

    async def check_critical_documents(self) -> list[str]:
        """Verify that key documents exist in Qdrant; return names of missing ones."""
        missing: list[str] = []
        for doc_name in _CRITICAL_DOCUMENTS:
            try:
                results = await self._qdrant.search(doc_name, limit=3)
                has_match = any(r.get("score", 0) >= 0.5 for r in results)
                if not has_match:
                    missing.append(doc_name)
            except DatabaseError:
                logger.exception("Failed to check for '%s' in Qdrant", doc_name)
                missing.append(doc_name)
        if missing:
            logger.warning("Missing critical documents: %s", missing)
        return missing

    def validate_business_rules(self, response: str) -> list[str]:
        """Check *response* against Acme Corp business rules."""
        violations: list[str] = []
        text_lower = response.lower()

        thickness_pattern = re.compile(r"am[-\s]?series.*?(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)
        for m in thickness_pattern.finditer(response):
            value = float(m.group(1))
            if value > _AM_MAX_THICKNESS_MM:
                violations.append(
                    f"AM thickness {value}mm exceeds architectural limit of {_AM_MAX_THICKNESS_MM}mm"
                )

        if "heavy gauge" in text_lower and "pf1" not in text_lower:
            violations.append("Heavy gauge application mentioned without recommending PF1")

        lead_time_pattern = re.compile(r"(\d+)[-–](\d+)\s*weeks?", re.IGNORECASE)
        for m in lead_time_pattern.finditer(response):
            low, high = int(m.group(1)), int(m.group(2))
            std_low, std_high = _STANDARD_LEAD_TIME_WEEKS
            if low < std_low - 4 or high > std_high + 8:
                violations.append(f"Lead time {low}-{high} weeks outside plausible range")

        if violations:
            logger.warning("Business rule violations: %s", violations)
        return violations

    async def detect_hallucinations(self, response: str) -> list[str]:
        """Flag marketing superlatives, fabricated models, and unrealistic prices."""
        flags: list[str] = []
        text_lower = response.lower()

        for phrase in _HALLUCINATION_PHRASES:
            if phrase in text_lower:
                flags.append(f"Superlative claim: '{phrase}'")

        mk = await self._get_machine_knowledge()
        catalog = mk.get("machine_catalog", {})
        model_pattern = re.compile(r"\b([A-Z]{2,4}[-\s]?\d{1,4}[A-Z]?(?:[-\s][A-Z0-9]+)*)\b")
        for m in model_pattern.finditer(response):
            model_str = m.group(1).replace(" ", "-")
            normalised = model_str.split("-")[0] + (
                "-" + "-".join(model_str.split("-")[1:]) if "-" in model_str else ""
            )
            if not any(normalised.upper().startswith(known.upper()) for known in catalog):
                flags.append(f"Unrecognised model reference: '{m.group(1)}'")

        price_pattern = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)")
        for m in price_pattern.finditer(response):
            value = float(m.group(1).replace(",", ""))
            if value > 5_000_000:
                flags.append(f"Unrealistic price: ${value:,.2f}")

        if flags:
            logger.warning("Hallucination flags: %s", flags)
        return flags

    async def verify_price(self, model: str, stated_price: float) -> dict[str, Any]:
        """Check *stated_price* against machine_knowledge.json within tolerance."""
        mk = await self._get_machine_knowledge()
        catalog = mk.get("machine_catalog", {})
        hints = mk.get("truth_hints", {})

        result: dict[str, Any] = {
            "model": model,
            "stated_price": stated_price,
            "verified": False,
            "reason": "no reference price found",
        }

        ref_key = next(
            (k for k in hints if model.lower() in k.lower() and "price" in k.lower()),
            None,
        )
        if ref_key:
            try:
                ref_str = re.search(r"[\d,]+(?:\.\d+)?", hints[ref_key])
                if ref_str:
                    ref_price = float(ref_str.group().replace(",", ""))
                    deviation = abs(stated_price - ref_price) / ref_price
                    result["reference_price"] = ref_price
                    result["deviation"] = round(deviation, 4)
                    if deviation <= _PRICE_TOLERANCE:
                        result["verified"] = True
                        result["reason"] = "within tolerance"
                    else:
                        result["reason"] = (
                            f"deviation {deviation:.1%} exceeds {_PRICE_TOLERANCE:.0%} tolerance"
                        )
            except (ValueError, ZeroDivisionError):
                pass

        if model not in catalog:
            result.setdefault("warning", "model not found in catalog")

        return result

    def track_recurring_issue(self, issue_type: str, details: str) -> int:
        """Record an issue occurrence and return total count for this type."""
        if not self._recurring_issues_loaded:
            self._load_recurring_issues()

        self._recurring_issues[issue_type].append(
            {
                "details": details,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        count = len(self._recurring_issues[issue_type])
        logger.debug("Recurring issue '%s' count: %d", issue_type, count)
        self._persist_issues()
        return count

    def get_chronic_issues(self, threshold: int = 3) -> list[dict[str, Any]]:
        """Return issues that have occurred more than *threshold* times."""
        if not self._recurring_issues_loaded:
            self._load_recurring_issues()

        chronic: list[dict[str, Any]] = []
        for issue_type, occurrences in self._recurring_issues.items():
            if len(occurrences) > threshold:
                chronic.append(
                    {
                        "type": issue_type,
                        "count": len(occurrences),
                        "latest": occurrences[-1],
                    }
                )
        return chronic

    async def send_alert(self, message: str) -> None:
        """Log a knowledge-health alert."""
        logger.warning("Knowledge health alert: %s", message)
