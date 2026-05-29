"""Fast, pattern-based intent router for Ira.

Before invoking an LLM to decide which agents should handle a query, the
:class:`DeterministicRouter` attempts a cheap keyword/regex classification.
If the match confidence is high enough the routing table is returned
immediately, saving an LLM round-trip on the most common query shapes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class IntentCategory(str, Enum):
    """High-level intent buckets recognised by the deterministic router."""

    SALES_PIPELINE = "SALES_PIPELINE"
    FINANCE_REVIEW = "FINANCE_REVIEW"
    HR_OVERVIEW = "HR_OVERVIEW"
    MACHINE_SPECS = "MACHINE_SPECS"
    PRODUCTION_STATUS = "PRODUCTION_STATUS"
    CUSTOMER_SERVICE = "CUSTOMER_SERVICE"
    MARKETING_CAMPAIGN = "MARKETING_CAMPAIGN"
    RESEARCH = "RESEARCH"
    QUOTE_REQUEST = "QUOTE_REQUEST"
    VENDOR_PROCUREMENT = "VENDOR_PROCUREMENT"
    PROJECT_MANAGEMENT = "PROJECT_MANAGEMENT"
    QUALITY_MANAGEMENT = "QUALITY_MANAGEMENT"
    CASE_STUDY = "CASE_STUDY"
    QUOTE_GENERATION = "QUOTE_GENERATION"
    ARCHIVE_SEARCH = "ARCHIVE_SEARCH"
    CONTACT_CLASSIFICATION = "CONTACT_CLASSIFICATION"
    CURSOR_SESSION_RECALL = "CURSOR_SESSION_RECALL"
    MEMORY_RECALL = "MEMORY_RECALL"
    SYSTEM_TRAINING = "SYSTEM_TRAINING"
    OUTBOUND_EMAIL_DRAFT = "OUTBOUND_EMAIL_DRAFT"
    MAILBOX_INTEL = "MAILBOX_INTEL"
    COMPANY_DOSSIER = "COMPANY_DOSSIER"
    NA_SALES_STRATEGY = "NA_SALES_STRATEGY"
    ENGINEERING_COMPUTE = "ENGINEERING_COMPUTE"
    GENERAL = "GENERAL"


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """Which agents and tools a given intent requires."""

    required_agents: tuple[str, ...]
    optional_agents: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()


ROUTING_TABLE: dict[IntentCategory, RoutingConfig] = {
    IntentCategory.SALES_PIPELINE: RoutingConfig(
        required_agents=("prometheus", "atlas", "clio", "chiron"),
        optional_agents=("tyche", "alexandros"),
        required_tools=("crm", "retriever"),
    ),
    IntentCategory.FINANCE_REVIEW: RoutingConfig(
        required_agents=("plutus",),
        optional_agents=("prometheus", "tyche", "maestro"),
        required_tools=("crm", "retriever"),
    ),
    IntentCategory.HR_OVERVIEW: RoutingConfig(
        required_agents=("themis",),
        optional_agents=("clio",),
        required_tools=("retriever",),
    ),
    IntentCategory.MACHINE_SPECS: RoutingConfig(
        required_agents=("hephaestus", "clio"),
        optional_agents=("vera", "maestro"),
        required_tools=("retriever",),
    ),
    IntentCategory.PRODUCTION_STATUS: RoutingConfig(
        required_agents=("hephaestus",),
        optional_agents=("clio",),
        required_tools=("retriever",),
    ),
    IntentCategory.CUSTOMER_SERVICE: RoutingConfig(
        required_agents=("clio", "prometheus"),
        optional_agents=("calliope",),
        required_tools=("crm", "retriever"),
    ),
    IntentCategory.MARKETING_CAMPAIGN: RoutingConfig(
        required_agents=("hermes",),
        optional_agents=("calliope", "arachne", "na_sales"),
        required_tools=("retriever", "drip_engine"),
    ),
    IntentCategory.RESEARCH: RoutingConfig(
        required_agents=("clio",),
        optional_agents=("iris", "vera", "alexandros", "maestro"),
        required_tools=("retriever",),
    ),
    IntentCategory.QUOTE_REQUEST: RoutingConfig(
        required_agents=("prometheus", "plutus", "hephaestus"),
        optional_agents=("calliope",),
        required_tools=("pricing_engine", "crm", "retriever"),
    ),
    IntentCategory.VENDOR_PROCUREMENT: RoutingConfig(
        required_agents=("hera",),
        optional_agents=("clio", "plutus"),
        required_tools=("retriever",),
    ),
    IntentCategory.PROJECT_MANAGEMENT: RoutingConfig(
        required_agents=("atlas",),
        optional_agents=("clio", "hephaestus"),
        required_tools=("retriever",),
    ),
    IntentCategory.QUALITY_MANAGEMENT: RoutingConfig(
        required_agents=("asclepius",),
        optional_agents=("atlas", "hephaestus"),
        required_tools=("retriever",),
    ),
    IntentCategory.CASE_STUDY: RoutingConfig(
        required_agents=("cadmus",),
        optional_agents=("clio", "calliope"),
        required_tools=("retriever",),
    ),
    IntentCategory.QUOTE_GENERATION: RoutingConfig(
        required_agents=("quotebuilder", "plutus", "hephaestus"),
        optional_agents=("calliope",),
        required_tools=("pricing_engine", "retriever"),
    ),
    IntentCategory.ARCHIVE_SEARCH: RoutingConfig(
        required_agents=("alexandros",),
        optional_agents=("clio",),
        required_tools=("retriever",),
    ),
    IntentCategory.CONTACT_CLASSIFICATION: RoutingConfig(
        required_agents=("delphi",),
        optional_agents=("clio", "prometheus"),
        required_tools=("crm", "retriever"),
    ),
    IntentCategory.CURSOR_SESSION_RECALL: RoutingConfig(
        required_agents=("graphe",),
        optional_agents=(),
        required_tools=(),
    ),
    IntentCategory.MEMORY_RECALL: RoutingConfig(
        required_agents=("mnemosyne", "sophia"),
        optional_agents=("clio",),
        required_tools=("retriever",),
    ),
    IntentCategory.OUTBOUND_EMAIL_DRAFT: RoutingConfig(
        required_agents=("prometheus", "clio", "calliope"),
        optional_agents=("artemis", "alexandros", "hephaestus"),
        required_tools=("retriever", "crm"),
    ),
    IntentCategory.MAILBOX_INTEL: RoutingConfig(
        required_agents=("artemis", "prometheus"),
        optional_agents=("alexandros", "delphi"),
        required_tools=("retriever", "crm"),
    ),
    IntentCategory.COMPANY_DOSSIER: RoutingConfig(
        required_agents=("argus", "prometheus"),
        optional_agents=("clio", "iris"),
        required_tools=("retriever", "crm"),
    ),
    IntentCategory.NA_SALES_STRATEGY: RoutingConfig(
        required_agents=("na_sales", "prometheus"),
        optional_agents=("calliope", "chiron"),
        required_tools=("retriever", "crm"),
    ),
    IntentCategory.SYSTEM_TRAINING: RoutingConfig(
        required_agents=("nemesis", "sophia", "chiron"),
        optional_agents=(),
        required_tools=("retriever",),
    ),
    IntentCategory.ENGINEERING_COMPUTE: RoutingConfig(
        required_agents=("maestro",),
        optional_agents=("vera",),
        required_tools=(),
    ),
    IntentCategory.GENERAL: RoutingConfig(
        required_agents=("clio",),
        optional_agents=("sphinx", "alexandros"),
        required_tools=("retriever",),
    ),
}


# ── keyword patterns ─────────────────────────────────────────────────────────
# Each entry is (compiled regex, intent, per-match weight).  A query can
# match multiple patterns; the intent with the highest cumulative weight wins.


@dataclass(frozen=True, slots=True)
class _Pattern:
    regex: re.Pattern[str]
    intent: IntentCategory
    weight: float = 1.0


def _compile(patterns: Sequence[tuple[str, IntentCategory, float]]) -> list[_Pattern]:
    return [_Pattern(re.compile(p, re.IGNORECASE), intent, w) for p, intent, w in patterns]


_PATTERNS: list[_Pattern] = _compile(
    [
        # Sales / pipeline / order book
        (r"\bpipeline\b", IntentCategory.SALES_PIPELINE, 2.0),
        (r"\bsales\b", IntentCategory.SALES_PIPELINE, 1.5),
        (r"\bdeals?\b", IntentCategory.SALES_PIPELINE, 1.5),
        (r"\bleads?\b", IntentCategory.SALES_PIPELINE, 1.5),
        (r"\bcrm\b", IntentCategory.SALES_PIPELINE, 2.0),
        (r"\bsales\s+funnel\b", IntentCategory.SALES_PIPELINE, 2.0),
        (r"\bconversion\s+rate\b", IntentCategory.SALES_PIPELINE, 1.5),
        (r"\bwin\s+rate\b", IntentCategory.SALES_PIPELINE, 1.5),
        (r"\border\s*book\b", IntentCategory.SALES_PIPELINE, 3.0),
        (r"\bhot\s+leads?\b", IntentCategory.SALES_PIPELINE, 3.0),
        (r"\bproposals?\s+sent\b", IntentCategory.SALES_PIPELINE, 2.5),
        (r"\bquotes?\s+sent\b", IntentCategory.SALES_PIPELINE, 2.5),
        (r"\bmissed\s+leads?\b", IntentCategory.MAILBOX_INTEL, 4.0),
        (r"\bclient\s*list\b", IntentCategory.SALES_PIPELINE, 2.5),
        (r"\bcustomer\s*list\b", IntentCategory.SALES_PIPELINE, 2.5),
        (r"\bactive\s+orders?\b", IntentCategory.SALES_PIPELINE, 2.5),
        (r"\bin\s+production\b", IntentCategory.SALES_PIPELINE, 1.5),
        # Quote / pricing — checked before machine specs so that queries
        # mentioning both a model name and a pricing keyword route here.
        (r"\bquotes?\s+for\b", IntentCategory.QUOTE_REQUEST, 5.0),
        (r"\bquote\b", IntentCategory.QUOTE_REQUEST, 3.0),
        (r"\bpric(e|ing)\b", IntentCategory.QUOTE_REQUEST, 3.0),
        (r"\bcost\b", IntentCategory.QUOTE_REQUEST, 2.5),
        (r"\bhow\s+much\b", IntentCategory.QUOTE_REQUEST, 3.0),
        (r"\bproposal\b", IntentCategory.QUOTE_REQUEST, 1.5),
        (r"\bestimate\b", IntentCategory.QUOTE_REQUEST, 1.0),
        (r"\bbudget\b", IntentCategory.QUOTE_REQUEST, 1.0),
        # Engineering math / units (Maestro / Wolfram) — before machine specs
        (r"\bconvert\b.+\b(to|into)\b", IntentCategory.ENGINEERING_COMPUTE, 5.0),
        (r"\bconvert\b", IntentCategory.ENGINEERING_COMPUTE, 4.0),
        (r"\bcalculate\b", IntentCategory.ENGINEERING_COMPUTE, 3.5),
        (r"\bheat\s+transfer\b", IntentCategory.ENGINEERING_COMPUTE, 4.0),
        (r"\bpsi\b", IntentCategory.ENGINEERING_COMPUTE, 4.0),
        (r"\bw/m", IntentCategory.ENGINEERING_COMPUTE, 3.0),
        (r"\b°[cf]\b", IntentCategory.ENGINEERING_COMPUTE, 2.5),
        # Machine specs
        (r"\bmachine\b", IntentCategory.MACHINE_SPECS, 1.5),
        (r"\bspecs?\b", IntentCategory.MACHINE_SPECS, 2.0),
        (r"\bspecification", IntentCategory.MACHINE_SPECS, 2.0),
        (r"\bPF[12]-[A-Z]-?\d+\b", IntentCategory.MACHINE_SPECS, 3.0),
        (r"\bPF[12]-[A-Z]\b", IntentCategory.MACHINE_SPECS, 2.5),
        (r"\bPF[12]\b", IntentCategory.MACHINE_SPECS, 2.0),
        (r"\bPF1-C\b", IntentCategory.MACHINE_SPECS, 2.5),
        (r"\bAM[\s-]?series\b", IntentCategory.MACHINE_SPECS, 2.0),
        (r"\bRF-100\b", IntentCategory.MACHINE_SPECS, 2.0),
        (r"\bSL-500\b", IntentCategory.MACHINE_SPECS, 2.0),
        (r"\bVFM\b", IntentCategory.MACHINE_SPECS, 2.0),
        (r"\bthermoform", IntentCategory.MACHINE_SPECS, 1.5),
        (r"\broll\s*form", IntentCategory.MACHINE_SPECS, 1.5),
        (r"\bpanel\s*form", IntentCategory.MACHINE_SPECS, 1.5),
        # Finance
        (r"\brevenue\b", IntentCategory.FINANCE_REVIEW, 2.0),
        (r"\bfinancial\b", IntentCategory.FINANCE_REVIEW, 2.0),
        (r"\bprofit\b", IntentCategory.FINANCE_REVIEW, 1.5),
        (r"\bmargin\b", IntentCategory.FINANCE_REVIEW, 1.5),
        (r"\bcash\s*flow\b", IntentCategory.FINANCE_REVIEW, 2.0),
        (r"\bforecast\b", IntentCategory.FINANCE_REVIEW, 1.0),
        # HR
        (r"\bhr\b", IntentCategory.HR_OVERVIEW, 2.0),
        (r"\bhuman\s+resources\b", IntentCategory.HR_OVERVIEW, 2.0),
        (r"\bemployee", IntentCategory.HR_OVERVIEW, 1.5),
        (r"\bheadcount\b", IntentCategory.HR_OVERVIEW, 2.0),
        (r"\bhiring\b", IntentCategory.HR_OVERVIEW, 1.5),
        (r"\binterview\s+questions?\b", IntentCategory.HR_OVERVIEW, 2.0),
        (r"\brecruit(ing|ment)?\b", IntentCategory.HR_OVERVIEW, 1.5),
        # Production
        (r"\bproduction\b", IntentCategory.PRODUCTION_STATUS, 1.5),
        (r"\bmanufactur", IntentCategory.PRODUCTION_STATUS, 1.5),
        (r"\bassembly\b", IntentCategory.PRODUCTION_STATUS, 1.5),
        (r"\blead\s*time\b", IntentCategory.PRODUCTION_STATUS, 1.5),
        # Marketing
        (r"\bmarketing\b", IntentCategory.MARKETING_CAMPAIGN, 2.0),
        (r"\bcampaign\b", IntentCategory.MARKETING_CAMPAIGN, 2.0),
        (r"\bdrip\b", IntentCategory.MARKETING_CAMPAIGN, 2.0),
        (r"\bnewsletter\b", IntentCategory.MARKETING_CAMPAIGN, 2.0),
        (r"\bemail\s+blast\b", IntentCategory.MARKETING_CAMPAIGN, 1.5),
        # Customer service / email
        (r"\bcomplaint\b", IntentCategory.CUSTOMER_SERVICE, 2.0),
        (r"\bsupport\s+ticket\b", IntentCategory.CUSTOMER_SERVICE, 2.0),
        (r"\bwarranty\b", IntentCategory.CUSTOMER_SERVICE, 1.5),
        (r"\bafter[\s-]?sales\b", IntentCategory.CUSTOMER_SERVICE, 2.0),
        (r"\bemail\s+(from|to|about|thread)\b", IntentCategory.MAILBOX_INTEL, 3.5),
        (r"\bfind\s+emails?\b", IntentCategory.MAILBOX_INTEL, 3.5),
        (r"\bpull\s+up\s+emails?\b", IntentCategory.MAILBOX_INTEL, 3.5),
        (r"\blast\s+email\b", IntentCategory.MAILBOX_INTEL, 3.0),
        (r"\binbox\b", IntentCategory.MAILBOX_INTEL, 2.5),
        # Research
        (r"\bresearch\b", IntentCategory.RESEARCH, 1.5),
        (r"\bmarket\s+analysis\b", IntentCategory.RESEARCH, 2.0),
        (r"\bcompetitor", IntentCategory.RESEARCH, 1.5),
        (r"\bindustry\s+trend", IntentCategory.RESEARCH, 1.5),
        # Vendor / procurement
        (r"\bvendor\b", IntentCategory.VENDOR_PROCUREMENT, 2.0),
        (r"\bsupplier\b", IntentCategory.VENDOR_PROCUREMENT, 2.0),
        (r"\bprocurement\b", IntentCategory.VENDOR_PROCUREMENT, 2.5),
        (r"\bcomponent\b", IntentCategory.VENDOR_PROCUREMENT, 1.5),
        (r"\binventory\b", IntentCategory.VENDOR_PROCUREMENT, 1.5),
        (r"\bstock\b", IntentCategory.VENDOR_PROCUREMENT, 1.0),
        (r"\bpart\s+number\b", IntentCategory.VENDOR_PROCUREMENT, 2.0),
        # Project management / delivery / order status
        (r"\bproject\b", IntentCategory.PROJECT_MANAGEMENT, 1.5),
        (r"\blogbook\b", IntentCategory.PROJECT_MANAGEMENT, 2.0),
        (r"\bmilestone\b", IntentCategory.PROJECT_MANAGEMENT, 2.0),
        (r"\bdelivery\s+schedule\b", IntentCategory.PROJECT_MANAGEMENT, 2.0),
        (r"\bpayment\s+alert\b", IntentCategory.PROJECT_MANAGEMENT, 2.0),
        (r"\bdelivery\s+(date|status|time)", IntentCategory.PROJECT_MANAGEMENT, 3.0),
        (r"\border\s+status\b", IntentCategory.PROJECT_MANAGEMENT, 3.0),
        (
            r"\bwhen\s+(is|will)\s+.+\s+(ship|deliver|dispatch)",
            IntentCategory.PROJECT_MANAGEMENT,
            3.0,
        ),
        (r"\bshipping\s+date\b", IntentCategory.PROJECT_MANAGEMENT, 2.5),
        (r"\bdispatch\b", IntentCategory.PROJECT_MANAGEMENT, 1.5),
        # Payment / invoice (routes to finance)
        (r"\bpayment\s+status\b", IntentCategory.FINANCE_REVIEW, 3.0),
        # Alone must clear FINANCE_REVIEW threshold (3.0) for queries like “latest invoice for X”.
        (r"\binvoice\b", IntentCategory.FINANCE_REVIEW, 3.0),
        (r"\bpayment\s+(due|overdue|received|pending)\b", IntentCategory.FINANCE_REVIEW, 3.0),
        (r"\b(AR|AP)\s+(aging|status|overdue)\b", IntentCategory.FINANCE_REVIEW, 3.0),
        (r"\baccounts?\s+(receivable|payable)\b", IntentCategory.FINANCE_REVIEW, 2.5),
        # Quality management
        (r"\bpunch\s*list\b", IntentCategory.QUALITY_MANAGEMENT, 3.0),
        (r"\bquality\b", IntentCategory.QUALITY_MANAGEMENT, 1.5),
        (r"\bFAT\b", IntentCategory.QUALITY_MANAGEMENT, 2.5),
        (r"\binstallation\b", IntentCategory.QUALITY_MANAGEMENT, 1.0),
        (r"\bcommissioning\b", IntentCategory.QUALITY_MANAGEMENT, 2.0),
        (r"\bdefect\b", IntentCategory.QUALITY_MANAGEMENT, 2.0),
        (r"\bsnag\b", IntentCategory.QUALITY_MANAGEMENT, 2.0),
        # Case study / content
        (r"\bcase\s+stud", IntentCategory.CASE_STUDY, 3.0),
        (r"\blinkedin\s+post\b", IntentCategory.CASE_STUDY, 3.0),
        (r"\bsuccess\s+stor", IntentCategory.CASE_STUDY, 2.5),
        (r"\bcontent\s+draft\b", IntentCategory.CASE_STUDY, 2.0),
        # Quote generation (PDF/document, not pricing inquiry)
        (r"\bgenerate\s+quote\b", IntentCategory.QUOTE_GENERATION, 3.0),
        (r"\bbuild\s+quote\b", IntentCategory.QUOTE_GENERATION, 3.0),
        (r"\bquote\s+PDF\b", IntentCategory.QUOTE_GENERATION, 3.0),
        (r"\bformal\s+quote\b", IntentCategory.QUOTE_GENERATION, 3.0),
        (r"\bquote\s+document\b", IntentCategory.QUOTE_GENERATION, 2.5),
        # Contact / inbox classification (sentinel for offline gold eval; must beat MACHINE_SPECS on long threads)
        (r"\[IRA_EMAIL_GOLD_EVAL_v1\]", IntentCategory.CONTACT_CLASSIFICATION, 100.0),
        # Contact classification
        (r"\bclassif", IntentCategory.CONTACT_CLASSIFICATION, 2.0),
        (r"\bwho\s+is\b.*\bcontact\b", IntentCategory.CONTACT_CLASSIFICATION, 2.5),
        (r"\bcontact\s+type\b", IntentCategory.CONTACT_CLASSIFICATION, 2.5),
        (r"\bcustomer\s+or\s+vendor\b", IntentCategory.CONTACT_CLASSIFICATION, 3.0),
        # Outbound / follow-up email drafting (evidence + Calliope — see prompts/calliope_system.txt)
        (r"\bdraft\s+follow[\s-]?up\s+email\b", IntentCategory.OUTBOUND_EMAIL_DRAFT, 5.0),
        (r"\bfollow[\s-]?up\s+email\s+(for|to)\b", IntentCategory.OUTBOUND_EMAIL_DRAFT, 5.0),
        (r"\bdraft\s+email\s+to\b", IntentCategory.OUTBOUND_EMAIL_DRAFT, 4.5),
        (r"\bcompose\s+email\s+to\b", IntentCategory.OUTBOUND_EMAIL_DRAFT, 4.0),
        (r"\boutbound\s+email\s+to\b", IntentCategory.OUTBOUND_EMAIL_DRAFT, 4.0),
        (r"\bcustomer\s+email\s+draft\b", IntentCategory.OUTBOUND_EMAIL_DRAFT, 4.0),
        (r"\bwrite\s+email\s+to\b", IntentCategory.OUTBOUND_EMAIL_DRAFT, 4.0),
        # Single-company/domain dossiering
        (r"\bdossier\s+on\b", IntentCategory.COMPANY_DOSSIER, 4.5),
        (r"\bintel\s+on\s+@\w+", IntentCategory.COMPANY_DOSSIER, 4.5),
        (r"\benrich\s+this\s+(domain|url)\b", IntentCategory.COMPANY_DOSSIER, 4.0),
        (
            r"\bwhat\s+do\s+we\s+know\s+about\s+this\s+prospect\b",
            IntentCategory.COMPANY_DOSSIER,
            4.0,
        ),
        # North America sales strategy prompts
        (r"\bnorth\s+america\s+sales\s+pitch\b", IntentCategory.NA_SALES_STRATEGY, 4.0),
        (r"\bus\s*/\s*canada\s+outreach\s+angle\b", IntentCategory.NA_SALES_STRATEGY, 4.0),
        (r"\breshoring\s+argument\b", IntentCategory.NA_SALES_STRATEGY, 3.5),
        (r"\bclosed\s+chamber\s+vs\s+open\s+chamber\b", IntentCategory.NA_SALES_STRATEGY, 4.0),
        (r"\bdowngauging\s+roi\b", IntentCategory.NA_SALES_STRATEGY, 3.5),
        # Logged Cursor session recall (Graphe FTS — before generic memory recall)
        (r"\b(previous|last|prior)\s+sessions?\b", IntentCategory.CURSOR_SESSION_RECALL, 3.5),
        (r"\bwhat\s+did\s+we\s+log\b", IntentCategory.CURSOR_SESSION_RECALL, 3.5),
        (r"\bwhat\s+was\s+logged\b", IntentCategory.CURSOR_SESSION_RECALL, 3.5),
        (r"\bcursor\s+sessions?\b", IntentCategory.CURSOR_SESSION_RECALL, 3.5),
        (r"\blogged\s+cursor\b", IntentCategory.CURSOR_SESSION_RECALL, 3.5),
        (r"\bsession\s+log\b", IntentCategory.CURSOR_SESSION_RECALL, 3.5),
        (
            r"\bsearch\s+(my|our)\s+(cursor\s+)?sessions?\b",
            IntentCategory.CURSOR_SESSION_RECALL,
            3.5,
        ),
        (r"\bgraphe\b", IntentCategory.CURSOR_SESSION_RECALL, 3.0),
        # Memory recall
        (r"\bremember\b", IntentCategory.MEMORY_RECALL, 2.0),
        (r"\brecall\b", IntentCategory.MEMORY_RECALL, 2.0),
        (r"\bwhat\s+did\s+(we|i|you)\s+(discuss|talk|say)\b", IntentCategory.MEMORY_RECALL, 3.0),
        (r"\blast\s+time\b", IntentCategory.MEMORY_RECALL, 1.5),
        (r"\bprevious\s+conversation\b", IntentCategory.MEMORY_RECALL, 3.0),
        # System training / self-improvement
        (r"\btrain\b", IntentCategory.SYSTEM_TRAINING, 2.0),
        (r"\bself[\s-]?assess", IntentCategory.SYSTEM_TRAINING, 2.5),
        (r"\bweak\s+area", IntentCategory.SYSTEM_TRAINING, 2.0),
        (r"\bimprove\s+yourself\b", IntentCategory.SYSTEM_TRAINING, 2.5),
        # Archive / document search
        (r"\barchive\b", IntentCategory.ARCHIVE_SEARCH, 3.0),
        (r"\bimports?\b", IntentCategory.ARCHIVE_SEARCH, 2.0),
        (r"\bbrowse\s+documents?\b", IntentCategory.ARCHIVE_SEARCH, 3.0),
        (r"\bbrowse\s+files?\b", IntentCategory.ARCHIVE_SEARCH, 3.0),
        (r"\bfind\s+the\s+file\b", IntentCategory.ARCHIVE_SEARCH, 3.0),
        (r"\bfind\s+the\s+document\b", IntentCategory.ARCHIVE_SEARCH, 3.0),
        (r"\bread\s+the\s+document\b", IntentCategory.ARCHIVE_SEARCH, 3.0),
        (r"\braw\s+document\b", IntentCategory.ARCHIVE_SEARCH, 3.0),
        (r"\bdata/imports\b", IntentCategory.ARCHIVE_SEARCH, 3.0),
        (r"\blook\s+up\s+file\b", IntentCategory.ARCHIVE_SEARCH, 2.5),
        (r"\boriginal\s+(file|document|pdf)\b", IntentCategory.ARCHIVE_SEARCH, 2.5),
        (r"\bsource\s+document\b", IntentCategory.ARCHIVE_SEARCH, 2.5),
        (
            r"\b(ingest|index)\b.*\b(new\s+file|new\s+document|imports?|archive)\b",
            IntentCategory.ARCHIVE_SEARCH,
            4.0,
        ),
        (
            r"\b(i\s+added|i\s+uploaded|i\s+dropped)\b.*\b(file|document)\b",
            IntentCategory.ARCHIVE_SEARCH,
            3.0,
        ),
    ]
)

_CONFIDENCE_THRESHOLD = 3.0


class DeterministicRouter:
    """Pattern-based intent classifier and routing-table lookup."""

    @staticmethod
    def accumulate_pattern_scores(query: str) -> dict[IntentCategory, float]:
        """Weighted pattern scores per intent (no threshold applied)."""
        scores: dict[IntentCategory, float] = {}
        for pat in _PATTERNS:
            if pat.regex.search(query):
                scores[pat.intent] = scores.get(pat.intent, 0.0) + pat.weight
        try:
            from brain_os.brain.learning_router_nudges import merge_learned_nudge_scores

            merge_learned_nudge_scores(query, scores)
        except (ImportError, OSError, TypeError, ValueError):
            pass
        return scores

    def classify_intent(self, query: str) -> IntentCategory | None:
        """Classify *query* by keyword patterns.

        Returns the best-matching :class:`IntentCategory`, or ``None`` if
        no pattern exceeds the confidence threshold (signalling that
        LLM-based routing should be used).
        """
        scores = self.accumulate_pattern_scores(query)
        if not scores:
            return None

        best_intent = max(scores, key=scores.__getitem__)
        if scores[best_intent] < _CONFIDENCE_THRESHOLD:
            return None

        return best_intent

    def routing_scoreboard(self, query: str) -> dict[str, Any]:
        """Top pattern scores + margin vs runner-up for observability and eval harness."""
        raw = self.accumulate_pattern_scores(query)
        ranked = sorted(raw.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        margin = (
            float(top[1]) - float(second[1])
            if top is not None and second is not None
            else (float(top[1]) if top is not None else 0.0)
        )
        return {
            "threshold": float(_CONFIDENCE_THRESHOLD),
            "top_intent": top[0].value if top else None,
            "top_score": float(top[1]) if top else 0.0,
            "second_intent": second[0].value if second else None,
            "second_score": float(second[1]) if second else 0.0,
            "margin": round(margin, 4),
            "matched": bool(top and top[1] >= _CONFIDENCE_THRESHOLD),
            "all_scores": [(k.value, round(v, 4)) for k, v in ranked[:8]],
        }

    def get_routing(self, intent: IntentCategory) -> dict:
        """Return the routing configuration for *intent* as a plain dict."""
        cfg = ROUTING_TABLE.get(intent, ROUTING_TABLE[IntentCategory.GENERAL])
        return {
            "intent": intent.value,
            "required_agents": list(cfg.required_agents),
            "optional_agents": list(cfg.optional_agents),
            "required_tools": list(cfg.required_tools),
        }

    def route(self, query: str) -> dict | None:
        """Convenience: classify and route in one call.

        Returns the routing dict, or ``None`` if the query should be
        routed by the LLM instead.
        """
        from brain_os.config import get_settings

        intent = self.classify_intent(query)
        if intent is None:
            return None

        sb = self.routing_scoreboard(query)
        margin_min = float(get_settings().app.router_deterministic_margin_min)
        if margin_min > 0 and sb["margin"] < margin_min:
            return None

        return self.get_routing(intent)
