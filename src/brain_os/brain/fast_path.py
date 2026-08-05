"""Fast-path classifier for simple conversational queries.

Detects greetings, identity questions, thanks, farewells, and simple
chat that do not require the full 11-stage pipeline.  Matched queries
skip routing, enrichment, metacognition, and reflection — returning in
1-3 seconds instead of 20+.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from brain_os.exceptions import LLMError
from brain_os.prompt_loader import load_soul_preamble
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class FastPathCategory(str, Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    THANKS = "thanks"
    IDENTITY = "identity"
    SIMPLE_CHAT = "simple_chat"


@dataclass(frozen=True, slots=True)
class FastPathResult:
    matched: bool
    response: str | None = None
    category: FastPathCategory | None = None


_GREETING_PATTERNS = re.compile(
    r"^\s*("
    r"h(i|ello|ey|owdy|ola)"
    r"|good\s*(morning|afternoon|evening|day)"
    r"|what'?s\s*up"
    r"|yo\b"
    r"|namaste"
    r"|salaam"
    r"|greetings"
    r")\s*[!.,?]*\s*$",
    re.IGNORECASE,
)

_FAREWELL_PATTERNS = re.compile(
    r"^\s*("
    r"bye\b|goodbye|good\s*night|see\s*you|later|take\s*care"
    r"|ciao|adios|cheerio"
    r"|that'?s\s*all|i'?m\s*done|nothing\s*else"
    r")\s*[!.,?]*\s*$",
    re.IGNORECASE,
)

_THANKS_PATTERNS = re.compile(
    r"^\s*("
    r"thanks?(\s*you)?|thank\s*you(\s*so\s*much)?|thx|ty"
    r"|much\s*appreciated|appreciate\s*it"
    r"|great(\s*job)?[!.]*"
    r"|perfect[!.]*"
    r"|awesome[!.]*"
    r")\s*[!.,?]*\s*$",
    re.IGNORECASE,
)

_IDENTITY_PATTERNS = re.compile(
    r"("
    r"who\s+(are\s+you|is\s+ira)"
    r"|what\s+(are\s+you|is\s+ira)"
    r"|tell\s+me\s+about\s+(yourself|ira)"
    r"|introduce\s+yourself"
    r"|what\s+do\s+you\s+do"
    r"|what\s+can\s+you\s+do"
    r"|what\s+are\s+your\s+capabilities"
    r"|describe\s+yourself"
    r")",
    re.IGNORECASE,
)

# Identity fast-path must not swallow compound questions (e.g. intro + pricing).
_COMPOUND_IDENTITY_DENY = re.compile(
    r"\band\b.+\b(price|pric(e|ing)|quote|cost|invoice|pipeline|deals?|leads?|PF\d)\b",
    re.IGNORECASE,
)

_SIMPLE_CHAT_PATTERNS = re.compile(
    r"^\s*("
    r"how\s+are\s+you"
    r"|how'?s\s+it\s+going"
    r"|how\s+do\s+you\s+feel"
    r"|are\s+you\s+(ok(ay)?|alright|there|awake|ready)"
    r"|you\s+there\??"
    r"|ping"
    r")\s*[!.,?]*\s*$",
    re.IGNORECASE,
)


_CANNED: dict[FastPathCategory, list[str]] = {
    FastPathCategory.GREETING: [
        "Hello! I'm Brain OS, the AI that runs Acme Corp. How can I help you today?",
        "Hi there! Brain OS here, ready to help. What would you like to know?",
        "Good to see you! What can I do for you today?",
    ],
    FastPathCategory.FAREWELL: [
        "Goodbye! Don't hesitate to reach out whenever you need anything.",
        "Take care! I'll be here whenever you need me.",
        "See you later! All systems will stay ready for your return.",
    ],
    FastPathCategory.THANKS: [
        "You're welcome! Let me know if there's anything else.",
        "Happy to help! Anything else you need?",
        "Glad I could assist. I'm here if you need more.",
    ],
    FastPathCategory.SIMPLE_CHAT: [
        "I'm doing well, thank you! All systems are running smoothly. How can I help you today?",
        "I'm here and ready! What would you like to work on?",
    ],
}

_IDENTITY_PROMPT = (
    "You are Brain OS, the AI operating system for Acme Corp — an Indian "
    "industrial machinery company. Introduce yourself warmly but concisely "
    "(3-4 sentences). Mention your role managing the business through 24 "
    "specialist agents, your philosophical foundation (Jain/Hindu heritage: "
    "cross-verification, truthfulness, mutual interdependence), and that you "
    "cover sales, production, finance, quality, HR, and knowledge management. "
    "Be decisive and professional, not chatbot-like."
)


def classify(query: str) -> FastPathResult:
    """Classify a query into a fast-path category, or return unmatched."""
    text = query.strip()

    if not text or len(text) > 200:
        return FastPathResult(matched=False)

    if _GREETING_PATTERNS.match(text):
        return FastPathResult(
            matched=True,
            response=_pick_canned(FastPathCategory.GREETING),
            category=FastPathCategory.GREETING,
        )

    if _FAREWELL_PATTERNS.match(text):
        return FastPathResult(
            matched=True,
            response=_pick_canned(FastPathCategory.FAREWELL),
            category=FastPathCategory.FAREWELL,
        )

    if _THANKS_PATTERNS.match(text):
        return FastPathResult(
            matched=True,
            response=_pick_canned(FastPathCategory.THANKS),
            category=FastPathCategory.THANKS,
        )

    if (
        _IDENTITY_PATTERNS.search(text)
        and len(text) < 60
        and not _COMPOUND_IDENTITY_DENY.search(text)
    ):
        return FastPathResult(
            matched=True,
            response=None,
            category=FastPathCategory.IDENTITY,
        )

    if _SIMPLE_CHAT_PATTERNS.match(text):
        return FastPathResult(
            matched=True,
            response=_pick_canned(FastPathCategory.SIMPLE_CHAT),
            category=FastPathCategory.SIMPLE_CHAT,
        )

    return FastPathResult(matched=False)


async def generate(
    query: str,
    category: FastPathCategory,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """Generate a response for categories that need a single LLM call."""
    if category == FastPathCategory.IDENTITY:
        return await _generate_identity(query, session_id=session_id, user_id=user_id)
    return _pick_canned(category)


async def _generate_identity(
    query: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """Single LLM call for identity questions."""
    llm = get_llm_client()
    soul = load_soul_preamble() or ""
    system = f"{soul}\n\n{_IDENTITY_PROMPT}" if soul else _IDENTITY_PROMPT
    try:
        return await llm.generate_text_with_fallback(
            system,
            f"The user asked: {query}\n\nIntroduce yourself.",
            primary="openai",
            temperature=0.4,
            name="fast_path.identity",
            session_id=session_id,
            user_id=user_id,
        )
    except LLMError:
        logger.exception("Fast-path identity LLM call failed")
        return (
            "I'm Brain OS, the AI that runs Acme Corp — an industrial machinery "
            "company based in India. I manage the entire business through 24 "
            "specialist agents covering sales, production, finance, quality, "
            "and more. How can I help you today?"
        )


def _pick_canned(category: FastPathCategory) -> str:
    """Pick a canned response, rotating through options."""
    import random

    options = _CANNED.get(category, [])
    return random.choice(options) if options else ""
