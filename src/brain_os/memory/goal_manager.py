"""Goal manager — tracks multi-turn goal-oriented dialogues with slot-filling."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import aiosqlite
from langfuse.decorators import observe
from pydantic import BaseModel, Field

from brain_os.memory.goal_backend import GoalManagerBackend, build_goal_backend
from brain_os.prompt_loader import load_prompt
from brain_os.schemas.llm_outputs import GoalDetection
from brain_os.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class GoalType(str, Enum):
    LEAD_QUALIFICATION = "LEAD_QUALIFICATION"
    MEETING_BOOKING = "MEETING_BOOKING"
    QUOTE_PREPARATION = "QUOTE_PREPARATION"
    FOLLOW_UP_SCHEDULING = "FOLLOW_UP_SCHEDULING"
    INFORMATION_GATHERING = "INFORMATION_GATHERING"


class GoalStatus(str, Enum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"


class Goal(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    goal_type: GoalType
    contact_id: str
    status: GoalStatus = GoalStatus.ACTIVE
    required_slots: dict[str, str | None]
    progress: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


_SLOT_TEMPLATES: dict[GoalType, dict[str, None]] = {
    GoalType.LEAD_QUALIFICATION: {
        "company_name": None,
        "industry": None,
        "machine_interest": None,
        "volume_requirements": None,
        "timeline": None,
        "budget_range": None,
    },
    GoalType.MEETING_BOOKING: {
        "preferred_date": None,
        "preferred_time": None,
        "meeting_type": None,
        "attendees": None,
        "agenda": None,
    },
    GoalType.QUOTE_PREPARATION: {
        "machine_model": None,
        "quantity": None,
        "destination_country": None,
        "customizations": None,
        "delivery_timeline": None,
    },
    GoalType.FOLLOW_UP_SCHEDULING: {
        "follow_up_date": None,
        "follow_up_channel": None,
        "follow_up_topic": None,
    },
    GoalType.INFORMATION_GATHERING: {
        "topic": None,
        "specific_questions": None,
        "context": None,
    },
}

_SLOT_QUESTIONS: dict[str, str] = {
    "company_name": "What is your company name?",
    "industry": "What industry are you in?",
    "machine_interest": "Which type of machine are you interested in?",
    "volume_requirements": "What are your volume requirements?",
    "timeline": "What is your timeline for this project?",
    "budget_range": "What is your budget range?",
    "preferred_date": "What date works best for the meeting?",
    "preferred_time": "What time works best for you?",
    "meeting_type": "What type of meeting would you prefer (e.g. video call, in-person)?",
    "attendees": "Who will be attending the meeting?",
    "agenda": "What would you like to discuss in the meeting?",
    "machine_model": "Which machine model are you looking for?",
    "quantity": "How many units do you need?",
    "destination_country": "What is the destination country for delivery?",
    "customizations": "Do you need any customizations or special configurations?",
    "delivery_timeline": "When do you need delivery?",
    "follow_up_date": "When would you like me to follow up?",
    "follow_up_channel": "How would you prefer to be contacted (email, phone, etc.)?",
    "follow_up_topic": "What should we follow up on?",
    "topic": "What topic would you like more information about?",
    "specific_questions": "What specific questions do you have?",
    "context": "What context or background would help me assist you better?",
}

_GOAL_PATTERNS: list[tuple[re.Pattern[str], GoalType]] = [
    (re.compile(r"\bmeeting\b", re.I), GoalType.MEETING_BOOKING),
    (re.compile(r"\bschedule\b", re.I), GoalType.MEETING_BOOKING),
    (re.compile(r"\bappointment\b", re.I), GoalType.MEETING_BOOKING),
    (re.compile(r"\bquote\b", re.I), GoalType.QUOTE_PREPARATION),
    (re.compile(r"\bpric(?:e|ing)\b", re.I), GoalType.QUOTE_PREPARATION),
    (re.compile(r"\bcost\b", re.I), GoalType.QUOTE_PREPARATION),
    (re.compile(r"\bproposal\b", re.I), GoalType.QUOTE_PREPARATION),
    (re.compile(r"\bfollow.?up\b", re.I), GoalType.FOLLOW_UP_SCHEDULING),
    (re.compile(r"\bremind\b", re.I), GoalType.FOLLOW_UP_SCHEDULING),
    (re.compile(r"\bcheck.?back\b", re.I), GoalType.FOLLOW_UP_SCHEDULING),
]

_DETECT_SYSTEM_PROMPT = load_prompt("goal_detect")

_EXTRACT_SYSTEM_PROMPT = load_prompt("goal_extract_slots")


def _goal_from_row(row: tuple[Any, ...]) -> Goal:
    return Goal(
        id=UUID(row[0]),
        goal_type=GoalType(row[1]),
        contact_id=row[2],
        status=GoalStatus(row[3]),
        required_slots=json.loads(row[4]),
        progress=row[5],
        created_at=datetime.fromisoformat(str(row[6]).replace("Z", "+00:00")),
        completed_at=(
            datetime.fromisoformat(str(row[7]).replace("Z", "+00:00")) if row[7] else None
        ),
    )


class GoalManager:
    def __init__(
        self,
        db_path: str = "data/goals.db",
    ) -> None:
        self._db_path = db_path
        self._llm = get_llm_client()
        self._backend: GoalManagerBackend = build_goal_backend(db_path)

    @property
    def _db(self):
        return self._backend.sqlite_db_connection

    async def initialize(self) -> None:
        await self._backend.initialize()

    @observe()
    async def detect_goal(self, query: str, context: dict[str, Any]) -> Goal | None:
        contact_id = context.get("contact_id")
        if not contact_id:
            return None
        existing = await self.get_active_goal(contact_id)
        if existing is not None:
            return None
        for pattern, goal_type in _GOAL_PATTERNS:
            if pattern.search(query):
                return await self._create_and_persist_goal(contact_id, goal_type)
        if context.get("is_new_lead"):
            return await self._create_and_persist_goal(contact_id, GoalType.LEAD_QUALIFICATION)
        result = await self._llm.generate_structured(
            _DETECT_SYSTEM_PROMPT,
            f"Query: {query}\nContext: {json.dumps(context)}",
            GoalDetection,
            name="goal_manager.detect",
        )
        try:
            if result.should_initiate and result.goal_type:
                gt = GoalType(result.goal_type)
                return await self._create_and_persist_goal(contact_id, gt)
        except (ValueError, KeyError):
            logger.warning("Goal detection LLM returned invalid goal_type: %s", result.goal_type)
        return None

    async def _create_and_persist_goal(self, contact_id: str, goal_type: GoalType) -> Goal:
        slots = dict(_SLOT_TEMPLATES[goal_type])
        goal = Goal(
            goal_type=goal_type,
            contact_id=contact_id,
            required_slots=slots,
        )
        await self._backend.insert_goal(
            str(goal.id),
            goal.goal_type.value,
            goal.contact_id,
            goal.status.value,
            json.dumps(goal.required_slots),
            goal.progress,
            goal.created_at.isoformat(),
            goal.completed_at.isoformat() if goal.completed_at else None,
        )
        return goal

    async def update_goal(self, goal_id: UUID, new_info: dict[str, Any]) -> Goal:
        row = await self._backend.get_row(str(goal_id))
        if row is None:
            raise ValueError(f"Goal {goal_id} not found")
        slots = json.loads(row[4])
        for key, value in new_info.items():
            if key in slots and slots[key] is None and value is not None:
                slots[key] = str(value) if not isinstance(value, str) else value
        filled = sum(1 for v in slots.values() if v is not None)
        total = len(slots)
        progress = filled / total if total else 0.0
        status = GoalStatus.COMPLETED if progress == 1.0 else GoalStatus(row[3])
        completed_at = datetime.now(UTC) if progress == 1.0 else None
        await self._db.execute(
            """
            UPDATE goals SET required_slots = ?, progress = ?, status = ?, completed_at = ?
            WHERE id = ?
            """,
            (
                json.dumps(slots),
                progress,
                status.value,
                completed_at.isoformat() if completed_at else None,
                str(goal_id),
            ),
        )
        await self._db.commit()
        return Goal(
            id=UUID(row[0]),
            goal_type=GoalType(row[1]),
            contact_id=row[2],
            status=status,
            required_slots=slots,
            progress=progress,
            created_at=datetime.fromisoformat(row[6].replace("Z", "+00:00")),
            completed_at=completed_at,
        )

    @observe()
    async def extract_slots(self, goal: Goal, message: str) -> dict[str, str]:
        unfilled = [k for k, v in goal.required_slots.items() if v is None]
        if not unfilled:
            return {}
        prompt = f"Goal type: {goal.goal_type.value}\nUnfilled slots: {unfilled}\nUser message: {message}"
        raw = await self._llm.generate_text(
            _EXTRACT_SYSTEM_PROMPT,
            prompt,
            name="goal_manager.extract_slots",
            temperature=0,
        )
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {k: str(v) for k, v in parsed.items() if v is not None}
        except (json.JSONDecodeError, TypeError):
            logger.warning("Slot extraction LLM returned non-JSON: %s", raw[:200])
        return {}

    def get_next_question(self, goal: Goal) -> str | None:
        for slot_name, value in goal.required_slots.items():
            if value is None:
                return _SLOT_QUESTIONS.get(slot_name)
        return None

    def is_goal_complete(self, goal: Goal) -> bool:
        return all(v is not None for v in goal.required_slots.values())

    async def get_active_goal(self, contact_id: str) -> Goal | None:
        row = await self._backend.get_active_row(contact_id)
        if row is None:
            return None
        return _goal_from_row(row)

    @observe()
    async def sweep_stalled_goals(self, stale_hours: int = 48) -> list[Goal]:
        """Find ACTIVE goals that haven't been updated in *stale_hours*."""
        cutoff = (datetime.now(UTC) - timedelta(hours=stale_hours)).isoformat()
        rows = await self._backend.list_stalled_active(cutoff)
        stalled = [_goal_from_row(row) for row in rows]
        logger.info(
            "Goal sweep found %d stalled goals (cutoff: %s)",
            len(stalled),
            cutoff,
        )
        return stalled

    async def abandon_goal(self, goal_id: UUID) -> None:
        await self._backend.abandon(str(goal_id), GoalStatus.ABANDONED.value)

    async def close(self) -> None:
        await self._backend.close()

    async def __aenter__(self) -> GoalManager:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
