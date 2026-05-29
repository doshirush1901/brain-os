"""Structured account journey extracted from full mailbox history."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class JourneyEvent(BaseModel):
    date: str = ""
    thread_id: str = ""
    direction: str = ""
    from_address: str = ""
    to_address: str = ""
    subject: str = ""
    summary: str = ""
    source: str = "gmail"
    attachment_count: int = 0
    attachment_excerpt: str = ""
    model_hints: list[str] = Field(default_factory=list)
    price_hints: list[str] = Field(default_factory=list)
    outcome_hint: str | None = None


class YearNarrative(BaseModel):
    year: int
    highlights: list[str] = Field(default_factory=list)


class MeetingContextPacket(BaseModel):
    current_need: str = ""
    constraints: list[str] = Field(default_factory=list)
    key_objections: list[str] = Field(default_factory=list)
    recommended_agenda: list[str] = Field(default_factory=list)
    draft_focus_points: list[str] = Field(default_factory=list)


class AccountJourney(BaseModel):
    company_name: str = ""
    contact_email: str | None = None
    domain: str | None = None
    mailbox: str | None = None
    query: str = ""
    total_messages_scanned: int = 0
    total_threads_scanned: int = 0
    events: list[JourneyEvent] = Field(default_factory=list)
    yearly_narrative: list[YearNarrative] = Field(default_factory=list)
    meeting_context: MeetingContextPacket = Field(default_factory=MeetingContextPacket)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    checkpoint_path: str | None = None
