"""Autonomous drip engine — automated multi-step email campaigns.

Evaluates active drip campaigns, sends pending steps via Gmail, and
checks for replies.  Integrated into the :class:`RespiratorySystem`
exhale cycle for nightly execution.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from googleapiclient.errors import HttpError as GoogleApiHttpError
from sqlalchemy.exc import SQLAlchemyError

from brain_os.config import get_settings
from brain_os.exceptions import BrainOSError
from brain_os.prompt_loader import load_prompt
from brain_os.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _as_utc_aware(dt: datetime | None) -> datetime | None:
    """Compare ORM/datetime values safely against ``datetime.now(timezone.utc)``."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_llm_json_object(raw: str) -> dict[str, Any] | None:
    """Parse JSON object from LLM output; tolerate optional markdown fences."""
    text = raw.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.split("\n")
        inner = "\n".join(lines[1:])
        if inner.rstrip().endswith("```"):
            inner = inner.rstrip()[:-3].rstrip()
        text = inner.strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Drip LLM output was not valid JSON (first 120 chars): %s", text[:120])
        return None
    return out if isinstance(out, dict) else None


class AutonomousDripEngine:
    """Manages automated drip email campaigns."""

    def __init__(
        self,
        crm: Any,
        quotes: Any | None = None,
        message_bus: Any | None = None,
        gmail: Any | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self._crm = crm
        self._quotes = quotes
        self._bus = message_bus
        self._gmail = gmail
        self._llm_client = llm_client

    @staticmethod
    def _extract_thread_id(draft: Any) -> str:
        if not isinstance(draft, dict):
            return ""
        message = draft.get("message")
        if isinstance(message, dict):
            return str(message.get("threadId", "") or "")
        return str(draft.get("threadId", "") or "")

    async def _create_draft(self, to: str, subject: str, body: str) -> dict[str, Any]:
        if self._gmail is None:
            return {}
        if hasattr(self._gmail, "create_draft"):
            result = await self._gmail.create_draft(to=to, subject=subject, body=body)
            return result if isinstance(result, dict) else {}
        if hasattr(self._gmail, "send_draft"):
            # Legacy adapter compatibility.
            await self._gmail.send_draft(to=to, subject=subject, body=body)
            return {}
        raise AttributeError("Gmail adapter must provide send_draft() or create_draft()")

    async def _has_reply(self, step: Any, to: str, subject: str) -> bool:
        if self._gmail is None:
            return False
        if hasattr(self._gmail, "check_replies"):
            marker = str(getattr(step, "reply_content", "") or "")
            if not marker.startswith("thread:"):
                return False
            thread_id = marker.split("thread:", 1)[1].strip()
            if not thread_id:
                return False
            messages = await self._gmail.check_replies(thread_id)
            return bool(messages)
        if hasattr(self._gmail, "check_reply"):
            return bool(await self._gmail.check_reply(to=to, subject=subject))
        raise AttributeError("Gmail adapter must provide check_reply() or check_replies()")

    async def _llm_evaluate_insights(self, summary: dict[str, Any]) -> dict[str, Any] | None:
        llm = self._llm_client if self._llm_client is not None else LLMClient()
        system = load_prompt("drip_evaluate")
        user = json.dumps(summary, default=str)
        raw = await llm.generate_text(
            system,
            user,
            temperature=0.2,
            max_tokens=1200,
            name="drip_evaluate",
        )
        if raw.strip().startswith("(No OpenAI") or raw.strip().startswith("(No Anthropic"):
            logger.info("Drip evaluate skipped: no LLM provider configured")
            return None
        parsed = _parse_llm_json_object(raw)
        if parsed is None:
            return {"error": "parse_failed", "raw_preview": raw[:500]}
        return parsed

    async def _build_adjust_user_payload(
        self,
        evaluation: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            campaigns = await self._crm.list_campaigns()
        except (SQLAlchemyError, OSError, RuntimeError, ValueError, TypeError):
            logger.exception("drip adjust: list_campaigns failed")
            return None
        active = [c for c in campaigns if getattr(c, "status", None) in ("ACTIVE", "active")]
        if not active:
            return None
        stat_by_name = {
            s["campaign"]: s for s in (evaluation.get("stats") or []) if "campaign" in s
        }
        campaigns_payload: list[dict[str, Any]] = []
        for c in active:
            try:
                steps = await self._crm.list_drip_steps(
                    filters={"campaign_id": str(c.id)},
                )
            except (SQLAlchemyError, OSError, RuntimeError, ValueError, TypeError):
                logger.exception(
                    "drip adjust: list_drip_steps failed for %s", getattr(c, "name", c)
                )
                continue
            outline: list[dict[str, Any]] = []
            for s in sorted(steps, key=lambda x: x.step_number):
                outline.append(
                    {
                        "step_number": s.step_number,
                        "subject": (s.email_subject or "")[:200],
                        "scheduled_at": s.scheduled_at.isoformat() if s.scheduled_at else None,
                        "sent": bool(s.sent_at),
                        "reply_received": bool(s.reply_received),
                    }
                )
            campaigns_payload.append(
                {
                    "name": c.name,
                    "target_segment": getattr(c, "target_segment", None),
                    "recent_metrics": stat_by_name.get(c.name, {}),
                    "steps_outline": outline,
                }
            )
        if not campaigns_payload:
            return None
        return {
            "campaigns": campaigns_payload,
            "aggregate": {
                "campaigns_total": evaluation.get("campaigns"),
                "active": evaluation.get("active"),
            },
        }

    async def _llm_suggest_adjustments(
        self,
        evaluation: dict[str, Any],
    ) -> dict[str, Any] | None:
        payload = await self._build_adjust_user_payload(evaluation)
        if not payload:
            return None
        llm = self._llm_client if self._llm_client is not None else LLMClient()
        system = load_prompt("drip_adjust")
        user = json.dumps(payload, default=str)
        raw = await llm.generate_text(
            system,
            user,
            temperature=0.2,
            max_tokens=2048,
            name="drip_adjust",
        )
        if raw.strip().startswith("(No OpenAI") or raw.strip().startswith("(No Anthropic"):
            logger.info("Drip adjust skipped: no LLM provider configured")
            return None
        parsed = _parse_llm_json_object(raw)
        if parsed is None:
            return {"error": "parse_failed", "raw_preview": raw[:500]}
        return parsed

    async def evaluate_campaigns(self) -> dict[str, Any]:
        """Check active campaigns and evaluate performance metrics."""
        try:
            campaigns = await self._crm.list_campaigns()
        except (SQLAlchemyError, OSError, RuntimeError, ValueError, TypeError):
            logger.exception("Failed to list campaigns")
            return {"campaigns": 0, "active": 0, "error": "CRM query failed"}

        active = [c for c in campaigns if getattr(c, "status", None) in ("ACTIVE", "active")]

        stats: list[dict[str, Any]] = []
        for campaign in active:
            try:
                steps = await self._crm.list_drip_steps(
                    filters={"campaign_id": str(campaign.id)},
                )
                sent = sum(1 for s in steps if s.sent_at)
                replied = sum(1 for s in steps if s.reply_received)
                reply_rate = replied / sent if sent > 0 else 0.0

                stats.append(
                    {
                        "campaign": campaign.name,
                        "total_steps": len(steps),
                        "sent": sent,
                        "replied": replied,
                        "reply_rate": round(reply_rate, 3),
                    }
                )
            except (SQLAlchemyError, OSError, RuntimeError, ValueError, TypeError):
                logger.exception("Failed to evaluate campaign %s", campaign.name)

        result: dict[str, Any] = {
            "campaigns": len(campaigns),
            "active": len(active),
            "stats": stats,
        }
        if get_settings().app.drip_llm_insights and "error" not in result:
            try:
                insights = await self._llm_evaluate_insights(result)
                if insights is not None:
                    result["llm_insights"] = insights
            except (TimeoutError, BrainOSError, OSError, RuntimeError, ValueError, TypeError):
                logger.exception("Drip LLM evaluate failed")
        return result

    async def send_pending_steps(self) -> dict[str, Any]:
        """Send drip steps that are due."""
        sent_count = 0
        errors: list[str] = []

        try:
            campaigns = await self._crm.list_campaigns()
        except (SQLAlchemyError, OSError, RuntimeError, ValueError, TypeError):
            return {"sent": 0, "errors": ["CRM query failed"]}

        active = [c for c in campaigns if getattr(c, "status", None) in ("ACTIVE", "active")]

        for campaign in active:
            try:
                steps = await self._crm.list_drip_steps(
                    filters={"campaign_id": str(campaign.id)},
                )
                pending = [s for s in steps if s.sent_at is None]

                for step in pending:
                    sched = _as_utc_aware(step.scheduled_at)
                    if sched and sched > datetime.now(UTC):
                        continue

                    if self._gmail is not None:
                        try:
                            contact = await self._crm.get_contact(step.contact_id)
                            if contact is None or not contact.email:
                                errors.append(
                                    f"No email for contact {step.contact_id} (step {step.step_number})"
                                )
                                continue

                            draft = await self._create_draft(
                                to=contact.email,
                                subject=step.email_subject,
                                body=step.email_body,
                            )
                            now = datetime.now(UTC)
                            step.sent_at = now
                            updates: dict[str, Any] = {"sent_at": now}
                            thread_id = self._extract_thread_id(draft)
                            if thread_id:
                                updates["reply_content"] = f"thread:{thread_id}"
                            await self._crm.update_drip_step(step.id, **updates)
                            sent_count += 1
                        except (
                            TimeoutError,
                            SQLAlchemyError,
                            BrainOSError,
                            GoogleApiHttpError,
                            AttributeError,
                            OSError,
                            RuntimeError,
                            ValueError,
                            TypeError,
                        ) as exc:
                            errors.append(f"Send failed for step {step.step_number}: {exc}")
                    else:
                        logger.debug(
                            "Gmail sender not configured — skipping step %d", step.step_number
                        )
            except (SQLAlchemyError, OSError, RuntimeError, ValueError, TypeError):
                logger.exception("Failed to process campaign %s", campaign.name)

        logger.info("Drip engine: sent %d steps, %d errors", sent_count, len(errors))
        return {"sent": sent_count, "errors": errors}

    async def check_replies(self) -> dict[str, Any]:
        """Poll for replies to sent drip steps."""
        reply_count = 0

        if self._gmail is None:
            return {"replies_detected": 0, "note": "Gmail not configured"}

        try:
            campaigns = await self._crm.list_campaigns()
            active = [c for c in campaigns if getattr(c, "status", None) in ("ACTIVE", "active")]

            for campaign in active:
                steps = await self._crm.list_drip_steps(
                    filters={"campaign_id": str(campaign.id)},
                )
                sent_unreplied = [s for s in steps if s.sent_at and not s.reply_received]

                for step in sent_unreplied:
                    try:
                        contact = await self._crm.get_contact(step.contact_id)
                        if contact is None or not contact.email:
                            continue

                        has_reply = await self._has_reply(
                            step=step,
                            to=contact.email,
                            subject=step.email_subject,
                        )
                        if has_reply:
                            step.reply_received = True
                            await self._crm.update_drip_step(step.id, reply_received=True)
                            reply_count += 1
                    except (
                        TimeoutError,
                        SQLAlchemyError,
                        GoogleApiHttpError,
                        AttributeError,
                        OSError,
                        RuntimeError,
                        ValueError,
                        TypeError,
                    ):
                        logger.warning(
                            "Reply check failed for step %d", step.step_number, exc_info=True
                        )

        except (SQLAlchemyError, BrainOSError, OSError, RuntimeError, ValueError, TypeError):
            logger.exception("Reply check cycle failed")

        return {"replies_detected": reply_count}

    async def run_cycle(self) -> dict[str, Any]:
        """Full drip cycle: evaluate, send pending, check replies."""
        evaluation = await self.evaluate_campaigns()
        send_result = await self.send_pending_steps()
        reply_result = await self.check_replies()

        out: dict[str, Any] = {
            "evaluation": evaluation,
            "sends": send_result,
            "replies": reply_result,
        }
        if get_settings().app.drip_llm_adjust and "error" not in evaluation:
            try:
                adjustments = await self._llm_suggest_adjustments(evaluation)
                if adjustments is not None:
                    out["llm_adjustments"] = adjustments
            except (TimeoutError, BrainOSError, OSError, RuntimeError, ValueError, TypeError):
                logger.exception("Drip LLM adjust failed")
        return out
