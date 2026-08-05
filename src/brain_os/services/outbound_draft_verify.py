"""Post-draft outbound verification — claim map + Aletheia outbound mode."""

from __future__ import annotations

import logging
from typing import Any

from brain_os.config import get_settings

logger = logging.getLogger(__name__)


async def verify_outbound_draft(
    draft_text: str,
    *,
    pantheon: Any | None = None,
    tag_filter: str = "",
) -> dict[str, Any]:
    """Run evidence claim map and optional Aletheia outbound provenance."""
    from brain_os.services.outbound_claim_map import build_outbound_claim_map
    from brain_os.services.sales_decisions import run_evidence_backing_check

    evidence = await run_evidence_backing_check(
        draft_text=draft_text,
        tag_filter=tag_filter,
    )
    claim_map = evidence.get("claim_map") or build_outbound_claim_map(
        draft_text,
        tag_filter=tag_filter,
    )

    aletheia_block: dict[str, Any] | None = None
    if get_settings().app.aletheia_outbound_mode_enabled and pantheon is not None:
        aletheia = pantheon.get_agent("aletheia")
        if aletheia is not None:
            try:
                aletheia_block = await aletheia.check_provenance_outbound(draft_text)
            except Exception as exc:
                logger.warning("Aletheia outbound check failed: %s", exc)
                aletheia_block = {"error": str(exc)[:200]}

    blocked = bool(claim_map.get("fail_closed"))
    if aletheia_block and aletheia_block.get("verdict") == "UNVERIFIED":
        blocked = blocked or len(aletheia_block.get("unverifiable") or []) >= 2

    if blocked or (
        aletheia_block
        and (
            aletheia_block.get("unverifiable")
            or aletheia_block.get("verdict") in {"UNVERIFIED", "BLOCK", "REVIEW"}
        )
    ):
        try:
            from brain_os.immune.registry import record_trigger

            record_trigger(
                "aletheia_provenance",
                {
                    "outbound": True,
                    "blocked": blocked,
                    "verdict": (aletheia_block or {}).get("verdict"),
                },
            )
        except Exception:
            logger.debug("immune record_trigger failed for aletheia outbound", exc_info=True)

    return {
        "ok": not blocked,
        "blocked": blocked,
        "evidence_backing": evidence,
        "claim_map": claim_map,
        "aletheia_outbound": aletheia_block,
    }
