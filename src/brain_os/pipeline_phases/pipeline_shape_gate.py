"""Pre-execute triangulation gate for Pipeline task: Shape fast path."""

from __future__ import annotations

import logging
from typing import Any

from brain_os.brain.triangulation_query_scope import extract_company_and_contact
from brain_os.config import get_settings
from brain_os.services.triangulation_enforcement import (
    TriangulationCheckResult,
    check_outbound_triangulation,
)

logger = logging.getLogger(__name__)


def _skip_triangulation_requested(meta: dict[str, Any]) -> bool:
    if meta.get("skip_triangulation") is True:
        return True
    gov = meta.get("governance")
    if isinstance(gov, dict) and gov.get("skip_triangulation") is True:
        return True
    return False


def _shape_block_on_gaps() -> bool:
    cfg = get_settings().app
    explicit = cfg.pipeline_shape_triangulation_block_on_gaps
    if explicit is not None:
        return bool(explicit)
    return bool(cfg.triangulation_block_on_gaps)


async def run_pipeline_shape_triangulation_gate(
    *,
    resolved_input: str,
    meta: dict[str, Any],
    pantheon: Any,
    email_processor: Any | None,
) -> tuple[bool, list[str], TriangulationCheckResult | None]:
    """Return (blocked, enrichment_prefixes, tri_result).

    When *blocked* is True, caller should return ``block_message`` to the user.
    """
    cfg = get_settings().app
    if not cfg.pipeline_shape_require_triangulation:
        return False, [], None
    if _skip_triangulation_requested(meta):
        return False, [], None

    company, contact = extract_company_and_contact(resolved_input)
    co = (company or "").strip()
    if not co:
        return False, [], None

    tri = await check_outbound_triangulation(
        company=co,
        contact_email=contact,
        pantheon=pantheon,
        email_processor=email_processor,
        hex=bool(cfg.pipeline_shape_triangulation_hex),
        enforce=True,
        block_on_gaps=_shape_block_on_gaps(),
    )
    prefixes: list[str] = []
    if tri.gaps_block:
        prefixes.append(tri.gaps_block)
    if tri.brief is not None and tri.brief.executive_summary.strip():
        prefixes.append(
            f"Account brief (shape gate):\n{tri.brief.executive_summary.strip()[:2000]}"
        )

    if not tri.allowed:
        msg = tri.block_message or "Triangulation gate blocked shape task."
        if tri.gaps_block and tri.gaps_block not in msg:
            msg = f"{msg}\n\n{tri.gaps_block}"
        logger.info(
            "PIPELINE_SHAPE_TRIANGULATION_BLOCK | company=%s gaps=%s",
            co[:40],
            tri.gap_keys,
        )
        return True, [msg], tri

    return False, prefixes, tri


__all__ = ["run_pipeline_shape_triangulation_gate"]
