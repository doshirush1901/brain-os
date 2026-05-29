"""Sync Letta-style memory blocks from relationship + episodic stores."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from brain_os.memory.blocks import BLOCK_SPECS, MemoryBlockStore
from brain_os.memory.relationship import Relationship

if TYPE_CHECKING:
    from brain_os.memory.episodic import EpisodicMemory
    from brain_os.memory.relationship import RelationshipMemory

logger = logging.getLogger(__name__)

_EMAIL_IN_TEXT = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


def scope_from_correction_text(*texts: str) -> str | None:
    """Extract a contact email scope from correction entity/source strings."""
    for text in texts:
        if not text:
            continue
        match = _EMAIL_IN_TEXT.search(text)
        if match:
            return match.group(0).lower()
    return None


def format_human_block(rel: Relationship, *, crm_prefix: str = "") -> str:
    lines: list[str] = []
    if crm_prefix.strip():
        lines.append(crm_prefix.strip())
        lines.append("")
    if rel.learned_preferences:
        lines.append("Preferences:")
        for key, val in list(rel.learned_preferences.items())[:12]:
            lines.append(f"- {key}: {val}")
    if rel.memorable_moments:
        lines.append("Memorable moments:")
        for moment in rel.memorable_moments[-8:]:
            lines.append(f"- {moment}")
    if not lines:
        return "(No profile facts yet.)"
    return "\n".join(lines)


def format_relationship_block(rel: Relationship) -> str:
    warmth = rel.warmth_level.value if hasattr(rel.warmth_level, "value") else str(rel.warmth_level)
    parts = [
        f"Warmth: {warmth}",
        f"Interactions: {rel.interaction_count}",
    ]
    if rel.first_interaction:
        parts.append(f"First contact: {rel.first_interaction.date().isoformat()}")
    if rel.last_interaction:
        parts.append(f"Last contact: {rel.last_interaction.date().isoformat()}")
    return "\n".join(parts)


async def gather_commitments(episodic: EpisodicMemory, scope: str) -> str:
    """Collect recent commitments from episodic rows."""
    assert episodic._db is not None
    cursor = await episodic._db.execute(
        """
        SELECT commitments, created_at FROM episodes
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 8
        """,
        (scope,),
    )
    rows = await cursor.fetchall()
    await cursor.close()
    items: list[str] = []
    for raw_commitments, created_at in rows:
        try:
            parsed = json.loads(raw_commitments) if isinstance(raw_commitments, str) else []
        except json.JSONDecodeError:
            parsed = []
        if not isinstance(parsed, list):
            continue
        for item in parsed:
            text = str(item).strip()
            if text and text not in items:
                items.append(f"- [{created_at}] {text}")
        if len(items) >= 12:
            break
    if not items:
        return "(No open commitments recorded.)"
    return "\n".join(items[:12])


async def build_crm_prefix(crm: Any | None, scope: str) -> str:
    """CRM snapshot for the human block (name, company, deal context)."""
    if crm is None or "@" not in scope:
        return ""
    try:
        contact = await crm.get_contact_by_email(scope.strip().lower())
    except Exception:
        logger.debug("CRM lookup failed for memory block scope %s", scope, exc_info=True)
        return ""
    if contact is None:
        return ""

    lines: list[str] = ["CRM profile:"]
    name = getattr(contact, "name", None)
    if name:
        lines.append(f"- Name: {name}")
    role = getattr(contact, "role", None)
    if role:
        lines.append(f"- Role: {role}")
    company_id = getattr(contact, "company_id", None)
    if company_id:
        try:
            company = await crm.get_company(company_id)
            if company is not None:
                co_name = getattr(company, "name", None)
                if co_name:
                    lines.append(f"- Company: {co_name}")
        except Exception:
            logger.debug("Company lookup failed for %s", company_id, exc_info=True)
    summary = getattr(contact, "account_summary", None)
    if summary and str(summary).strip():
        lines.append(f"- Account summary: {str(summary).strip()[:400]}")
    jtbd = getattr(contact, "job_to_be_done", None)
    if jtbd and str(jtbd).strip():
        lines.append(f"- Job to be done: {str(jtbd).strip()[:200]}")
    buyer = getattr(contact, "buyer_role", None)
    if buyer:
        lines.append(f"- Buyer role: {buyer}")
    return "\n".join(lines)


async def ensure_contact_blocks_rendered(
    store: MemoryBlockStore,
    scope: str,
    *,
    relationship: RelationshipMemory | None = None,
    episodic: EpisodicMemory | None = None,
    crm: Any | None = None,
) -> str:
    """Return pinned block XML, seeding from backing stores when blocks are empty."""
    xml = await store.render_for_scope(scope)
    if xml or relationship is None:
        return xml
    await refresh_contact_blocks(
        store,
        scope,
        relationship=relationship,
        episodic=episodic,
        crm=crm,
    )
    return await store.render_for_scope(scope)


async def refresh_contact_blocks(
    store: MemoryBlockStore,
    scope: str,
    *,
    relationship: RelationshipMemory,
    episodic: EpisodicMemory | None = None,
    crm: Any | None = None,
) -> None:
    """Rebuild all default blocks for a contact from backing stores."""
    rel = await relationship.get_relationship(scope)
    crm_prefix = await build_crm_prefix(crm, scope)
    await store.set_block(scope, "human", format_human_block(rel, crm_prefix=crm_prefix))
    await store.set_block(scope, "relationship", format_relationship_block(rel))

    if episodic is None:
        return

    try:
        timeline = await episodic.weave_episodes(scope)
        if timeline and not timeline.startswith("No episodes"):
            await store.set_block(scope, "timeline", timeline)
    except Exception:
        logger.debug("Timeline block refresh failed for %s", scope, exc_info=True)

    try:
        commitments = await gather_commitments(episodic, scope)
        await store.set_block(scope, "commitments", commitments)
    except Exception:
        logger.debug("Commitments block refresh failed for %s", scope, exc_info=True)


async def merge_preferences_from_facts(
    relationship: RelationshipMemory,
    scope: str,
    facts: list[str],
) -> None:
    """Lightweight preference merge from LEARN fact lines (no extra LLM)."""
    if not facts:
        return
    rel = await relationship.get_relationship(scope)
    changed = False
    for fact in facts[:5]:
        text = fact.strip()
        if len(text) < 12:
            continue
        key = f"fact_{len(rel.learned_preferences) + 1}"
        if text not in rel.learned_preferences.values():
            rel.learned_preferences[key] = text[:200]
            changed = True
    if not changed:
        return
    assert relationship._db is not None
    import json as _json

    first_iso = rel.first_interaction.isoformat() if rel.first_interaction else None
    last_iso = rel.last_interaction.isoformat() if rel.last_interaction else None
    warmth = rel.warmth_level.value if hasattr(rel.warmth_level, "value") else str(rel.warmth_level)
    await relationship._db.execute(
        """
        INSERT OR REPLACE INTO relationships
        (contact_id, warmth_level, interaction_count, memorable_moments, learned_preferences,
         first_interaction, last_interaction)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            scope,
            warmth,
            rel.interaction_count,
            _json.dumps(rel.memorable_moments),
            _json.dumps(rel.learned_preferences),
            first_iso,
            last_iso,
        ),
    )
    await relationship._db.commit()


async def apply_corrections_to_blocks(
    store: MemoryBlockStore,
    corrections: list[dict[str, Any]],
    *,
    default_label: str = "human",
) -> int:
    """Push corrections into contact memory blocks when scope is known."""
    if default_label not in BLOCK_SPECS:
        default_label = "human"
    applied = 0
    for correction in corrections:
        scope = scope_from_correction_text(
            str(correction.get("entity", "")),
            str(correction.get("source", "")),
        )
        new_value = str(correction.get("new_value", "")).strip()
        if not scope or not new_value:
            continue
        label = default_label
        existing = await store.get_block(scope, label)
        prior = (existing.value if existing else "").strip()
        header = f"[Corrected {correction.get('id', '?')}]"
        merged = f"{header} {new_value}"
        if prior:
            merged = f"{merged}\n\n{prior}"
        await store.apply_correction(scope, label, merged)
        applied += 1
    return applied
