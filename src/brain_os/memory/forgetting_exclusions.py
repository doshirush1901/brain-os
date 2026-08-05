"""Exclusion rules for Mem0 two-stage archival forgetting.

These rules are the fail-closed allowlist of what must **never** be archived
or hard-deleted. Keep this module the single source of truth; docs should
point here rather than duplicating policy prose.

Never touch
-----------
1. **Mnemon corrections** — ``metadata.type == "correction"`` or
   ``provenance_class == "correction"``.
2. **Preferences** — ``metadata.type == "preference"``.
3. **Verified facts** — ``metadata.verified is True``.
4. **Any recorded access** — hit count > 0 in ``mem0_access_tracker``.
5. **High salience** — ``metadata.salience_score`` ≥
   ``APP__MEM0_FORGET_SALIENCE_PROTECT`` (default 0.35).
6. **Recently reconsolidated** — rewritten within the reconsolidation window
   (checked by the forgetting runner via the access tracker).
7. **Referenced by an active procedure** — memory id or a substantial content
   snippet appears in a procedural memory trigger/steps.
8. **Referenced by an active standing goal** — memory id or content snippet
   appears in an active standing-goal objective.

Age / confidence gates (``too_young``, ``high_confidence``) are separate
candidate filters, not exclusion rules — they simply mean "not yet stale."
"""

from __future__ import annotations

from typing import Any, Final

#: metadata.type values that are never archived/deleted.
PROTECTED_TYPES: Final[frozenset[str]] = frozenset({"correction", "preference"})

#: provenance_class values that are never archived/deleted.
PROTECTED_PROVENANCE: Final[frozenset[str]] = frozenset({"correction"})

#: Minimum content length for substring reference matching against procedures/goals.
_MIN_CONTENT_ANCHOR_CHARS: Final[int] = 40

#: Documented exclusion reason codes (stable for audit JSONL + CLI breakdown).
EXCLUSION_REASONS: Final[tuple[str, ...]] = (
    "protected_type",
    "protected_provenance",
    "verified",
    "accessed",
    "high_salience",
    "referenced_by_procedure",
    "referenced_by_goal",
    "recently_reconsolidated",
)


def exclusion_policy_doc() -> dict[str, Any]:
    """Machine-readable exclusion policy for CLI / ops snapshots."""
    return {
        "never_touch": [
            {
                "code": "protected_type",
                "description": "Mnemon corrections and preferences (metadata.type)",
                "types": sorted(PROTECTED_TYPES),
            },
            {
                "code": "protected_provenance",
                "description": "provenance_class=correction",
                "classes": sorted(PROTECTED_PROVENANCE),
            },
            {
                "code": "verified",
                "description": "metadata.verified is True",
            },
            {
                "code": "accessed",
                "description": "Any recorded hit in mem0_access_tracker",
            },
            {
                "code": "high_salience",
                "description": "salience_score >= APP__MEM0_FORGET_SALIENCE_PROTECT",
            },
            {
                "code": "referenced_by_procedure",
                "description": "Memory id/content referenced by an active procedure",
            },
            {
                "code": "referenced_by_goal",
                "description": "Memory id/content referenced by an active standing goal",
            },
            {
                "code": "recently_reconsolidated",
                "description": "Rewritten inside APP__MEM0_RECONSOLIDATION_WINDOW_HOURS",
            },
        ],
        "stages": {
            "archive": "Move candidate out of live Mem0 into local archive (reversible)",
            "hard_delete": (
                "Permanently drop archive rows after N days still never accessed "
                "(APP__MEM0_FORGET_HARD_DELETE_AFTER_DAYS, default 30)"
            ),
        },
    }


def _meta(memory: dict[str, Any]) -> dict[str, Any]:
    meta = memory.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _content(memory: dict[str, Any]) -> str:
    return str(memory.get("memory", memory.get("content", "")) or "")


def check_hard_exclusions(
    memory: dict[str, Any],
    *,
    access_count: int,
    salience_protect: float,
    procedure_anchors: frozenset[str] | set[str] | None = None,
    goal_anchors: frozenset[str] | set[str] | None = None,
) -> tuple[bool, str]:
    """Return ``(excluded, reason)`` for hard never-touch rules.

    Does **not** evaluate age/confidence candidate gates.
    """
    meta = _meta(memory)
    mem_type = str(meta.get("type", "") or "").strip().lower()
    if mem_type in PROTECTED_TYPES:
        return True, "protected_type"

    provenance = str(meta.get("provenance_class", "") or "").strip().lower()
    if provenance in PROTECTED_PROVENANCE:
        return True, "protected_provenance"

    if meta.get("verified") is True:
        return True, "verified"

    if access_count > 0:
        return True, "accessed"

    try:
        salience = float(meta.get("salience_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        salience = 0.0
    if salience >= float(salience_protect):
        return True, "high_salience"

    mem_id = str(memory.get("id", "") or "").strip().lower()
    content = _content(memory)
    content_l = content.strip().lower()
    anchor_snip = content_l[:200] if len(content_l) >= _MIN_CONTENT_ANCHOR_CHARS else ""

    if procedure_anchors:
        if mem_id and mem_id in procedure_anchors:
            return True, "referenced_by_procedure"
        if anchor_snip and any(anchor_snip in a or mem_id in a for a in procedure_anchors if a):
            return True, "referenced_by_procedure"

    if goal_anchors:
        if mem_id and mem_id in goal_anchors:
            return True, "referenced_by_goal"
        if anchor_snip and any(anchor_snip in a or mem_id in a for a in goal_anchors if a):
            return True, "referenced_by_goal"

    # Explicit metadata links (when writers attach them).
    if meta.get("procedure_id") or meta.get("procedure_ids"):
        return True, "referenced_by_procedure"
    if meta.get("goal_id") or meta.get("standing_goal_id"):
        return True, "referenced_by_goal"

    return False, ""


async def load_procedure_anchors() -> frozenset[str]:
    """Best-effort text anchors from procedural memory (empty on failure)."""
    try:
        from brain_os.memory.procedural import ProceduralMemory

        pm = ProceduralMemory()
        await pm.initialize()
        try:
            procs = await pm.list_procedures(limit=500)
        finally:
            await pm.close()
        anchors: set[str] = set()
        for p in procs:
            blob = " ".join([p.trigger_pattern, *list(p.steps or [])]).strip().lower()
            if blob:
                anchors.add(blob)
        return frozenset(anchors)
    except Exception:
        return frozenset()


async def load_goal_anchors() -> frozenset[str]:
    """Best-effort text anchors from active standing goals (empty on failure)."""
    try:
        from brain_os.systems.standing_goal import StandingGoalStatus, list_standing_goal_snapshots

        anchors: set[str] = set()
        for snap in list_standing_goal_snapshots():
            status = str(snap.get("status") or "").lower()
            # list_standing_goal_snapshots already drops achieved/cleared; keep
            # paused out of "active" protection to avoid hoarding forever.
            if status and status != StandingGoalStatus.ACTIVE.value:
                continue
            obj = str(snap.get("standing_objective") or snap.get("objective") or "").strip().lower()
            if obj:
                anchors.add(obj)
        return frozenset(anchors)
    except Exception:
        return frozenset()
