"""Closed-loop weight proposals from Sophia lessons and operator tune CLI.

Lessons propose adjustments; operators approve before any weights JSON changes.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from brain_os.config import get_settings
from brain_os.contracts.learning_paths import (
    prediction_tuning_log_path,
    predictions_tuning_dir,
)
from brain_os.contracts.learning_paths import (
    weight_proposals_path as contract_weight_proposals_path,
)
from brain_os.memory.prediction_log import (
    HOT_LEAD_BOARD_PATTERN,
    MUST_ACT_TODAY_PATTERN,
    TYCHE_DEAL_FORECAST_PATTERN,
    PredictionLog,
)
from brain_os.schemas.llm_outputs import PredictionReflection
from brain_os.systems.data_dir_lock import get_data_dir

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_DRIFT_ACCURACY_FLOOR = 0.55
_BOARD_HEALTH_FLOOR = 0.60
_WEIGHT_DELTA_UNIT = 0.03
_WEIGHT_DELTA_SCORE = 5.0

_OVER_PRED_RE = re.compile(
    r"(?i)(?:pattern\s+)?[\"']?(.{3,80}?)[\"']?\s+(?:consistently\s+)?over[- ]?predicts"
)
_UNDER_PRED_RE = re.compile(
    r"(?i)(?:pattern\s+)?[\"']?(.{3,80}?)[\"']?\s+(?:consistently\s+)?under[- ]?predicts"
)
_CONVERTS_BETTER_RE = re.compile(
    r"(?i)(?:corridor\s+)?([a-z][a-z0-9_\- /]{2,40}?)\s+converts better(?:\s+than scored)?"
)
_SCORE_TOO_HIGH_RE = re.compile(
    r"(?i)\b(S_[a-z_]+|[a-z][a-z0-9_]{2,30})\b\s+(?:scores?|weights?)\s+too\s+high"
)
_SCORE_TOO_LOW_RE = re.compile(
    r"(?i)\b(S_[a-z_]+|[a-z][a-z0-9_]{2,30})\b\s+(?:scores?|weights?)\s+too\s+low"
)

PREDICTION_PATTERN_LABELS: dict[str, str] = {
    HOT_LEAD_BOARD_PATTERN: "hot-lead-board",
    MUST_ACT_TODAY_PATTERN: "must-act-today",
    TYCHE_DEAL_FORECAST_PATTERN: "deal-forecast",
}

ProposalStatus = Literal["pending", "approved", "rejected", "deferred"]


@dataclass(frozen=True, slots=True)
class WeightTarget:
    weights_file: str
    json_path: tuple[str, ...]
    scale: Literal["unit", "score"] = "unit"


def predictions_dir() -> Path:
    """``{data}/predictions`` — weight proposals and tuning audit."""
    return predictions_tuning_dir()


def weight_proposals_path() -> Path:
    return contract_weight_proposals_path()


def tuning_log_path() -> Path:
    return prediction_tuning_log_path()


def morning_brain_inbox_path() -> Path:
    rel = (get_settings().app.morning_brain_inbox_path or "").strip()
    path = Path(rel) if rel else get_data_dir() / "operations" / "morning_brain_inbox.json"
    if not path.is_absolute():
        path = get_data_dir() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_weights_path(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    legacy = get_data_dir() / rel
    if legacy.is_file():
        return legacy
    repo = _REPO_ROOT / rel
    return repo if repo.is_file() else legacy


def _normalize_token(raw: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (raw or "").strip().lower()).strip("_")


def _weight_targets_for_token(token: str) -> list[WeightTarget]:
    t = _normalize_token(token)
    if not t:
        return []

    na_path = "data/knowledge/na_matchmaker_weights.json"
    fit_path = "data/knowledge/customer_profile_formula_weights.json"

    subscores = {
        "s_process": ("weights", "S_process"),
        "process": ("weights", "S_process"),
        "s_scale": ("weights", "S_scale"),
        "scale": ("weights", "S_scale"),
        "s_cohort": ("weights", "S_cohort"),
        "cohort": ("weights", "S_cohort"),
        "s_geo": ("weights", "S_geo"),
        "geo": ("weights", "S_geo"),
    }
    if t in subscores:
        return [WeightTarget(na_path, subscores[t], scale="unit")]

    corridors = {"great_lakes", "midwest", "on_qc", "default"}
    if t in corridors:
        return [WeightTarget(na_path, ("corridor_geo_scores", t), scale="score")]

    gauge = {
        "heavy_gauge": ("gauge_tier_scores", "heavy_gauge"),
        "thin_gauge": ("gauge_tier_scores", "thin_gauge"),
        "mixed": ("gauge_tier_scores", "mixed"),
    }
    if t in gauge:
        return [WeightTarget(na_path, gauge[t], scale="score")]

    prediction_map: dict[str, WeightTarget] = {
        "hot_lead_board": WeightTarget(
            fit_path, ("math_mode_bridge", "relationship", "mail_threads"), scale="unit"
        ),
        "hot_lead": WeightTarget(
            fit_path, ("math_mode_bridge", "relationship", "mail_threads"), scale="unit"
        ),
        "hot_lead_board_pattern": WeightTarget(
            fit_path, ("math_mode_bridge", "relationship", "mail_threads"), scale="unit"
        ),
        "must_act_today": WeightTarget(
            fit_path, ("math_mode_bridge", "momentum", "mail_threads"), scale="unit"
        ),
        "must_act": WeightTarget(
            fit_path, ("math_mode_bridge", "momentum", "mail_threads"), scale="unit"
        ),
        "tyche_deal_forecast": WeightTarget(
            fit_path, ("math_mode_bridge", "intent", "project_count"), scale="unit"
        ),
        "deal_forecast": WeightTarget(
            fit_path, ("math_mode_bridge", "intent", "project_count"), scale="unit"
        ),
    }
    if t in prediction_map:
        return [prediction_map[t]]

    if t.startswith("s_") and len(t) > 2:
        key = t.upper().replace("S_", "S_") if t.startswith("s_") else t
        if t.startswith("s_"):
            key = "S_" + t[2:].upper()
        return [WeightTarget(na_path, ("weights", key), scale="unit")]

    return []


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return data


def _get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any | None:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: float) -> None:
    cur: Any = data
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _proposed_value(
    current: float, direction: Literal["increase", "decrease"], scale: str
) -> float:
    if scale == "score":
        delta = _WEIGHT_DELTA_SCORE if direction == "increase" else -_WEIGHT_DELTA_SCORE
        return round(max(0.0, min(100.0, current + delta)), 2)
    delta = _WEIGHT_DELTA_UNIT if direction == "increase" else -_WEIGHT_DELTA_UNIT
    return round(max(0.0, min(1.0, current + delta)), 4)


def _confidence_from_evidence(*, accuracy: float | None, sample_size: int) -> float:
    base = 0.45
    if accuracy is not None:
        base += min(0.25, abs(0.5 - accuracy))
    if sample_size >= 10:
        base += 0.15
    elif sample_size >= 5:
        base += 0.08
    return round(min(0.95, base), 2)


def _parse_lesson_direction(lesson: str) -> tuple[str, Literal["increase", "decrease"]] | None:
    text = (lesson or "").strip()
    if not text:
        return None
    for regex, direction in (
        (_OVER_PRED_RE, "decrease"),
        (_UNDER_PRED_RE, "increase"),
        (_CONVERTS_BETTER_RE, "increase"),
        (_SCORE_TOO_HIGH_RE, "decrease"),
        (_SCORE_TOO_LOW_RE, "increase"),
    ):
        match = regex.search(text)
        if match:
            return match.group(1).strip(), direction
    lower = text.lower()
    if "over-predict" in lower or "over predict" in lower or "false positive" in lower:
        token = text.split(" over", 1)[0].strip(" .\"'")
        if len(token) >= 3:
            return token, "decrease"
    if "under-predict" in lower or "under predict" in lower:
        token = text.split(" under", 1)[0].strip(" .\"'")
        if len(token) >= 3:
            return token, "increase"
    if "converts better" in lower:
        m = _CONVERTS_BETTER_RE.search(text)
        if m:
            return m.group(1).strip(), "increase"
    return None


def _existing_proposal_keys() -> set[str]:
    keys: set[str] = set()
    path = weight_proposals_path()
    if not path.is_file():
        return keys
    for row in _read_jsonl(path):
        if str(row.get("status") or "") in {"pending", "deferred"}:
            keys.add(
                f"{row.get('pattern')}|{row.get('weights_file')}|{row.get('weight_key')}|"
                f"{row.get('direction')}"
            )
    return keys


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def _update_proposal_status(
    proposal_id: str, status: ProposalStatus, **extra: Any
) -> dict[str, Any] | None:
    path = weight_proposals_path()
    if not path.is_file():
        return None
    rows = _read_jsonl(path)
    updated: dict[str, Any] | None = None
    for row in rows:
        if str(row.get("proposal_id") or "") != proposal_id:
            continue
        row["status"] = status
        row["updated_at"] = datetime.now(UTC).isoformat()
        row.update(extra)
        updated = row
        break
    if updated is None:
        return None
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    return updated


async def rolling_accuracy_by_pattern(
    log: PredictionLog,
    *,
    days: int = 30,
    min_predictions: int = 3,
) -> dict[str, dict[str, Any]]:
    """Rolling window accuracy with optional prior-window trend."""
    return await log.accuracy_by_pattern_window(
        days=days,
        min_predictions=min_predictions,
    )


def build_weight_proposal(
    *,
    pattern: str,
    target: WeightTarget,
    direction: Literal["increase", "decrease"],
    evidence: dict[str, Any],
    lesson: str = "",
    source: str = "sophia_lesson",
) -> dict[str, Any] | None:
    path = _resolve_weights_path(target.weights_file)
    if not path.is_file():
        logger.debug("Weights file missing for proposal: %s", path)
        return None
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    current = _get_nested(data, target.json_path)
    if not isinstance(current, (int, float)):
        return None
    current_f = float(current)
    proposed = _proposed_value(current_f, direction, target.scale)
    if abs(proposed - current_f) < 1e-6:
        return None
    sample_size = int(evidence.get("sample_size") or 0)
    accuracy = evidence.get("accuracy")
    acc_f = float(accuracy) if isinstance(accuracy, (int, float)) else None
    return {
        "proposal_id": f"wp_{uuid.uuid4().hex[:10]}",
        "status": "pending",
        "pattern": pattern,
        "weight_key": ".".join(target.json_path),
        "weights_file": target.weights_file,
        "direction": direction,
        "current_weight": current_f,
        "proposed_weight": proposed,
        "confidence": _confidence_from_evidence(accuracy=acc_f, sample_size=sample_size),
        "evidence": evidence,
        "lesson": lesson[:400],
        "source": source,
        "created_at": datetime.now(UTC).isoformat(),
    }


async def extract_weight_proposals_from_lessons(
    *,
    lessons: list[str],
    reflection: PredictionReflection | None,
    log: PredictionLog,
    reconciliation_accuracy: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Parse Sophia lessons (+ false-positive patterns) into queued weight proposals."""
    rolling = await rolling_accuracy_by_pattern(log, days=30, min_predictions=1)
    seen = _existing_proposal_keys()
    proposals: list[dict[str, Any]] = []

    recon_acc = reconciliation_accuracy or {}
    aliases = prediction_pattern_aliases()
    for lesson in lessons:
        parsed = _parse_lesson_direction(lesson)
        if not parsed:
            continue
        token, direction = parsed
        targets = _weight_targets_for_token(token)
        pattern_label = _normalize_token(token) or token[:80]
        roll_key = aliases.get(pattern_label, pattern_label)
        roll = rolling.get(roll_key) or rolling.get(pattern_label) or {}
        evidence = {
            "accuracy": roll.get("accuracy")
            or recon_acc.get(roll_key)
            or recon_acc.get(pattern_label),
            "sample_size": roll.get("sample_size") or 0,
            "period": roll.get("period") or "30d",
            "trend": roll.get("trend"),
        }
        for target in targets:
            key = f"{pattern_label}|{target.weights_file}|{'.'.join(target.json_path)}|{direction}"
            if key in seen:
                continue
            proposal = build_weight_proposal(
                pattern=pattern_label,
                target=target,
                direction=direction,
                evidence=evidence,
                lesson=lesson,
            )
            if proposal:
                proposals.append(proposal)
                seen.add(key)

    if reflection:
        for fp in reflection.false_positive_patterns:
            parsed = _parse_lesson_direction(fp) or (fp, "decrease")
            token = parsed[0] if isinstance(parsed, tuple) else fp
            direction: Literal["increase", "decrease"] = (
                parsed[1] if isinstance(parsed, tuple) and len(parsed) > 1 else "decrease"
            )
            for target in _weight_targets_for_token(token):
                pattern_label = _normalize_token(token) or "hot_lead_board"
                roll = rolling.get(HOT_LEAD_BOARD_PATTERN, {})
                evidence = {
                    "accuracy": roll.get("accuracy"),
                    "sample_size": roll.get("sample_size") or 0,
                    "period": roll.get("period") or "30d",
                    "trend": roll.get("trend"),
                }
                key = f"{pattern_label}|{target.weights_file}|{'.'.join(target.json_path)}|{direction}"
                if key in seen:
                    continue
                proposal = build_weight_proposal(
                    pattern=pattern_label,
                    target=target,
                    direction=direction,
                    evidence=evidence,
                    lesson=fp,
                    source="false_positive_pattern",
                )
                if proposal:
                    proposals.append(proposal)
                    seen.add(key)

    return proposals


def prediction_pattern_aliases() -> dict[str, str]:
    return {
        "hot_lead": HOT_LEAD_BOARD_PATTERN,
        "hot_lead_board": HOT_LEAD_BOARD_PATTERN,
        "must_act": MUST_ACT_TODAY_PATTERN,
        "must_act_today": MUST_ACT_TODAY_PATTERN,
        "deal_forecast": TYCHE_DEAL_FORECAST_PATTERN,
        "tyche": TYCHE_DEAL_FORECAST_PATTERN,
    }


def append_weight_proposals(proposals: list[dict[str, Any]]) -> int:
    if not proposals:
        return 0
    path = weight_proposals_path()
    for proposal in proposals:
        _append_jsonl(path, proposal)
    logger.info("Queued %d weight proposal(s) at %s", len(proposals), path)
    return len(proposals)


async def detect_prediction_drift(
    log: PredictionLog,
    *,
    floor: float = _DRIFT_ACCURACY_FLOOR,
    days: int = 30,
    min_predictions: int = 3,
) -> list[dict[str, Any]]:
    """Patterns whose rolling accuracy is below *floor*."""
    rolling = await rolling_accuracy_by_pattern(log, days=days, min_predictions=min_predictions)
    alerts: list[dict[str, Any]] = []
    for pattern, stats in rolling.items():
        accuracy = float(stats.get("accuracy") or 0.0)
        if accuracy >= floor:
            continue
        label = PREDICTION_PATTERN_LABELS.get(pattern, pattern.replace("_", "-"))
        trend = stats.get("trend")
        trend_txt = f"{float(trend):+.0%}" if isinstance(trend, (int, float)) else "unknown"
        alerts.append(
            {
                "pattern": pattern,
                "pattern_label": label,
                "accuracy": accuracy,
                "sample_size": int(stats.get("sample_size") or 0),
                "period": stats.get("period") or f"{days}d",
                "trend": trend,
                "suggested_investigation": (
                    f"Review {label} scoring inputs and recent false positives; "
                    f"30d accuracy {accuracy:.0%} (trend {trend_txt}, n={stats.get('sample_size')}). "
                    "Run `brain predictions tune` for pending weight proposals."
                ),
            }
        )
    return alerts


def enqueue_prediction_drift_for_morning_brain(alerts: list[dict[str, Any]]) -> int:
    """Persist drift alerts for the next Morning Brain run."""
    if not alerts:
        return 0
    path = morning_brain_inbox_path()
    payload: dict[str, Any] = {"prediction_drift": [], "updated_at": datetime.now(UTC).isoformat()}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload = existing
        except (OSError, json.JSONDecodeError):
            pass
    queue = list(payload.get("prediction_drift") or [])
    existing_ids = {str(a.get("alert_id") or "") for a in queue if isinstance(a, dict)}
    added = 0
    for alert in alerts:
        alert_id = f"drift_{alert.get('pattern')}_{alert.get('period', '30d')}"
        if alert_id in existing_ids:
            continue
        queue.append(
            {
                "alert_id": alert_id,
                "action_type": "prediction_drift",
                "consumed": False,
                "created_at": datetime.now(UTC).isoformat(),
                **alert,
            }
        )
        existing_ids.add(alert_id)
        added += 1
    payload["prediction_drift"] = queue[-50:]
    payload["updated_at"] = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return added


def load_pending_drift_inbox() -> list[dict[str, Any]]:
    path = morning_brain_inbox_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = data.get("prediction_drift") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict) and not r.get("consumed")]


def mark_drift_inbox_consumed(alert_ids: list[str]) -> None:
    if not alert_ids:
        return
    path = morning_brain_inbox_path()
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    ids = set(alert_ids)
    rows = data.get("prediction_drift") or []
    for row in rows:
        if isinstance(row, dict) and str(row.get("alert_id") or "") in ids:
            row["consumed"] = True
            row["consumed_at"] = datetime.now(UTC).isoformat()
    data["updated_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def list_weight_proposals(*, status: ProposalStatus | None = "pending") -> list[dict[str, Any]]:
    rows = _read_jsonl(weight_proposals_path())
    if status is None:
        return rows
    return [r for r in rows if str(r.get("status") or "pending") == status]


def append_tuning_log(entry: dict[str, Any]) -> None:
    row = {"logged_at": datetime.now(UTC).isoformat(), **entry}
    _append_jsonl(tuning_log_path(), row)


def apply_weight_proposal(proposal_id: str) -> dict[str, Any]:
    """Apply an approved proposal to its weights JSON (operator gate)."""
    rows = _read_jsonl(weight_proposals_path())
    proposal = next((r for r in rows if str(r.get("proposal_id") or "") == proposal_id), None)
    if proposal is None:
        return {"ok": False, "error": "proposal_not_found", "proposal_id": proposal_id}
    if str(proposal.get("status") or "") != "pending":
        return {
            "ok": False,
            "error": "invalid_status",
            "status": proposal.get("status"),
            "proposal_id": proposal_id,
        }

    rel = str(proposal.get("weights_file") or "")
    path = _resolve_weights_path(rel)
    if not path.is_file():
        return {"ok": False, "error": "weights_file_missing", "path": str(path)}

    weight_key = str(proposal.get("weight_key") or "")
    json_path = tuple(weight_key.split(".")) if weight_key else ()
    if not json_path:
        return {"ok": False, "error": "missing_weight_key", "proposal_id": proposal_id}

    data = _read_json(path)
    previous = _get_nested(data, json_path)
    proposed = proposal.get("proposed_weight")
    if not isinstance(proposed, (int, float)):
        return {"ok": False, "error": "invalid_proposed_weight", "proposal_id": proposal_id}

    backup_dir = predictions_dir() / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"{path.stem}_{stamp}.json"
    backup_path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    _set_nested(data, json_path, float(proposed))
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    updated = _update_proposal_status(
        proposal_id,
        "approved",
        applied_at=datetime.now(UTC).isoformat(),
        applied_path=str(path),
        backup_path=str(backup_path),
    )
    log_entry = {
        "action": "approve",
        "proposal_id": proposal_id,
        "pattern": proposal.get("pattern"),
        "weights_file": rel,
        "weight_key": weight_key,
        "previous_weight": previous,
        "applied_weight": proposed,
        "backup_path": str(backup_path),
    }
    append_tuning_log(log_entry)
    return {
        "ok": True,
        "proposal": updated,
        "backup_path": str(backup_path),
        "applied_path": str(path),
    }


def reject_weight_proposal(proposal_id: str, *, reason: str = "") -> dict[str, Any]:
    updated = _update_proposal_status(
        proposal_id,
        "rejected",
        reject_reason=reason[:240] if reason else "",
    )
    if updated is None:
        return {"ok": False, "error": "proposal_not_found", "proposal_id": proposal_id}
    append_tuning_log({"action": "reject", "proposal_id": proposal_id, "reason": reason[:240]})
    return {"ok": True, "proposal": updated}


def defer_weight_proposal(proposal_id: str) -> dict[str, Any]:
    updated = _update_proposal_status(proposal_id, "deferred")
    if updated is None:
        return {"ok": False, "error": "proposal_not_found", "proposal_id": proposal_id}
    append_tuning_log({"action": "defer", "proposal_id": proposal_id})
    return {"ok": True, "proposal": updated}


def prediction_health_snapshot(*, drift_floor: float = _BOARD_HEALTH_FLOOR) -> dict[str, Any]:
    """Accuracy rollup for board brief and operator dashboards."""
    import asyncio

    from brain_os.contracts.learning_paths import predictions_db_path

    async def _load() -> tuple[dict[str, float], dict[str, Any], dict[str, dict[str, Any]]]:
        db_path = predictions_db_path()
        if not db_path.is_file():
            return {}, {}, {}
        log = PredictionLog(db_path)
        await log.initialize()
        try:
            by_pattern = await log.accuracy_by_pattern(min_predictions=1)
            overall = await log.overall_accuracy()
            rolling = await rolling_accuracy_by_pattern(log, days=30, min_predictions=1)
            return by_pattern, overall, rolling
        finally:
            await log.close()

    by_pattern, overall, rolling = asyncio.run(_load())
    patterns = [
        (HOT_LEAD_BOARD_PATTERN, "hot-lead-board", by_pattern.get(HOT_LEAD_BOARD_PATTERN)),
        (MUST_ACT_TODAY_PATTERN, "must-act-today", by_pattern.get(MUST_ACT_TODAY_PATTERN)),
        (TYCHE_DEAL_FORECAST_PATTERN, "deal-forecast", by_pattern.get(TYCHE_DEAL_FORECAST_PATTERN)),
    ]

    rows: list[dict[str, Any]] = []
    below_floor: list[str] = []
    for pattern_id, label, all_time in patterns:
        roll = rolling.get(pattern_id) or {}
        accuracy = roll.get("accuracy") if roll else all_time
        sample = int(roll.get("sample_size") or 0)
        correct = int(roll.get("correct") or 0)
        if accuracy is None and all_time is not None:
            accuracy = all_time
        acc_f = float(accuracy) if isinstance(accuracy, (int, float)) else None
        if sample > 0 and correct == 0 and acc_f is not None:
            correct = round(acc_f * sample)
        row = {
            "pattern": label,
            "pattern_id": pattern_id,
            "accuracy": acc_f,
            "sample_size": sample,
            "correct": correct if sample > 0 else None,
            "period": roll.get("period") or "all_time",
            "trend": roll.get("trend"),
            "below_floor": acc_f is not None and acc_f < drift_floor,
        }
        rows.append(row)
        if row["below_floor"]:
            below_floor.append(label)

    return {
        "patterns": rows,
        "below_floor": below_floor,
        "floor": drift_floor,
        "pending_proposals": len(list_weight_proposals(status="pending")),
        "overall": overall,
    }


async def process_reconciliation_weight_loop(
    *,
    lessons: list[str],
    reflection: PredictionReflection | None,
    log: PredictionLog,
    reconciliation_accuracy: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Post-Sophia hook: proposals queue + drift inbox (no auto-apply)."""
    proposals = await extract_weight_proposals_from_lessons(
        lessons=lessons,
        reflection=reflection,
        log=log,
        reconciliation_accuracy=reconciliation_accuracy,
    )
    written = append_weight_proposals(proposals)
    drift = await detect_prediction_drift(log)
    drift_enqueued = enqueue_prediction_drift_for_morning_brain(drift)
    return {
        "weight_proposals_written": written,
        "drift_alerts": drift,
        "drift_enqueued": drift_enqueued,
    }
