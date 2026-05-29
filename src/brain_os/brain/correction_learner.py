"""Maintains learned lists of competitors, customers, and prospects from corrections.

Parses natural-language corrections to build up entity lists and
name/price mappings that other agents can query at runtime.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_PATH = Path("data/brain/learned_corrections.json")

_EMPTY_STATE: dict = {
    "competitors": [],
    "customers": [],
    "prospects": [],
    "entity_corrections": {},
    "price_corrections": [],
}


class CorrectionLearner:
    """Accumulates structured knowledge from free-text corrections."""

    def __init__(self, data_path: Path | None = None) -> None:
        self._path = data_path or _DATA_PATH
        self._state: dict = json.loads(json.dumps(_EMPTY_STATE))

    # ── public API ───────────────────────────────────────────────────────

    async def learn_from_correction(self, correction_text: str) -> dict:
        """Parse *correction_text* and update internal state.

        Returns a summary dict describing what was learned.
        """
        learned: dict[str, list[str]] = {
            "competitors_added": [],
            "customers_added": [],
            "prospects_added": [],
            "entity_corrections_added": [],
            "price_corrections_added": [],
        }

        self._detect_entity_role(correction_text, learned)
        self._detect_entity_rename(correction_text, learned)
        self._detect_price_correction(correction_text, learned)

        if any(learned.values()):
            await self._save()
            logger.info("Learned from correction: %s", learned)

        return learned

    def is_competitor(self, name: str) -> bool:
        return self._normalised_in(name, self._state["competitors"])

    def is_customer(self, name: str) -> bool:
        return self._normalised_in(name, self._state["customers"])

    def get_entity_correction(self, entity: str) -> str | None:
        key = entity.strip().lower()
        for old, new in self._state["entity_corrections"].items():
            if old.lower() == key:
                return new
        return None

    def get_all_learned(self) -> dict:
        return dict(self._state)

    # ── pattern detectors ────────────────────────────────────────────────

    def _detect_entity_role(self, text: str, learned: dict[str, list[str]]) -> None:
        role_patterns: list[tuple[str, str, str]] = [
            (r"(.+?)\s+is\s+a\s+competitor", "competitors", "competitors_added"),
            (r"(.+?)\s+is\s+a\s+customer", "customers", "customers_added"),
            (r"(.+?)\s+is\s+a\s+prospect", "prospects", "prospects_added"),
            (r"(.+?)\s+are\s+competitors", "competitors", "competitors_added"),
            (r"(.+?)\s+are\s+customers", "customers", "customers_added"),
            (r"(.+?)\s+are\s+prospects", "prospects", "prospects_added"),
        ]
        for pattern, list_key, learned_key in role_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                name = m.group(1).strip().strip("\"'")
                if not self._normalised_in(name, self._state[list_key]):
                    self._state[list_key].append(name)
                    learned[learned_key].append(name)

    def _detect_entity_rename(self, text: str, learned: dict[str, list[str]]) -> None:
        rename_patterns = [
            r"(.+?)\s+is\s+not\s+(.+?)[,;.]\s*it'?s\s+(.+)",
            r"(.+?)\s+is\s+not\s+(.+?)[,;.]\s*(?:it\s+is|they\s+are)\s+(.+)",
            r"not\s+(.+?)[,;.]\s*(?:it'?s|it\s+is)\s+(.+)",
        ]
        for pattern in rename_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                groups = m.groups()
                if len(groups) == 3:
                    old_val = groups[1].strip().rstrip(".")
                    new_val = groups[2].strip().rstrip(".")
                elif len(groups) == 2:
                    old_val = groups[0].strip().rstrip(".")
                    new_val = groups[1].strip().rstrip(".")
                else:
                    continue
                self._state["entity_corrections"][old_val] = new_val
                learned["entity_corrections_added"].append(f"{old_val} -> {new_val}")
                break

    def _detect_price_correction(self, text: str, learned: dict[str, list[str]]) -> None:
        price_patterns = [
            r"the\s+price\s+is\s+(.+?)(?:\.|$)",
            r"price\s+should\s+be\s+(.+?)(?:\.|$)",
            r"correct\s+price\s+is\s+(.+?)(?:\.|$)",
            r"costs?\s+(.+?)(?:\s+not\s+.+)?(?:\.|$)",
        ]
        for pattern in price_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                price_info = m.group(1).strip()
                if price_info and price_info not in self._state["price_corrections"]:
                    self._state["price_corrections"].append(price_info)
                    learned["price_corrections_added"].append(price_info)
                break

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalised_in(name: str, items: list[str]) -> bool:
        key = name.strip().lower()
        return any(item.lower() == key for item in items)

    async def on_knowledge_corrected(self, event: object) -> None:
        """DataEventBus handler — reload learned corrections from disk."""
        logger.info("CorrectionLearner: reloading after KNOWLEDGE_CORRECTED event")
        self._state = await self._load()

    async def _load(self) -> dict:
        if self._path.exists():
            try:
                raw = await asyncio.to_thread(self._path.read_text)
                data = json.loads(raw)
                for key in _EMPTY_STATE:
                    data.setdefault(key, _EMPTY_STATE[key])
                return data
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not load corrections; starting fresh")
        return json.loads(json.dumps(_EMPTY_STATE))

    async def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self._state, indent=2)
            await asyncio.to_thread(self._path.write_text, payload)
        except OSError:
            logger.exception("Failed to persist learned corrections")
