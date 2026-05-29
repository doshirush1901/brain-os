"""Typed state bag for the request pipeline (LangGraph-style, framework-free).

``PipelineState`` is the canonical per-turn state object. Phase modules should
accept and return slices of it as the monolith in ``pipeline.py`` is carved up
(see ``docs/PIPELINE_SPLIT_PLAN.md``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PipelineTerminal = Literal["continue", "return", "interrupt"]


class PipelineState(BaseModel):
    """Mutable-in-place pipeline context for one ``process_request`` turn."""

    run_id: str
    raw_input: str
    channel: str
    sender_id: str
    meta: dict[str, Any] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)
    resolved_input: str = ""
    agents_used: list[str] = Field(default_factory=list)
    terminal: PipelineTerminal = "continue"
    early_exit: str | None = None
    route_method: str = ""
    current_node: str = "entry"

    model_config = {"extra": "forbid"}

    @classmethod
    def from_bootstrap(
        cls,
        *,
        run_id: str,
        raw_input: str,
        channel: str,
        sender_id: str,
        meta: dict[str, Any],
        trace: dict[str, Any],
    ) -> PipelineState:
        """Build initial state after ``attach_pipeline_run_id`` / trace init."""
        return cls(
            run_id=run_id,
            raw_input=raw_input,
            channel=channel,
            sender_id=sender_id,
            meta=meta,
            trace=trace,
            resolved_input=raw_input,
        )

    def with_terminal(
        self,
        *,
        terminal: PipelineTerminal,
        early_exit: str | None = None,
        route_method: str | None = None,
        current_node: str | None = None,
        agents_used: list[str] | None = None,
    ) -> PipelineState:
        """Return a copy with terminal / exit metadata (immutable update)."""
        updates: dict[str, Any] = {"terminal": terminal}
        if early_exit is not None:
            updates["early_exit"] = early_exit
        if route_method is not None:
            updates["route_method"] = route_method
        if current_node is not None:
            updates["current_node"] = current_node
        if agents_used is not None:
            updates["agents_used"] = agents_used
        return self.model_copy(update=updates)

    @classmethod
    def from_clarification_checkpoint(
        cls,
        *,
        checkpoint: dict[str, Any],
        clarification_answer: str,
        run_id: str,
        channel: str,
        sender_id: str,
        meta: dict[str, Any],
        trace: dict[str, Any],
    ) -> PipelineState:
        """Rehydrate state after a clarification interrupt (merged query)."""
        from brain_os.pipeline_checkpoint import merge_clarification_turn

        original = str(checkpoint.get("original_query") or checkpoint.get("raw_input") or "")
        merged = merge_clarification_turn(
            original_query=original,
            answer=clarification_answer,
        )
        return cls(
            run_id=run_id,
            raw_input=clarification_answer,
            channel=channel,
            sender_id=sender_id,
            meta=meta,
            trace=trace,
            resolved_input=merged,
            agents_used=list(checkpoint.get("agents_used") or []),
            route_method=str(checkpoint.get("route_method") or ""),
            current_node="route",
        )


def merge_agents_used(existing: list[str], added: list[str]) -> list[str]:
    """Reducer: append agent names preserving first-seen order."""
    seen = set(existing)
    out = list(existing)
    for name in added:
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def merge_trace_fields(trace: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """Reducer: shallow-merge observability fields into ``trace``."""
    merged = dict(trace)
    merged.update(fields)
    return merged
