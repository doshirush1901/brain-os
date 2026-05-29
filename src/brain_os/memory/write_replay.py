"""Mem0 write-contract replay handler (registers against ``brain_os.contracts.write_receipt``)."""

from __future__ import annotations

from typing import Any

from brain_os.contracts.write_receipt import register_write_replay_handler


async def replay_mem0_store(row: dict[str, Any]) -> bool:
    meta = dict(row.get("metadata") or {})
    body = (meta.get("replay_content") or "").strip()
    if not body:
        return False
    uid = str(meta.get("user_id") or "global").strip() or "global"
    outbound_meta = {k: v for k, v in meta.items() if k not in {"replay_content"}}
    from brain_os.memory.long_term import LongTermMemory

    mem = LongTermMemory()
    out = await mem.store(
        body, user_id=uid, metadata=outbound_meta, run_id=str(row.get("run_id") or "")
    )
    return isinstance(out, list) and len(out) > 0


def install_mem0_write_replay_handler() -> None:
    register_write_replay_handler("mem0", "store", replay_mem0_store)


install_mem0_write_replay_handler()
