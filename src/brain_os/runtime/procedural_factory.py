"""Factory for ProceduralMemory with Voyage embedder injected (memory↛brain)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain_os.memory.procedural import ProceduralMemory


def build_procedural_memory(db_path: str | None = None) -> ProceduralMemory:
    """Construct ProceduralMemory with EmbeddingService when available.

    Lives in ``brain_os.runtime`` so ``brain_os.memory`` never imports ``brain_os.brain``.
    Embedding failures at match time degrade to Jaccard inside ProceduralMemory.
    """
    from brain_os.memory.procedural import ProceduralMemory

    embedder = None
    try:
        from brain_os.brain.embeddings import EmbeddingService

        embedder = EmbeddingService()
    except Exception:
        embedder = None
    if db_path is None:
        return ProceduralMemory(embedder=embedder)
    return ProceduralMemory(db_path=db_path, embedder=embedder)
