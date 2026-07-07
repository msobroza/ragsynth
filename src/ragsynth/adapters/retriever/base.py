"""Retriever Protocol and its registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from ragsynth.pipeline.registry import Registry

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class Retriever(Protocol):
    """Embedding -> ranked ``(chunk_id, score)`` list (SPEC §12)."""

    def search(self, query_emb: NDArray[np.float64], k: int) -> list[tuple[str, float]]:
        """Return the top-``k`` chunks by score, best first."""
        ...


RETRIEVERS: Registry[Any] = Registry("retriever")
