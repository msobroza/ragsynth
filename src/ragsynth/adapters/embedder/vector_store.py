"""Vector-store abstraction the embedding cache writes through (DIP).

:class:`~ragsynth.adapters.embedder.cached_embedder.CachedEmbedder` depends only
on this interface; concrete stores (ChromaDB today) implement it. Adding a new
store never touches the cache (OCP): implement the two methods plus ``location``
and register a composition class like ``cached_chroma``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray


class EmbeddingVectorStore(ABC):
    """Keyed, idempotent vector persistence for the embedding cache.

    Contract: ``fetch_vectors`` is order-agnostic (a mapping, never a positional
    list — stores may reorder), returns only the ids it holds, and returns
    exactly what ``store_vectors`` persisted (so a repeat ``encode`` on the same
    texts is byte-identical even if the store narrows precision, e.g. Chroma's
    float32). ``store_vectors`` has upsert semantics.
    """

    @property
    @abstractmethod
    def location(self) -> str:
        """Human-readable store location (e.g. collection name) for error messages."""

    @abstractmethod
    def fetch_vectors(self, ids: Sequence[str]) -> dict[str, NDArray[np.float64]]:
        """Return ``id -> vector`` for every id in ``ids`` present in the store."""

    @abstractmethod
    def store_vectors(
        self,
        ids: Sequence[str],
        vectors: NDArray[np.float64],
        documents: Sequence[str],
        metadata: Mapping[str, str],
    ) -> None:
        """Upsert ``vectors`` under ``ids`` with source ``documents`` and shared metadata."""
