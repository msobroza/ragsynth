"""Exact dense retrieval over an in-memory matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ragsynth.adapters.retriever.base import RETRIEVERS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ragsynth.datasets.base import DatasetBundle


@RETRIEVERS.register("dense_inmemory")
class DenseInMemoryRetriever:
    """Exact top-k dense retrieval via matmul (SPEC §12).

    Scores are dot products; with unit-norm rows (the store invariant)
    that is cosine similarity.
    """

    def __init__(self, chunk_ids: Sequence[str], matrix: NDArray[np.float64]) -> None:
        if len(chunk_ids) != matrix.shape[0]:
            raise ValueError(
                f"chunk_ids ({len(chunk_ids)}) and matrix rows ({matrix.shape[0]}) mismatch"
            )
        self.chunk_ids = list(chunk_ids)
        self.matrix = np.asarray(matrix, dtype=np.float64)

    def search(self, query_emb: NDArray[np.float64], k: int) -> list[tuple[str, float]]:
        """Return the exact top-``k`` chunks by dot-product score, best first."""
        scores = self.matrix @ np.asarray(query_emb, dtype=np.float64)
        k_eff = min(k, len(scores))
        top = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.chunk_ids[i], float(scores[i])) for i in top]

    def to_config(self) -> dict[str, Any]:
        """JSON-safe params (the matrix itself is rebuilt from the bundle)."""
        return {}

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> DenseInMemoryRetriever:
        """Build over the bundle's chunk embeddings (composition root fills them first).

        Raises:
            ValueError: If the bundle's embedding store is not populated yet.
        """
        if bundle.embeddings is None:
            raise ValueError("dense_inmemory retriever needs a populated bundle.embeddings store")
        ids = [chunk.chunk_id for chunk in bundle.chunks]
        return cls(chunk_ids=ids, matrix=bundle.embeddings.get(ids).astype(np.float64))
