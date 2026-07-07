"""Lexical BM25 retriever over the ``bm25s`` optional extra (SPEC §12, §3.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragsynth.adapters.retriever.base import RETRIEVERS
from ragsynth.optional_deps import require_optional

try:
    import bm25s
except ImportError:  # pragma: no cover - exercised via require_optional tests
    bm25s = None

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from ragsynth.datasets.base import DatasetBundle

_DEFAULT_K1 = 1.5
_DEFAULT_B = 0.75


@RETRIEVERS.register("bm25s")
class BM25sRetriever:
    """BM25 retrieval via the ``bm25s`` package (``uv sync --extra bm25``).

    v1 limitation: the :class:`~ragsynth.adapters.retriever.base.Retriever`
    Protocol's ``search`` takes an *embedding*, but BM25 scores *text* --
    so this adapter exposes ``search_text(query_text, k)`` and raises
    ``NotImplementedError`` from ``search``. Wire lexical zoo systems via
    ``search_text``; dense retrievers serve the embedding path.
    """

    def __init__(
        self,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        k1: float = _DEFAULT_K1,
        b: float = _DEFAULT_B,
    ) -> None:
        require_optional(bm25s, "BM25sRetriever", "bm25")
        if len(chunk_ids) != len(texts):
            raise ValueError(f"chunk_ids ({len(chunk_ids)}) and texts ({len(texts)}) mismatch")
        self.chunk_ids = list(chunk_ids)
        self.k1 = k1
        self.b = b
        self._index = bm25s.BM25(k1=k1, b=b)
        self._index.index(bm25s.tokenize(list(texts), show_progress=False))

    def search(self, query_emb: NDArray[np.float64], k: int) -> list[tuple[str, float]]:
        """Unsupported embedding entrypoint (see class docstring).

        Raises:
            NotImplementedError: Always; BM25 is lexical, not dense.
        """
        raise NotImplementedError(
            "BM25sRetriever requires text queries; wire via search_text -- "
            "dense retrievers serve the embedding path"
        )

    def search_text(self, query_text: str, k: int) -> list[tuple[str, float]]:
        """Return the top-``k`` chunks for a *text* query, best first."""
        k_eff = min(k, len(self.chunk_ids))
        indices, scores = self._index.retrieve(
            bm25s.tokenize([query_text], show_progress=False), k=k_eff, show_progress=False
        )
        return [
            (self.chunk_ids[int(idx)], float(score))
            for idx, score in zip(indices[0], scores[0], strict=True)
        ]

    def to_config(self) -> dict[str, Any]:
        """JSON-safe params (the index is rebuilt from the bundle's chunks)."""
        return {"k1": self.k1, "b": self.b}

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> BM25sRetriever:
        """Build the index over the bundle's chunk texts."""
        return cls(
            chunk_ids=[chunk.chunk_id for chunk in bundle.chunks],
            texts=[chunk.text for chunk in bundle.chunks],
            k1=float(params.get("k1", _DEFAULT_K1)),
            b=float(params.get("b", _DEFAULT_B)),
        )
