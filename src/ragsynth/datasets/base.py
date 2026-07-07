"""Dataset bundle contract: what every dataset hands the composition root."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ragsynth.pipeline.registry import Registry

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ragsynth.domain import Chunk, ProductionQuery
    from ragsynth.io.embeddings import EmbeddingStore


class EmbeddingBank:
    """Mutable text -> vector lookup shared between coupled adapters.

    The toy world's ChatModel writes each emitted query's geometric embedding
    here; the passthrough Embedder reads it back by text (SPEC §10, PLAN D12).
    Text corpora leave the bank ``None`` and embed with a real featurizer.
    """

    def __init__(self) -> None:
        self._vectors: dict[str, NDArray[np.float64]] = {}

    def put(self, text: str, vector: NDArray[np.float64]) -> None:
        """Store the vector for ``text`` (idempotent for identical content)."""
        self._vectors[text] = np.asarray(vector, dtype=np.float64)

    def get(self, text: str) -> NDArray[np.float64]:
        """Look up the vector for ``text``.

        Raises:
            KeyError: If ``text`` was never registered.
        """
        if text not in self._vectors:
            raise KeyError(f"no embedding banked for text: {text[:60]!r}")
        return self._vectors[text]

    def __contains__(self, text: str) -> bool:
        return text in self._vectors


@dataclass(frozen=True)
class DatasetBundle:
    """Everything a dataset contributes to :class:`~ragsynth.pipeline.base.Resources`.

    Production queries arrive pre-split train/anchor/oracle = 0.60/0.25/0.15
    (PLAN D10): demand estimation and exemplars use ``queries_train`` only;
    ``queries_anchor`` is the validation reference; the ORACLE arm draws from
    ``queries_oracle``. Qrels may be empty -- the composition root falls back
    to nearest-chunk gate-style relabeling (PLAN D17).
    """

    chunks: tuple[Chunk, ...]
    queries_train: tuple[ProductionQuery, ...]
    queries_anchor: tuple[ProductionQuery, ...]
    queries_oracle: tuple[ProductionQuery, ...]
    anchor_qrels: dict[str, dict[str, int]] = field(default_factory=dict)
    oracle_qrels: dict[str, dict[str, int]] = field(default_factory=dict)
    embeddings: EmbeddingStore | None = None
    bank: EmbeddingBank | None = None


DATASETS: Registry[Any] = Registry("dataset")
