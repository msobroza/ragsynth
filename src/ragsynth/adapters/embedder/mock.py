"""Deterministic offline Embedder for tests/CI."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np

from ragsynth.adapters.embedder.base import EMBEDDERS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ragsynth.datasets.base import DatasetBundle


@EMBEDDERS.register("mock")
class MockEmbedder:
    """Hash-seeded random unit vectors: same text (and seed) -> same vector.

    Intentionally content-blind (no semantics); use ``hashed_ngram`` when
    surface similarity should correlate with cosine similarity.
    """

    def __init__(self, dim: int = 32, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return deterministic unit-norm vectors, one per text."""
        rows = np.empty((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(f"{self.seed}|{text}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            rows[i] = rng.standard_normal(self.dim)
        norms = np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
        return np.asarray(rows / norms, dtype=np.float64)

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {"dim": self.dim, "seed": self.seed}

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> MockEmbedder:
        """Build from a config params block."""
        return cls(dim=int(params.get("dim", 32)), seed=int(params.get("seed", 0)))
