"""Pure-numpy character n-gram hashing embedder (SPEC §12, PLAN D3).

The v1 default text featurizer for real corpora: no model downloads, no
sklearn -- character n-grams hashed into a fixed-width signed feature
vector (the signed hashing trick), log1p-scaled and L2-normalized, so
surface-similar texts land close in cosine space.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np

from ragsynth.adapters.embedder.base import EMBEDDERS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ragsynth.datasets.base import DatasetBundle

_DEFAULT_DIM = 256
_DEFAULT_NGRAM_RANGE = (3, 5)
_BUCKET_BYTES = 8  # sha256 prefix used for the bucket index
_SIGN_BYTE = 8  # next byte's parity gives the +/-1 sign
_NORM_EPS = 1e-12


@EMBEDDERS.register("hashed_ngram")
class HashedNGramEmbedder:
    """Deterministic character n-gram hashing featurizer.

    Pipeline per text: lowercase, extract all character n-grams for
    ``n in ngram_range``, hash each via ``sha256(f"{seed}|{ngram}")`` to a
    bucket in ``[0, dim)`` with a +/-1 sign bit, accumulate signed counts,
    apply signed ``log1p`` scaling, L2-normalize. Text with no n-grams
    (e.g. empty string) maps to a deterministic one-hot fallback vector.
    """

    def __init__(
        self,
        dim: int = _DEFAULT_DIM,
        ngram_range: tuple[int, int] = _DEFAULT_NGRAM_RANGE,
        seed: int = 0,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        low, high = ngram_range
        if low < 1 or high < low:
            raise ValueError(f"ngram_range must satisfy 1 <= low <= high, got {ngram_range}")
        self.dim = dim
        self.ngram_range = (int(low), int(high))
        self.seed = seed

    def _featurize(self, text: str) -> NDArray[np.float64]:
        """Return the unit-norm feature vector for one text."""
        lowered = text.lower()
        counts = np.zeros(self.dim, dtype=np.float64)
        low, high = self.ngram_range
        for n in range(low, high + 1):
            for start in range(len(lowered) - n + 1):
                ngram = lowered[start : start + n]
                digest = hashlib.sha256(f"{self.seed}|{ngram}".encode()).digest()
                bucket = int.from_bytes(digest[:_BUCKET_BYTES], "big") % self.dim
                sign = 1.0 if digest[_SIGN_BYTE] % 2 == 0 else -1.0
                counts[bucket] += sign
        if not counts.any():
            # No n-grams (or full cancellation): deterministic unit fallback.
            counts[0] = 1.0
            return counts
        scaled = np.sign(counts) * np.log1p(np.abs(counts))
        norm = max(float(np.linalg.norm(scaled)), _NORM_EPS)
        return np.asarray(scaled / norm, dtype=np.float64)

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return an ``(len(texts), dim)`` matrix of unit-norm rows."""
        rows = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            rows[i] = self._featurize(text)
        return rows

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {"dim": self.dim, "ngram_range": list(self.ngram_range), "seed": self.seed}

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> HashedNGramEmbedder:
        """Build from a config params block."""
        low, high = params.get("ngram_range", _DEFAULT_NGRAM_RANGE)
        return cls(
            dim=int(params.get("dim", _DEFAULT_DIM)),
            ngram_range=(int(low), int(high)),
            seed=int(params.get("seed", 0)),
        )
