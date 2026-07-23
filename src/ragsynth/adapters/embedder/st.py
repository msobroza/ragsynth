"""Sentence-Transformers embedder behind the ``st`` optional extra (SPEC §12, §3.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ragsynth.adapters.embedder.base import EMBEDDERS
from ragsynth.optional_deps import require_optional

try:
    import sentence_transformers
except ImportError:  # pragma: no cover - exercised via require_optional tests
    sentence_transformers = None

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ragsynth.datasets.base import DatasetBundle

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_NORM_EPS = 1e-12


@EMBEDDERS.register("sentence_transformer")
class SentenceTransformerEmbedder:
    """Neural text embedder via ``sentence-transformers`` (``uv sync --extra st``).

    Delegates encoding to the named model and L2-normalizes rows, honoring
    the Embedder Protocol invariant regardless of the model's own settings.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME, device: str = "cpu") -> None:
        require_optional(sentence_transformers, "SentenceTransformerEmbedder", "st")
        self.model_name = model_name
        self.device = device
        self._model = sentence_transformers.SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return an ``(len(texts), d)`` matrix of unit-norm rows."""
        raw = self._model.encode(list(texts), convert_to_numpy=True)
        matrix = np.asarray(raw, dtype=np.float64)
        norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), _NORM_EPS)
        return np.asarray(matrix / norms, dtype=np.float64)

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params, plus the model's embedding ``dim``.

        The dim is resolved lazily from the loaded model so ``cached_chroma``
        can derive its embedder identity; models that report no dimension keep
        the bare two-key shape (``from_config`` never requires ``dim``).
        """
        config: dict[str, Any] = {"model_name": self.model_name, "device": self.device}
        dim = self._model.get_sentence_embedding_dimension()
        if dim is not None:
            config["dim"] = int(dim)
        return config

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> SentenceTransformerEmbedder:
        """Build from a config params block."""
        return cls(
            model_name=str(params.get("model_name", _DEFAULT_MODEL_NAME)),
            device=str(params.get("device", "cpu")),
        )
