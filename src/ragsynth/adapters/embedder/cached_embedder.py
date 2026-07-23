"""Store-agnostic embedding cache decorator (spec01 Task 4, SOLID refactor).

Provenance: extracted from the original single-file ``cached_chroma.py`` so the
cache policy (sha256 ids, dedup, miss-only embedding, read-back, input-order
assembly) depends only on the :class:`EmbeddingVectorStore` abstraction — any
embedder (Gemini, OpenAI-compatible, sentence-transformers) rides any store.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ragsynth.adapters.embedder.base import Embedder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ragsynth.adapters.embedder.vector_store import EmbeddingVectorStore

_MODEL_KEYS = ("model", "model_name")  # backend to_config keys that name the model
_DIM_KEYS = ("output_dimensionality", "dim")  # ...and that carry its dimensionality


class ConfigurableEmbedder(Embedder, Protocol):
    """An :class:`Embedder` that also exposes JSON-safe params via ``to_config``."""

    def to_config(self) -> dict[str, Any]:
        """Return the backend's JSON-safe constructor params."""
        ...


def derive_embedder_identity(backend_type: str, config: dict[str, Any]) -> tuple[str, int]:
    """Return ``("<type>:<model>:<dim>", dim)`` from a backend's config params.

    Example: ``derive_embedder_identity("gemini", {"model": "gemini-embedding-2",
    "output_dimensionality": 768})`` -> ``("gemini:gemini-embedding-2:768", 768)``.
    """
    model = next((str(config[k]) for k in _MODEL_KEYS if config.get(k)), backend_type)
    dim_val = next((config[k] for k in _DIM_KEYS if config.get(k) is not None), None)
    if dim_val is None:
        raise ValueError(
            f"backend {backend_type!r} to_config() must expose an embedding "
            f"dimension via one of {_DIM_KEYS}; got keys {sorted(config)}"
        )
    dim = int(dim_val)
    return f"{backend_type}:{model}:{dim}", dim


class CachedEmbedder:
    """Cache a backend embedder's output in any :class:`EmbeddingVectorStore`.

    ``encode`` hashes each text to a ``sha256`` id, fetches stored vectors,
    embeds only the misses with the backend once, stores them, and returns the
    full matrix in input order. Returned rows are always read back *from the
    store* so two calls on the same texts yield a byte-identical matrix even
    when the store narrows precision. Duplicate texts within one call share a
    single id and are embedded exactly once.
    """

    def __init__(
        self,
        backend: ConfigurableEmbedder,
        backend_type: str,
        store: EmbeddingVectorStore,
        embedder_id: str,
        dim: int,
        item_metadata: dict[str, str] | None = None,
    ) -> None:
        self.backend = backend
        self.backend_type = backend_type
        self.store = store
        self.embedder_id = embedder_id
        self.dim = dim
        # Attached to every stored vector; compositions extend it (e.g. dataset).
        self.item_metadata = item_metadata or {"embedder_id": embedder_id}

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return an ``(len(texts), dim)`` float64 matrix, embedding only misses.

        Raises:
            RuntimeError: If the backend returns a different number of vectors
                than the misses requested, or if a vector is still absent after
                store -- rows must never be silently corrupted or left blank.
        """
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float64)
        ids = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        unique_ids, unique_texts = _first_occurrences(ids, texts)
        id_to_vec = self.store.fetch_vectors(unique_ids)
        id_to_vec = self._embed_misses(unique_ids, unique_texts, id_to_vec)
        self._check_complete(unique_ids, id_to_vec)
        rows = np.empty((len(texts), self.dim), dtype=np.float64)
        for i, text_id in enumerate(ids):
            rows[i] = id_to_vec[text_id]
        return rows

    def _embed_misses(
        self,
        unique_ids: list[str],
        unique_texts: list[str],
        id_to_vec: dict[str, NDArray[np.float64]],
    ) -> dict[str, NDArray[np.float64]]:
        """Embed and store ids absent from ``id_to_vec``; return the completed map."""
        miss = [(i, t) for i, t in zip(unique_ids, unique_texts, strict=True) if i not in id_to_vec]
        if not miss:
            return id_to_vec
        miss_ids = [i for i, _ in miss]
        miss_texts = [t for _, t in miss]
        vectors = np.asarray(self.backend.encode(miss_texts), dtype=np.float64)
        if vectors.shape[0] != len(miss_ids):
            raise RuntimeError(
                f"backend {self.backend_type!r} returned {vectors.shape[0]} vectors "
                f"for {len(miss_ids)} texts; inputs and outputs mismatch"
            )
        self.store.store_vectors(miss_ids, vectors, miss_texts, self.item_metadata)
        # Read the misses back so returned rows are the stored values: a repeat
        # call then yields a byte-identical matrix (store may narrow precision).
        id_to_vec.update(self.store.fetch_vectors(miss_ids))
        return id_to_vec

    def _check_complete(
        self, unique_ids: list[str], id_to_vec: dict[str, NDArray[np.float64]]
    ) -> None:
        """Fail loudly if any id is still missing after the store round-trip."""
        absent = [text_id for text_id in unique_ids if text_id not in id_to_vec]
        if absent:
            raise RuntimeError(
                f"{len(absent)} id(s) missing from the cache after store "
                f"({self.store.location!r}); the store may be corrupt"
            )


def _first_occurrences(ids: list[str], texts: Sequence[str]) -> tuple[list[str], list[str]]:
    """Distinct ids in first-occurrence order (duplicate texts collapse here)."""
    unique_ids: list[str] = []
    unique_texts: list[str] = []
    seen: set[str] = set()
    for text_id, text in zip(ids, texts, strict=True):
        if text_id not in seen:
            seen.add(text_id)
            unique_ids.append(text_id)
            unique_texts.append(text)
    return unique_ids, unique_texts
