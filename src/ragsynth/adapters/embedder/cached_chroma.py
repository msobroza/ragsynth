"""ChromaDB-backed embedding cache behind the ``chromadb`` extra (spec01 Task 4).

Wraps *any* backend embedder built via the ``EMBEDDERS`` registry so a text is
never embedded twice: ids are ``sha256(text)`` and vectors live in a Chroma
collection keyed by an embedder identity hash plus a dataset slug. Collection
metadata records the embedder identity and dataset so the store is
self-describing to a human inspecting it.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ragsynth.adapters.embedder.base import EMBEDDERS, Embedder
from ragsynth.optional_deps import require_optional

chromadb: Any
try:
    import chromadb
except ImportError:  # pragma: no cover - exercised via require_optional tests
    chromadb = None

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ragsynth.datasets.base import DatasetBundle

_DEFAULT_PATH = "data/embeddings/chroma"
_GET_BATCH = 500  # Chroma get() is sliced to keep id lists bounded.
_MODEL_KEYS = ("model", "model_name")  # backend to_config keys that name the model
_DIM_KEYS = ("output_dimensionality", "dim")  # ...and that carry its dimensionality
_NAME_MAX = 63  # conservative Chroma collection-name limit (3-63, alnum-bounded)
_HASH_LEN = 12  # sha256 prefix length used in the collection name


class _ConfigurableEmbedder(Embedder, Protocol):
    """An :class:`Embedder` that also exposes JSON-safe params via ``to_config``."""

    def to_config(self) -> dict[str, Any]:
        """Return the backend's JSON-safe constructor params."""
        ...


def _slugify(text: str) -> str:
    """Lowercase ``text`` to a Chroma-safe ``[a-z0-9_]`` slug (runs collapsed)."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


@EMBEDDERS.register("cached_chroma")
class CachedChromaEmbedder:
    """Cache the output of a backend embedder in a ChromaDB collection.

    ``encode`` hashes each text to a ``sha256`` id, fetches any already-stored
    vectors, embeds only the misses with the backend once, upserts them, and
    returns the full matrix in input order. Because Chroma stores vectors as
    ``float32``, the returned rows are always read back *from the store* so two
    calls on the same texts yield a byte-identical matrix. Duplicate texts
    within one call share a single id and are embedded exactly once.

    Accepts an injected Chroma ``client`` for offline testing (excluded from
    ``to_config``); without one, a ``PersistentClient`` is built at ``path``.
    """

    def __init__(
        self,
        backend: _ConfigurableEmbedder,
        backend_type: str,
        dataset: str,
        path: str = _DEFAULT_PATH,
        client: Any | None = None,
    ) -> None:
        if not dataset:
            raise ValueError("dataset must be a non-empty string")
        self.backend = backend
        self.backend_type = backend_type
        self.dataset = dataset
        self.path = path
        self.embedder_id, self.dim = self._derive_identity()
        self.collection_name = self._collection_name()
        if client is not None:
            self._client = client
        else:
            require_optional(chromadb, "CachedChromaEmbedder", "chromadb")
            self._client = chromadb.PersistentClient(path=path)
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "embedder_id": self.embedder_id,
                "dataset": self.dataset,
                "dim": self.dim,
            },
        )

    def _derive_identity(self) -> tuple[str, int]:
        """Return ``("<type>:<model>:<dim>", dim)`` from the backend's config."""
        cfg = self.backend.to_config()
        model = next((str(cfg[k]) for k in _MODEL_KEYS if cfg.get(k)), self.backend_type)
        dim_val = next((cfg[k] for k in _DIM_KEYS if cfg.get(k) is not None), None)
        if dim_val is None:
            raise ValueError(
                f"backend {self.backend_type!r} to_config() must expose an embedding "
                f"dimension via one of {_DIM_KEYS}; got keys {sorted(cfg)}"
            )
        dim = int(dim_val)
        return f"{self.backend_type}:{model}:{dim}", dim

    def _collection_name(self) -> str:
        """Build a Chroma-safe collection name from the identity and dataset."""
        digest = hashlib.sha256(self.embedder_id.encode()).hexdigest()[:_HASH_LEN]
        prefix = f"emb_{digest}"  # 16 chars, always alphanumeric-bounded
        slug = _slugify(self.dataset)
        name = f"{prefix}_{slug}" if slug else prefix
        return name[:_NAME_MAX].rstrip("_-.")

    def _fetch(self, ids: list[str]) -> dict[str, NDArray[np.float64]]:
        """Return ``id -> vector`` for every stored id in ``ids`` (order-agnostic)."""
        found: dict[str, NDArray[np.float64]] = {}
        for start in range(0, len(ids), _GET_BATCH):
            chunk = ids[start : start + _GET_BATCH]
            result = self.collection.get(ids=chunk, include=["embeddings"])
            for got_id, vector in zip(result["ids"], result["embeddings"], strict=True):
                found[got_id] = np.asarray(vector, dtype=np.float64)
        return found

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return an ``(len(texts), dim)`` float64 matrix, embedding only misses.

        Raises:
            RuntimeError: If the backend returns a different number of vectors
                than the misses requested, or if a vector is still absent after
                upsert -- rows must never be silently corrupted or left blank.
        """
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float64)

        ids = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        # Distinct ids in first-occurrence order (duplicate texts collapse here).
        unique_ids: list[str] = []
        unique_texts: list[str] = []
        seen: set[str] = set()
        for text_id, text in zip(ids, texts, strict=True):
            if text_id not in seen:
                seen.add(text_id)
                unique_ids.append(text_id)
                unique_texts.append(text)

        id_to_vec = self._fetch(unique_ids)
        miss_pairs = [
            (text_id, text)
            for text_id, text in zip(unique_ids, unique_texts, strict=True)
            if text_id not in id_to_vec
        ]
        miss_ids = [text_id for text_id, _ in miss_pairs]
        miss_texts = [text for _, text in miss_pairs]
        if miss_ids:
            vectors = np.asarray(self.backend.encode(miss_texts), dtype=np.float64)
            if vectors.shape[0] != len(miss_ids):
                raise RuntimeError(
                    f"backend {self.backend_type!r} returned {vectors.shape[0]} vectors "
                    f"for {len(miss_ids)} texts; inputs and outputs mismatch"
                )
            self.collection.upsert(
                ids=miss_ids,
                embeddings=vectors.tolist(),
                documents=miss_texts,
                metadatas=[
                    {"embedder_id": self.embedder_id, "dataset": self.dataset} for _ in miss_ids
                ],
            )
            # Read the misses back so returned rows are the stored (float32) values:
            # a repeat call then yields a byte-identical matrix.
            id_to_vec.update(self._fetch(miss_ids))

        absent = [text_id for text_id in unique_ids if text_id not in id_to_vec]
        if absent:
            raise RuntimeError(
                f"{len(absent)} id(s) missing from the cache after upsert "
                f"(collection {self.collection_name!r}); the store may be corrupt"
            )
        rows = np.empty((len(texts), self.dim), dtype=np.float64)
        for i, text_id in enumerate(ids):
            rows[i] = id_to_vec[text_id]
        return rows

    def to_config(self) -> dict[str, Any]:
        """JSON-safe params; the injected ``client`` is excluded.

        The backend is serialized as its own ``{"type", "params"}`` block so the
        whole config round-trips through :meth:`from_config`.
        """
        return {
            "backend": {"type": self.backend_type, "params": self.backend.to_config()},
            "path": self.path,
            "dataset": self.dataset,
        }

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> CachedChromaEmbedder:
        """Build from a config params block, constructing the nested backend.

        The backend is built through the ``EMBEDDERS`` registry from its nested
        ``{"type", "params"}`` block; no ``client`` is injected, so a real
        ``PersistentClient`` is created at ``path``.
        """
        backend_block = params["backend"]
        backend_type = str(backend_block["type"])
        backend = EMBEDDERS.get(backend_type).from_config(
            backend_block.get("params") or {}, bundle, rng
        )
        return cls(
            backend=backend,
            backend_type=backend_type,
            dataset=str(params["dataset"]),
            path=str(params.get("path", _DEFAULT_PATH)),
        )
