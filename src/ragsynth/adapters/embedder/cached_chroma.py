"""ChromaDB-backed embedding cache: the registered ``cached_chroma`` composition.

Wires :class:`CachedEmbedder` (store-agnostic cache policy) to
:class:`ChromaEmbeddingVectorStore` (Chroma persistence). Config schema and the
public surface (``embedder_id``, ``dim``, ``collection_name``, ``collection``)
are unchanged from the original single-file implementation (spec01 Task 4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragsynth.adapters.embedder.base import EMBEDDERS
from ragsynth.adapters.embedder.cached_embedder import (
    CachedEmbedder,
    ConfigurableEmbedder,
    derive_embedder_identity,
)
from ragsynth.adapters.embedder.chroma_vector_store import ChromaEmbeddingVectorStore

if TYPE_CHECKING:
    import numpy as np

    from ragsynth.datasets.base import DatasetBundle

_DEFAULT_PATH = "data/embeddings/chroma"


@EMBEDDERS.register("cached_chroma")
class CachedChromaEmbedder(CachedEmbedder):
    """Cache any backend embedder's vectors in a ChromaDB collection.

    Example: ``CachedChromaEmbedder(backend=GeminiEmbedder(...), backend_type=
    "gemini", dataset="fiqa")`` — a text embedded once is never re-embedded.
    Accepts an injected Chroma ``client`` for offline testing (excluded from
    ``to_config``); without one, a ``PersistentClient`` is built at ``path``.
    """

    def __init__(
        self,
        backend: ConfigurableEmbedder,
        backend_type: str,
        dataset: str,
        path: str = _DEFAULT_PATH,
        client: Any | None = None,
    ) -> None:
        if not dataset:
            raise ValueError("dataset must be a non-empty string")
        embedder_id, dim = derive_embedder_identity(backend_type, backend.to_config())
        store = ChromaEmbeddingVectorStore(
            embedder_id=embedder_id, dim=dim, dataset=dataset, path=path, client=client
        )
        super().__init__(
            backend=backend,
            backend_type=backend_type,
            store=store,
            embedder_id=embedder_id,
            dim=dim,
            item_metadata={"embedder_id": embedder_id, "dataset": dataset},
        )
        self.dataset = dataset
        self.path = path
        self._chroma_store = store

    @property
    def collection_name(self) -> str:
        """The Chroma collection name derived from identity hash + dataset slug."""
        return self._chroma_store.collection_name

    @property
    def collection(self) -> Any:
        """The underlying Chroma collection (self-describing metadata included)."""
        return self._chroma_store.collection

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
