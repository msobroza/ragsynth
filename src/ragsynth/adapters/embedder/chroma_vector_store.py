"""ChromaDB implementation of :class:`EmbeddingVectorStore` (``chromadb`` extra).

Owns everything Chroma-specific: client construction, collection naming and
metadata, and the batched ``get``/``upsert`` calls. The cache policy lives in
``cached_embedder.py`` and never imports this module (DIP).
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from ragsynth.adapters.embedder.vector_store import EmbeddingVectorStore
from ragsynth.optional_deps import require_optional

chromadb: Any
try:
    import chromadb
except ImportError:  # pragma: no cover - exercised via require_optional tests
    chromadb = None

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

_GET_BATCH = 500  # Chroma get() is sliced to keep id lists bounded.
_NAME_MAX = 63  # conservative Chroma collection-name limit (3-63, alnum-bounded)
_HASH_LEN = 12  # sha256 prefix length used in the collection name


def _slugify(text: str) -> str:
    """Lowercase ``text`` to a Chroma-safe ``[a-z0-9_]`` slug (runs collapsed)."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def chroma_collection_name(embedder_id: str, dataset: str) -> str:
    """Build a Chroma-safe collection name from the embedder identity and dataset."""
    digest = hashlib.sha256(embedder_id.encode()).hexdigest()[:_HASH_LEN]
    prefix = f"emb_{digest}"  # 16 chars, always alphanumeric-bounded
    slug = _slugify(dataset)
    name = f"{prefix}_{slug}" if slug else prefix
    return name[:_NAME_MAX].rstrip("_-.")


class ChromaEmbeddingVectorStore(EmbeddingVectorStore):
    """Persist cache vectors in a ChromaDB collection.

    The collection is named from the embedder identity hash plus a dataset slug,
    and carries ``{embedder_id, dataset, dim}`` metadata so the store is
    self-describing to a human inspecting it. Accepts an injected ``client``
    for offline testing; without one, a ``PersistentClient`` is built at
    ``path``.
    """

    def __init__(
        self,
        embedder_id: str,
        dim: int,
        dataset: str,
        path: str,
        client: Any | None = None,
    ) -> None:
        self.collection_name = chroma_collection_name(embedder_id, dataset)
        if client is None:
            require_optional(chromadb, "ChromaEmbeddingVectorStore", "chromadb")
            client = chromadb.PersistentClient(path=path)
        self._client = client
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"embedder_id": embedder_id, "dataset": dataset, "dim": dim},
        )

    @property
    def location(self) -> str:
        """The Chroma collection name, used in cache error messages."""
        return self.collection_name

    def fetch_vectors(self, ids: Sequence[str]) -> dict[str, NDArray[np.float64]]:
        """Return ``id -> vector`` via batched ``get`` (Chroma may reorder; map by id)."""
        found: dict[str, NDArray[np.float64]] = {}
        for start in range(0, len(ids), _GET_BATCH):
            chunk = list(ids[start : start + _GET_BATCH])
            result = self.collection.get(ids=chunk, include=["embeddings"])
            for got_id, vector in zip(result["ids"], result["embeddings"], strict=True):
                found[got_id] = np.asarray(vector, dtype=np.float64)
        return found

    def store_vectors(
        self,
        ids: Sequence[str],
        vectors: NDArray[np.float64],
        documents: Sequence[str],
        metadata: Mapping[str, str],
    ) -> None:
        """Upsert vectors with per-item copies of the shared ``metadata``."""
        self.collection.upsert(
            ids=list(ids),
            embeddings=np.asarray(vectors, dtype=np.float64).tolist(),
            documents=list(documents),
            metadatas=[dict(metadata) for _ in ids],
        )
