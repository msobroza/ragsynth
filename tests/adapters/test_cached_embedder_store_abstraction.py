"""CachedEmbedder works with ANY embedder and ANY EmbeddingVectorStore (SOLID).

The cache policy (dedup, miss-only embedding, read-back, order assembly) must
not depend on ChromaDB or on a specific embedder family: these tests drive it
with a named FakeVectorStore and OpenAI-/sentence-transformers-shaped fake
embedders, fully air-gapped — no ``chromadb`` import anywhere in this file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ragsynth.adapters.embedder.cached_embedder import (
    CachedEmbedder,
    derive_embedder_identity,
)
from ragsynth.adapters.embedder.vector_store import EmbeddingVectorStore

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

DIM = 8


class FakeVectorStore(EmbeddingVectorStore):
    """In-memory store recording every fetch/store call (F.I.R.S.T-friendly)."""

    def __init__(self) -> None:
        self.vectors: dict[str, NDArray[np.float64]] = {}
        self.stored_documents: dict[str, str] = {}
        self.stored_metadata: list[Mapping[str, str]] = []
        self.fetch_calls: list[list[str]] = []

    @property
    def location(self) -> str:
        return "fake-store"

    def fetch_vectors(self, ids: Sequence[str]) -> dict[str, NDArray[np.float64]]:
        self.fetch_calls.append(list(ids))
        return {i: self.vectors[i] for i in ids if i in self.vectors}

    def store_vectors(
        self,
        ids: Sequence[str],
        vectors: NDArray[np.float64],
        documents: Sequence[str],
        metadata: Mapping[str, str],
    ) -> None:
        for i, vec_id in enumerate(ids):
            self.vectors[vec_id] = np.asarray(vectors[i], dtype=np.float64)
            self.stored_documents[vec_id] = documents[i]
        self.stored_metadata.append(dict(metadata))


@dataclass
class FakeOpenAIEmbedder:
    """OpenAI-API-shaped backend: config keys ``model`` + ``output_dimensionality``."""

    model: str = "text-embedding-3-small"
    dim: int = DIM
    calls: list[list[str]] = field(default_factory=list)

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        self.calls.append(list(texts))
        rows = np.empty((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(f"{self.model}|{text}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            rows[i] = rng.standard_normal(self.dim)
        return rows

    def to_config(self) -> dict[str, Any]:
        return {"model": self.model, "output_dimensionality": self.dim}


@dataclass
class FakeSentenceTransformerEmbedder:
    """sentence-transformers-shaped backend: config keys ``model_name`` + ``dim``."""

    model_name: str = "all-MiniLM-L6-v2"
    dim: int = DIM

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        rng = np.random.default_rng(7)
        return rng.standard_normal((len(texts), self.dim))

    def to_config(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "device": "cpu", "dim": self.dim}


def _cached(backend: Any, backend_type: str) -> tuple[CachedEmbedder, FakeVectorStore]:
    embedder_id, dim = derive_embedder_identity(backend_type, backend.to_config())
    store = FakeVectorStore()
    cache = CachedEmbedder(
        backend=backend,
        backend_type=backend_type,
        store=store,
        embedder_id=embedder_id,
        dim=dim,
    )
    return cache, store


def test_identity_derives_from_openai_style_config_keys() -> None:
    embedder_id, dim = derive_embedder_identity("openai", FakeOpenAIEmbedder().to_config())
    assert embedder_id == f"openai:text-embedding-3-small:{DIM}"
    assert dim == DIM


def test_identity_derives_from_sentence_transformer_style_config_keys() -> None:
    embedder_id, dim = derive_embedder_identity(
        "sentence_transformer", FakeSentenceTransformerEmbedder().to_config()
    )
    assert embedder_id == f"sentence_transformer:all-MiniLM-L6-v2:{DIM}"
    assert dim == DIM


def test_miss_then_hit_with_fake_store_never_reembeds() -> None:
    backend = FakeOpenAIEmbedder()
    cache, store = _cached(backend, "openai")
    texts = ["alpha", "beta", "gamma"]

    out1 = cache.encode(texts)
    out2 = cache.encode(texts)

    assert sum(len(batch) for batch in backend.calls) == 3
    np.testing.assert_array_equal(out1, out2)
    assert set(store.stored_documents.values()) == set(texts)


def test_metadata_carries_embedder_id_and_reaches_the_store() -> None:
    cache, store = _cached(FakeOpenAIEmbedder(), "openai")
    cache.encode(["alpha"])
    assert store.stored_metadata[0]["embedder_id"] == cache.embedder_id


def test_duplicates_in_one_call_embed_once_matrix_row_per_input() -> None:
    backend = FakeOpenAIEmbedder()
    cache, _ = _cached(backend, "openai")
    out = cache.encode(["a", "a", "b"])
    assert sum(len(batch) for batch in backend.calls) == 2
    assert out.shape == (3, DIM)
    np.testing.assert_array_equal(out[0], out[1])


def test_empty_input_returns_zero_by_dim_matrix() -> None:
    cache, _ = _cached(FakeOpenAIEmbedder(), "openai")
    assert cache.encode([]).shape == (0, DIM)
