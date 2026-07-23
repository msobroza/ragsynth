"""Tests for the ChromaDB-backed embedding cache (spec01 Task 4, user-directed).

The cache wraps ANY backend embedder built via the EMBEDDERS registry and
never re-embeds a text it has already seen. Chroma is exercised air-gapped via
``pytest.importorskip`` on a tmp-path ``PersistentClient`` / ``EphemeralClient``;
the backend is a stub that counts the texts it is asked to embed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ragsynth.adapters.embedder.base import EMBEDDERS
from ragsynth.adapters.embedder.cached_chroma import CachedChromaEmbedder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

DIM = 8


@dataclass
class _StubBackend:
    """Deterministic offline backend that records every batch it embeds."""

    dim: int = DIM
    model: str = "stub-model"
    calls: list[list[str]] = field(default_factory=list)

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        self.calls.append(list(texts))
        rows = np.empty((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(f"{self.model}|{text}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            rows[i] = rng.standard_normal(self.dim)
        norms = np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
        return np.asarray(rows / norms, dtype=np.float64)

    def to_config(self) -> dict[str, Any]:
        return {"model": self.model, "output_dimensionality": self.dim}

    @property
    def n_texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.calls)


@pytest.fixture
def client(tmp_path: Any) -> Any:
    # A tmp-path PersistentClient isolates each test: EphemeralClient shares one
    # in-memory system across instances in a process, leaking collections.
    chromadb = pytest.importorskip("chromadb")
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


def _embedder(
    client: Any, backend: _StubBackend | None = None, **kwargs: Any
) -> CachedChromaEmbedder:
    backend = backend if backend is not None else _StubBackend()
    kwargs.setdefault("dataset", "fiqa")
    return CachedChromaEmbedder(backend=backend, backend_type="stub", client=client, **kwargs)


def test_registered_in_embedders_registry() -> None:
    assert EMBEDDERS.get("cached_chroma") is CachedChromaEmbedder


def test_miss_then_hit_no_second_backend_call_byte_identical(client: Any) -> None:
    backend = _StubBackend()
    emb = _embedder(client, backend)
    texts = ["alpha", "beta", "gamma"]

    out1 = emb.encode(texts)
    assert out1.shape == (3, DIM)
    assert out1.dtype == np.float64
    assert backend.n_texts_embedded == 3

    out2 = emb.encode(texts)
    # Second identical call: zero further backend work, byte-identical matrix.
    assert backend.n_texts_embedded == 3
    assert out2.tobytes() == out1.tobytes()
    np.testing.assert_array_equal(out2, out1)


def test_partial_hit_only_misses_are_embedded(client: Any) -> None:
    backend = _StubBackend()
    emb = _embedder(client, backend)

    emb.encode(["a", "b"])
    assert backend.calls == [["a", "b"]]

    emb.encode(["a", "b", "c"])
    # Only "c" is new; "a"/"b" served from the cache.
    assert backend.calls == [["a", "b"], ["c"]]


def test_duplicate_texts_within_one_call_embedded_once(client: Any) -> None:
    backend = _StubBackend()
    emb = _embedder(client, backend)

    out = emb.encode(["a", "a", "b"])
    # Deduplicated: the backend sees each distinct text exactly once...
    assert backend.calls == [["a", "b"]]
    # ...but the returned matrix still has one row per input, in input order.
    assert out.shape == (3, DIM)
    np.testing.assert_array_equal(out[0], out[1])
    assert not np.array_equal(out[0], out[2])


def test_different_embedder_id_uses_separate_collection(client: Any) -> None:
    b1 = _StubBackend(model="model-one")
    b2 = _StubBackend(model="model-two")
    e1 = _embedder(client, b1)
    e2 = _embedder(client, b2)

    assert e1.embedder_id != e2.embedder_id
    assert e1.collection_name != e2.collection_name

    v1 = e1.encode(["shared"])
    v2 = e2.encode(["shared"])
    # Both had to embed: no cross-contamination between the two caches.
    assert b1.calls == [["shared"]]
    assert b2.calls == [["shared"]]
    assert not np.array_equal(v1, v2)


def test_collection_metadata_records_embedder_and_dataset(client: Any) -> None:
    backend = _StubBackend()
    emb = _embedder(client, backend, dataset="fiqa")
    meta = emb.collection.metadata
    assert meta["embedder_id"] == emb.embedder_id
    assert meta["embedder_id"] == "stub:stub-model:8"
    assert meta["dataset"] == "fiqa"
    assert meta["dim"] == DIM


def test_collection_name_is_chroma_safe(client: Any) -> None:
    emb = _embedder(client, dataset="FiQA-2018!!")
    name = emb.collection_name
    assert 3 <= len(name) <= 63
    assert re.fullmatch(r"emb_[0-9a-f]{12}_[a-z0-9_]*[a-z0-9]", name)
    assert "fiqa_2018" in name


def test_empty_input_returns_empty_matrix_with_dim(client: Any) -> None:
    emb = _embedder(client)
    out = emb.encode([])
    assert out.shape == (0, DIM)
    assert out.dtype == np.float64


def test_per_item_upsert_metadata_stored(client: Any) -> None:
    backend = _StubBackend()
    emb = _embedder(client, backend, dataset="fiqa")
    emb.encode(["alpha"])
    stored = emb.collection.get(ids=[hashlib.sha256(b"alpha").hexdigest()], include=["metadatas"])
    assert stored["metadatas"][0]["embedder_id"] == emb.embedder_id
    assert stored["metadatas"][0]["dataset"] == "fiqa"


def test_to_config_round_trip_byte_stable(tmp_path: Any) -> None:
    pytest.importorskip("chromadb")
    from ragsynth.datasets.base import DatasetBundle

    bundle = DatasetBundle(chunks=(), queries_train=(), queries_anchor=(), queries_oracle=())
    params = {
        "backend": {"type": "mock", "params": {"dim": 8, "seed": 1}},
        "path": str(tmp_path / "chroma"),
        "dataset": "fiqa",
    }
    emb = CachedChromaEmbedder.from_config(params, bundle, np.random.default_rng(0))
    config = emb.to_config()
    assert config == params
    # A fresh build from the produced config yields the same config (byte-stable).
    rebuilt = CachedChromaEmbedder.from_config(config, bundle, np.random.default_rng(0))
    assert rebuilt.to_config() == config


def test_client_excluded_from_to_config(client: Any) -> None:
    emb = _embedder(client, dataset="fiqa", path="data/embeddings/chroma")
    config = emb.to_config()
    assert "client" not in config
    assert config == {
        "backend": {"type": "stub", "params": {"model": "stub-model", "output_dimensionality": 8}},
        "path": "data/embeddings/chroma",
        "dataset": "fiqa",
    }
