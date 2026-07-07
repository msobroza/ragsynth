"""Tests for the EmbeddingStore (npz-backed id -> vector map)."""

from pathlib import Path

import numpy as np
import pytest

from ragsynth.io.embeddings import EmbeddingStore


@pytest.fixture
def vectors() -> np.ndarray:
    rng = np.random.default_rng(0)
    v = rng.standard_normal((3, 8)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_add_get_roundtrip_preserves_order_and_dtype(vectors: np.ndarray) -> None:
    store = EmbeddingStore()
    store.add(["a", "b", "c"], vectors)
    out = store.get(["c", "a"])
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[0], vectors[2])
    np.testing.assert_array_equal(out[1], vectors[0])


def test_contains_len_dim(vectors: np.ndarray) -> None:
    store = EmbeddingStore()
    assert store.dim is None
    store.add(["a", "b", "c"], vectors)
    assert "a" in store
    assert "z" not in store
    assert len(store) == 3
    assert store.dim == 8


def test_duplicate_id_rejected(vectors: np.ndarray) -> None:
    store = EmbeddingStore()
    store.add(["a", "b", "c"], vectors)
    with pytest.raises(ValueError, match="duplicate"):
        store.add(["c"], vectors[:1])


def test_missing_id_keyerror(vectors: np.ndarray) -> None:
    store = EmbeddingStore()
    store.add(["a", "b", "c"], vectors)
    with pytest.raises(KeyError, match="unknown"):
        store.get(["a", "nope"])


def test_mismatched_lengths_rejected(vectors: np.ndarray) -> None:
    store = EmbeddingStore()
    with pytest.raises(ValueError, match="ids"):
        store.add(["a", "b"], vectors)


def test_save_load_roundtrip(tmp_path: Path, vectors: np.ndarray) -> None:
    store = EmbeddingStore()
    store.add(["a", "b", "c"], vectors)
    path = tmp_path / "emb.npz"
    store.save(path)
    loaded = EmbeddingStore.load(path)
    assert len(loaded) == 3
    assert loaded.dim == 8
    np.testing.assert_array_equal(loaded.get(["a", "b", "c"]), vectors)
