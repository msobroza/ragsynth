"""Tests for the pure-numpy HashedNGramEmbedder (SPEC §12, PLAN D3)."""

import numpy as np

from ragsynth.adapters.embedder.base import EMBEDDERS
from ragsynth.adapters.embedder.hashed import HashedNGramEmbedder

DIM = 64


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def test_registered_in_embedders_registry() -> None:
    assert EMBEDDERS.get("hashed_ngram") is HashedNGramEmbedder


def test_shape_and_unit_rows() -> None:
    emb = HashedNGramEmbedder(dim=DIM)
    out = emb.encode(["interest rate swap", "zebra migration patterns", "hello"])
    assert out.shape == (3, DIM)
    assert out.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-9)


def test_deterministic_across_instances() -> None:
    a = HashedNGramEmbedder(dim=DIM, seed=0).encode(["abc def"])
    b = HashedNGramEmbedder(dim=DIM, seed=0).encode(["abc def"])
    np.testing.assert_array_equal(a, b)


def test_same_text_same_vector_within_batch() -> None:
    out = HashedNGramEmbedder(dim=DIM).encode(["repeat me", "other", "repeat me"])
    np.testing.assert_array_equal(out[0], out[2])


def test_different_texts_differ() -> None:
    out = HashedNGramEmbedder(dim=DIM).encode(["interest rate swap", "zebra migration patterns"])
    assert not np.allclose(out[0], out[1])


def test_case_insensitive() -> None:
    out = HashedNGramEmbedder(dim=DIM).encode(["Interest Rate Swap", "interest rate swap"])
    np.testing.assert_array_equal(out[0], out[1])


def test_similar_texts_have_higher_cosine_than_dissimilar() -> None:
    emb = HashedNGramEmbedder(dim=DIM)
    out = emb.encode(["interest rate swap", "interest rate swaps", "zebra migration patterns"])
    assert _cos(out[0], out[1]) > _cos(out[0], out[2])


def test_seed_changes_vectors() -> None:
    a = HashedNGramEmbedder(dim=DIM, seed=0).encode(["interest rate swap"])
    b = HashedNGramEmbedder(dim=DIM, seed=1).encode(["interest rate swap"])
    assert not np.allclose(a, b)


def test_empty_string_safe_and_unit_norm() -> None:
    out = HashedNGramEmbedder(dim=DIM).encode([""])
    assert out.shape == (1, DIM)
    np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0, atol=1e-9)
    # Deterministic fallback across calls.
    again = HashedNGramEmbedder(dim=DIM).encode([""])
    np.testing.assert_array_equal(out, again)


def test_ngram_range_respected() -> None:
    # With (3, 3) grams, a 2-char text yields no n-grams -> same fallback as "".
    emb33 = HashedNGramEmbedder(dim=DIM, ngram_range=(3, 3))
    out = emb33.encode(["ab", ""])
    np.testing.assert_array_equal(out[0], out[1])
    # Widening the range changes the featurization of a longer text.
    emb35 = HashedNGramEmbedder(dim=DIM, ngram_range=(3, 5))
    text = ["interest rate swap"]
    assert not np.allclose(emb33.encode(text), emb35.encode(text))


def test_from_config_round_trip() -> None:
    from ragsynth.datasets.base import DatasetBundle

    bundle = DatasetBundle(chunks=(), queries_train=(), queries_anchor=(), queries_oracle=())
    original = HashedNGramEmbedder(dim=32, ngram_range=(2, 4), seed=7)
    rebuilt = HashedNGramEmbedder.from_config(
        original.to_config(), bundle, np.random.default_rng(0)
    )
    assert rebuilt.to_config() == original.to_config()
    np.testing.assert_array_equal(rebuilt.encode(["abc"]), original.encode(["abc"]))
