"""Tests for the dense in-memory retriever."""

import numpy as np
import pytest

from ragsynth.adapters.retriever.dense_inmemory import DenseInMemoryRetriever


@pytest.fixture
def retriever() -> DenseInMemoryRetriever:
    # Four unit vectors with hand-computable cosines to the query [1, 0].
    matrix = np.array(
        [
            [1.0, 0.0],  # c-exact: cos 1.0
            [0.8, 0.6],  # c-close: cos 0.8
            [0.0, 1.0],  # c-orth:  cos 0.0
            [-1.0, 0.0],  # c-anti: cos -1.0
        ]
    )
    return DenseInMemoryRetriever(
        chunk_ids=["c-exact", "c-close", "c-orth", "c-anti"], matrix=matrix
    )


def test_exact_topk_order_and_scores(retriever: DenseInMemoryRetriever) -> None:
    hits = retriever.search(np.array([1.0, 0.0]), k=2)
    assert [h[0] for h in hits] == ["c-exact", "c-close"]
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[1][1] == pytest.approx(0.8)


def test_k_larger_than_corpus(retriever: DenseInMemoryRetriever) -> None:
    hits = retriever.search(np.array([1.0, 0.0]), k=100)
    assert len(hits) == 4
    assert [h[0] for h in hits] == ["c-exact", "c-close", "c-orth", "c-anti"]


def test_mismatched_ids_matrix_rejected() -> None:
    with pytest.raises(ValueError, match="chunk_ids"):
        DenseInMemoryRetriever(chunk_ids=["a"], matrix=np.eye(2))
