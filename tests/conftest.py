"""Shared fixtures: minimal Resources, fake steps, tiny dataset.

Test-only registry entries use the ``test.`` key prefix; the registry
contract test (tests/pipeline/test_contract.py) skips that prefix.
"""

from typing import Any

import numpy as np
import pytest

from ragsynth.adapters.embedder.mock import MockEmbedder
from ragsynth.adapters.judge.mock import MockJudge
from ragsynth.adapters.llm.mock import MockChatModel
from ragsynth.adapters.retriever.dense_inmemory import DenseInMemoryRetriever
from ragsynth.domain import Chunk, ProductionQuery
from ragsynth.io.artifacts import ArtifactStore
from ragsynth.io.embeddings import EmbeddingStore
from ragsynth.pipeline.base import Resources


def make_min_resources(tmp_path: Any, seed: int = 0, n_chunks: int = 4) -> Resources:
    """A tiny but structurally-complete Resources for unit tests.

    partition/demand/zoo are filled with plain placeholders -- tests that
    need real fitted objects build them explicitly (or use the small-world
    fixture from Phase 3).
    """
    chunks = tuple(
        Chunk.create(text=f"chunk text number {i} about topic {i % 2}", doc_id=f"doc{i % 2}")
        for i in range(n_chunks)
    )
    train = tuple(ProductionQuery(query_id=f"qt{i}", text=f"train query {i}?") for i in range(6))
    anchor = tuple(ProductionQuery(query_id=f"qa{i}", text=f"anchor query {i}?") for i in range(3))
    oracle = tuple(ProductionQuery(query_id=f"qo{i}", text=f"oracle query {i}?") for i in range(3))
    embedder = MockEmbedder(dim=16, seed=seed)
    store = EmbeddingStore()
    store.add([c.chunk_id for c in chunks], embedder.encode([c.text for c in chunks]))
    for split in (train, anchor, oracle):
        store.add([q.query_id for q in split], embedder.encode([q.text for q in split]))
    return Resources(
        chunks=chunks,
        queries_train=train,
        queries_anchor=anchor,
        queries_oracle=oracle,
        anchor_qrels={},
        oracle_qrels={},
        embedder=embedder,
        generator_llm=MockChatModel(seed=seed),
        judge=MockJudge(),
        retriever=DenseInMemoryRetriever(
            chunk_ids=[c.chunk_id for c in chunks],
            matrix=store.get([c.chunk_id for c in chunks]).astype(np.float64),
        ),
        embeddings=store,
        partition=object(),  # type: ignore[arg-type]
        demand=object(),  # type: ignore[arg-type]
        zoo={},
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        seed=seed,
    )


@pytest.fixture
def min_resources(tmp_path: Any) -> Resources:
    return make_min_resources(tmp_path)
