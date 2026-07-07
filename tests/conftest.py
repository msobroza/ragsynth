"""Shared fixtures: minimal Resources, fake steps, tiny dataset.

Test-only registry entries use the ``test.`` key prefix; the registry
contract test (tests/pipeline/test_contract.py) skips that prefix.
"""

from pathlib import Path
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
from ragsynth.pipeline.base import DemandArtifact, Resources
from ragsynth.sampling.demand import demand_from_responsibilities, nn_cos_threshold, tilt_weights
from ragsynth.sampling.movmf import MovMF
from ragsynth.sampling.partition import ReferencePartition
from ragsynth.sampling.vmf import l2_normalize


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


# ---- two-cluster hand-built world (moved from tests/steps/conftest.py) ----

D = 16
N_PER_CLUSTER = 10


def _jitter(base: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    return l2_normalize(base + scale * rng.standard_normal(base.shape))


@pytest.fixture
def steps_world(tmp_path: Path) -> Resources:
    rng = np.random.default_rng(0)
    e0 = np.zeros(D)
    e0[0] = 1.0
    e1 = np.zeros(D)
    e1[1] = 1.0

    chunks: list[Chunk] = []
    chunk_vecs: list[np.ndarray] = []
    for cluster, base in enumerate((e0, e1)):
        for i in range(N_PER_CLUSTER):
            # Two chunks per doc so p_group has same-doc neighbors.
            chunk = Chunk.create(
                text=f"cluster {cluster} chunk {i} content", doc_id=f"doc-{cluster}-{i // 2}"
            )
            chunks.append(chunk)
            chunk_vecs.append(_jitter(base, 0.05, rng))

    train_specs = [(0, 8), (1, 4)]
    train: list[ProductionQuery] = []
    train_vecs: list[np.ndarray] = []
    for cluster, count in train_specs:
        base = (e0, e1)[cluster]
        for i in range(count):
            train.append(
                ProductionQuery(query_id=f"qt-{cluster}-{i}", text=f"train c{cluster} q{i}?")
            )
            train_vecs.append(_jitter(base, 0.05, rng))

    anchor = tuple(ProductionQuery(query_id=f"qa-{i}", text=f"anchor q{i}?") for i in range(4))
    anchor_vecs = [_jitter(e0 if i < 2 else e1, 0.05, rng) for i in range(4)]
    oracle = tuple(ProductionQuery(query_id=f"qo-{i}", text=f"oracle q{i}?") for i in range(4))
    oracle_vecs = [_jitter(e0 if i < 2 else e1, 0.05, rng) for i in range(4)]

    store = EmbeddingStore()
    store.add([c.chunk_id for c in chunks], np.stack(chunk_vecs))
    store.add([q.query_id for q in train], np.stack(train_vecs))
    store.add([q.query_id for q in anchor], np.stack(anchor_vecs))
    store.add([q.query_id for q in oracle], np.stack(oracle_vecs))

    train_matrix = np.stack(train_vecs)
    partition = ReferencePartition.fit(train_matrix, n_clusters=2, seed=0)
    movmf = MovMF(n_components=2, seed=0).fit(train_matrix)
    movmf_demand = demand_from_responsibilities(movmf.responsibilities(train_matrix))
    demand = DemandArtifact(
        p_hat=partition.proportions(train_matrix),
        movmf=movmf,
        movmf_demand=movmf_demand,
        tilted=tilt_weights(movmf_demand, 0.7),
        tau_r=nn_cos_threshold(train_matrix, pct=5.0),
        lam=0.7,
    )
    return Resources(
        chunks=tuple(chunks),
        queries_train=tuple(train),
        queries_anchor=anchor,
        queries_oracle=oracle,
        anchor_qrels={},
        oracle_qrels={},
        embedder=MockEmbedder(dim=D, seed=0),
        generator_llm=MockChatModel(seed=0),
        judge=MockJudge(),
        retriever=DenseInMemoryRetriever(
            chunk_ids=[c.chunk_id for c in chunks], matrix=np.stack(chunk_vecs)
        ),
        embeddings=store,
        partition=partition,
        demand=demand,
        zoo={},
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        seed=0,
    )


def cluster_of(resources: Resources, chunk_id: str) -> int:
    """Reference-partition cluster of a chunk (helper for assertions)."""
    emb = resources.embeddings.get([chunk_id]).astype(np.float64)
    return int(resources.partition.assign(emb)[0])
