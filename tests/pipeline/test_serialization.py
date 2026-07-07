"""Tests for config loading/validation and the composition root."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from ragsynth.datasets.base import DATASETS, DatasetBundle
from ragsynth.domain import Chunk, ProductionQuery
from ragsynth.pipeline.registry import RegistryError
from ragsynth.pipeline.serialization import (
    build_resources,
    config_hash,
    dump_config,
    load_config,
    validate_config,
)

try:
    DATASETS.get("test.tiny")
except RegistryError:

    @DATASETS.register("test.tiny")
    class _TinyDataset:
        """Deterministic 8-chunk / 12-query dataset for composition tests."""

        @classmethod
        def build(cls, params: dict[str, Any], seed: int) -> DatasetBundle:
            chunks = tuple(
                Chunk.create(text=f"the topic {i % 2} fact number {i}", doc_id=f"d{i % 2}")
                for i in range(8)
            )
            queries = [
                ProductionQuery(query_id=f"q{i}", text=f"question about topic {i % 2} nr {i}?")
                for i in range(12)
            ]
            return DatasetBundle(
                chunks=chunks,
                queries_train=tuple(queries[:6]),
                queries_anchor=tuple(queries[6:9]),
                queries_oracle=tuple(queries[9:]),
            )


def _tiny_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "ragsynth": {"schema_version": 1, "name": "tiny", "seed": 0},
        "resources": {
            "dataset": {"type": "test.tiny", "params": {}},
            "embedder": {"type": "mock", "params": {"dim": 16}},
            "generator_llm": {"type": "mock"},
            "judge_llm": {"type": "mock"},
            "retriever": {"type": "dense_inmemory"},
            "partition": {"n_clusters": 2},
            "demand": {"n_components": 2},
        },
        "artifacts_dir": str(tmp_path / "artifacts"),
        "pipeline": [{"type": "test.count", "params": {"amount": 1}}],
    }


@pytest.fixture
def tiny_config(tmp_path: Path) -> dict[str, Any]:
    import tests.helpers  # noqa: F401  (registers test.count)

    return _tiny_config(tmp_path)


def test_load_config_valid(tmp_path: Path, tiny_config: dict[str, Any]) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(tiny_config))
    assert load_config(path)["ragsynth"]["name"] == "tiny"


def test_load_config_bad_schema_version(tmp_path: Path, tiny_config: dict[str, Any]) -> None:
    tiny_config["ragsynth"]["schema_version"] = 2
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(tiny_config))
    with pytest.raises(ValueError, match="schema_version"):
        load_config(path)


def test_unknown_step_type_lists_known_keys(tiny_config: dict[str, Any]) -> None:
    tiny_config["pipeline"] = [{"type": "no.such.step", "params": {}}]
    with pytest.raises(RegistryError, match=r"unknown pipeline step 'no\.such\.step'"):
        validate_config(tiny_config)


def test_missing_block_raises(tiny_config: dict[str, Any]) -> None:
    del tiny_config["artifacts_dir"]
    with pytest.raises(ValueError, match="artifacts_dir"):
        validate_config(tiny_config)


def test_same_family_judge_warns(
    tiny_config: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        warnings = validate_config(tiny_config)
    assert any("same-family" in w or "model family" in w for w in warnings)
    assert any("SIGIR 2025" in r.message for r in caplog.records)


def test_dump_config_byte_stable_round_trip(tiny_config: dict[str, Any]) -> None:
    once = dump_config(tiny_config)
    twice = dump_config(yaml.safe_load(once))
    assert once == twice
    assert config_hash(yaml.safe_load(once)) == config_hash(tiny_config)


def test_build_resources_assembles_everything(tiny_config: dict[str, Any]) -> None:
    resources = build_resources(tiny_config)
    assert len(resources.chunks) == 8
    assert len(resources.queries_train) == 6
    # Every chunk and query is embedded.
    for chunk in resources.chunks:
        assert chunk.chunk_id in resources.embeddings
    # Partition + demand fitted on the train split with configured sizes.
    assert resources.partition.n_clusters == 2
    assert resources.demand.p_hat.shape == (2,)
    assert resources.demand.p_hat.sum() == pytest.approx(1.0)
    assert resources.demand.tilted.shape == (2,)
    assert 0.0 < resources.demand.tau_r <= 1.0
    # Zoo: exact + 7 distortions + 4 low-rank = 12 systems.
    assert len(resources.zoo) == 12
    # Qrels fell back to nearest-chunk relabeling (PLAN D17).
    assert set(resources.anchor_qrels) == {q.query_id for q in resources.queries_anchor}
    assert all(len(v) == 1 for v in resources.anchor_qrels.values())
    # Artifacts persisted with manifest hashes.
    assert "partition-c2.npz" in resources.artifacts.manifest
    assert "demand-movmf.npz" in resources.artifacts.manifest


def test_build_resources_deterministic(tmp_path: Path, tiny_config: dict[str, Any]) -> None:
    r1 = build_resources(tiny_config)
    cfg2 = _tiny_config(tmp_path)
    r2 = build_resources(cfg2)
    np.testing.assert_array_equal(r1.demand.p_hat, r2.demand.p_hat)
    np.testing.assert_array_equal(r1.demand.movmf_demand, r2.demand.movmf_demand)
