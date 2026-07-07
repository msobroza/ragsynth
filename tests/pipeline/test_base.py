"""Tests for Resources, PipelineState, and the PipelineStep contract surface."""

from typing import Any

import numpy as np

from ragsynth.pipeline.base import PipelineState, PipelineStep, Resources, stable_hash64
from tests.conftest import make_min_resources


def test_stable_hash64_is_process_independent_and_distinct() -> None:
    assert stable_hash64("seed_sampler") == stable_hash64("seed_sampler")
    assert stable_hash64("a") != stable_hash64("b")


def test_rng_streams_deterministic_across_instances(tmp_path: Any) -> None:
    r1 = make_min_resources(tmp_path, seed=7)
    r2 = make_min_resources(tmp_path, seed=7)
    assert r1.rng("gate").standard_normal(4).tolist() == r2.rng("gate").standard_normal(4).tolist()
    assert (
        r1.rng("gate").standard_normal(4).tolist() != r1.rng("curator").standard_normal(4).tolist()
    )


def test_with_overrides_swaps_one_field(min_resources: Resources) -> None:
    other = min_resources.with_overrides(seed=99)
    assert other.seed == 99
    assert other.chunks == min_resources.chunks
    assert other.embedder is min_resources.embedder


def test_chunk_and_query_embedding_views(min_resources: Resources) -> None:
    embs = min_resources.chunk_embs()
    assert embs.shape == (4, 16)
    np.testing.assert_array_equal(
        embs[0], min_resources.embeddings.get([min_resources.chunks[0].chunk_id])[0]
    )
    assert min_resources.query_embs("train").shape == (6, 16)
    assert min_resources.query_embs("anchor").shape == (3, 16)


def test_query_embs_unknown_split(min_resources: Resources) -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown query split"):
        min_resources.query_embs("dev")


def test_pipeline_state_defaults_are_independent() -> None:
    a = PipelineState()
    b = PipelineState()
    a.metrics["x"] = 1
    a.seeds.append(None)  # type: ignore[arg-type]
    assert b.metrics == {}
    assert b.seeds == []


def test_pipeline_step_default_fit_returns_self(min_resources: Resources) -> None:
    class Dummy(PipelineStep):
        name = "test.dummy"

        def run(self, state: PipelineState) -> PipelineState:
            return state

        def to_config(self) -> dict[str, Any]:
            return {}

        @classmethod
        def from_config(cls, config: dict[str, Any], resources: Resources) -> "Dummy":
            return cls()

    step = Dummy()
    assert step.fit(min_resources) is step
    assert step.version == "1"
