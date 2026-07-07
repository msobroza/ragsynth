"""Tests for the three seed samplers (SPEC §6.1)."""

import numpy as np
import pytest

from ragsynth.pipeline.base import PipelineState, Resources
from ragsynth.steps.seed_sampler import (
    QuotaSeedSampler,
    SpecSeedSampler,
    UniformSeedSampler,
)
from tests.steps.conftest import cluster_of

STRATA = ["factoid", "howto", "keyword"]


class TestUniform:
    def test_produces_n_seeds_with_clusters_and_strata(self, steps_world: Resources) -> None:
        step = UniformSeedSampler(steps_world, n_seeds=9, p_group=0.0, strata=STRATA)
        state = step.run(PipelineState())
        assert len(state.seeds) == 9
        for seed in state.seeds:
            assert len(seed.chunk_ids) == 1
            assert seed.cluster_id == cluster_of(steps_world, seed.chunk_ids[0])
            assert seed.stratum.dims["query_type"] in STRATA
            assert seed.z is None

    def test_p_group_one_pairs_same_doc_neighbors(self, steps_world: Resources) -> None:
        step = UniformSeedSampler(steps_world, n_seeds=10, p_group=1.0, strata=STRATA)
        state = step.run(PipelineState())
        for seed in state.seeds:
            assert len(seed.chunk_ids) == 2
            docs = {steps_world.chunk_index[cid].doc_id for cid in seed.chunk_ids}
            assert len(docs) == 1  # same-doc neighbor

    def test_deterministic_under_seed(self, steps_world: Resources) -> None:
        run1 = UniformSeedSampler(steps_world, n_seeds=6, strata=STRATA).run(PipelineState())
        run2 = UniformSeedSampler(steps_world, n_seeds=6, strata=STRATA).run(PipelineState())
        assert [s.chunk_ids for s in run1.seeds] == [s.chunk_ids for s in run2.seeds]


class TestQuota:
    def test_allocation_matches_mixture_with_floor(self, steps_world: Resources) -> None:
        # p_hat = [2/3, 1/3]; tilt(lam=.7) = [.7*2/3+.15, .7/3+.15] = [.61667, .38333].
        # n_seeds=12, n_min=3: remaining 6 shared by largest remainder over tilt:
        # 6*.61667=3.7 -> floor 3 (+1 for the .7 remainder), 6*.38333=2.3 -> floor 2.
        # Totals: [3+4, 3+2] = [7, 5].
        step = QuotaSeedSampler(steps_world, n_seeds=12, lam=0.7, n_min=3, p_group=0.0)
        state = step.run(PipelineState())
        counts = np.bincount([s.cluster_id for s in state.seeds], minlength=2)
        assert counts.tolist() == [7, 5]

    def test_seed_chunks_come_from_their_cluster(self, steps_world: Resources) -> None:
        state = QuotaSeedSampler(steps_world, n_seeds=12, p_group=0.0).run(PipelineState())
        for seed in state.seeds:
            assert cluster_of(steps_world, seed.chunk_ids[0]) == seed.cluster_id

    def test_n_seeds_below_floor_capacity_raises(self, steps_world: Resources) -> None:
        with pytest.raises(ValueError, match="n_min"):
            QuotaSeedSampler(steps_world, n_seeds=5, n_min=3).run(PipelineState())

    def test_round_robin_strata_within_cluster(self, steps_world: Resources) -> None:
        state = QuotaSeedSampler(steps_world, n_seeds=12, n_min=3, p_group=0.0, strata=STRATA).run(
            PipelineState()
        )
        by_cluster: dict[int, list[str]] = {}
        for seed in state.seeds:
            by_cluster.setdefault(seed.cluster_id, []).append(seed.stratum.dims["query_type"])
        for values in by_cluster.values():
            expected = [STRATA[i % 3] for i in range(len(values))]
            assert values == expected


class TestSpec:
    def test_seeds_carry_z_and_knn_chunks(self, steps_world: Resources) -> None:
        step = SpecSeedSampler(steps_world, n_seeds=8, n_chunks_per_seed=3, strata=STRATA)
        step.fit(steps_world)
        state = step.run(PipelineState())
        assert len(state.seeds) == 8
        chunk_matrix = steps_world.chunk_embs()
        chunk_ids = [c.chunk_id for c in steps_world.chunks]
        for seed in state.seeds:
            assert seed.z is not None
            z = np.asarray(seed.z)
            assert z.shape == (16,)
            assert np.linalg.norm(z) == pytest.approx(1.0, abs=1e-6)
            assert len(seed.chunk_ids) == 3
            # chunk_ids are exactly the top-3 chunks by cosine to z.
            top3 = np.argsort(-(chunk_matrix @ z))[:3]
            assert set(seed.chunk_ids) == {chunk_ids[i] for i in top3}
            # Cluster is the partition cluster of the target z.
            assert seed.cluster_id == int(steps_world.partition.assign(z[None, :])[0])

    def test_config_round_trip(self, steps_world: Resources) -> None:
        step = SpecSeedSampler(steps_world, n_seeds=8, n_chunks_per_seed=3, strata=STRATA)
        assert step.to_config() == {
            "n_seeds": 8,
            "n_chunks_per_seed": 3,
            "strata": STRATA,
        }
        rebuilt = SpecSeedSampler.from_config(step.to_config(), steps_world)
        assert rebuilt.to_config() == step.to_config()
