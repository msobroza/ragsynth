"""Seed samplers: what to generate from (SPEC §6.1).

Three strategies as separate registry entries (Strategy pattern):
``seed_sampler.uniform`` (A0), ``seed_sampler.quota`` (A1, lambda-mixture
allocation), ``seed_sampler.spec`` (A2, demand-tilted movMF targets).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np

from ragsynth.domain import Seed, Stratum
from ragsynth.pipeline.base import STEPS, PipelineStep
from ragsynth.sampling.demand import tilt_weights
from ragsynth.sampling.spec_sampler import SpecSampler

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ragsynth.pipeline.base import PipelineState, Resources

DEFAULT_STRATA = ["factoid", "howto", "keyword"]


def _round_robin_strata(cluster_ids: list[int], strata: list[str]) -> list[Stratum]:
    """Cycle the strata list independently within each cluster (SPEC §6.1)."""
    counters: dict[int, int] = {}
    out: list[Stratum] = []
    for cluster in cluster_ids:
        i = counters.get(cluster, 0)
        counters[cluster] = i + 1
        out.append(Stratum(dims={"query_type": strata[i % len(strata)]}))
    return out


def _maybe_group(
    resources: Resources,
    chunk_idx: int,
    p_group: float,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """With prob ``p_group``, pair the chunk with a same-doc neighbor."""
    chunk = resources.chunks[chunk_idx]
    if p_group > 0 and rng.random() < p_group:
        neighbors = [
            c.chunk_id
            for c in resources.chunks
            if c.doc_id == chunk.doc_id and c.chunk_id != chunk.chunk_id
        ]
        if neighbors:
            partner = neighbors[int(rng.integers(0, len(neighbors)))]
            return (chunk.chunk_id, partner)
    return (chunk.chunk_id,)


def _chunk_clusters(resources: Resources) -> NDArray[np.int_]:
    return resources.partition.assign(resources.chunk_embs())


@STEPS.register("seed_sampler.uniform")
class UniformSeedSampler(PipelineStep):
    """Uniform chunk seeds, no demand steering (arm A0)."""

    name = "seed_sampler.uniform"

    def __init__(
        self,
        resources: Resources,
        n_seeds: int,
        p_group: float = 0.0,
        strata: list[str] | None = None,
    ) -> None:
        self._resources = resources
        self.n_seeds = n_seeds
        self.p_group = p_group
        self.strata = list(strata) if strata is not None else list(DEFAULT_STRATA)

    def run(self, state: PipelineState) -> PipelineState:
        """Sample ``n_seeds`` uniform chunk seeds into ``state.seeds``."""
        resources = self._resources
        rng = resources.rng(self.name)
        clusters = _chunk_clusters(resources)
        indices = rng.integers(0, len(resources.chunks), size=self.n_seeds)
        cluster_ids = [int(clusters[i]) for i in indices]
        strata = _round_robin_strata(cluster_ids, self.strata)
        for n, (idx, cluster, stratum) in enumerate(zip(indices, cluster_ids, strata, strict=True)):
            state.seeds.append(
                Seed(
                    seed_id=f"seed-uniform-{n:05d}",
                    chunk_ids=_maybe_group(resources, int(idx), self.p_group, rng),
                    cluster_id=cluster,
                    stratum=stratum,
                )
            )
        return state

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {"n_seeds": self.n_seeds, "p_group": self.p_group, "strata": self.strata}

    @classmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> Self:
        """Build from a config params block."""
        return cls(resources, **config)


@STEPS.register("seed_sampler.quota")
class QuotaSeedSampler(PipelineStep):
    """Lambda-mixture cluster quotas over the frozen partition (arm A1).

    Allocation: every cluster gets the ``n_min`` floor; the remainder is
    distributed proportionally to ``tilt_weights(p_hat, lam)`` by largest
    remainder (Jelinek-Mercer-style smoothing; coverage floors per BCG
    arXiv 2510.00001).
    """

    name = "seed_sampler.quota"

    def __init__(
        self,
        resources: Resources,
        n_seeds: int,
        lam: float = 0.7,
        n_min: int = 3,
        p_group: float = 0.2,
        strata: list[str] | None = None,
    ) -> None:
        self._resources = resources
        self.n_seeds = n_seeds
        self.lam = lam
        self.n_min = n_min
        self.p_group = p_group
        self.strata = list(strata) if strata is not None else list(DEFAULT_STRATA)

    def _allocate(self, p_hat: NDArray[np.float64]) -> NDArray[np.int_]:
        """Proportional largest-remainder allocation with a per-cluster floor.

        ``n_c`` is proportional to ``tilt_weights(p_hat, lam)`` (SPEC §6.1);
        clusters below ``n_min`` are raised to the floor, the excess taken
        back from the largest allocations (deterministic waterfill).
        """
        n_clusters = len(p_hat)
        if self.n_seeds < n_clusters * self.n_min:
            raise ValueError(
                f"n_seeds={self.n_seeds} cannot satisfy the n_min={self.n_min} floor "
                f"across {n_clusters} clusters (need >= {n_clusters * self.n_min})"
            )
        tilted = tilt_weights(p_hat, self.lam)
        shares = tilted * self.n_seeds
        counts = np.floor(shares).astype(np.int_)
        shortfall = self.n_seeds - int(counts.sum())
        if shortfall > 0:
            order = np.argsort(-(shares - counts), kind="stable")
            counts[order[:shortfall]] += 1
        deficits = np.flatnonzero(counts < self.n_min)
        need = int((self.n_min - counts[deficits]).sum())
        counts[deficits] = self.n_min
        for _ in range(need):
            donor = int(np.argmax(np.where(counts > self.n_min, counts, -1)))
            counts[donor] -= 1
        return counts

    def run(self, state: PipelineState) -> PipelineState:
        """Sample quota-allocated seeds, chunks drawn within their cluster."""
        resources = self._resources
        rng = resources.rng(self.name)
        chunk_clusters = _chunk_clusters(resources)
        counts = self._allocate(resources.demand.p_hat)
        cluster_ids: list[int] = []
        chosen: list[int] = []
        for cluster, count in enumerate(counts):
            pool = np.flatnonzero(chunk_clusters == cluster)
            if len(pool) == 0:
                pool = np.arange(len(resources.chunks))
            picks = rng.choice(pool, size=int(count))
            chosen.extend(int(i) for i in picks)
            cluster_ids.extend([cluster] * int(count))
        strata = _round_robin_strata(cluster_ids, self.strata)
        for n, (idx, cluster, stratum) in enumerate(zip(chosen, cluster_ids, strata, strict=True)):
            state.seeds.append(
                Seed(
                    seed_id=f"seed-quota-{n:05d}",
                    chunk_ids=_maybe_group(resources, idx, self.p_group, rng),
                    cluster_id=cluster,
                    stratum=stratum,
                )
            )
        return state

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {
            "n_seeds": self.n_seeds,
            "lam": self.lam,
            "n_min": self.n_min,
            "p_group": self.p_group,
            "strata": self.strata,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> Self:
        """Build from a config params block."""
        return cls(resources, **config)


@STEPS.register("seed_sampler.spec")
class SpecSeedSampler(PipelineStep):
    """Spec-first target sampling from the demand-tilted movMF (arm A2).

    Samples guarded targets ``z`` (SPEC §7.5), attaches the kNN chunks as
    the seed's evidence/gold set, and stores ``z`` for the generator's
    target check. Cluster id is the partition cluster of ``z`` itself.
    """

    name = "seed_sampler.spec"

    def __init__(
        self,
        resources: Resources,
        n_seeds: int,
        n_chunks_per_seed: int = 5,
        strata: list[str] | None = None,
    ) -> None:
        self._resources = resources
        self.n_seeds = n_seeds
        self.n_chunks_per_seed = n_chunks_per_seed
        self.strata = list(strata) if strata is not None else list(DEFAULT_STRATA)
        self._sampler: SpecSampler | None = None

    def fit(self, resources: Resources) -> Self:
        """Assemble the guarded sampler from the demand artifact."""
        demand = resources.demand
        self._sampler = SpecSampler(
            model=demand.movmf,
            tilted_weights=demand.tilted,
            prod_emb=resources.query_embs("train"),
            tau_r=demand.tau_r,
        )
        return self

    def run(self, state: PipelineState) -> PipelineState:
        """Sample targets and attach kNN chunks."""
        resources = self._resources
        if self._sampler is None:
            self.fit(resources)
        assert self._sampler is not None  # noqa: S101 - narrowing for mypy
        rng = resources.rng(self.name)
        z_all, _components = self._sampler.sample(self.n_seeds, rng)
        chunk_embs = resources.chunk_embs()
        chunk_ids = [c.chunk_id for c in resources.chunks]
        cluster_ids = [int(c) for c in resources.partition.assign(z_all)]
        strata = _round_robin_strata(cluster_ids, self.strata)
        for n, (z, cluster, stratum) in enumerate(zip(z_all, cluster_ids, strata, strict=True)):
            top = np.argsort(-(chunk_embs @ z), kind="stable")[: self.n_chunks_per_seed]
            state.seeds.append(
                Seed(
                    seed_id=f"seed-spec-{n:05d}",
                    chunk_ids=tuple(chunk_ids[int(i)] for i in top),
                    cluster_id=cluster,
                    stratum=stratum,
                    z=[float(v) for v in z],
                )
            )
        return state

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {
            "n_seeds": self.n_seeds,
            "n_chunks_per_seed": self.n_chunks_per_seed,
            "strata": self.strata,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> Self:
        """Build from a config params block."""
        return cls(resources, **config)
