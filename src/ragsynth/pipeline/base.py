"""Pipeline contract: Resources, PipelineState, PipelineStep (SPEC §3.1).

``Resources`` is frozen and injected at the composition root
(``pipeline.serialization``); steps never construct adapters (DIP).
All randomness flows from the single config seed through
:meth:`Resources.rng` substreams so runs are reproducible end to end.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Runtime imports: pydantic resolves PipelineState's field annotations when
# the model class is built, so the domain types cannot live in TYPE_CHECKING.
from ragsynth.domain import (
    AnnotationRecord,
    GenerationContext,
    Rejection,
    Seed,
    SyntheticQuery,
)
from ragsynth.pipeline.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from ragsynth.adapters.embedder.base import Embedder
    from ragsynth.adapters.judge.base import RelevanceJudge
    from ragsynth.adapters.llm.base import ChatModel
    from ragsynth.adapters.retriever.base import Retriever
    from ragsynth.datasets.base import DatasetBundle
    from ragsynth.domain import Chunk, ProductionQuery
    from ragsynth.io.artifacts import ArtifactStore
    from ragsynth.io.embeddings import EmbeddingStore
    from ragsynth.metrics.validity.systems import RetrievalSystem
    from ragsynth.sampling.movmf import MovMF
    from ragsynth.sampling.partition import ReferencePartition


def stable_hash64(name: str) -> int:
    """Stable 64-bit integer hash of ``name`` (process-independent).

    Python's builtin ``hash`` is salted per process; reproducible rng
    substreams need a deterministic alternative.
    """
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class DemandArtifact:
    """Fitted demand map: the versioned frozen artifact per benchmark epoch (SPEC §7).

    Attributes:
        p_hat: Demand over the reference partition's clusters (hard labels),
            used by quotas / ESS / reporting.
        movmf: The finer planning density fitted on train queries (A2).
        movmf_demand: Demand over the movMF components
            (``demand_from_responsibilities``).
        tilted: ``tilt_weights(movmf_demand, lam)`` -- the sampling mixture.
        tau_r: On-manifold guard threshold (percentile of production
            NN-cosines, SPEC §7.5).
        lam: The mixture coefficient used for ``tilted``.
    """

    p_hat: NDArray[np.float64]
    movmf: MovMF
    movmf_demand: NDArray[np.float64]
    tilted: NDArray[np.float64]
    tau_r: float
    lam: float


@dataclass(frozen=True)
class Resources:
    """Frozen dependency bundle injected at the composition root (SPEC §3.1).

    Production queries are pre-split train/anchor/oracle (PLAN D10): demand,
    partition, and exemplars derive from ``queries_train`` only.
    """

    chunks: tuple[Chunk, ...]
    queries_train: tuple[ProductionQuery, ...]
    queries_anchor: tuple[ProductionQuery, ...]
    queries_oracle: tuple[ProductionQuery, ...]
    anchor_qrels: Mapping[str, Mapping[str, int]]
    oracle_qrels: Mapping[str, Mapping[str, int]]
    embedder: Embedder
    generator_llm: ChatModel
    judge: RelevanceJudge
    retriever: Retriever
    embeddings: EmbeddingStore
    partition: ReferencePartition
    demand: DemandArtifact
    zoo: Mapping[str, RetrievalSystem]
    artifacts: ArtifactStore
    seed: int
    bundle: DatasetBundle | None = None

    def rng(self, name: str) -> np.random.Generator:
        """Deterministic per-name random substream.

        Two Resources with the same seed give identical streams for the same
        ``name`` and independent streams for different names.
        """
        return np.random.default_rng([self.seed, stable_hash64(name)])

    def with_overrides(self, **kwargs: Any) -> Resources:
        """Return a copy with the given fields replaced (arms use this, PLAN D12)."""
        return replace(self, **kwargs)

    @cached_property
    def chunk_index(self) -> dict[str, Chunk]:
        """Chunk lookup by id (cached; writes instance __dict__, frozen-safe)."""
        return {c.chunk_id: c for c in self.chunks}

    def chunk_embs(self) -> NDArray[np.float64]:
        """Chunk embedding matrix, row i = ``chunks[i]``."""
        return self.embeddings.get([c.chunk_id for c in self.chunks]).astype(np.float64)

    def query_embs(self, which: str) -> NDArray[np.float64]:
        """Query embedding matrix for one split (``train``/``anchor``/``oracle``).

        Raises:
            ValueError: If ``which`` is not a known split name.
        """
        splits = {
            "train": self.queries_train,
            "anchor": self.queries_anchor,
            "oracle": self.queries_oracle,
        }
        if which not in splits:
            raise ValueError(f"unknown query split '{which}'; known: {sorted(splits)}")
        return self.embeddings.get([q.query_id for q in splits[which]]).astype(np.float64)


class PipelineState(BaseModel):
    """Mutable state threaded through the steps (SPEC §3.1).

    ``gate_accepted`` is the gate's working set (dedup compares against it);
    ``accepted`` holds finished :class:`AnnotationRecord` objects.
    ``metrics['gate_reject_reasons']`` carries the per-check tallies -- the
    v2 optimizer's routing signal (SPEC §6.4).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    seeds: list[Seed] = Field(default_factory=list)
    contexts: list[GenerationContext] = Field(default_factory=list)
    candidates: list[SyntheticQuery] = Field(default_factory=list)
    gate_accepted: list[SyntheticQuery] = Field(default_factory=list)
    accepted: list[AnnotationRecord] = Field(default_factory=list)
    rejected: list[Rejection] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class PipelineStep(ABC):
    """One stage of the synthetic-annotation pipeline (SPEC §3.1).

    Lifecycle: ``fit(resources)`` (optional, idempotent) learns anything
    data-dependent; ``run(state)`` consumes and returns the state. Both must
    be side-effect-free outside ``state`` and the step's own artifacts.
    Concrete steps take ``resources`` as their first constructor argument
    and expose their config params via ``to_config`` (JSON-safe only).
    """

    name: ClassVar[str]
    version: ClassVar[str] = "1"

    def fit(self, resources: Resources) -> Self:
        """Learn data-dependent parameters (default: no-op)."""
        return self

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """Consume and return the pipeline state."""

    @abstractmethod
    def to_config(self) -> dict[str, Any]:
        """Return JSON-safe constructor params (round-trips via from_config)."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> Self:
        """Build the step from a config params block plus injected resources."""


STEPS: Registry[PipelineStep] = Registry("pipeline step")
