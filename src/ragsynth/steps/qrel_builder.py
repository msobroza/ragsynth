"""Qrel builder: binary v1 annotations (SPEC §6.5)."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from ragsynth.domain import AnnotationRecord
from ragsynth.pipeline.base import STEPS, PipelineStep
from ragsynth.pipeline.registry import Registry

if TYPE_CHECKING:
    from ragsynth.domain import SyntheticQuery
    from ragsynth.pipeline.base import PipelineState, Resources


class QrelStrategy(ABC):
    """How qrels are graded; ``binary`` is the only v1 strategy.

    Graded/pooled qrels (UMBRELA, Upadhyay/Clarke/Lin 2024) plug in here
    as a v2 registry entry (SPEC §2.2, R5).
    """

    name: str

    @abstractmethod
    def build(self, candidate: SyntheticQuery, resources: Resources) -> dict[str, int]:
        """Return chunk_id -> grade for one accepted candidate."""


QREL_STRATEGIES: Registry[QrelStrategy] = Registry("qrel strategy")


@QREL_STRATEGIES.register("binary")
class BinaryQrels(QrelStrategy):
    """Gold = seed chunks + uniqueness promotions, all at grade 1."""

    name = "binary"

    def build(self, candidate: SyntheticQuery, resources: Resources) -> dict[str, int]:
        """Collect seed chunks and gate promotions at grade 1."""
        gold = list(candidate.seed.chunk_ids) + list(candidate.gen_meta.get("promoted", []))
        return dict.fromkeys(gold, 1)


@QREL_STRATEGIES.register("relabel_nearest")
class RelabelNearestQrels(QrelStrategy):
    """Gold = the single nearest chunk of the EMITTED query (grade 1).

    The gate-style relabeling the toy world requires (SPEC §10; prototype
    arm docstrings: "mirroring what the round-trip + uniqueness gate
    produces", Gecko-style). Promotions are intentionally ignored -- the
    emitted query's nearest chunk IS the annotation.
    """

    name = "relabel_nearest"

    def build(self, candidate: SyntheticQuery, resources: Resources) -> dict[str, int]:
        """Relabel to the nearest chunk by cosine; fall back to seed gold."""
        ref = candidate.embedding_ref
        if ref is None or ref not in resources.embeddings:
            return dict.fromkeys(candidate.seed.chunk_ids, 1)
        emb = resources.embeddings.get([ref])[0]
        nearest = int(np.argmax(resources.chunk_embs() @ emb.astype(np.float64)))
        return {resources.chunks[nearest].chunk_id: 1}


@STEPS.register("qrel_builder")
class QrelBuilder(PipelineStep):
    """Turn gate-accepted candidates into AnnotationRecords (SPEC §6.5).

    ``content_hashes`` snapshot every annotated chunk's text hash so the v2
    churn lifecycle needs no migration; ``crucial`` = all gold in v1 (the
    v3 masking-ablation check refines it).
    """

    name = "qrel_builder"

    def __init__(self, resources: Resources, strategy: str = "binary") -> None:
        self._resources = resources
        self.strategy = strategy
        self._strategy_impl = QREL_STRATEGIES.get(strategy)()

    def run(self, state: PipelineState) -> PipelineState:
        """Emit one AnnotationRecord per gate-accepted candidate."""
        resources = self._resources
        benchmark_version = str(state.provenance.get("benchmark_version", "v1"))
        created_at = datetime.now(tz=UTC)
        for candidate in state.gate_accepted:
            qrels = self._strategy_impl.build(candidate, resources)
            record_id = hashlib.sha256(
                f"{candidate.query_id}|{candidate.text}".encode()
            ).hexdigest()[:16]
            state.accepted.append(
                AnnotationRecord(
                    record_id=f"rec-{record_id}",
                    query=candidate,
                    qrels=qrels,
                    crucial=tuple(sorted(qrels)),
                    stratum=candidate.seed.stratum,
                    gate_meta=dict(candidate.gen_meta.get("gate", {})),
                    content_hashes={
                        cid: resources.chunk_index[cid].content_hash
                        for cid in qrels
                        if cid in resources.chunk_index
                    },
                    benchmark_version=benchmark_version,
                    created_at=created_at,
                )
            )
        return state

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {"strategy": self.strategy}

    @classmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> Self:
        """Build from a config params block."""
        return cls(resources, **config)
