"""Context assembler: what the generator sees (SPEC §6.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np

from ragsynth.domain import GenerationContext
from ragsynth.pipeline.base import STEPS, PipelineStep

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ragsynth.domain import Seed
    from ragsynth.pipeline.base import PipelineState, Resources

_INSTRUCTIONS = {
    "factoid": "Write one specific factual question answerable from the evidence.",
    "howto": "Write one how-do-I procedural question answerable from the evidence.",
    "keyword": "Write one terse keyword-style search (not a full sentence) targeting the evidence.",
}
_DEFAULT_INSTRUCTION = "Write one realistic search query answerable from the evidence."


@STEPS.register("context_assembler")
class ContextAssembler(PipelineStep):
    """Chunk texts + nearest-production-query exemplars + stratum instruction.

    Exemplars prefer train queries from the seed's own cluster (falling back
    to the global pool); similarity anchor is the seed's ``z`` when present
    (A2), else the mean of its chunk embeddings. The ``two_step`` (tuple ->
    query phrasing decoupling, Hamel/Shankar) and ``blind_summary``
    (ViDoRe V2) modes are v2 stubs and fail fast if enabled (SPEC §6.2).
    """

    name = "context_assembler"

    def __init__(
        self,
        resources: Resources,
        k_style: int = 3,
        *,
        two_step: bool = False,
        blind_summary: bool = False,
    ) -> None:
        if two_step:
            raise NotImplementedError("two_step phrasing decoupling is a v2 feature (SPEC §6.2)")
        if blind_summary:
            raise NotImplementedError("blind_summary mode is a v2 feature (SPEC §6.2)")
        self._resources = resources
        self.k_style = k_style
        self.two_step = two_step
        self.blind_summary = blind_summary
        self._train_labels: NDArray[np.int_] | None = None

    def fit(self, resources: Resources) -> Self:
        """Cache the train queries' partition labels for in-cluster preference."""
        self._train_labels = resources.partition.assign(resources.query_embs("train"))
        return self

    def _anchor(self, seed: Seed) -> NDArray[np.float64]:
        if seed.z is not None:
            return np.asarray(seed.z, dtype=np.float64)
        embs = self._resources.embeddings.get(list(seed.chunk_ids)).astype(np.float64)
        mean = embs.mean(axis=0)
        return np.asarray(mean / max(float(np.linalg.norm(mean)), 1e-12), dtype=np.float64)

    def _exemplars(self, seed: Seed) -> tuple[str, ...]:
        if self.k_style <= 0:
            return ()
        resources = self._resources
        if self._train_labels is None:
            self.fit(resources)
        assert self._train_labels is not None  # noqa: S101 - narrowing for mypy
        anchor = self._anchor(seed)
        train_embs = resources.query_embs("train")
        sims = train_embs @ anchor
        in_cluster = np.flatnonzero(self._train_labels == seed.cluster_id)
        order = in_cluster[np.argsort(-sims[in_cluster], kind="stable")].tolist()
        if len(order) < self.k_style:
            rest = np.argsort(-sims, kind="stable")
            order += [int(i) for i in rest if int(i) not in set(order)]
        return tuple(resources.queries_train[int(i)].text for i in order[: self.k_style])

    def run(self, state: PipelineState) -> PipelineState:
        """Assemble one GenerationContext per seed."""
        resources = self._resources
        for seed in state.seeds:
            state.contexts.append(
                GenerationContext(
                    seed=seed,
                    chunk_texts=tuple(resources.chunk_index[cid].text for cid in seed.chunk_ids),
                    style_exemplars=self._exemplars(seed),
                    instruction=_INSTRUCTIONS.get(
                        seed.stratum.dims.get("query_type", ""), _DEFAULT_INSTRUCTION
                    ),
                )
            )
        return state

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {
            "k_style": self.k_style,
            "two_step": self.two_step,
            "blind_summary": self.blind_summary,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> Self:
        """Build from a config params block."""
        return cls(resources, **config)
