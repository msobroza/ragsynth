"""A2: spec-first generation from demand-tilted movMF targets (SPEC §10)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragsynth.arms.base import ARMS, GenerativeArmPreset
from ragsynth.steps.seed_sampler import SpecSeedSampler

if TYPE_CHECKING:
    from ragsynth.pipeline.base import PipelineStep, Resources


@ARMS.register("a2")
class A2Spec(GenerativeArmPreset):
    """Guarded z-sampling + kNN evidence + generator target check."""

    name = "a2"

    def sampler(self, resources: Resources, params: dict[str, Any]) -> PipelineStep:
        """Spec-first seeds (kNN chunk count is the main knob; PLAN notes the
        toy config uses 1 to reproduce the prototype's single-gold table).
        """
        spec = params.get("spec", {})
        return SpecSeedSampler(
            resources,
            n_seeds=params.get("n_seeds", 200),
            n_chunks_per_seed=spec.get("n_chunks_per_seed", 5),
            strata=spec.get("strata"),
        )
