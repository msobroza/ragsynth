"""A0: naive chunk-first generation -- uniform seeds, no steering (SPEC §10)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from ragsynth.arms.base import ARMS, GenerativeArmPreset
from ragsynth.steps.seed_sampler import UniformSeedSampler

if TYPE_CHECKING:
    from ragsynth.pipeline.base import PipelineStep, Resources


@ARMS.register("a0")
class A0Naive(GenerativeArmPreset):
    """Uniform chunk seeds, zero style exemplars: the untuned baseline."""

    name = "a0"
    k_style_default: ClassVar[int] = 0

    def sampler(self, resources: Resources, params: dict[str, Any]) -> PipelineStep:
        """Uniform seeds with no chunk-grouping by default."""
        uniform = params.get("uniform", {})
        return UniformSeedSampler(
            resources,
            n_seeds=params.get("n_seeds", 200),
            p_group=uniform.get("p_group", 0.0),
            strata=uniform.get("strata"),
        )
