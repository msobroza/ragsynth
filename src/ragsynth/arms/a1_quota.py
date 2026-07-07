"""A1: cluster-quota chunk-first with exemplar steering (SPEC §10)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragsynth.arms.base import ARMS, GenerativeArmPreset
from ragsynth.steps.seed_sampler import QuotaSeedSampler

if TYPE_CHECKING:
    from ragsynth.pipeline.base import PipelineStep, Resources


@ARMS.register("a1")
class A1Quota(GenerativeArmPreset):
    """Lambda-mixture quotas over the frozen partition + style exemplars."""

    name = "a1"

    def sampler(self, resources: Resources, params: dict[str, Any]) -> PipelineStep:
        """Quota seeds with the SPEC §6.1 defaults."""
        quota = params.get("quota", {})
        return QuotaSeedSampler(
            resources,
            n_seeds=params.get("n_seeds", 200),
            lam=quota.get("lam", 0.7),
            n_min=quota.get("n_min", 3),
            p_group=quota.get("p_group", 0.2),
            strata=quota.get("strata"),
        )
