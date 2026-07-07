"""Arm presets: thin compositions of steps + configs (SPEC §10)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from ragsynth.pipeline.base import PipelineState
from ragsynth.pipeline.pipeline import Pipeline
from ragsynth.pipeline.registry import Registry
from ragsynth.steps.context_assembler import ContextAssembler
from ragsynth.steps.curator import Curator
from ragsynth.steps.gate import VerificationGate
from ragsynth.steps.generator import QueryGenerator
from ragsynth.steps.qrel_builder import QrelBuilder

if TYPE_CHECKING:
    from ragsynth.domain import AnnotationRecord
    from ragsynth.pipeline.base import PipelineStep, Resources

DEFAULT_GATE: dict[str, Any] = {
    "checks": ["dedup", "zero_context", "answerability", "round_trip", "uniqueness"],
    "dedup": {"cos_threshold": 0.95},
    "round_trip": {"k": 10},
    "uniqueness": {"mode": "promote", "top_m": 5},
}


class ArmPreset(ABC):
    """One experiment arm: how to produce an AnnotationRecord set.

    Presets compose existing steps with arm-specific defaults -- no new
    logic lives here (SPEC §10). ``llm_override`` in params swaps the
    generator ChatModel via ``Resources.with_overrides`` (PLAN D12).
    """

    name: ClassVar[str]

    @abstractmethod
    def run(self, resources: Resources, params: dict[str, Any]) -> list[AnnotationRecord]:
        """Produce the arm's records."""


ARMS: Registry[ArmPreset] = Registry("arm")


class GenerativeArmPreset(ArmPreset):
    """Shared machinery: sampler + assembler + generator + gate + qrels + curator."""

    k_style_default: ClassVar[int] = 3

    @abstractmethod
    def sampler(self, resources: Resources, params: dict[str, Any]) -> PipelineStep:
        """The arm-specific seed sampler."""

    def build_steps(
        self, resources: Resources, params: dict[str, Any]
    ) -> tuple[Resources, list[PipelineStep]]:
        """Resolve overrides and assemble the generation sub-pipeline."""
        override = params.get("llm_override")
        if override is not None:
            from ragsynth.adapters.llm.base import CHAT_MODELS

            chat = CHAT_MODELS.get(override["type"]).from_config(
                override.get("params") or {}, resources.bundle, resources.rng("llm_override")
            )
            resources = resources.with_overrides(generator_llm=chat)
        gate_config = {**DEFAULT_GATE, **params.get("gate", {})}
        steps: list[PipelineStep] = [
            self.sampler(resources, params),
            ContextAssembler(resources, k_style=params.get("k_style", self.k_style_default)),
            QueryGenerator(resources, **params.get("generator", {})),
            VerificationGate.from_config(gate_config, resources),
            QrelBuilder(resources, **params.get("qrels", {})),
            Curator(resources, **params.get("curator", {})),
        ]
        return resources, steps

    def run(self, resources: Resources, params: dict[str, Any]) -> list[AnnotationRecord]:
        """Fit and run the sub-pipeline; return its accepted records."""
        arm_resources, steps = self.build_steps(resources, params)
        pipeline = Pipeline(steps)
        pipeline.fit(arm_resources)
        state = pipeline.run(PipelineState(provenance={"benchmark_version": f"arm-{self.name}"}))
        return state.accepted


def run_arm(
    name: str, resources: Resources, params: dict[str, Any] | None = None
) -> list[AnnotationRecord]:
    """Run one arm preset by registry name and return its records."""
    preset = ARMS.get(name)()
    return preset.run(resources, params or {})
