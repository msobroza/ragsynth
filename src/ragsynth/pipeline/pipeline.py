"""The Pipeline runner: sklearn-style composition over PipelineSteps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragsynth.pipeline.base import PipelineState

if TYPE_CHECKING:
    from pathlib import Path

    from ragsynth.pipeline.base import PipelineStep, Resources


class Pipeline:
    """Ordered step chain with sklearn ergonomics (SPEC §3.1).

    ``Pipeline.from_yaml`` (see :mod:`ragsynth.pipeline.serialization`) is
    the ONLY deserialization entrypoint (SPEC §13); construction from step
    instances is for tests and presets.
    """

    def __init__(self, steps: list[PipelineStep], config: dict[str, Any] | None = None) -> None:
        self.steps = steps
        self.config = config

    @property
    def named_steps(self) -> dict[str, PipelineStep]:
        """Steps keyed by their registry name."""
        return {step.name: step for step in self.steps}

    def fit(self, resources: Resources) -> Pipeline:
        """Fit every step in order (idempotent per the step contract)."""
        for step in self.steps:
            step.fit(resources)
        return self

    def run(self, state: PipelineState | None = None) -> PipelineState:
        """Thread the state through every step in order."""
        current = state if state is not None else PipelineState()
        for step in self.steps:
            current = step.run(current)
        return current

    def get_params(self) -> dict[str, Any]:
        """Flat ``step__param`` view of every step's config (read-only in v1)."""
        return {
            f"{step.name}__{key}": value
            for step in self.steps
            for key, value in step.to_config().items()
        }

    def to_step_configs(self) -> list[dict[str, Any]]:
        """The ``pipeline:`` config section regenerated from live steps."""
        return [{"type": step.name, "params": step.to_config()} for step in self.steps]

    def to_yaml(self) -> str:
        """Serialize the full config (byte-stable; SPEC §13).

        Raises:
            ValueError: If the pipeline was built without a config snapshot.
        """
        from ragsynth.pipeline.serialization import dump_config

        if self.config is None:
            raise ValueError("Pipeline.to_yaml needs the config snapshot (build via from_yaml)")
        config = dict(self.config)
        config["pipeline"] = self.to_step_configs()
        return dump_config(config)

    @classmethod
    def from_yaml(cls, path: Path) -> tuple[Pipeline, Resources]:
        """Load config, build resources and pipeline (SPEC §13 entrypoint)."""
        from ragsynth.pipeline.serialization import build_pipeline, build_resources, load_config

        config = load_config(path)
        resources = build_resources(config)
        return build_pipeline(config, resources), resources
