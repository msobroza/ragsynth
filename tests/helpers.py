"""Test-only pipeline steps, registered under the ``test.`` prefix.

Import this module (as ``tests.helpers``) to make the fake steps available
in the STEPS registry; registration is idempotent-guarded so accidental
double-imports don't explode.
"""

import contextlib
from typing import Any

from ragsynth.pipeline.base import STEPS, PipelineState, PipelineStep, Resources
from ragsynth.pipeline.registry import RegistryError


class TagStep(PipelineStep):
    """Appends its tag to state.metrics['trace'] (order probe)."""

    name = "test.tag"

    def __init__(self, tag: str = "t") -> None:
        self.tag = tag
        self.fit_calls = 0

    def fit(self, resources: Resources) -> "TagStep":
        self.fit_calls += 1
        return self

    def run(self, state: PipelineState) -> PipelineState:
        state.metrics.setdefault("trace", []).append(self.tag)
        return state

    def to_config(self) -> dict[str, Any]:
        return {"tag": self.tag}

    @classmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> "TagStep":
        return cls(tag=str(config.get("tag", "t")))


class CountStep(PipelineStep):
    """Adds its amount to state.metrics['count']."""

    name = "test.count"

    def __init__(self, amount: int = 1) -> None:
        self.amount = amount

    def run(self, state: PipelineState) -> PipelineState:
        state.metrics["count"] = state.metrics.get("count", 0) + self.amount
        return state

    def to_config(self) -> dict[str, Any]:
        return {"amount": self.amount}

    @classmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> "CountStep":
        return cls(amount=int(config.get("amount", 1)))


for _step_class in (TagStep, CountStep):
    with contextlib.suppress(RegistryError):  # double-import guard
        STEPS.register(_step_class.name)(_step_class)
