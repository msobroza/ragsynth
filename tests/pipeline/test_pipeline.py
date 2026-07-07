"""Tests for the Pipeline runner."""

import pytest

from ragsynth.pipeline.base import STEPS, Resources
from ragsynth.pipeline.pipeline import Pipeline
from ragsynth.pipeline.registry import RegistryError
from tests.helpers import CountStep, TagStep


def test_run_threads_state_in_order() -> None:
    pipe = Pipeline([TagStep("a"), TagStep("b"), CountStep(2)])
    state = pipe.run()
    assert state.metrics["trace"] == ["a", "b"]
    assert state.metrics["count"] == 2


def test_fit_visits_every_step(min_resources: Resources) -> None:
    steps = [TagStep("a"), TagStep("b")]
    Pipeline(steps).fit(min_resources)
    assert [s.fit_calls for s in steps] == [1, 1]


def test_named_steps_and_get_params() -> None:
    pipe = Pipeline([TagStep("x"), CountStep(5)])
    assert set(pipe.named_steps) == {"test.tag", "test.count"}
    assert pipe.get_params() == {"test.tag__tag": "x", "test.count__amount": 5}


def test_to_step_configs_round_trip_shape() -> None:
    pipe = Pipeline([CountStep(3)])
    assert pipe.to_step_configs() == [{"type": "test.count", "params": {"amount": 3}}]


def test_steps_registered_and_from_config_round_trip(min_resources: Resources) -> None:
    step_class = STEPS.get("test.count")
    step = step_class.from_config({"amount": 4}, min_resources)
    assert step.to_config() == {"amount": 4}


def test_to_yaml_without_config_raises() -> None:
    with pytest.raises(ValueError, match="config snapshot"):
        Pipeline([CountStep()]).to_yaml()


def test_registry_unknown_step_error_lists_known() -> None:
    with pytest.raises(RegistryError, match="unknown pipeline step"):
        STEPS.get("does.not.exist")
