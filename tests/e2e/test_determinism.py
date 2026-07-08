"""SPEC §15.1: same seed => byte-identical metrics.json."""

from pathlib import Path

import pytest

from ragsynth.pipeline.serialization import (
    build_pipeline,
    build_resources,
    make_initial_state,
)
from tests.conftest import TOY_ARM_PARAMS, toy_config


def _run_once(tmp_path: Path) -> bytes:
    config = toy_config(tmp_path)
    config["pipeline"].append(
        {
            "type": "validator",
            "params": {
                "arms": ["a1", "oracle"],
                "n_boot": 30,
                "n_per_arm": 16,
                "reuse_pipeline_for": "a1",
                "arm_params": {key: TOY_ARM_PARAMS[key] for key in ("a1", "oracle")},
                "wc2st_min_per_side": 5,
            },
        }
    )
    resources = build_resources(config)
    pipeline = build_pipeline(config, resources)
    pipeline.fit(resources)
    pipeline.run(make_initial_state(config))
    return (tmp_path / "exp" / "metrics.json").read_bytes()


def test_same_seed_identical_metrics_json(tmp_path: Path) -> None:
    first = _run_once(tmp_path)
    second = _run_once(tmp_path)  # same config, same dirs, fresh objects
    assert first == second


def test_metrics_json_has_no_wallclock(tmp_path: Path) -> None:
    payload = _run_once(tmp_path).decode()
    # PLAN D14: no timestamps/provenance may leak into metrics.json.
    assert "provenance" not in payload
    assert "created_at" not in payload


@pytest.mark.parametrize("seed", [1])
def test_different_seed_different_benchmark(tmp_path: Path, seed: int) -> None:
    base = _run_once(tmp_path)
    config = toy_config(tmp_path)
    config["ragsynth"]["seed"] = seed
    config["pipeline"].append(
        {
            "type": "validator",
            "params": {
                "arms": ["oracle"],
                "n_boot": 30,
                "n_per_arm": 16,
                "arm_params": {"oracle": TOY_ARM_PARAMS["oracle"]},
                "wc2st_min_per_side": 5,
            },
        }
    )
    resources = build_resources(config)
    pipeline = build_pipeline(config, resources)
    pipeline.fit(resources)
    pipeline.run(make_initial_state(config))
    assert (tmp_path / "exp" / "metrics.json").read_bytes() != base
