"""SPEC §13/§15.1: from_yaml(to_yaml(p)) - identical configs AND identical outputs."""

from pathlib import Path

import yaml

from ragsynth.pipeline.pipeline import Pipeline
from ragsynth.pipeline.serialization import dump_config, make_initial_state
from tests.conftest import TOY_ARM_PARAMS, toy_config


def _with_validator(tmp_path: Path) -> dict:
    config = toy_config(tmp_path)
    config["pipeline"].append(
        {
            "type": "validator",
            "params": {
                "arms": ["a1", "oracle"],
                "n_boot": 20,
                "n_per_arm": 12,
                "reuse_pipeline_for": "a1",
                "arm_params": {key: TOY_ARM_PARAMS[key] for key in ("a1", "oracle")},
                "wc2st_min_per_side": 5,
            },
        }
    )
    return config


def test_yaml_round_trip_identical_config_and_outputs(tmp_path: Path) -> None:
    original_path = tmp_path / "original.yaml"
    original_path.write_text(dump_config(_with_validator(tmp_path)))

    # Cycle 1 canonicalizes: step defaults get materialized into the yaml.
    pipeline_a, _resources_a = Pipeline.from_yaml(original_path)
    canonical = pipeline_a.to_yaml()
    canonical_path = tmp_path / "canonical.yaml"
    canonical_path.write_text(canonical)
    assert dump_config(yaml.safe_load(canonical)) == canonical  # byte-stable dump

    # Cycle 2: from_yaml(to_yaml(p)) is a fixed point on configs (SPEC §13).
    pipeline_b, resources_b = Pipeline.from_yaml(canonical_path)
    regenerated = pipeline_b.to_yaml()
    assert regenerated == canonical
    pipeline_b.fit(resources_b)
    pipeline_b.run(make_initial_state(yaml.safe_load(canonical)))
    first = (tmp_path / "exp" / "metrics.json").read_bytes()

    # Cycle 3: ...and a fixed point on outputs under the fixed seed.
    third_path = tmp_path / "third.yaml"
    third_path.write_text(regenerated)
    pipeline_c, resources_c = Pipeline.from_yaml(third_path)
    pipeline_c.fit(resources_c)
    pipeline_c.run(make_initial_state(yaml.safe_load(regenerated)))
    second = (tmp_path / "exp" / "metrics.json").read_bytes()
    assert first == second


def test_schema_version_2_round_trips_through_pipeline_from_yaml(tmp_path: Path) -> None:
    """schema_version 2 configs load via Pipeline.from_yaml and preserve 2 (v2 README §8)."""
    config = toy_config(tmp_path)
    config["ragsynth"] = {**config["ragsynth"], "schema_version": 2}
    path = tmp_path / "v2_config.yaml"
    path.write_text(dump_config(config))

    pipeline, _resources = Pipeline.from_yaml(path)
    state = make_initial_state(yaml.safe_load(path.read_text()))
    assert state.provenance["config"]["ragsynth"]["schema_version"] == 2

    regenerated = pipeline.to_yaml()
    assert yaml.safe_load(regenerated)["ragsynth"]["schema_version"] == 2
