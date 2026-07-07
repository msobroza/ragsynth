"""Tests for the validator step (SPEC §6.7) on a small toy world."""

import json
from pathlib import Path

import pytest

from ragsynth.pipeline.base import PipelineState, Resources
from ragsynth.pipeline.serialization import build_pipeline, build_resources, config_hash
from ragsynth.steps.validator import Validator
from tests.conftest import TOY_ARM_PARAMS, toy_config

ARM_NAMES = ["a0", "a1", "a2", "oracle"]


@pytest.fixture(scope="module")
def validated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Resources, PipelineState, Path]:
    """Run the small toy pipeline + validator once; share across tests."""
    tmp_path = tmp_path_factory.mktemp("validator")
    config = toy_config(tmp_path)
    resources = build_resources(config)
    pipeline = build_pipeline(config, resources)
    state = PipelineState(
        provenance={
            "name": config["ragsynth"]["name"],
            "seed": config["ragsynth"]["seed"],
            "config": config,
            "config_hash": config_hash(config),
            "benchmark_version": f"{config['ragsynth']['name']}@{config_hash(config)[:8]}",
        }
    )
    pipeline.fit(resources)
    state = pipeline.run(state)
    assert state.accepted, "outer pipeline produced no records - toy world misconfigured"
    validator = Validator(
        resources,
        arms=ARM_NAMES,
        n_boot=50,
        n_per_arm=24,
        reuse_pipeline_for="a1",
        arm_params=TOY_ARM_PARAMS,
    )
    validator.fit(resources)
    state = validator.run(state)
    return resources, state, tmp_path / "exp"


def test_report_has_one_block_per_arm(validated: tuple[Resources, PipelineState, Path]) -> None:
    _resources, state, _out = validated
    report = state.metrics["eval_report"]
    assert set(report["arms"]) == set(ARM_NAMES)


def test_reuse_pipeline_for_marks_a1(validated: tuple[Resources, PipelineState, Path]) -> None:
    _resources, state, _out = validated
    report = state.metrics["eval_report"]
    assert report["arms"]["a1"]["reused_pipeline_records"] is True
    assert all(
        report["arms"][a]["reused_pipeline_records"] is False for a in ("a0", "a2", "oracle")
    )
    # The a1 block was computed from the outer pipeline's accepted set.
    assert report["arms"]["a1"]["n_records"] == len(state.accepted)


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_metric_blocks_complete(validated: tuple[Resources, PipelineState, Path], arm: str) -> None:
    _resources, state, _out = validated
    block = state.metrics["eval_report"]["arms"][arm]
    fidelity = block["fidelity"]
    for key in ("kl", "c2st_auc", "wc2st_mean", "wc2st_per_cluster", "mmd"):
        assert key in fidelity
    efficiency = block["efficiency"]
    for key in (
        "ess_ratio",
        "coverage_gap",
        "demand_weighted_coverage",
        "zero_query_clusters",
        "post_stratified_ndcg",
        "per_cluster",
        "worst_clusters",
    ):
        assert key in efficiency
    assert 0.0 <= efficiency["ess_ratio"] <= 1.0 + 1e-9
    validity = block["validity"]
    for key in ("tau", "tau_ci", "tau_ap", "rbo", "controls", "per_stratum"):
        assert key in validity
    assert set(validity["controls"]) == {"drop_index", "noise", "truncate_topk"}
    for control in validity["controls"].values():
        assert 0.0 <= control["p_value"] <= 1.0
    diversity = block["diversity"]
    for key in ("distinct_1", "distinct_2", "semantic_dedup_rate"):
        assert key in diversity
    assert block["n_records"] > 0
    assert "gates_passed" in block


def test_dual_view_reporting_rule(validated: tuple[Resources, PipelineState, Path]) -> None:
    # SPEC §8-9: demand-weighted headline AND unweighted per-cluster table.
    _resources, state, _out = validated
    block = state.metrics["eval_report"]["arms"]["a1"]
    per_cluster = block["efficiency"]["per_cluster"]
    assert len(per_cluster) == 4  # one row per reference cluster
    for row in per_cluster:
        assert set(row) >= {"cluster", "p_hat", "n_synth", "mean_ndcg"}


def test_oracle_tau_is_high(validated: tuple[Resources, PipelineState, Path]) -> None:
    # At n=24 the oracle's own tau is noisy (the ceiling has a CI - prototype
    # epilogue), so this only asserts an absolute floor; the full qualitative
    # arm ordering is locked at proper scale in tests/e2e (Phase 4).
    _resources, state, _out = validated
    arms = state.metrics["eval_report"]["arms"]
    assert arms["oracle"]["validity"]["tau"] >= 0.6


def test_outputs_written(validated: tuple[Resources, PipelineState, Path]) -> None:
    _resources, _state, out_dir = validated
    assert (out_dir / "metrics.json").exists()
    payload = json.loads((out_dir / "metrics.json").read_text())
    assert "provenance" not in payload  # determinism rule (PLAN D14)
    assert set(payload["arms"]) == set(ARM_NAMES)
    report_md = (out_dir / "report.md").read_text()
    assert "a2" in report_md
    assert "ESS" in report_md
    for figure in ("fidelity_bars", "tau_ci", "ess_coverage", "gate_rejects"):
        path = out_dir / "figures" / f"{figure}.png"
        assert path.exists()
        assert path.stat().st_size > 0
    records_path = out_dir / "records.jsonl"
    assert records_path.exists()
    assert len(records_path.read_text().splitlines()) > 0


def test_config_round_trip(min_resources: Resources) -> None:
    step = Validator(
        min_resources,
        arms=["a0", "oracle"],
        n_boot=10,
        gates={"tau": 0.8, "tau_ap": 0.7},
        n_per_arm=5,
        reuse_pipeline_for=None,
        arm_params={"a0": {"n_seeds": 5}},
    )
    config = step.to_config()
    rebuilt = Validator.from_config(config, min_resources)
    assert rebuilt.to_config() == config


def test_validator_registered() -> None:
    from ragsynth.pipeline.base import STEPS

    assert STEPS.get("validator") is Validator
