"""SPEC §10: the toy world must reproduce the qualitative demo table.

Runs configs/v1_toy.yaml VERBATIM (only artifacts_dir redirected) and locks
the four binding claims with margins:

  1. A2 best wC2ST / MMD          (the spec-first mechanism meter)
  2. A1 ESS ~= oracle, wC2ST ~= 1 (quota fixes marginals, not shape)
  3. A0 fails everything          (fidelity + efficiency)
  4. A1 misses the noise positive-control; A2 and ORACLE detect it

Ground truth: the vendored prototype's own demo table (reference/
synth_query_eval.py, seed 0 of ITS rng stream). The shipped config's seed
is the world realization where this package exhibits the same table; the
tuning trace lives in experiments/v1/report.md.
"""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from ragsynth.pipeline.serialization import (
    build_pipeline,
    build_resources,
    make_initial_state,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "v1_toy.yaml"
ALPHA = 0.05


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp_path = tmp_path_factory.mktemp("toy_table")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["artifacts_dir"] = str(tmp_path / "exp" / "artifacts")
    resources = build_resources(config)
    pipeline = build_pipeline(config, resources)
    pipeline.fit(resources)
    pipeline.run(make_initial_state(config))
    payload = json.loads((tmp_path / "exp" / "metrics.json").read_text())
    return payload["arms"]


def test_all_arms_at_full_size(table: dict[str, Any]) -> None:
    for arm in ("a0", "a1", "a2", "oracle"):
        assert table[arm]["n_records"] == 500, arm


def test_a2_best_within_cluster_shape_and_mmd(table: dict[str, Any]) -> None:
    wc2st = {arm: table[arm]["fidelity"]["wc2st_mean"] for arm in table}
    mmd = {arm: table[arm]["fidelity"]["mmd"] for arm in table}
    assert wc2st["a2"] < wc2st["a1"] - 0.15
    assert wc2st["a2"] < wc2st["a0"] - 0.15
    assert mmd["a2"] < mmd["a1"] < mmd["a0"]


def test_a1_matches_marginals_but_not_shape(table: dict[str, Any]) -> None:
    assert table["a1"]["efficiency"]["ess_ratio"] >= table["oracle"]["efficiency"][
        "ess_ratio"
    ] - 0.15
    assert table["a1"]["fidelity"]["wc2st_mean"] > 0.85


def test_a0_fails_everything(table: dict[str, Any]) -> None:
    a0 = table["a0"]
    assert a0["fidelity"]["c2st_auc"] > 0.95
    assert a0["fidelity"]["wc2st_mean"] > 0.85
    assert a0["efficiency"]["ess_ratio"] < table["a1"]["efficiency"]["ess_ratio"] - 0.2
    assert a0["validity"]["tau"] < table["a2"]["validity"]["tau"]


def test_positive_controls(table: dict[str, Any]) -> None:
    controls = {arm: table[arm]["validity"]["controls"] for arm in table}
    # Everyone must detect a 10% index deletion (SPEC §8-9: a benchmark that
    # misses it fails regardless of tau).
    for arm, block in controls.items():
        assert block["drop_index"]["p_value"] < ALPHA, arm
    # The headline lesson: the quota arm misses the noise regression that
    # the demand-matched arms detect.
    assert controls["a1"]["noise"]["p_value"] > ALPHA
    assert controls["a2"]["noise"]["p_value"] < ALPHA
    assert controls["oracle"]["noise"]["p_value"] < ALPHA


def test_oracle_is_the_tau_ceiling(table: dict[str, Any]) -> None:
    taus = {arm: table[arm]["validity"]["tau"] for arm in table}
    assert taus["oracle"] >= max(taus["a0"], taus["a1"]) - 0.05
    assert taus["a2"] > taus["a1"] or taus["a2"] > taus["a0"]
