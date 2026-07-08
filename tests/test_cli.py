"""Tests for the typer CLI (SPEC §5: ragsynth run|validate|report)."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from ragsynth.cli import app
from tests.conftest import TOY_ARM_PARAMS, toy_config

runner = CliRunner()


@pytest.fixture(scope="module")
def cli_config_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small toy config WITH a validator step, written to disk."""
    tmp_path = tmp_path_factory.mktemp("cli")
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
            },
        }
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _out_dir(config_path: Path) -> Path:
    config: dict[str, Any] = yaml.safe_load(config_path.read_text())
    return Path(config["artifacts_dir"]).parent


def test_validate_ok(cli_config_path: Path) -> None:
    result = runner.invoke(app, ["validate", "--config", str(cli_config_path)])
    assert result.exit_code == 0, result.output
    assert "toy_world" in result.output
    assert "validator" in result.output


def test_validate_unknown_type_lists_known(tmp_path: Path, cli_config_path: Path) -> None:
    config: dict[str, Any] = yaml.safe_load(cli_config_path.read_text())
    config["pipeline"][0]["type"] = "seed_sampler.bogus"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(config))
    result = runner.invoke(app, ["validate", "--config", str(bad)])
    assert result.exit_code == 1
    assert "seed_sampler.bogus" in result.output
    assert "seed_sampler.quota" in result.output  # known keys listed


@pytest.fixture(scope="module")
def cli_run(cli_config_path: Path) -> Path:
    """Invoke `ragsynth run` once; tests below inspect its outputs."""
    result = runner.invoke(app, ["run", "--config", str(cli_config_path)])
    assert result.exit_code == 0, result.output
    # The console summary shows the per-arm table.
    assert "oracle" in " ".join(result.output.split())
    return cli_config_path


def test_run_produces_outputs(cli_run: Path) -> None:
    out = _out_dir(cli_run)
    for name in ("metrics.json", "report.md", "records.jsonl"):
        assert (out / name).exists(), name
    assert (out / "figures" / "tau_ci.png").exists()
    assert (out / "artifacts" / "manifest.json").exists()
    payload = json.loads((out / "metrics.json").read_text())
    assert set(payload["arms"]) == {"a1", "oracle"}


def test_report_rerenders_from_metrics(cli_run: Path) -> None:
    out = _out_dir(cli_run)
    report_md = out / "report.md"
    original = report_md.read_text()
    report_md.unlink()
    result = runner.invoke(app, ["report", "--config", str(cli_run)])
    assert result.exit_code == 0, result.output
    assert report_md.read_text() == original


def test_report_without_run_fails_actionably(tmp_path: Path) -> None:
    config = toy_config(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = runner.invoke(app, ["report", "--config", str(path)])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())  # rich wraps lines at terminal width
    assert "missing metrics.json" in flat
    assert "ragsynth run" in flat


def test_missing_config_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0
