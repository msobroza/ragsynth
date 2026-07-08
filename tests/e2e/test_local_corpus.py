"""SPEC §15.1: the bundled 200-chunk sample corpus runs end-to-end with mocks."""

import json
from pathlib import Path

import pytest
import yaml

from ragsynth.pipeline.serialization import (
    build_pipeline,
    build_resources,
    make_initial_state,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "v1_local_corpus.yaml"


@pytest.fixture(scope="module")
def local_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The shipped config, shrunk for CI speed, run into a tmp dir."""
    tmp_path = tmp_path_factory.mktemp("local")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["artifacts_dir"] = str(tmp_path / "exp" / "artifacts")
    config["pipeline"][0]["params"]["n_seeds"] = 48
    validator = config["pipeline"][-1]["params"]
    validator["n_boot"] = 50
    validator["n_per_arm"] = 30
    resources = build_resources(config)
    pipeline = build_pipeline(config, resources)
    pipeline.fit(resources)
    state = pipeline.run(make_initial_state(config))
    assert state.accepted, "local corpus pipeline produced no records"
    return tmp_path / "exp"


def test_all_arms_have_records(local_run: Path) -> None:
    payload = json.loads((local_run / "metrics.json").read_text())
    assert set(payload["arms"]) == {"a0", "a1", "a2", "oracle"}
    for arm, block in payload["arms"].items():
        assert not block.get("skipped"), arm
        assert block["n_records"] >= 1, arm


def test_outputs_complete(local_run: Path) -> None:
    for name in ("metrics.json", "report.md", "records.jsonl"):
        assert (local_run / name).exists(), name
    records = (local_run / "records.jsonl").read_text().splitlines()
    assert len(records) >= 1
    first = json.loads(records[0])
    assert first["qrels"]
    assert first["content_hashes"]
