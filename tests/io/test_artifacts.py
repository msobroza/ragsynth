"""Tests for the ArtifactStore (hashed artifact persistence) and hash helpers."""

import json
from pathlib import Path

import numpy as np
import pytest

from ragsynth.io.artifacts import ArtifactStore, canonical_json, sha256_hex


def test_sha256_hex_known_value() -> None:
    # sha256("") is a published constant.
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_canonical_json_is_key_order_independent() -> None:
    a = canonical_json({"b": 1, "a": [1.0, 2]})
    b = canonical_json({"a": [1.0, 2], "b": 1})
    assert a == b
    assert a == '{"a":[1.0,2],"b":1}'


def test_save_json_and_manifest_hash_stable(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    p1 = store.save_json("config-snapshot", {"x": 1, "y": [2, 3]})
    h1 = store.manifest["config-snapshot.json"]
    p2 = store.save_json("config-snapshot", {"x": 1, "y": [2, 3]})
    assert p1 == p2
    assert store.manifest["config-snapshot.json"] == h1
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["config-snapshot.json"] == h1


def test_save_load_npz_roundtrip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    store.save_npz("demand-map", p_hat=x)
    out = store.load_npz("demand-map")
    np.testing.assert_array_equal(out["p_hat"], x)


def test_save_load_joblib_roundtrip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.save_joblib("obj", {"k": [1, 2, 3]})
    assert store.load_joblib("obj") == {"k": [1, 2, 3]}


def test_load_missing_artifact_raises(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="absent-artifact"):
        store.load_json("absent-artifact")


def test_manifest_reloaded_from_disk(tmp_path: Path) -> None:
    ArtifactStore(tmp_path).save_json("a", {"v": 1})
    reopened = ArtifactStore(tmp_path)
    assert "a.json" in reopened.manifest
