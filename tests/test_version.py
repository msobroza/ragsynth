"""Smoke test: the package imports and exposes a version."""

import ragsynth


def test_version() -> None:
    assert ragsynth.__version__ == "0.1.0"
