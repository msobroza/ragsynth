"""Tests for the optional-dependency helper."""

import pytest

from ragsynth.optional_deps import require_optional


def test_require_optional_passes_when_module_present() -> None:
    import json

    require_optional(json, "JSONFeature", "optimization")


def test_require_optional_raises_with_install_hint() -> None:
    with pytest.raises(ImportError, match=r"uv sync --extra optimization"):
        require_optional(None, "MIPROv2Optimizer", "optimization")


def test_require_optional_names_the_feature() -> None:
    with pytest.raises(ImportError, match=r"BM25sRetriever"):
        require_optional(None, "BM25sRetriever", "bm25")
