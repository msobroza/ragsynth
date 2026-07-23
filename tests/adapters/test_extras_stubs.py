"""Extras-gated adapters: import always works, construction needs the extra."""

import importlib.util

import pytest


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        # find_spec on a dotted name raises (rather than returning None) when
        # the parent package itself isn't importable.
        return False


def test_adapter_packages_import_without_extras() -> None:
    import ragsynth.adapters.embedder
    import ragsynth.adapters.retriever

    assert hasattr(ragsynth.adapters.retriever, "BM25sRetriever")
    assert hasattr(ragsynth.adapters.embedder, "SentenceTransformerEmbedder")
    assert hasattr(ragsynth.adapters.embedder, "GeminiEmbedder")


def test_registered_even_without_extras() -> None:
    from ragsynth.adapters.embedder.base import EMBEDDERS
    from ragsynth.adapters.retriever.base import RETRIEVERS

    assert "bm25s" in RETRIEVERS.keys()  # noqa: SIM118
    assert "sentence_transformer" in EMBEDDERS.keys()  # noqa: SIM118
    assert "gemini" in EMBEDDERS.keys()  # noqa: SIM118


def test_bm25s_retriever_without_extra_raises_actionable_import_error() -> None:
    if _has("bm25s"):
        pytest.skip("bm25s extra installed; missing-extra path untestable")
    from ragsynth.adapters.retriever.bm25s import BM25sRetriever

    with pytest.raises(ImportError, match=r"uv sync --extra bm25"):
        BM25sRetriever(chunk_ids=["c1"], texts=["some text"])


def test_st_embedder_without_extra_raises_actionable_import_error() -> None:
    if _has("sentence_transformers"):
        pytest.skip("st extra installed; missing-extra path untestable")
    from ragsynth.adapters.embedder.st import SentenceTransformerEmbedder

    with pytest.raises(ImportError, match=r"uv sync --extra st"):
        SentenceTransformerEmbedder()


def test_gemini_embedder_without_extra_raises_actionable_import_error() -> None:
    if _has("google.genai"):
        pytest.skip("gemini extra installed; missing-extra path untestable")
    from ragsynth.adapters.embedder.gemini import GeminiEmbedder

    with pytest.raises(ImportError, match=r"uv sync --extra gemini"):
        GeminiEmbedder()


def test_bm25s_search_requires_text_queries_when_installed() -> None:
    if not _has("bm25s"):
        pytest.skip("bm25s extra not installed")
    import numpy as np

    from ragsynth.adapters.retriever.bm25s import BM25sRetriever

    retriever = BM25sRetriever(chunk_ids=["c1", "c2"], texts=["alpha beta", "gamma delta"])
    with pytest.raises(NotImplementedError, match="search_text"):
        retriever.search(np.zeros(4), k=1)
    hits = retriever.search_text("alpha", k=1)
    assert hits[0][0] == "c1"
