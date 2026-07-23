"""Tests for the Gemini API embedder (spec01 Task 3, user-directed D39 amendment)."""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from ragsynth.adapters.embedder.base import EMBEDDERS
from ragsynth.adapters.embedder.gemini import GeminiEmbedder

DIM = 8


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        # find_spec on a dotted name raises (rather than returning None) when
        # the parent package itself isn't importable.
        return False


def _vector_for(text: str) -> list[float]:
    """Deterministic, non-unit-norm vector so normalization is actually exercised."""
    digest = hashlib.sha256(text.encode()).digest()
    return [float(b) for b in digest[:DIM]]


@dataclass
class _StubEmbedding:
    values: list[float]


@dataclass
class _StubResponse:
    embeddings: list[_StubEmbedding]


@dataclass
class _StubModels:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def embed_content(self, *, model: str, contents: list[str], config: Any) -> _StubResponse:
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        return _StubResponse([_StubEmbedding(_vector_for(text)) for text in contents])


@dataclass
class _StubClient:
    models: _StubModels = field(default_factory=_StubModels)


def _embedder(**kwargs: Any) -> GeminiEmbedder:
    kwargs.setdefault("output_dimensionality", DIM)
    kwargs.setdefault("client", _StubClient())
    return GeminiEmbedder(**kwargs)


def test_registered_in_embedders_registry() -> None:
    assert EMBEDDERS.get("gemini") is GeminiEmbedder


def test_shape_dtype_and_unit_rows() -> None:
    emb = _embedder()
    out = emb.encode(["alpha", "beta", "gamma"])
    assert out.shape == (3, DIM)
    assert out.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-9)


def test_batching_splits_calls_by_batch_size() -> None:
    client = _StubClient()
    emb = GeminiEmbedder(output_dimensionality=DIM, batch_size=100, client=client)
    texts = [f"text-{i}" for i in range(250)]
    emb.encode(texts)
    assert [len(call["contents"]) for call in client.models.calls] == [100, 100, 50]


def test_input_order_preserved_across_batches() -> None:
    client = _StubClient()
    emb = GeminiEmbedder(output_dimensionality=DIM, batch_size=2, client=client)
    texts = ["a", "b", "c", "d", "e"]
    out = emb.encode(texts)
    for i, text in enumerate(texts):
        expected = np.asarray(_vector_for(text), dtype=np.float64)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(out[i], expected)
    # Confirm batching actually happened (3 calls for 5 texts at batch_size 2).
    assert [len(call["contents"]) for call in client.models.calls] == [2, 2, 1]


def test_short_api_response_raises_actionable_error() -> None:
    """A batch answered with fewer embeddings than requested must fail loudly.

    The API can silently drop items (e.g. per-item filtering); without the
    length check the missing rows would be uninitialized memory.
    """

    @dataclass
    class _ShortModels(_StubModels):
        def embed_content(self, *, model: str, contents: list[str], config: Any) -> _StubResponse:
            response = super().embed_content(model=model, contents=contents, config=config)
            if len(self.calls) == 2:  # drop one embedding from the second batch only
                return _StubResponse(response.embeddings[:-1])
            return response

    client = _StubClient(models=_ShortModels())
    emb = GeminiEmbedder(output_dimensionality=DIM, batch_size=2, client=client)
    with pytest.raises(RuntimeError, match=r"batch at offset 2.*expected 2.*got 1"):
        emb.encode(["a", "b", "c", "d"])


def test_empty_input_returns_empty_matrix_with_output_dim() -> None:
    emb = _embedder()
    out = emb.encode([])
    assert out.shape == (0, DIM)
    assert out.dtype == np.float64


def test_model_and_config_passed_to_embed_content() -> None:
    client = _StubClient()
    emb = GeminiEmbedder(model="gemini-embedding-2", output_dimensionality=DIM, client=client)
    emb.encode(["hello"])
    call = client.models.calls[0]
    assert call["model"] == "gemini-embedding-2"
    assert call["config"].output_dimensionality == DIM


def test_to_config_round_trip_excludes_client() -> None:
    # Offline/air-gapped: from_config is the real-resource factory (registry
    # contract used by the pipeline's build_resources, matching production
    # config loading), so it legitimately requires the extra + API key. The
    # round-trip property under test here is that to_config()'s dict, fed
    # back through the constructor (with a fresh injected client), yields an
    # identical config -- the client itself is never part of it.
    original = GeminiEmbedder(
        model="gemini-embedding-2",
        output_dimensionality=256,
        api_key_env="MY_KEY",
        batch_size=50,
        client=_StubClient(),
    )
    config = original.to_config()
    assert config == {
        "model": "gemini-embedding-2",
        "output_dimensionality": 256,
        "api_key_env": "MY_KEY",
        "batch_size": 50,
    }
    rebuilt = GeminiEmbedder(**config, client=_StubClient())
    assert rebuilt.to_config() == config


def test_from_config_round_trip_with_real_lib(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry's real factory path (spec01 §3.2): needs the extra + API key."""
    if not _has("google.genai"):
        pytest.skip("gemini extra not installed; from_config builds a real client")
    from ragsynth.datasets.base import DatasetBundle

    monkeypatch.setenv("MY_KEY", "dummy-key-for-construction-only")
    bundle = DatasetBundle(chunks=(), queries_train=(), queries_anchor=(), queries_oracle=())
    params = {
        "model": "gemini-embedding-2",
        "output_dimensionality": 256,
        "api_key_env": "MY_KEY",
        "batch_size": 50,
    }
    rebuilt = GeminiEmbedder.from_config(params, bundle, np.random.default_rng(0))
    assert rebuilt.to_config() == params


def test_to_config_defaults() -> None:
    emb = GeminiEmbedder(client=_StubClient())
    assert emb.to_config() == {
        "model": "gemini-embedding-2",
        "output_dimensionality": 768,
        "api_key_env": "GEMINI_API_KEY",
        "batch_size": 100,
    }


def test_missing_env_var_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _has("google.genai"):
        pytest.skip("gemini extra not installed; missing-extra path takes precedence")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiEmbedder()


def test_missing_env_var_names_custom_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _has("google.genai"):
        pytest.skip("gemini extra not installed; missing-extra path takes precedence")
    monkeypatch.delenv("MY_CUSTOM_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MY_CUSTOM_KEY"):
        GeminiEmbedder(api_key_env="MY_CUSTOM_KEY")


def test_constructs_with_real_lib_present(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-construction-only")
    emb = GeminiEmbedder()
    assert isinstance(emb, GeminiEmbedder)
