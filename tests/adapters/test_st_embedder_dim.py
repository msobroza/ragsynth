"""SentenceTransformerEmbedder exposes its embedding dim so cached_chroma can wrap it.

The dim is resolved lazily from the loaded model (not stored at init) and lands
in ``to_config`` under the ``dim`` key that ``CachedChromaEmbedder._derive_identity``
already accepts. ``from_config`` keeps accepting configs without the key. The
sentence-transformers library is stubbed at the module handle, so these tests run
air-gapped without the ``st`` extra; one ``importorskip`` test exercises the real
library where installed.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import ragsynth.adapters.embedder.st as st_mod
from ragsynth.adapters.embedder.st import SentenceTransformerEmbedder

if TYPE_CHECKING:
    from collections.abc import Sequence

MINILM_DIM = 384


def _stub_module(dim: int | None) -> types.SimpleNamespace:
    """A stand-in for the sentence_transformers module with a fixed model dim."""

    class _Model:
        def __init__(self, model_name: str, device: str = "cpu") -> None:
            self.model_name = model_name
            self.device = device

        def get_sentence_embedding_dimension(self) -> int | None:
            return dim

        def encode(self, texts: Sequence[str], convert_to_numpy: bool = True) -> Any:
            rng = np.random.default_rng(0)
            return rng.standard_normal((len(texts), dim or 4))

    return types.SimpleNamespace(SentenceTransformer=_Model)


@pytest.fixture
def stub_st(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(st_mod, "sentence_transformers", _stub_module(MINILM_DIM))


def test_to_config_exposes_lazily_resolved_dim(stub_st: None) -> None:
    emb = SentenceTransformerEmbedder()
    cfg = emb.to_config()
    assert cfg == {"model_name": "all-MiniLM-L6-v2", "device": "cpu", "dim": MINILM_DIM}
    assert isinstance(cfg["dim"], int)


def test_to_config_omits_dim_when_model_reports_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(st_mod, "sentence_transformers", _stub_module(None))
    cfg = SentenceTransformerEmbedder().to_config()
    assert cfg == {"model_name": "all-MiniLM-L6-v2", "device": "cpu"}


def test_from_config_accepts_config_without_dim(stub_st: None) -> None:
    emb = SentenceTransformerEmbedder.from_config(
        {"model_name": "all-MiniLM-L6-v2", "device": "cpu"},
        None,  # type: ignore[arg-type]  # bundle unused by st
        np.random.default_rng(0),
    )
    assert emb.to_config()["dim"] == MINILM_DIM


def test_round_trip_dim_stable(stub_st: None) -> None:
    cfg = SentenceTransformerEmbedder().to_config()
    emb2 = SentenceTransformerEmbedder.from_config(
        cfg,
        None,  # type: ignore[arg-type]
        np.random.default_rng(0),
    )
    assert emb2.to_config() == cfg


def test_cached_chroma_wraps_stubbed_st(stub_st: None, tmp_path: Any) -> None:
    chromadb = pytest.importorskip("chromadb")
    from ragsynth.adapters.embedder.cached_chroma import CachedChromaEmbedder

    cache = CachedChromaEmbedder(
        backend=SentenceTransformerEmbedder(),
        backend_type="sentence_transformer",
        dataset="fiqa",
        client=chromadb.PersistentClient(path=str(tmp_path / "chroma")),
    )
    assert cache.embedder_id == f"sentence_transformer:all-MiniLM-L6-v2:{MINILM_DIM}"
    assert cache.dim == MINILM_DIM


def test_real_st_dim_when_installed() -> None:
    pytest.importorskip("sentence_transformers")
    cfg = SentenceTransformerEmbedder().to_config()
    assert cfg.get("dim") == MINILM_DIM
