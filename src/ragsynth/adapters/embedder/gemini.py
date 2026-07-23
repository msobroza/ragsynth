"""Gemini API embedder behind the ``gemini`` optional extra (SPEC §12, user-directed D39 amendment).

Uses the ``google-genai`` SDK (package ``google.genai``) -- the legacy
``google-generativeai`` package is deprecated and must never be used.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from ragsynth.adapters.embedder.base import EMBEDDERS
from ragsynth.optional_deps import require_optional

genai: Any
genai_types: Any
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised via require_optional tests
    genai = None
    genai_types = None

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ragsynth.datasets.base import DatasetBundle

_DEFAULT_MODEL = "gemini-embedding-2"
_DEFAULT_OUTPUT_DIM = 768
_DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
_DEFAULT_BATCH_SIZE = 100
_NORM_EPS = 1e-12


@EMBEDDERS.register("gemini")
class GeminiEmbedder:
    """Neural text embedder via the Gemini API (``uv sync --extra gemini``).

    Calls ``client.models.embed_content`` in ``batch_size``-sized slices and
    defensively L2-normalizes rows, honoring the Embedder Protocol invariant
    regardless of the API's own normalization. Accepts an injected ``client``
    for offline testing (excluded from ``to_config``); without one, the real
    client is built from ``api_key_env`` at construction time.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        output_dimensionality: int = _DEFAULT_OUTPUT_DIM,
        api_key_env: str = _DEFAULT_API_KEY_ENV,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        client: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model = model
        self.output_dimensionality = output_dimensionality
        self.api_key_env = api_key_env
        self.batch_size = batch_size
        if client is not None:
            self._client = client
        else:
            require_optional(genai, "GeminiEmbedder", "gemini")
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"environment variable {api_key_env!r} is not set; "
                    f"export it with a Gemini API key "
                    f"(e.g. `export {api_key_env}=...`)"
                )
            self._client = genai.Client(api_key=api_key)

    def _embed_content_config(self) -> Any:
        """Build the request config, real ``EmbedContentConfig`` if available."""
        if genai_types is not None:
            return genai_types.EmbedContentConfig(output_dimensionality=self.output_dimensionality)
        # Extra not installed but a client was injected (offline logic tests):
        # a duck-typed stand-in with the one attribute callers read.
        return SimpleNamespace(output_dimensionality=self.output_dimensionality)

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return an ``(len(texts), output_dimensionality)`` matrix of unit-norm rows."""
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.output_dimensionality), dtype=np.float64)
        rows = np.empty((n, self.output_dimensionality), dtype=np.float64)
        config = self._embed_content_config()
        for start in range(0, n, self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self._client.models.embed_content(
                model=self.model,
                contents=batch,
                config=config,
            )
            for offset, embedding in enumerate(response.embeddings):
                rows[start + offset] = np.asarray(embedding.values, dtype=np.float64)
        norms = np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), _NORM_EPS)
        return np.asarray(rows / norms, dtype=np.float64)

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params (the injected ``client`` is excluded)."""
        return {
            "model": self.model,
            "output_dimensionality": self.output_dimensionality,
            "api_key_env": self.api_key_env,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> GeminiEmbedder:
        """Build from a config params block."""
        return cls(
            model=str(params.get("model", _DEFAULT_MODEL)),
            output_dimensionality=int(params.get("output_dimensionality", _DEFAULT_OUTPUT_DIM)),
            api_key_env=str(params.get("api_key_env", _DEFAULT_API_KEY_ENV)),
            batch_size=int(params.get("batch_size", _DEFAULT_BATCH_SIZE)),
        )
