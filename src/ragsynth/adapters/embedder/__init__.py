"""Embedding adapters: Protocol, hashed n-gram featurizer, mocks, extras."""

from ragsynth.adapters.embedder.base import EMBEDDERS, Embedder
from ragsynth.adapters.embedder.mock import MockEmbedder

__all__ = ["EMBEDDERS", "Embedder", "MockEmbedder"]
