"""Embedding adapters: Protocol, hashed n-gram featurizer, mocks, extras."""

from ragsynth.adapters.embedder.base import EMBEDDERS, Embedder
from ragsynth.adapters.embedder.gemini import GeminiEmbedder
from ragsynth.adapters.embedder.hashed import HashedNGramEmbedder
from ragsynth.adapters.embedder.mock import MockEmbedder
from ragsynth.adapters.embedder.st import SentenceTransformerEmbedder

__all__ = [
    "EMBEDDERS",
    "Embedder",
    "GeminiEmbedder",
    "HashedNGramEmbedder",
    "MockEmbedder",
    "SentenceTransformerEmbedder",
]
