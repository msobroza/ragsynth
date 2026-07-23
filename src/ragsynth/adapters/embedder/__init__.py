"""Embedding adapters: Protocol, hashed n-gram featurizer, mocks, extras."""

from ragsynth.adapters.embedder.base import EMBEDDERS, Embedder
from ragsynth.adapters.embedder.cached_chroma import CachedChromaEmbedder
from ragsynth.adapters.embedder.cached_embedder import CachedEmbedder
from ragsynth.adapters.embedder.chroma_vector_store import ChromaEmbeddingVectorStore
from ragsynth.adapters.embedder.gemini import GeminiEmbedder
from ragsynth.adapters.embedder.hashed import HashedNGramEmbedder
from ragsynth.adapters.embedder.mock import MockEmbedder
from ragsynth.adapters.embedder.st import SentenceTransformerEmbedder
from ragsynth.adapters.embedder.vector_store import EmbeddingVectorStore

__all__ = [
    "EMBEDDERS",
    "CachedChromaEmbedder",
    "CachedEmbedder",
    "ChromaEmbeddingVectorStore",
    "Embedder",
    "EmbeddingVectorStore",
    "GeminiEmbedder",
    "HashedNGramEmbedder",
    "MockEmbedder",
    "SentenceTransformerEmbedder",
]
