"""Retrieval adapters: Protocol, dense in-memory, bm25s extra."""

from ragsynth.adapters.retriever.base import RETRIEVERS, Retriever
from ragsynth.adapters.retriever.dense_inmemory import DenseInMemoryRetriever

__all__ = ["RETRIEVERS", "DenseInMemoryRetriever", "Retriever"]
