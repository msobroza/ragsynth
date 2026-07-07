"""Datasets: bundle contract, toy world, JSONL corpus loader, sample corpus."""

from ragsynth.datasets.base import DATASETS, DatasetBundle, EmbeddingBank
from ragsynth.datasets.jsonl_loader import (
    JsonlDataset,
    load_anchor_qrels,
    load_chunks,
    load_queries,
)
from ragsynth.datasets.sample_corpus import generate_sample_corpus
from ragsynth.datasets.toy_world import (
    PassthroughEmbedder,
    ToyChatModel,
    ToyJudge,
    ToyWorldDataset,
)

__all__ = [
    "DATASETS",
    "DatasetBundle",
    "EmbeddingBank",
    "JsonlDataset",
    "PassthroughEmbedder",
    "ToyChatModel",
    "ToyJudge",
    "ToyWorldDataset",
    "generate_sample_corpus",
    "load_anchor_qrels",
    "load_chunks",
    "load_queries",
]
