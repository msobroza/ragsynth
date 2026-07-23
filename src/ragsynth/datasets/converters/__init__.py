"""Benchmark converters: one module per dataset behind ``BenchmarkConverter`` (spec01 §5).

``scripts/convert_benchmark.py <name>`` is a thin wrapper over
``CONVERTERS[<name>].convert(raw_dir, out_dir)``; all conversion logic lives
in these importable modules so mypy/ruff/pytest cover it.
"""

from __future__ import annotations

from ragsynth.datasets.converters import fiqa, legalbench_rag, nfcorpus
from ragsynth.datasets.converters.base import BenchmarkConverter, ConversionManifest

CONVERTERS: dict[str, BenchmarkConverter] = {
    "fiqa": fiqa,
    "legalbench_rag": legalbench_rag,
    "nfcorpus": nfcorpus,
}

__all__ = [
    "CONVERTERS",
    "BenchmarkConverter",
    "ConversionManifest",
    "fiqa",
    "legalbench_rag",
    "nfcorpus",
]
