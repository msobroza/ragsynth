"""Argument-parsing shell: download a BEIR benchmark archive (spec01 D32).

Usage:
    uv run python scripts/fetch_benchmarks.py <fiqa|nfcorpus> [--out-dir DIR]

Downloads the upstream BEIR zip via plain ``urllib`` (no ``beir`` runtime
dependency), extracts it to ``<out-dir>/<name>/raw/``, and records its
sha256 + license note in ``<out-dir>/README.md``. All logic lives in
``ragsynth.datasets.converters.base`` (importable, tested); this script is
not unit-tested itself since it touches the network.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ragsynth.datasets.converters.base import BENCHMARK_URLS, download_benchmark

_DEFAULT_OUT_DIR = "data/benchmarks"


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ``python scripts/fetch_benchmarks.py <name> [--out-dir DIR]``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=sorted(BENCHMARK_URLS), help="Benchmark to fetch")
    parser.add_argument(
        "--out-dir", type=Path, default=Path(_DEFAULT_OUT_DIR), help="Benchmarks root directory"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_benchmark(args.name, args.out_dir)


if __name__ == "__main__":
    main()
