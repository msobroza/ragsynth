r"""Argument-parsing shell over ``ragsynth.datasets.converters.<name>`` (spec01 D32).

Usage:
    uv run python scripts/convert_benchmark.py <fiqa|nfcorpus> \
        [--raw-dir DIR] [--out-dir DIR]

Defaults: ``--raw-dir data/benchmarks/<name>/raw``,
``--out-dir data/benchmarks/<name>``. All conversion logic lives in
``ragsynth.datasets.converters``; this script only parses arguments and
delegates.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ragsynth.datasets.converters import CONVERTERS

logger = logging.getLogger(__name__)

_DEFAULT_BENCHMARKS_ROOT = Path("data/benchmarks")


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ``python scripts/convert_benchmark.py <name> [--raw-dir DIR] [--out-dir DIR]``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=sorted(CONVERTERS), help="Benchmark to convert")
    parser.add_argument("--raw-dir", type=Path, default=None, help="BEIR raw layout directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output jsonl directory")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw_dir = args.raw_dir or (_DEFAULT_BENCHMARKS_ROOT / args.name / "raw")
    out_dir = args.out_dir or (_DEFAULT_BENCHMARKS_ROOT / args.name)

    manifest = CONVERTERS[args.name].convert(raw_dir, out_dir)
    logger.info(
        f"converted {args.name}: {manifest.n_chunks} chunks, {manifest.n_queries} queries, "
        f"{manifest.n_qrel_entries} qrel entries -> {out_dir}"
    )


if __name__ == "__main__":
    main()
