"""NFCorpus converter: abstracts 1:1, graded qrels binarized at grade >= 1 (spec01 §5, D33)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragsynth.datasets.converters.base import LICENSE_NOTES, convert_beir_corpus_1to1

if TYPE_CHECKING:
    from pathlib import Path

    from ragsynth.datasets.converters.base import ConversionManifest

name = "nfcorpus"


def convert(raw_dir: Path, out_dir: Path) -> ConversionManifest:
    """Convert an NFCorpus BEIR ``raw_dir`` into ``chunks``/``queries``/``anchor_qrels`` jsonl.

    Args:
        raw_dir: Directory containing ``corpus.jsonl``, ``queries.jsonl``,
            and ``qrels/*.tsv`` (BEIR layout).
        out_dir: Destination directory for the three emitted jsonl files
            and ``manifest.json``.

    Returns:
        The :class:`~ragsynth.datasets.converters.base.ConversionManifest`
        written alongside the output.
    """
    return convert_beir_corpus_1to1(
        raw_dir,
        out_dir,
        dataset_name=name,
        license_note=LICENSE_NOTES[name],
        binarize_threshold=1,
    )
