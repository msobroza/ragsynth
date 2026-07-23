"""CONVERTERS registry: both dataset modules satisfy BenchmarkConverter (spec01 §5)."""

from pathlib import Path

from ragsynth.datasets.converters import CONVERTERS, fiqa, nfcorpus
from ragsynth.datasets.converters.base import ConversionManifest


def test_registry_keys() -> None:
    assert set(CONVERTERS) == {"fiqa", "nfcorpus"}
    assert CONVERTERS["fiqa"] is fiqa
    assert CONVERTERS["nfcorpus"] is nfcorpus


def test_every_entry_has_a_name_and_a_convert_callable() -> None:
    for converter in CONVERTERS.values():
        assert isinstance(converter.name, str)
        assert converter.name
        assert callable(converter.convert)


def test_convert_returns_conversion_manifest(tmp_path: Path) -> None:
    fixture_raw = Path(__file__).resolve().parents[2] / "fixtures" / "benchmarks" / "fiqa" / "raw"
    manifest = CONVERTERS["fiqa"].convert(fixture_raw, tmp_path)
    assert isinstance(manifest, ConversionManifest)
