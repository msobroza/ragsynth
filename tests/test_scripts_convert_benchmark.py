"""Smoke test for the scripts/convert_benchmark.py argument-parsing shell.

``scripts/`` is not a package (no __init__.py, not covered by mypy's
``files = ["src"]``); this loads the script by file path with
``importlib`` to prove its arg-parsing + delegation wiring works end to
end over the bundled fixture, with no network involved.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "convert_benchmark.py"
FIXTURE_RAW = REPO_ROOT / "tests" / "fixtures" / "benchmarks" / "fiqa" / "raw"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("convert_benchmark_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_converts_the_named_benchmark_end_to_end(tmp_path: Path) -> None:
    module = _load_script()
    out_dir = tmp_path / "out"

    module.main(["fiqa", "--raw-dir", str(FIXTURE_RAW), "--out-dir", str(out_dir)])

    assert (out_dir / "chunks.jsonl").is_file()
    assert (out_dir / "queries.jsonl").is_file()
    assert (out_dir / "anchor_qrels.jsonl").is_file()
    assert (out_dir / "manifest.json").is_file()
