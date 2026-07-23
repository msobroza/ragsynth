"""The three committed schema-2 benchmark configs (spec01 §8, §13.4, D39).

These tests exercise config *parsing/serialization* and the dataset+embedder
resource path with offline stubs only. The Qwen/Llama LLM adapters are never
constructed (no ``RAGSYNTH_LLM_BASE_URL`` / API keys needed) -- validate_config
only resolves registry keys, and the build_resources check swaps in offline
``mock``/``hashed_ngram`` adapters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from ragsynth.datasets.jsonl_loader import JsonlDataset
from ragsynth.pipeline.serialization import (
    build_resources,
    config_hash,
    dump_config,
    load_config,
    validate_config,
)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
V2_CONFIGS = ("v2_fiqa.yaml", "v2_nfcorpus.yaml", "v2_legalbench_rag.yaml")
PLACEHOLDER = "${RAGSYNTH_LLM_BASE_URL}"


@pytest.mark.parametrize("name", V2_CONFIGS)
def test_v2_config_loads_as_schema_2(name: str) -> None:
    config = load_config(CONFIGS_DIR / name)
    assert config["ragsynth"]["schema_version"] == 2


@pytest.mark.parametrize("name", V2_CONFIGS)
def test_v2_config_round_trips_byte_stable(name: str) -> None:
    """dump is idempotent and hash-stable (SPEC §13 round-trip rule)."""
    config = load_config(CONFIGS_DIR / name)
    once = dump_config(config)
    twice = dump_config(yaml.safe_load(once))
    assert once == twice
    assert config_hash(yaml.safe_load(once)) == config_hash(config)


@pytest.mark.parametrize("name", V2_CONFIGS)
def test_v2_config_placeholder_survives_unresolved(name: str) -> None:
    """${RAGSYNTH_LLM_BASE_URL} is never resolved by load/round-trip (T2 §15.4)."""
    raw = dump_config(load_config(CONFIGS_DIR / name))
    assert PLACEHOLDER in raw
    # generator + judge each carry it once; round-trip must not expand or drop it.
    assert raw.count(PLACEHOLDER) == 2


def test_v2_configs_do_not_require_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading must not read RAGSYNTH_LLM_BASE_URL (no adapter construction)."""
    monkeypatch.delenv("RAGSYNTH_LLM_BASE_URL", raising=False)
    for name in V2_CONFIGS:
        load_config(CONFIGS_DIR / name)  # must not raise


@pytest.mark.parametrize("name", V2_CONFIGS)
def test_v2_config_no_same_family_warning(name: str, caplog: pytest.LogCaptureFixture) -> None:
    """§6.4 unwrap sees Qwen (gen) vs Llama (judge) through `cached`: zero warnings (D39)."""
    config = yaml.safe_load((CONFIGS_DIR / name).read_text())
    with caplog.at_level(logging.WARNING):
        warnings = validate_config(config)
    assert warnings == []
    assert not any("SIGIR 2025" in r.message for r in caplog.records)


def test_legal_config_carries_split_stratify_by() -> None:
    config = load_config(CONFIGS_DIR / "v2_legalbench_rag.yaml")
    assert config["resources"]["dataset"]["params"]["split_stratify_by"] == "subcorpus"
    for name in ("v2_fiqa.yaml", "v2_nfcorpus.yaml"):
        other = load_config(CONFIGS_DIR / name)
        assert "split_stratify_by" not in other["resources"]["dataset"]["params"]


# --- dataset + embedder resource path with offline stubs (build_resources) -----------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _stub_schema2_config(tmp_path: Path) -> dict[str, Any]:
    """A schema-2 config that uses jsonl + split_stratify_by + partition.ladder but
    swaps the gemini/Qwen/Llama stack for offline mock/hashed adapters."""
    chunks = [
        {"text": f"legal clause number {i} about liability", "doc_id": f"d{i}"} for i in range(12)
    ]
    queries = [
        {
            "query_id": f"q{i:02d}",
            "text": f"is clause {i} enforceable?",
            "metadata": {"subcorpus": "cuad" if i < 12 else "maud"},
        }
        for i in range(20)
    ]
    _write_jsonl(tmp_path / "chunks.jsonl", chunks)
    _write_jsonl(tmp_path / "queries.jsonl", queries)
    return {
        "ragsynth": {"schema_version": 2, "name": "v2-stub", "seed": 0},
        "resources": {
            "dataset": {
                "type": "jsonl",
                "params": {
                    "chunks_path": str(tmp_path / "chunks.jsonl"),
                    "queries_path": str(tmp_path / "queries.jsonl"),
                    "split_stratify_by": "subcorpus",
                },
            },
            "embedder": {"type": "hashed_ngram", "params": {"dim": 64, "seed": 0}},
            "generator_llm": {"type": "mock"},
            "judge_llm": {"type": "mock"},
            "retriever": {"type": "dense_inmemory"},
            "partition": {"n_clusters": 2, "ladder": {"candidates": [2], "min_per_side": 1}},
            "demand": {"n_components": 2, "lam": 0.7, "tau_r_pct": 5.0},
        },
        "artifacts_dir": str(tmp_path / "artifacts"),
        "pipeline": [
            {"type": "seed_sampler.quota", "params": {"lam": 0.7, "n_min": 1, "n_seeds": 4}},
        ],
    }


def test_build_resources_flows_split_stratify_by_through_dataset(tmp_path: Path) -> None:
    """build_resources honors the schema-2 dataset param (stratified 60/25/15)."""
    config = _stub_schema2_config(tmp_path)
    validate_config(config)  # accepts the schema-2 params under version 2
    resources = build_resources(config)
    # cuad 12 -> 7/3/2, maud 8 -> 4/2/2  =>  11/5/4 (a plain split of 20 gives 12/5/3).
    counts = (
        len(resources.queries_train),
        len(resources.queries_anchor),
        len(resources.queries_oracle),
    )
    assert counts == (11, 5, 4)
    # Embedder path built and every chunk/query embedded through hashed_ngram.
    assert all(c.chunk_id in resources.embeddings for c in resources.chunks)
    assert resources.partition.n_clusters == 2


def test_build_resources_default_split_when_absent(tmp_path: Path) -> None:
    """Dropping split_stratify_by reverts to the plain 20-query split (12/5/3)."""
    config = _stub_schema2_config(tmp_path)
    del config["resources"]["dataset"]["params"]["split_stratify_by"]
    bundle = JsonlDataset.build(config["resources"]["dataset"]["params"], seed=0)
    assert (len(bundle.queries_train), len(bundle.queries_anchor), len(bundle.queries_oracle)) == (
        12,
        5,
        3,
    )
