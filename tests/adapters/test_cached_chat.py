"""Tests for CachedChatModel: transcript record/replay determinism boundary (D40).

The backend is an offline spy that counts calls, so the contract is proven
without any network (air-gap rule). §13.4 known-value fixtures live here.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragsynth.adapters.llm.base import CHAT_MODELS
from ragsynth.adapters.llm.cached import CachedChatModel
from ragsynth.adapters.llm.mock import MockChatModel
from ragsynth.datasets.base import DatasetBundle


class SpyChat:
    """ChatModel spy: unique response per call, records how often it was hit."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        self.calls += 1
        return f"resp::{system}|{user}|{sorted(kwargs.items())}|#{self.calls}"


def _bundle() -> DatasetBundle:
    return DatasetBundle(chunks=(), queries_train=(), queries_anchor=(), queries_oracle=())


def _lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_registered_in_chat_models_registry() -> None:
    assert CHAT_MODELS.get("cached") is CachedChatModel


def test_record_run_calls_backend_once_per_distinct_input_and_writes_jsonl(
    tmp_path: Path,
) -> None:
    import json

    path = tmp_path / "transcripts" / "gen.jsonl"
    spy = SpyChat()
    cached = CachedChatModel(backend=spy, transcript_path=str(path), mode="record")

    outputs = [cached.complete("sys", f"user {i}") for i in range(3)]

    assert spy.calls == 3
    lines = _lines(path)
    assert len(lines) == 3
    records = [json.loads(ln) for ln in lines]
    for rec, out in zip(records, outputs, strict=True):
        assert set(rec) == {"key", "system", "user", "kwargs", "response"}
        assert rec["response"] == out
        assert rec["system"] == "sys"


def test_replay_of_recorded_inputs_makes_zero_backend_calls_identical_outputs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gen.jsonl"
    rec_spy = SpyChat()
    recorder = CachedChatModel(backend=rec_spy, transcript_path=str(path), mode="record")
    recorded = [recorder.complete("s", f"u{i}", temperature=0.7) for i in range(4)]

    replay_spy = SpyChat()
    replayer = CachedChatModel(backend=replay_spy, transcript_path=str(path), mode="replay")
    replayed = [replayer.complete("s", f"u{i}", temperature=0.7) for i in range(4)]

    assert replay_spy.calls == 0
    assert replayed == recorded


def test_record_then_replay_single_backend_call(tmp_path: Path) -> None:
    """§13.4 fixture: record then replay ⇒ 1 backend call, identical outputs."""
    path = tmp_path / "gen.jsonl"
    spy = SpyChat()
    recorder = CachedChatModel(backend=spy, transcript_path=str(path), mode="record")
    first = recorder.complete("s", "u")

    replayer = CachedChatModel(backend=SpyChat(), transcript_path=str(path), mode="replay")
    second = replayer.complete("s", "u")

    assert spy.calls == 1
    assert first == second


def test_kwargs_order_independent_key_yields_one_line_and_one_call(tmp_path: Path) -> None:
    path = tmp_path / "gen.jsonl"
    spy = SpyChat()
    cached = CachedChatModel(backend=spy, transcript_path=str(path), mode="record")

    first = cached.complete("s", "u", alpha=1, beta=2)
    second = cached.complete("s", "u", beta=2, alpha=1)

    assert spy.calls == 1
    assert first == second
    assert len(_lines(path)) == 1


def test_hit_in_record_mode_returns_stored_without_backend_call(tmp_path: Path) -> None:
    path = tmp_path / "gen.jsonl"
    spy = SpyChat()
    cached = CachedChatModel(backend=spy, transcript_path=str(path), mode="record")

    first = cached.complete("s", "u")
    again = cached.complete("s", "u")

    assert spy.calls == 1
    assert first == again


def test_replay_miss_raises_actionable_error_naming_path(tmp_path: Path) -> None:
    path = tmp_path / "transcripts" / "gen.jsonl"
    cached = CachedChatModel(backend=SpyChat(), transcript_path=str(path), mode="replay")

    with pytest.raises(RuntimeError, match=str(path)) as excinfo:
        cached.complete("s", "never recorded")
    assert "record" in str(excinfo.value)


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        CachedChatModel(backend=SpyChat(), transcript_path="x.jsonl", mode="bogus")


def test_to_config_serializes_mode_path_and_backend_block() -> None:
    cached = CachedChatModel(
        backend=MockChatModel(seed=7), transcript_path="data/t/gen.jsonl", mode="replay"
    )
    config = cached.to_config()
    assert config["mode"] == "replay"
    assert config["transcript_path"] == "data/t/gen.jsonl"
    assert config["backend"] == {"type": "mock", "params": {"seed": 7}}


def test_from_config_builds_backend_through_registry() -> None:
    params = {
        "mode": "record",
        "transcript_path": "data/t/gen.jsonl",
        "backend": {"type": "mock", "params": {"seed": 4}},
    }
    cached = CachedChatModel.from_config(params, _bundle(), np.random.default_rng(0))
    assert isinstance(cached.backend, MockChatModel)
    assert cached.backend.seed == 4
    assert cached.mode == "record"


def test_from_config_round_trip_byte_stable() -> None:
    original = CachedChatModel(
        backend=MockChatModel(seed=2), transcript_path="data/t/judge.jsonl", mode="replay"
    )
    rebuilt = CachedChatModel.from_config(original.to_config(), _bundle(), np.random.default_rng(0))
    assert rebuilt.to_config() == original.to_config()
