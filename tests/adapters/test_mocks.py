"""Tests for adapter Protocols and their deterministic mocks (SPEC §12)."""

import numpy as np

from ragsynth.adapters.embedder.mock import MockEmbedder
from ragsynth.adapters.judge.base import JudgeVerdict
from ragsynth.adapters.judge.mock import MockJudge
from ragsynth.adapters.llm.mock import MockChatModel


def test_mock_chat_is_deterministic_across_instances() -> None:
    a = MockChatModel(seed=0)
    b = MockChatModel(seed=0)
    out1 = a.complete("system prompt", "write a question about fees")
    out2 = b.complete("system prompt", "write a question about fees")
    assert out1 == out2
    assert isinstance(out1, str)
    assert out1


def test_mock_chat_varies_with_input_and_seed() -> None:
    m = MockChatModel(seed=0)
    assert m.complete("s", "u1") != m.complete("s", "u2")
    assert MockChatModel(seed=1).complete("s", "u1") != m.complete("s", "u1")


def test_mock_embedder_unit_norm_and_deterministic() -> None:
    e = MockEmbedder(dim=16, seed=0)
    out = e.encode(["alpha", "beta", "alpha"])
    assert out.shape == (3, 16)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(out[0], out[2])
    again = MockEmbedder(dim=16, seed=0).encode(["alpha"])
    np.testing.assert_array_equal(again[0], out[0])


def test_mock_judge_default_rules() -> None:
    j = MockJudge()
    with_evidence = j.judge("q", ["some evidence"])
    without = j.judge("q", [])
    assert with_evidence == JudgeVerdict(answerable=True, answer="mock answer", confidence=1.0)
    assert without.answerable is False


def test_mock_judge_configurable_rules() -> None:
    j = MockJudge(answerable_with_evidence=False, answerable_without_evidence=True)
    assert j.judge("q", ["e"]).answerable is False
    assert j.judge("q", []).answerable is True
