"""Tests for LLMJudge: strict-JSON verdict parsing over any ChatModel."""

import logging
from typing import Any

import numpy as np
import pytest

from ragsynth.adapters.judge.base import JUDGES, JudgeVerdict
from ragsynth.adapters.judge.llm_judge import LLMJudge
from ragsynth.adapters.llm.mock import MockChatModel
from ragsynth.datasets.base import DatasetBundle


class StubChat:
    """ChatModel stub replaying a canned response and recording calls."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        self.calls.append((system, user))
        return self.response


def _bundle() -> DatasetBundle:
    return DatasetBundle(chunks=(), queries_train=(), queries_anchor=(), queries_oracle=())


def test_registered_in_judges_registry() -> None:
    assert JUDGES.get("llm") is LLMJudge


def test_happy_path_parses_strict_json() -> None:
    chat = StubChat('{"answerable": true, "answer": "42 bps", "confidence": 0.9}')
    judge = LLMJudge(chat=chat)
    verdict = judge.judge("what is the spread?", ["The spread is 42 bps."])
    assert verdict == JudgeVerdict(answerable=True, answer="42 bps", confidence=0.9)


def test_prompt_contains_query_and_evidence() -> None:
    chat = StubChat('{"answerable": false, "answer": "", "confidence": 0.1}')
    judge = LLMJudge(chat=chat)
    judge.judge("what is the spread?", ["evidence alpha", "evidence beta"])
    (_, user) = chat.calls[0]
    assert "what is the spread?" in user
    assert "evidence alpha" in user
    assert "evidence beta" in user


def test_json_wrapped_in_prose_is_still_parsed() -> None:
    chat = StubChat(
        'Sure! Here is my verdict:\n{"answerable": true, "answer": "yes", "confidence": 0.5}\n'
        "Hope that helps."
    )
    verdict = LLMJudge(chat=chat).judge("q", ["e"])
    assert verdict == JudgeVerdict(answerable=True, answer="yes", confidence=0.5)


def test_malformed_output_falls_back_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    chat = StubChat("I cannot answer in JSON, sorry.")
    with caplog.at_level(logging.WARNING):
        verdict = LLMJudge(chat=chat).judge("q", ["e"])
    assert verdict == JudgeVerdict(answerable=False, answer="", confidence=0.0)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_broken_json_falls_back_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    chat = StubChat('{"answerable": true, "answer": ')
    with caplog.at_level(logging.WARNING):
        verdict = LLMJudge(chat=chat).judge("q", ["e"])
    assert verdict == JudgeVerdict(answerable=False, answer="", confidence=0.0)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_missing_answerable_key_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    chat = StubChat('{"answer": "yes", "confidence": 0.5}')
    with caplog.at_level(logging.WARNING):
        verdict = LLMJudge(chat=chat).judge("q", ["e"])
    assert verdict == JudgeVerdict(answerable=False, answer="", confidence=0.0)


def test_from_config_builds_nested_chat_from_registry() -> None:
    params = {"chat": {"type": "mock", "params": {"seed": 3}}, "prompt_version": "judge_v1"}
    judge = LLMJudge.from_config(params, _bundle(), np.random.default_rng(0))
    assert isinstance(judge.chat, MockChatModel)
    assert judge.chat.seed == 3
    # The mock replies non-JSON prose -> deterministic fallback verdict.
    verdict = judge.judge("q", ["e"])
    assert verdict == JudgeVerdict(answerable=False, answer="", confidence=0.0)


def test_to_config_serializes_nested_chat() -> None:
    judge = LLMJudge(chat=MockChatModel(seed=5), prompt_version="judge_v1")
    config = judge.to_config()
    assert config["prompt_version"] == "judge_v1"
    assert config["chat"] == {"type": "mock", "params": {"seed": 5}}
