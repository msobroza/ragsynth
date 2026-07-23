"""Offline tests for OpenAICompatibleChat: the `_post_json` seam is monkeypatched."""

from typing import Any

import pytest

from ragsynth.adapters.llm.base import CHAT_MODELS
from ragsynth.adapters.llm.openai_compatible import OpenAICompatibleChat

BASE_URL = "http://localhost:8000/v1"


def _fake_response(content: str = "the answer") -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch) -> OpenAICompatibleChat:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return OpenAICompatibleChat(base_url=BASE_URL, model="my-model")


def test_registered_in_chat_models_registry() -> None:
    assert CHAT_MODELS.get("openai_compatible") is OpenAICompatibleChat


def test_complete_posts_to_chat_completions_and_parses_content(
    chat: OpenAICompatibleChat, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen["url"] = url
        seen["payload"] = payload
        return _fake_response("hello world")

    monkeypatch.setattr(chat, "_post_json", fake_post)
    out = chat.complete("be terse", "what is a swap?")

    assert out == "hello world"
    assert seen["url"] == BASE_URL + "/chat/completions"
    payload = seen["payload"]
    assert payload["model"] == "my-model"
    assert payload["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "what is a swap?"},
    ]


def test_defaults_land_in_payload(
    chat: OpenAICompatibleChat, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = payload
        return _fake_response()

    monkeypatch.setattr(chat, "_post_json", fake_post)
    chat.complete("s", "u")

    assert seen["payload"]["temperature"] == 0.7
    assert seen["payload"]["max_tokens"] == 512


def test_kwargs_override_temperature_and_max_tokens(
    chat: OpenAICompatibleChat, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = payload
        return _fake_response()

    monkeypatch.setattr(chat, "_post_json", fake_post)
    chat.complete("s", "u", temperature=0.1, max_tokens=64)

    assert seen["payload"]["temperature"] == 0.1
    assert seen["payload"]["max_tokens"] == 64


def test_missing_env_var_raises_actionable_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MY_GATEWAY_KEY", raising=False)
    chat = OpenAICompatibleChat(base_url=BASE_URL, model="m", api_key_env="MY_GATEWAY_KEY")
    with pytest.raises(RuntimeError, match="MY_GATEWAY_KEY"):
        chat.complete("s", "u")


def test_to_config_never_serializes_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret")
    chat = OpenAICompatibleChat(base_url=BASE_URL, model="m")
    config = chat.to_config()
    assert config["api_key_env"] == "OPENAI_API_KEY"
    assert "super-secret" not in repr(config)


def test_from_config_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from ragsynth.datasets.base import DatasetBundle

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    bundle = DatasetBundle(chunks=(), queries_train=(), queries_anchor=(), queries_oracle=())
    original = OpenAICompatibleChat(
        base_url=BASE_URL, model="m", temperature=0.2, max_tokens=99, timeout=5.0
    )
    rebuilt = OpenAICompatibleChat.from_config(
        original.to_config(), bundle, np.random.default_rng(0)
    )
    assert rebuilt.to_config() == original.to_config()


def test_env_placeholder_in_base_url_resolves_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("RAGSYNTH_LLM_BASE_URL", "http://gateway.internal:9000/v1")
    chat = OpenAICompatibleChat(base_url="${RAGSYNTH_LLM_BASE_URL}", model="m")
    assert chat.base_url == "http://gateway.internal:9000/v1"


def test_env_placeholder_missing_var_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAGSYNTH_LLM_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="RAGSYNTH_LLM_BASE_URL"):
        OpenAICompatibleChat(base_url="${RAGSYNTH_LLM_BASE_URL}", model="m")


def test_to_config_preserves_unresolved_env_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGSYNTH_LLM_BASE_URL", "http://gateway.internal:9000/v1")
    chat = OpenAICompatibleChat(base_url="${RAGSYNTH_LLM_BASE_URL}", model="m")
    assert chat.to_config()["base_url"] == "${RAGSYNTH_LLM_BASE_URL}"


def test_env_placeholder_round_trip_preserves_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from ragsynth.datasets.base import DatasetBundle

    monkeypatch.setenv("RAGSYNTH_LLM_BASE_URL", "http://gateway.internal:9000/v1")
    bundle = DatasetBundle(chunks=(), queries_train=(), queries_anchor=(), queries_oracle=())
    original = OpenAICompatibleChat(base_url="${RAGSYNTH_LLM_BASE_URL}", model="m")
    rebuilt = OpenAICompatibleChat.from_config(
        original.to_config(), bundle, np.random.default_rng(0)
    )
    assert rebuilt.to_config()["base_url"] == "${RAGSYNTH_LLM_BASE_URL}"
