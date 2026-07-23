"""LLM chat adapters: Protocol, OpenAI-compatible client, offline mock."""

from ragsynth.adapters.llm.base import CHAT_MODELS, ChatModel
from ragsynth.adapters.llm.cached import CachedChatModel
from ragsynth.adapters.llm.mock import MockChatModel
from ragsynth.adapters.llm.openai_compatible import OpenAICompatibleChat

__all__ = [
    "CHAT_MODELS",
    "CachedChatModel",
    "ChatModel",
    "MockChatModel",
    "OpenAICompatibleChat",
]
