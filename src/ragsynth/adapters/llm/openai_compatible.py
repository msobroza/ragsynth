"""ChatModel over any OpenAI-compatible ``/chat/completions`` endpoint (SPEC §12).

Covers vLLM, LiteLLM, and internal gateways. Implemented over stdlib
``urllib.request`` (air-gap rule: no HTTP client dependency); all network
I/O funnels through the private ``_post_json`` seam, which tests
monkeypatch so CI never touches the network.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import TYPE_CHECKING, Any

from ragsynth.adapters.llm.base import CHAT_MODELS

if TYPE_CHECKING:
    import numpy as np

    from ragsynth.datasets.base import DatasetBundle

_CHAT_COMPLETIONS_PATH = "/chat/completions"
_ALLOWED_SCHEMES = ("http://", "https://")
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TIMEOUT_S = 60.0


@CHAT_MODELS.register("openai_compatible")
class OpenAICompatibleChat:
    """Chat completions against an OpenAI-compatible HTTP gateway.

    The API key is never stored on the instance nor serialized: only the
    *name* of the environment variable (``api_key_env``) is configuration;
    the value is read from ``os.environ`` at call time.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        if not base_url.startswith(_ALLOWED_SCHEMES):
            raise ValueError(f"base_url must start with http:// or https://, got {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _api_key(self) -> str:
        """Read the API key from the configured environment variable.

        Raises:
            RuntimeError: If the environment variable is not set.
        """
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"environment variable {self.api_key_env!r} is not set; "
                f"export it with the API key for {self.base_url} "
                f"(e.g. `export {self.api_key_env}=...`)"
            )
        return key

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``payload`` as JSON and return the decoded JSON response.

        The single network seam of this adapter; tests monkeypatch it.
        """
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - scheme validated in __init__
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key()}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            decoded: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        return decoded

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        """Return the assistant text for a (system, user) exchange.

        Args:
            system: System prompt.
            user: User prompt.
            **kwargs: Per-call overrides for ``temperature`` / ``max_tokens``.

        Raises:
            RuntimeError: If the configured API-key environment variable is
                missing (checked before any network activity).
        """
        self._api_key()  # Fail fast, before building or sending the request.
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        data = self._post_json(f"{self.base_url}{_CHAT_COMPLETIONS_PATH}", payload)
        return str(data["choices"][0]["message"]["content"])

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params (env var *name* only, never the key)."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> OpenAICompatibleChat:
        """Build from a config params block (composition-root factory contract)."""
        return cls(
            base_url=str(params["base_url"]),
            model=str(params["model"]),
            api_key_env=str(params.get("api_key_env", "OPENAI_API_KEY")),
            temperature=float(params.get("temperature", _DEFAULT_TEMPERATURE)),
            max_tokens=int(params.get("max_tokens", _DEFAULT_MAX_TOKENS)),
            timeout=float(params.get("timeout", _DEFAULT_TIMEOUT_S)),
        )
