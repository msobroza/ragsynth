"""Transcript record/replay ChatModel wrapper -- the determinism boundary (D40).

The first (``record``) run against a live endpoint appends every distinct
request/response to a JSONL transcript keyed by ``sha256(system, user,
sorted kwargs)``. Every later (``replay``) run reads that transcript and never
touches the network, so ``same seed + same transcripts ⇒ identical metrics``
(SPEC §15, spec01 §7). Generator and judge each point at their own transcript
file -- they are just different paths, no special handling here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragsynth.adapters.llm.base import CHAT_MODELS

if TYPE_CHECKING:
    import numpy as np

    from ragsynth.adapters.llm.base import ChatModel
    from ragsynth.datasets.base import DatasetBundle

_MODES = ("record", "replay")


@CHAT_MODELS.register("cached")
class CachedChatModel:
    """Transcript-record/replay decorator around any ChatModel (D40).

    Key = sha256 over (system, user, sorted kwargs). Modes:
    record (miss -> call backend, append jsonl, return), replay (miss -> raise
    actionable error naming the transcript path). to_config/from_config
    serialize the backend block + transcript_path + mode.
    """

    def __init__(self, backend: ChatModel, transcript_path: str, mode: str = "record") -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.backend = backend
        self.transcript_path = Path(transcript_path)
        self.mode = mode
        self._cache: dict[str, str] | None = None

    @staticmethod
    def _key(system: str, user: str, kwargs: dict[str, Any]) -> str:
        """sha256 over the canonical JSON of ``[system, user, sorted(kwargs)]``."""
        canonical = json.dumps(
            [system, user, sorted(kwargs.items())],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _load(self) -> dict[str, str]:
        """Read the transcript once, lazily; return the in-memory key->response map."""
        if self._cache is None:
            cache: dict[str, str] = {}
            if self.transcript_path.exists():
                for line in self.transcript_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    cache[str(record["key"])] = str(record["response"])
            self._cache = cache
        return self._cache

    def _append(
        self, key: str, system: str, user: str, kwargs: dict[str, Any], response: str
    ) -> None:
        """Append one transcript record as a single JSONL line (record mode)."""
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"key": key, "system": system, "user": user, "kwargs": kwargs, "response": response},
            ensure_ascii=False,
        )
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def complete(self, system: str, user: str, **kw: Any) -> str:
        """Return the transcript hit, or (record mode) call the backend and store it.

        Raises:
            RuntimeError: In replay mode on a transcript miss, naming the path
                and the one-line fix (rerun once with ``mode: record``).
        """
        cache = self._load()
        key = self._key(system, user, kw)
        if key in cache:
            return cache[key]
        if self.mode == "replay":
            raise RuntimeError(
                f"transcript miss in replay mode: no recorded response for this request in "
                f"{self.transcript_path}. Run once with mode: record against a live backend "
                f"to populate the transcript, then replay."
            )
        response = self.backend.complete(system, user, **kw)
        self._append(key, system, user, kw, response)
        cache[key] = response
        return response

    def to_config(self) -> dict[str, Any]:
        """JSON-safe params: mode, transcript path, and the nested backend block."""
        backend_type = next(
            (key for key in CHAT_MODELS.keys() if CHAT_MODELS.get(key) is type(self.backend)),  # noqa: SIM118
            type(self.backend).__name__,
        )
        to_config = getattr(self.backend, "to_config", None)
        backend_params: dict[str, Any] = to_config() if callable(to_config) else {}
        return {
            "mode": self.mode,
            "transcript_path": str(self.transcript_path),
            "backend": {"type": backend_type, "params": backend_params},
        }

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> CachedChatModel:
        """Build the wrapper and its backend through the CHAT_MODELS registry."""
        backend_config = params["backend"]
        backend_cls = CHAT_MODELS.get(str(backend_config["type"]))
        backend: ChatModel = backend_cls.from_config(
            dict(backend_config.get("params", {})), bundle, rng
        )
        return cls(
            backend=backend,
            transcript_path=str(params["transcript_path"]),
            mode=str(params.get("mode", "record")),
        )
