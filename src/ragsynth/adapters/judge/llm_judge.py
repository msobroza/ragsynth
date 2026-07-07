"""LLM-backed RelevanceJudge with strict-JSON verdict parsing (SPEC §12, §6.4).

The judge owns its OWN ChatModel instance, built from a distinct config
key (``params["chat"]``) -- never shared with the generator LLM, so judge
and generator can differ (and be swapped) independently.

Note:
    The judging prompt is an inline string template for now; the SPEC
    steps/prompts jinja templates (``judge_v1.j2``) arrive in Phase 3 and
    will replace it behind the same ``prompt_version`` knob.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from ragsynth.adapters.judge.base import JUDGES, JudgeVerdict
from ragsynth.adapters.llm.base import CHAT_MODELS

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from ragsynth.adapters.llm.base import ChatModel
    from ragsynth.datasets.base import DatasetBundle

logger = logging.getLogger(__name__)

_FALLBACK_VERDICT = JudgeVerdict(answerable=False, answer="", confidence=0.0)

_SYSTEM_PROMPT = (
    "You are a strict relevance judge for retrieval evaluation. "
    "Decide whether the query is answerable from the given evidence alone. "
    'Reply ONLY with a JSON object: {"answerable": bool, "answer": str, '
    '"confidence": float}. No prose, no markdown, no explanation.'
)

_USER_TEMPLATE = """Query:
{query}

Evidence:
{evidence}

Is the query answerable from the evidence above alone? Reply ONLY with the JSON object."""

_NO_EVIDENCE_PLACEHOLDER = "(no evidence provided)"


def _parse_verdict(raw: str) -> JudgeVerdict | None:
    """Parse the first JSON object found in ``raw``; ``None`` if unusable."""
    decoder = json.JSONDecoder()
    start = raw.find("{")
    while start != -1:
        try:
            data, _ = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            start = raw.find("{", start + 1)
            continue
        if not isinstance(data, dict):
            return None
        try:
            return JudgeVerdict(
                answerable=bool(data["answerable"]),
                answer=str(data.get("answer", "")),
                confidence=float(data.get("confidence", 0.0)),
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None


@JUDGES.register("llm")
class LLMJudge:
    """RelevanceJudge that asks a ChatModel for a strict-JSON verdict.

    Malformed or unparseable model output degrades safely to
    ``JudgeVerdict(False, "", 0.0)`` with a warning log, so a flaky judge
    rejects rather than pollutes the gate.
    """

    def __init__(self, chat: ChatModel, prompt_version: str = "judge_v1") -> None:
        self.chat = chat
        self.prompt_version = prompt_version

    def judge(self, query: str, evidence_texts: Sequence[str]) -> JudgeVerdict:
        """Return the verdict for one (query, evidence) pair."""
        evidence = (
            "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(evidence_texts))
            or _NO_EVIDENCE_PLACEHOLDER
        )
        raw = self.chat.complete(
            _SYSTEM_PROMPT, _USER_TEMPLATE.format(query=query, evidence=evidence)
        )
        verdict = _parse_verdict(raw)
        if verdict is None:
            logger.warning(
                f"LLMJudge: unparseable verdict for query {query[:60]!r}; "
                f"falling back to not-answerable (raw head: {raw[:120]!r})"
            )
            return _FALLBACK_VERDICT
        return verdict

    def to_config(self) -> dict[str, Any]:
        """JSON-safe params with the nested chat config under its own key."""
        chat_type = next(
            (key for key in CHAT_MODELS.keys() if CHAT_MODELS.get(key) is type(self.chat)),  # noqa: SIM118
            type(self.chat).__name__,
        )
        to_config = getattr(self.chat, "to_config", None)
        chat_params: dict[str, Any] = to_config() if callable(to_config) else {}
        return {
            "prompt_version": self.prompt_version,
            "chat": {"type": chat_type, "params": chat_params},
        }

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> LLMJudge:
        """Build the judge and its OWN nested ChatModel from ``params["chat"]``.

        Enforces the distinct-config-key rule (SPEC §12, §6.4): the judge
        never reuses the generator's model instance.
        """
        chat_config = params["chat"]
        chat_cls = CHAT_MODELS.get(str(chat_config["type"]))
        chat: ChatModel = chat_cls.from_config(dict(chat_config.get("params", {})), bundle, rng)
        return cls(chat=chat, prompt_version=str(params.get("prompt_version", "judge_v1")))
