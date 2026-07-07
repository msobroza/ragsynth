"""RelevanceJudge Protocol, JudgeVerdict, and the judge registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ragsynth.pipeline.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class JudgeVerdict:
    """Outcome of a relevance/answerability judgment (SPEC §12).

    Attributes:
        answerable: Whether the query is answerable from the given evidence
            (or from nothing, in the zero-context self-test).
        answer: The judge's answer text (may be empty).
        confidence: Judge-reported confidence in [0, 1].
    """

    answerable: bool
    answer: str
    confidence: float


class RelevanceJudge(Protocol):
    """Judges whether ``query`` is answerable from ``evidence_texts`` (SPEC §12).

    Called with empty evidence for the zero-context common-knowledge
    self-test (CiteVQA, arXiv 2605.12882).
    """

    def judge(self, query: str, evidence_texts: Sequence[str]) -> JudgeVerdict:
        """Return the verdict for one (query, evidence) pair."""
        ...


JUDGES: Registry[Any] = Registry("judge")
