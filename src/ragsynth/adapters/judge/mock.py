"""Deterministic offline RelevanceJudge for tests/CI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragsynth.adapters.judge.base import JUDGES, JudgeVerdict

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from ragsynth.datasets.base import DatasetBundle


@JUDGES.register("mock")
class MockJudge:
    """Rule-based judge with configurable answers.

    Defaults model the well-behaved case: answerable from evidence, not
    answerable from nothing -- which passes both the zero-context and
    answerability gate checks.
    """

    def __init__(
        self,
        *,
        answerable_with_evidence: bool = True,
        answerable_without_evidence: bool = False,
        confidence: float = 1.0,
    ) -> None:
        self.answerable_with_evidence = answerable_with_evidence
        self.answerable_without_evidence = answerable_without_evidence
        self.confidence = confidence

    def judge(self, query: str, evidence_texts: Sequence[str]) -> JudgeVerdict:
        """Apply the configured rule for evidence/no-evidence calls."""
        answerable = (
            self.answerable_with_evidence if evidence_texts else self.answerable_without_evidence
        )
        return JudgeVerdict(
            answerable=answerable,
            answer="mock answer" if answerable else "",
            confidence=self.confidence,
        )

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {
            "answerable_with_evidence": self.answerable_with_evidence,
            "answerable_without_evidence": self.answerable_without_evidence,
            "confidence": self.confidence,
        }

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> MockJudge:
        """Build from a config params block."""
        return cls(
            answerable_with_evidence=bool(params.get("answerable_with_evidence", True)),
            answerable_without_evidence=bool(params.get("answerable_without_evidence", False)),
            confidence=float(params.get("confidence", 1.0)),
        )
