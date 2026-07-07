"""Zero-context check: reject common-knowledge questions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragsynth.gate.checks.base import CHECKS, CheckResult, GateCheck

if TYPE_CHECKING:
    from ragsynth.domain import SyntheticQuery
    from ragsynth.pipeline.base import PipelineState, Resources


@CHECKS.register("zero_context")
class ZeroContextCheck(GateCheck):
    """Reject candidates the judge answers WITHOUT evidence (SPEC §6.4.2).

    A query answerable from nothing is common knowledge and cannot measure
    retrieval (CiteVQA zero-document self-test, arXiv 2605.12882).
    """

    name = "zero_context"

    def check(
        self, candidate: SyntheticQuery, state: PipelineState, resources: Resources
    ) -> CheckResult:
        """Judge with empty evidence; answerable means reject."""
        verdict = resources.judge.judge(candidate.text, [])
        if verdict.answerable:
            return CheckResult(
                passed=False,
                score=verdict.confidence,
                reason="answerable without evidence (common knowledge)",
            )
        return CheckResult(passed=True, score=verdict.confidence, reason="needs evidence")
