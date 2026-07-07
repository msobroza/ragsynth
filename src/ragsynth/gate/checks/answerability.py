"""Answerability check: the gold evidence must actually answer the query."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragsynth.gate.checks.base import CHECKS, CheckResult, GateCheck

if TYPE_CHECKING:
    from ragsynth.domain import SyntheticQuery
    from ragsynth.pipeline.base import PipelineState, Resources


@CHECKS.register("answerability")
class AnswerabilityCheck(GateCheck):
    """Reject candidates the judge cannot answer FROM the evidence (SPEC §6.4.3).

    The counterpart of the zero-context test (CiteVQA, arXiv 2605.12882):
    the seed chunks must suffice to answer, or the qrel is wrong.
    """

    name = "answerability"

    def check(
        self, candidate: SyntheticQuery, state: PipelineState, resources: Resources
    ) -> CheckResult:
        """Judge with the seed chunks as evidence; unanswerable means reject."""
        evidence = [
            resources.chunk_index[chunk_id].text
            for chunk_id in candidate.seed.chunk_ids
            if chunk_id in resources.chunk_index
        ]
        verdict = resources.judge.judge(candidate.text, evidence)
        if not verdict.answerable:
            return CheckResult(
                passed=False,
                score=verdict.confidence,
                reason="not answerable from gold evidence",
            )
        return CheckResult(passed=True, score=verdict.confidence, reason="answerable")
