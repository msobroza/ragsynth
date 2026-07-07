"""Dedup check: exact + semantic duplicate rejection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ragsynth.gate.checks.base import CHECKS, CheckResult, GateCheck

if TYPE_CHECKING:
    from ragsynth.domain import SyntheticQuery
    from ragsynth.pipeline.base import PipelineState, Resources


@CHECKS.register("dedup")
class DedupCheck(GateCheck):
    """Reject exact-text and near-duplicate (cosine) candidates (SPEC §6.4.1).

    Compares against the gate's already-accepted working set
    (``state.gate_accepted``) in arrival order -- a greedy simplification of
    MMR selection (Carbonell & Goldstein, SIGIR 1998; PLAN D13).
    """

    name = "dedup"

    def __init__(self, cos_threshold: float = 0.95) -> None:
        self.cos_threshold = cos_threshold

    def check(
        self, candidate: SyntheticQuery, state: PipelineState, resources: Resources
    ) -> CheckResult:
        """Reject if the candidate duplicates an accepted candidate."""
        accepted = state.gate_accepted
        if not accepted:
            return CheckResult(passed=True, score=0.0, reason="first candidate")
        if any(candidate.text == a.text for a in accepted):
            return CheckResult(passed=False, score=1.0, reason="exact duplicate text")
        if candidate.embedding_ref is None:
            return CheckResult(passed=True, score=None, reason="no embedding; text is novel")
        cand = resources.embeddings.get([candidate.embedding_ref])[0].astype(np.float64)
        refs = [a.embedding_ref for a in accepted if a.embedding_ref is not None]
        if not refs:
            return CheckResult(passed=True, score=None, reason="no accepted embeddings")
        max_cos = float(np.max(resources.embeddings.get(refs).astype(np.float64) @ cand))
        if max_cos >= self.cos_threshold:
            return CheckResult(
                passed=False,
                score=max_cos,
                reason=f"cosine {max_cos:.3f} >= {self.cos_threshold} vs accepted set",
            )
        return CheckResult(passed=True, score=max_cos, reason="novel")

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {"cos_threshold": self.cos_threshold}
