"""Uniqueness check: detect (and promote or reject) leaky non-gold answers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ragsynth.gate.checks.base import CHECKS, CheckResult, GateCheck

if TYPE_CHECKING:
    from ragsynth.domain import SyntheticQuery
    from ragsynth.pipeline.base import PipelineState, Resources

_MODES = ("promote", "reject")


@CHECKS.register("uniqueness")
class UniquenessCheck(GateCheck):
    """Judge whether top non-gold retrieved chunks also answer (SPEC §6.4.5).

    Corrupted golds invalidate retrieval eval: if another chunk answers the
    query, either add it to the qrels (``promote``) or drop the candidate
    (``reject``) -- the anti-leakage rule.
    """

    name = "uniqueness"

    def __init__(self, mode: str = "promote", top_m: int = 5) -> None:
        if mode not in _MODES:
            raise ValueError(f"uniqueness mode must be one of {_MODES}, got '{mode}'")
        self.mode = mode
        self.top_m = top_m

    def check(
        self, candidate: SyntheticQuery, state: PipelineState, resources: Resources
    ) -> CheckResult:
        """Inspect the top non-gold retrieved chunks with the judge."""
        if candidate.embedding_ref is None:
            return CheckResult(passed=True, score=None, reason="no embedding; skipped")
        emb = resources.embeddings.get([candidate.embedding_ref])[0].astype(np.float64)
        gold = set(candidate.seed.chunk_ids)
        hits = resources.retriever.search(emb, self.top_m + len(gold))
        non_gold = [cid for cid, _ in hits if cid not in gold][: self.top_m]
        leaky: list[str] = []
        for chunk_id in non_gold:
            chunk = resources.chunk_index.get(chunk_id)
            if chunk is None:
                continue
            if resources.judge.judge(candidate.text, [chunk.text]).answerable:
                leaky.append(chunk_id)
        if leaky and self.mode == "reject":
            return CheckResult(
                passed=False,
                score=float(len(leaky)),
                reason=f"leaky gold: {len(leaky)} non-gold chunk(s) also answer",
            )
        return CheckResult(
            passed=True,
            score=float(len(leaky)),
            reason="promoted leaky chunks into qrels" if leaky else "gold is unique",
            promoted=tuple(leaky),
        )

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {"mode": self.mode, "top_m": self.top_m}
