"""Round-trip check: the configured retriever must find the gold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ragsynth.gate.checks.base import CHECKS, CheckResult, GateCheck

if TYPE_CHECKING:
    from ragsynth.domain import SyntheticQuery
    from ragsynth.pipeline.base import PipelineState, Resources


@CHECKS.register("round_trip")
class RoundTripCheck(GateCheck):
    """Reject candidates whose gold chunks miss the retriever top-k (SPEC §6.4.4).

    The Promptagator consistency filter (Dai et al., ICLR 2023); filtering
    beats generating more (Doc2Query--, Gospodinov et al., ECIR 2023).
    """

    name = "round_trip"

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def check(
        self, candidate: SyntheticQuery, state: PipelineState, resources: Resources
    ) -> CheckResult:
        """Pass with score 1/rank of the best-ranked gold chunk in top-k."""
        if candidate.embedding_ref is None:
            return CheckResult(passed=False, score=None, reason="candidate has no embedding")
        emb = resources.embeddings.get([candidate.embedding_ref])[0].astype(np.float64)
        hits = resources.retriever.search(emb, self.k)
        gold = set(candidate.seed.chunk_ids)
        for rank, (chunk_id, _score) in enumerate(hits, start=1):
            if chunk_id in gold:
                return CheckResult(passed=True, score=1.0 / rank, reason=f"gold at rank {rank}")
        return CheckResult(passed=False, score=0.0, reason=f"gold not in retriever top-{self.k}")

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {"k": self.k}
