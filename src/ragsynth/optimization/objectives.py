"""Optimization objectives: fidelity-driven prompt fitness (SPEC §11).

:class:`FidelityObjective` is the objective the v2 closed loop (MIPROv2,
Opsahl-Ong et al., "Optimizing Instructions and Demonstrations for
Multi-Stage Language Model Programs", EMNLP 2024; roadmap R2) searches
over. It is dependency-injected -- callers pass a ``pipeline_factory``
building a mini pipeline for a candidate prompt -- so this module never
imports Phase-3 pipeline steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ragsynth.metrics.fidelity import c2st_auc, kl_similarity_distributions

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ragsynth.io.embeddings import EmbeddingStore
    from ragsynth.pipeline.pipeline import Pipeline

HARD_PENALTY = -1e9
"""Score when the gate collapses or too few accepted embeddings exist to measure."""

_MIN_EMBEDDINGS = 2
"""Minimum accepted-query embeddings for the fidelity metrics to be defined."""


class FidelityObjective:
    """Prompt fitness = negated fidelity gap of the induced synthetic cloud.

    Structurally satisfies
    :class:`~ragsynth.optimization.base.OptimizationMetric`. Each call runs
    a mini pipeline built for the candidate prompt, collects the accepted
    queries' embeddings, and scores them against the real reference::

        score = -(KL + alpha * max(0, C2ST_AUC - 0.5))

    subject to ``gate_pass_rate >= min_pass_rate`` -- a prompt that
    collapses the gate (or yields too few accepted queries to measure)
    earns :data:`HARD_PENALTY` instead of a spuriously good fidelity
    number. This is the v2 closed loop's objective (MIPROv2, Opsahl-Ong
    et al., EMNLP 2024); the companion routing signal -- which gate check
    rejected what, ``state.metrics["gate_reject_reasons"]`` (SPEC §6.4) --
    tells that optimizer *what* to fix, while this scalar tells it *how
    much* better a candidate is.

    Attributes:
        pipeline_factory: Builds the mini pipeline for a candidate prompt
            (dependency injection; no Phase-3 step imports here).
        reference_embs: Real reference query embeddings, ``(n_ref, d)``.
        chunk_embs: Chunk embeddings, ``(n_chunks, d)``.
        embeddings: Store resolving accepted queries' ``embedding_ref``.
        alpha: Weight of the C2ST excess-AUC term.
        min_pass_rate: Gate pass-rate floor below which the prompt is
            hard-penalized.
        seed: Seed forwarded to the C2ST cross-validation splitter.
    """

    def __init__(
        self,
        pipeline_factory: Callable[[str], Pipeline],
        reference_embs: NDArray[np.float64],
        chunk_embs: NDArray[np.float64],
        embeddings: EmbeddingStore,
        alpha: float = 1.0,
        min_pass_rate: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.pipeline_factory = pipeline_factory
        self.reference_embs = reference_embs
        self.chunk_embs = chunk_embs
        self.embeddings = embeddings
        self.alpha = alpha
        self.min_pass_rate = min_pass_rate
        self.seed = seed

    def __call__(self, prompt: str) -> float:
        """Score a candidate prompt (higher is better).

        Runs ``pipeline_factory(prompt)``, gathers accepted-query
        embeddings via each record's ``query.embedding_ref`` (records
        without a ref are skipped), and reads
        ``state.metrics["gate_pass_rate"]`` (default 1.0 when absent).

        Args:
            prompt: The candidate prompt text.

        Returns:
            ``-(KL + alpha * max(0, C2ST_AUC - 0.5))``, or
            :data:`HARD_PENALTY` when the gate pass rate falls below
            ``min_pass_rate`` or fewer than 2 embeddings survive.
        """
        state = self.pipeline_factory(prompt).run()
        refs = [
            record.query.embedding_ref
            for record in state.accepted
            if record.query.embedding_ref is not None
        ]
        pass_rate = float(state.metrics.get("gate_pass_rate", 1.0))
        if pass_rate < self.min_pass_rate or len(refs) < _MIN_EMBEDDINGS:
            return HARD_PENALTY
        synth_embs = self.embeddings.get(refs).astype(np.float64)
        kl = kl_similarity_distributions(self.reference_embs, synth_embs, self.chunk_embs)
        auc = c2st_auc(self.reference_embs, synth_embs, seed=self.seed)
        return -(kl + self.alpha * max(0.0, auc - 0.5))
