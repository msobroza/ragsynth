"""Tests for the prompt-optimization contract (SPEC §11 -- v1 is contract only)."""

import dataclasses
from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from ragsynth.domain import AnnotationRecord, Seed, Stratum, SyntheticQuery
from ragsynth.io.embeddings import EmbeddingStore
from ragsynth.optimization import (
    BaseOptimizerConfig,
    FidelityObjective,
    MIPROv2Optimizer,
    NoOpOptimizer,
    OptimizationResult,
    TrialRecord,
    create_optimizer,
    mipro_adapter,
)
from ragsynth.optimization.base import OPTIMIZERS
from ragsynth.optimization.objectives import HARD_PENALTY
from ragsynth.pipeline.base import PipelineState, PipelineStep, Resources
from ragsynth.pipeline.pipeline import Pipeline
from ragsynth.pipeline.registry import RegistryError

# ---------------------------------------------------------------------------
# NoOpOptimizer
# ---------------------------------------------------------------------------


class TestNoOpOptimizer:
    def test_without_metric_returns_baseline_with_zero_score(self):
        result = NoOpOptimizer(BaseOptimizerConfig()).optimize("keep this prompt", None, None)
        assert result.optimized_prompt == "keep this prompt"
        assert result.baseline_score == 0.0
        assert result.optimized_score == 0.0
        assert result.improvement == 0.0
        assert result.num_trials == 1
        assert len(result.trial_history) == 1
        assert result.trial_history[0].score == 0.0
        assert result.optimizer_name == "noop"
        assert result.config == {"max_trials": 10, "seed": 0}

    def test_metric_evaluated_exactly_once_on_current_prompt(self):
        calls: list[str] = []

        def metric(prompt: str) -> float:
            calls.append(prompt)
            return float(len(prompt))

        result = NoOpOptimizer(BaseOptimizerConfig()).optimize("abc", None, metric)
        assert calls == ["abc"]
        assert result.baseline_score == metric("abc")
        assert result.optimized_score == result.baseline_score
        assert result.improvement == 0.0
        assert result.trial_history[0].prompt == "abc"
        assert result.trial_history[0].score == result.baseline_score

    def test_config_dict_reflects_custom_knobs(self):
        result = NoOpOptimizer(BaseOptimizerConfig(max_trials=3, seed=7)).optimize("p", None, None)
        assert result.config == {"max_trials": 3, "seed": 7}

    def test_trial_timestamp_is_timezone_aware_iso8601(self):
        result = NoOpOptimizer(BaseOptimizerConfig()).optimize("p", None, None)
        parsed = datetime.fromisoformat(result.trial_history[0].timestamp)
        assert parsed.tzinfo is not None


# ---------------------------------------------------------------------------
# Frozen result types / frozen config
# ---------------------------------------------------------------------------


class TestFrozenTypes:
    def test_trial_record_rejects_mutation(self):
        trial = TrialRecord(trial_id=1, prompt="p", score=None, timestamp="2026-01-01T00:00:00")
        with pytest.raises(dataclasses.FrozenInstanceError):
            trial.score = 1.0  # type: ignore[misc]

    def test_optimization_result_rejects_mutation(self):
        result = OptimizationResult(
            optimized_prompt="p",
            baseline_score=0.0,
            optimized_score=0.0,
            improvement=0.0,
            num_trials=1,
            trial_history=[],
            optimizer_name="noop",
            config={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.improvement = 1.0  # type: ignore[misc]

    def test_base_optimizer_config_is_frozen(self):
        config = BaseOptimizerConfig()
        with pytest.raises(ValidationError):
            config.max_trials = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------


class TestCreateOptimizer:
    def test_creates_noop_instance(self):
        optimizer = create_optimizer("noop", BaseOptimizerConfig())
        assert isinstance(optimizer, NoOpOptimizer)

    def test_unknown_name_raises_registry_error_listing_known_keys(self):
        with pytest.raises(RegistryError, match=r"unknown prompt optimizer.*noop"):
            create_optimizer("does.not.exist", BaseOptimizerConfig())

    def test_mipro_v2_is_registered(self):
        assert OPTIMIZERS.get("mipro_v2") is MIPROv2Optimizer


# ---------------------------------------------------------------------------
# MIPROv2 stub (optional extra)
# ---------------------------------------------------------------------------


class TestMIPROv2Stub:
    def test_missing_dspy_raises_actionable_import_error(self):
        if mipro_adapter.dspy is not None:
            pytest.skip("dspy is installed; the missing-extra path is unreachable")
        with pytest.raises(ImportError, match="uv sync --extra optimization"):
            MIPROv2Optimizer(BaseOptimizerConfig())

    def test_optimize_is_a_v2_stub(self):
        if mipro_adapter.dspy is None:
            pytest.skip("dspy not installed; construction raises before optimize")
        with pytest.raises(NotImplementedError, match="v2"):
            MIPROv2Optimizer(BaseOptimizerConfig()).optimize("p", None, None)


# ---------------------------------------------------------------------------
# FidelityObjective
# ---------------------------------------------------------------------------


class FillAcceptedStep(PipelineStep):
    """Fills state.accepted and gate metrics; local-only, deliberately unregistered."""

    name = "test.fill_accepted"

    def __init__(self, records: list[AnnotationRecord], pass_rate: float = 1.0) -> None:
        self.records = records
        self.pass_rate = pass_rate

    def run(self, state: PipelineState) -> PipelineState:
        state.accepted.extend(self.records)
        state.metrics["gate_pass_rate"] = self.pass_rate
        state.metrics["gate_reject_reasons"] = {}
        return state

    def to_config(self):
        return {"pass_rate": self.pass_rate}

    @classmethod
    def from_config(cls, config, resources: Resources) -> "FillAcceptedStep":
        return cls([], pass_rate=float(config.get("pass_rate", 1.0)))


def _make_record(i: int, embedding_ref: str | None) -> AnnotationRecord:
    stratum = Stratum(dims={"query_type": "factoid"})
    seed = Seed(seed_id=f"s{i}", chunk_ids=("c",), cluster_id=0, stratum=stratum)
    query = SyntheticQuery(
        query_id=f"q{i}", text=f"synthetic query {i}", seed=seed, embedding_ref=embedding_ref
    )
    return AnnotationRecord(
        record_id=f"r{i}",
        query=query,
        qrels={"c": 1},
        stratum=stratum,
        benchmark_version="test",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _build_objective(pass_rate: float = 1.0, n_records: int = 6) -> FidelityObjective:
    """FidelityObjective over a one-step fake pipeline (d=8, ~40 reference rows).

    The reference cloud tiles 6 rng-drawn unit vectors to 40 rows and the
    synthetic vectors are its first 6 rows (a subset): only 6 synthetic
    samples exist, so KL ~ 0 requires the reference top-1-cosine support to
    fit in <= 6 of the 50 histogram bins -- a plain 40-point cloud cannot be
    covered by 6 samples (eps-smoothed empty bins push KL to ~9).
    """
    d = 8
    rng = np.random.default_rng(0)
    base = _unit(rng.normal(size=(6, d)))
    reference = np.tile(base, (7, 1))[:40]
    chunk_embs = _unit(rng.normal(size=(10, d)))

    embeddings = EmbeddingStore()
    records = []
    if n_records:
        refs = [f"emb{i}" for i in range(n_records)]
        embeddings.add(refs, reference[:n_records])
        records = [_make_record(i, ref) for i, ref in enumerate(refs)]
        # A ref-less record must be skipped, not crash the collection.
        records.append(_make_record(99, None))

    def pipeline_factory(prompt: str) -> Pipeline:
        return Pipeline([FillAcceptedStep(records, pass_rate=pass_rate)])

    return FidelityObjective(
        pipeline_factory=pipeline_factory,
        reference_embs=reference,
        chunk_embs=chunk_embs,
        embeddings=embeddings,
        alpha=1.0,
        min_pass_rate=0.5,
        seed=0,
    )


class TestFidelityObjective:
    def test_matching_synth_cloud_scores_near_zero(self):
        score = _build_objective()("candidate prompt")
        assert np.isfinite(score)
        assert score > -1.0
        # synth == subset of reference => KL ~ 0, C2ST clamp = 0
        assert score > -0.5

    def test_deterministic_under_seed(self):
        assert _build_objective()("p") == _build_objective()("p")

    def test_low_gate_pass_rate_hits_hard_penalty(self):
        assert _build_objective(pass_rate=0.1)("p") == HARD_PENALTY

    def test_no_accepted_records_hits_hard_penalty(self):
        assert _build_objective(n_records=0)("p") == HARD_PENALTY
