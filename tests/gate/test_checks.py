"""Tests for the five verification gate checks (SPEC §6.4).

Each check gets a pass fixture and a reject fixture with a reason string.
The min_resources world: 4 chunks embedded with MockEmbedder(dim=16),
DenseInMemoryRetriever over them, MockJudge (answerable with evidence,
not answerable without).
"""

import numpy as np
import pytest

from ragsynth.adapters.judge.mock import MockJudge
from ragsynth.domain import Seed, Stratum, SyntheticQuery
from ragsynth.gate.checks.answerability import AnswerabilityCheck
from ragsynth.gate.checks.base import CHECKS
from ragsynth.gate.checks.dedup import DedupCheck
from ragsynth.gate.checks.round_trip import RoundTripCheck
from ragsynth.gate.checks.uniqueness import UniquenessCheck
from ragsynth.gate.checks.zero_context import ZeroContextCheck
from ragsynth.pipeline.base import PipelineState, Resources


def _candidate(
    resources: Resources, text: str, emb_like_chunk: int, gold_chunk: int
) -> SyntheticQuery:
    """Candidate whose embedding equals a chunk's vector (hand-steerable geometry)."""
    qid = f"cand-{abs(hash(text)) % 10**8}"
    if qid not in resources.embeddings:
        resources.embeddings.add(
            [qid], resources.embeddings.get([resources.chunks[emb_like_chunk].chunk_id])
        )
    seed = Seed(
        seed_id=f"seed-{qid}",
        chunk_ids=(resources.chunks[gold_chunk].chunk_id,),
        cluster_id=0,
        stratum=Stratum(dims={"query_type": "factoid"}),
    )
    return SyntheticQuery(query_id=qid, text=text, seed=seed, embedding_ref=qid, gen_meta={})


class TestDedup:
    def test_passes_when_novel(self, min_resources: Resources) -> None:
        state = PipelineState()
        cand = _candidate(min_resources, "novel question?", 0, 0)
        result = DedupCheck(cos_threshold=0.95).check(cand, state, min_resources)
        assert result.passed

    def test_rejects_exact_text_duplicate(self, min_resources: Resources) -> None:
        state = PipelineState()
        first = _candidate(min_resources, "same question?", 0, 0)
        state.gate_accepted.append(first)
        dupe = _candidate(min_resources, "same question?", 1, 1)
        result = DedupCheck(cos_threshold=0.95).check(dupe, state, min_resources)
        assert not result.passed
        assert "exact" in result.reason

    def test_rejects_semantic_near_duplicate(self, min_resources: Resources) -> None:
        state = PipelineState()
        state.gate_accepted.append(_candidate(min_resources, "first phrasing?", 0, 0))
        near = _candidate(min_resources, "second phrasing?", 0, 0)  # same embedding
        result = DedupCheck(cos_threshold=0.95).check(near, state, min_resources)
        assert not result.passed
        assert "cosine" in result.reason
        assert result.score == pytest.approx(1.0)

    def test_below_threshold_passes(self, min_resources: Resources) -> None:
        state = PipelineState()
        state.gate_accepted.append(_candidate(min_resources, "about topic zero?", 0, 0))
        other = _candidate(min_resources, "about topic one?", 1, 1)
        result = DedupCheck(cos_threshold=0.999).check(other, state, min_resources)
        assert result.passed


class TestZeroContext:
    def test_passes_when_not_common_knowledge(self, min_resources: Resources) -> None:
        cand = _candidate(min_resources, "what is the fee?", 0, 0)
        result = ZeroContextCheck().check(cand, PipelineState(), min_resources)
        assert result.passed

    def test_rejects_common_knowledge(self, min_resources: Resources) -> None:
        resources = min_resources.with_overrides(judge=MockJudge(answerable_without_evidence=True))
        cand = _candidate(resources, "what is water?", 0, 0)
        result = ZeroContextCheck().check(cand, PipelineState(), resources)
        assert not result.passed
        assert "without evidence" in result.reason


class TestAnswerability:
    def test_passes_when_answerable_from_evidence(self, min_resources: Resources) -> None:
        cand = _candidate(min_resources, "what does chunk 0 say?", 0, 0)
        result = AnswerabilityCheck().check(cand, PipelineState(), min_resources)
        assert result.passed

    def test_rejects_unanswerable(self, min_resources: Resources) -> None:
        resources = min_resources.with_overrides(judge=MockJudge(answerable_with_evidence=False))
        cand = _candidate(resources, "unanswerable?", 0, 0)
        result = AnswerabilityCheck().check(cand, PipelineState(), resources)
        assert not result.passed
        assert "not answerable" in result.reason


class TestRoundTrip:
    def test_gold_in_topk_passes_with_reciprocal_rank(self, min_resources: Resources) -> None:
        cand = _candidate(min_resources, "hits its own gold?", 0, 0)
        result = RoundTripCheck(k=2).check(cand, PipelineState(), min_resources)
        assert result.passed
        assert result.score == pytest.approx(1.0)  # gold at rank 1

    def test_gold_outside_topk_rejects(self, min_resources: Resources) -> None:
        # Embedding = chunk 0's vector, but gold is the most dissimilar chunk.
        sims = min_resources.chunk_embs() @ min_resources.chunk_embs()[0]
        farthest = int(np.argmin(sims))
        cand = _candidate(min_resources, "misses its gold?", 0, farthest)
        result = RoundTripCheck(k=1).check(cand, PipelineState(), min_resources)
        assert not result.passed
        assert "top-1" in result.reason


class TestUniqueness:
    def test_promote_mode_promotes_leaky_chunks(self, min_resources: Resources) -> None:
        cand = _candidate(min_resources, "leaky in promote mode?", 0, 0)
        result = UniquenessCheck(mode="promote", top_m=2).check(
            cand, PipelineState(), min_resources
        )
        assert result.passed
        assert len(result.promoted) >= 1
        assert min_resources.chunks[0].chunk_id not in result.promoted

    def test_reject_mode_rejects_leaky_gold(self, min_resources: Resources) -> None:
        cand = _candidate(min_resources, "leaky in reject mode?", 0, 0)
        result = UniquenessCheck(mode="reject", top_m=2).check(cand, PipelineState(), min_resources)
        assert not result.passed
        assert "leaky" in result.reason

    def test_no_leak_passes_clean(self, min_resources: Resources) -> None:
        resources = min_resources.with_overrides(judge=MockJudge(answerable_with_evidence=False))
        cand = _candidate(resources, "clean uniqueness?", 0, 0)
        result = UniquenessCheck(mode="promote", top_m=2).check(cand, PipelineState(), resources)
        assert result.passed
        assert result.promoted == ()

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            UniquenessCheck(mode="banana")


class TestRegistryAndConfig:
    @pytest.mark.parametrize(
        "key", ["dedup", "zero_context", "answerability", "round_trip", "uniqueness"]
    )
    def test_all_checks_registered(self, key: str) -> None:
        registered = CHECKS.keys()
        assert key in registered

    def test_config_round_trip(self, min_resources: Resources) -> None:
        for key in CHECKS.keys():  # noqa: SIM118 - Registry method, not dict.keys
            check_class = CHECKS.get(key)
            check = check_class.from_config({}, min_resources)
            rebuilt = check_class.from_config(check.to_config(), min_resources)
            assert rebuilt.to_config() == check.to_config()
