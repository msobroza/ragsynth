"""Tests for the query generator step (SPEC §6.3)."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ragsynth.domain import Seed, Stratum
from ragsynth.pipeline.base import PipelineState, Resources
from ragsynth.steps.context_assembler import ContextAssembler
from ragsynth.steps.generator import QueryGenerator
from ragsynth.steps.seed_sampler import QuotaSeedSampler


class _SpyChat:
    """Records prompts; returns distinct deterministic text per call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        self.calls.append((system, user))
        return f"spy query {len(self.calls)}?"


class _TargetEmbedder:
    """Always returns the target direction (=> cos_to_target == 1)."""

    def __init__(self, direction: NDArray[np.float64]) -> None:
        self.direction = direction

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        return np.tile(self.direction, (len(texts), 1))


def _contexts(resources: Resources, n_seeds: int = 4) -> PipelineState:
    state = QuotaSeedSampler(resources, n_seeds=n_seeds, n_min=2, p_group=0.0).run(PipelineState())
    assembler = ContextAssembler(resources, k_style=2)
    assembler.fit(resources)
    return assembler.run(state)


def test_n_candidates_per_seed_with_meta(steps_world: Resources) -> None:
    state = _contexts(steps_world)
    step = QueryGenerator(steps_world, n_candidates=3)
    state = step.run(state)
    assert len(state.candidates) == 3 * len(state.seeds)
    for cand in state.candidates:
        assert cand.gen_meta["prompt_version"] == "answer_first_v1"
        assert cand.gen_meta["model"] == "MockChatModel"
        assert cand.gen_meta["candidate_index"] in (0, 1, 2)
        assert cand.embedding_ref is not None
        assert cand.embedding_ref in steps_world.embeddings


def test_prompts_vary_by_candidate_index(steps_world: Resources) -> None:
    spy = _SpyChat()
    resources = steps_world.with_overrides(generator_llm=spy)
    state = _contexts(resources, n_seeds=4)
    QueryGenerator(resources, n_candidates=2).run(state)
    # Same seed's two prompts differ (variant marker), so mock LLMs produce
    # distinct texts and dedup stays meaningful.
    per_seed = spy.calls[:2]
    assert per_seed[0][1] != per_seed[1][1]
    assert "Variant 1 of 2" in per_seed[0][1]
    assert "Variant 2 of 2" in per_seed[1][1]


def test_prompt_renders_evidence_exemplars_and_rules(steps_world: Resources) -> None:
    spy = _SpyChat()
    resources = steps_world.with_overrides(generator_llm=spy)
    state = _contexts(resources, n_seeds=4)
    QueryGenerator(resources, n_candidates=1).run(state)
    _system, user = spy.calls[0]
    ctx = state.contexts[0]
    for chunk_text in ctx.chunk_texts:
        assert chunk_text in user
    for exemplar in ctx.style_exemplars:
        assert f"- {exemplar}" in user
    assert ctx.instruction in user
    assert "according to the document" in user  # the forbidden-phrase rule is stated
    assert "answer" in user.lower()  # answer-first scaffold


def test_revision_triggered_when_below_tau_t(steps_world: Resources) -> None:
    # Random mock embeddings are ~orthogonal to z => cos < tau_t => one revision.
    z = np.zeros(16)
    z[0] = 1.0
    seed = Seed(
        seed_id="s-a2",
        chunk_ids=(steps_world.chunks[0].chunk_id,),
        cluster_id=0,
        stratum=Stratum(dims={"query_type": "factoid"}),
        z=z.tolist(),
    )
    spy = _SpyChat()
    resources = steps_world.with_overrides(generator_llm=spy)
    state = PipelineState(seeds=[seed])
    assembler = ContextAssembler(resources, k_style=1)
    assembler.fit(resources)
    state = assembler.run(state)
    state = QueryGenerator(resources, n_candidates=1, tau_t=0.6, max_revisions=1).run(state)
    cand = state.candidates[0]
    assert cand.gen_meta["revisions"] == 1
    assert cand.gen_meta["cos_to_target"] < 0.6
    assert len(spy.calls) == 2
    assert "REVISE_REQUEST" in spy.calls[1][1]
    assert "spy query 1?" in spy.calls[1][1]  # previous query shown to the reviser


def test_no_revision_when_target_met(steps_world: Resources) -> None:
    z = np.zeros(16)
    z[0] = 1.0
    seed = Seed(
        seed_id="s-a2-good",
        chunk_ids=(steps_world.chunks[0].chunk_id,),
        cluster_id=0,
        stratum=Stratum(dims={"query_type": "factoid"}),
        z=z.tolist(),
    )
    resources = steps_world.with_overrides(embedder=_TargetEmbedder(z))
    state = PipelineState(seeds=[seed])
    assembler = ContextAssembler(resources, k_style=0)
    assembler.fit(resources)
    state = assembler.run(state)
    state = QueryGenerator(resources, n_candidates=1, tau_t=0.6).run(state)
    cand = state.candidates[0]
    assert cand.gen_meta["revisions"] == 0
    assert cand.gen_meta["cos_to_target"] == pytest.approx(1.0)


def test_no_target_check_for_chunk_first_seeds(steps_world: Resources) -> None:
    state = _contexts(steps_world, n_seeds=4)
    state = QueryGenerator(steps_world, n_candidates=1).run(state)
    for cand in state.candidates:
        assert cand.gen_meta["revisions"] == 0
        assert cand.gen_meta["cos_to_target"] is None


def test_config_round_trip(steps_world: Resources) -> None:
    step = QueryGenerator(steps_world, n_candidates=2, tau_t=0.5, max_revisions=2)
    config = step.to_config()
    assert config == {
        "n_candidates": 2,
        "prompt_version": "answer_first_v1",
        "tau_t": 0.5,
        "max_revisions": 2,
    }
    assert QueryGenerator.from_config(config, steps_world).to_config() == config
