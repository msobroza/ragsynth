"""Tests for the context assembler (SPEC §6.2)."""

import numpy as np
import pytest

from ragsynth.pipeline.base import PipelineState, Resources
from ragsynth.steps.context_assembler import ContextAssembler
from ragsynth.steps.seed_sampler import QuotaSeedSampler


def _seeded_state(resources: Resources) -> PipelineState:
    return QuotaSeedSampler(resources, n_seeds=8, n_min=3, p_group=0.0).run(PipelineState())


def test_contexts_align_with_seeds(steps_world: Resources) -> None:
    state = _seeded_state(steps_world)
    step = ContextAssembler(steps_world, k_style=3)
    step.fit(steps_world)
    state = step.run(state)
    assert len(state.contexts) == len(state.seeds)
    for ctx, seed in zip(state.contexts, state.seeds, strict=True):
        assert ctx.seed == seed
        assert ctx.chunk_texts == tuple(steps_world.chunk_index[cid].text for cid in seed.chunk_ids)
        assert len(ctx.style_exemplars) == 3


def test_exemplars_prefer_same_cluster(steps_world: Resources) -> None:
    state = _seeded_state(steps_world)
    step = ContextAssembler(steps_world, k_style=3)
    step.fit(steps_world)
    state = step.run(state)
    train_by_text = {q.text: q.query_id for q in steps_world.queries_train}
    labels = steps_world.partition.assign(steps_world.query_embs("train"))
    label_by_id = {q.query_id: int(labels[i]) for i, q in enumerate(steps_world.queries_train)}
    for ctx in state.contexts:
        for exemplar in ctx.style_exemplars:
            assert label_by_id[train_by_text[exemplar]] == ctx.seed.cluster_id


def test_instruction_reflects_stratum(steps_world: Resources) -> None:
    state = _seeded_state(steps_world)
    step = ContextAssembler(steps_world, k_style=0)
    step.fit(steps_world)
    state = step.run(state)
    for ctx in state.contexts:
        query_type = ctx.seed.stratum.dims["query_type"]
        marker = {"factoid": "factual", "howto": "how", "keyword": "keyword"}[query_type]
        assert marker in ctx.instruction.lower()
        assert ctx.style_exemplars == ()


def test_stubbed_flags_fail_fast(steps_world: Resources) -> None:
    with pytest.raises(NotImplementedError, match="two_step"):
        ContextAssembler(steps_world, two_step=True)
    with pytest.raises(NotImplementedError, match="blind_summary"):
        ContextAssembler(steps_world, blind_summary=True)


def test_config_round_trip(steps_world: Resources) -> None:
    step = ContextAssembler(steps_world, k_style=5)
    assert step.to_config() == {"k_style": 5, "two_step": False, "blind_summary": False}
    rebuilt = ContextAssembler.from_config(step.to_config(), steps_world)
    assert rebuilt.to_config() == step.to_config()


def test_exemplars_use_z_when_present(steps_world: Resources) -> None:
    # A2-style seed: z exactly on cluster-1's direction => exemplars from cluster 1.
    from ragsynth.domain import Seed, Stratum

    z = np.zeros(16)
    z[1] = 1.0
    seed = Seed(
        seed_id="s-z",
        chunk_ids=(steps_world.chunks[-1].chunk_id,),
        cluster_id=int(steps_world.partition.assign(z[None, :])[0]),
        stratum=Stratum(dims={"query_type": "factoid"}),
        z=z.tolist(),
    )
    state = PipelineState(seeds=[seed])
    step = ContextAssembler(steps_world, k_style=2)
    step.fit(steps_world)
    state = step.run(state)
    exemplars = state.contexts[0].style_exemplars
    assert len(exemplars) == 2
    assert all("c1" in text for text in exemplars)  # cluster-1 train queries
