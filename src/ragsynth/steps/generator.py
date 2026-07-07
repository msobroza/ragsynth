"""Query generator: overgenerate candidates with an answer-first prompt (SPEC §6.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np
from jinja2 import Environment, PackageLoader, StrictUndefined

from ragsynth.domain import SyntheticQuery
from ragsynth.pipeline.base import STEPS, PipelineStep

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ragsynth.domain import GenerationContext
    from ragsynth.pipeline.base import PipelineState, Resources

SYSTEM_PROMPT = (
    "You write realistic search queries for a retrieval evaluation benchmark. "
    "Output only the query text."
)

_ENV = Environment(
    loader=PackageLoader("ragsynth.steps", "prompts"),
    autoescape=False,  # noqa: S701 - plain-text prompt templates, not HTML
    undefined=StrictUndefined,
    keep_trailing_newline=False,
)


@STEPS.register("generator")
class QueryGenerator(PipelineStep):
    """Overgenerate ``n_candidates`` per seed; A2 seeds get a target check.

    Answer-first prompting: extract a claim, then write the question a user
    with that need would type (Alberti et al., ACL 2019; Promptagator, Dai
    et al., ICLR 2023; InPars, Bonifacio et al., SIGIR 2022). When the seed
    carries a target ``z`` and ``cos(emb(q), z) < tau_t``, one revision pass
    is requested (vec2text lineage; SPEC R1). Prompt templates are versioned
    jinja2 files under ``steps/prompts/``; ``prompt_version`` is recorded in
    ``gen_meta``.
    """

    name = "generator"

    def __init__(
        self,
        resources: Resources,
        n_candidates: int = 3,
        prompt_version: str = "answer_first_v1",
        tau_t: float = 0.6,
        max_revisions: int = 1,
    ) -> None:
        self._resources = resources
        self.n_candidates = n_candidates
        self.prompt_version = prompt_version
        self.tau_t = tau_t
        self.max_revisions = max_revisions
        self._template = _ENV.get_template(f"{prompt_version}.j2")
        self._revise_template = _ENV.get_template("revise_v1.j2")

    def _embed(self, ref: str, text: str) -> NDArray[np.float64]:
        emb = self._resources.embedder.encode([text]).astype(np.float64)
        self._resources.embeddings.add([ref], emb)
        return np.asarray(emb[0], dtype=np.float64)

    def _generate_one(self, context: GenerationContext, candidate_index: int) -> SyntheticQuery:
        resources = self._resources
        seed = context.seed
        base_kwargs = {
            "chunk_texts": context.chunk_texts,
            "style_exemplars": context.style_exemplars,
            "instruction": context.instruction,
            "candidate_index": candidate_index,
            "n_candidates": self.n_candidates,
        }
        user = self._template.render(**base_kwargs)
        text = resources.generator_llm.complete(SYSTEM_PROMPT, user)
        query_id = f"synq-{seed.seed_id}-{candidate_index}"
        ref = f"{query_id}#a0"
        emb = self._embed(ref, text)

        z = np.asarray(seed.z, dtype=np.float64) if seed.z is not None else None
        cos: float | None = float(emb @ z) if z is not None else None
        revisions = 0
        while (
            z is not None
            and cos is not None
            and cos < self.tau_t
            and revisions < self.max_revisions
        ):
            revisions += 1
            revise_user = self._revise_template.render(
                previous_query=text, cos_to_target=cos, tau_t=self.tau_t, **base_kwargs
            )
            text = resources.generator_llm.complete(SYSTEM_PROMPT, revise_user)
            ref = f"{query_id}#a{revisions}"
            emb = self._embed(ref, text)
            cos = float(emb @ z)

        return SyntheticQuery(
            query_id=query_id,
            text=text,
            seed=seed,
            embedding_ref=ref,
            gen_meta={
                "model": type(resources.generator_llm).__name__,
                "prompt_version": self.prompt_version,
                "candidate_index": candidate_index,
                "cos_to_target": cos,
                "revisions": revisions,
            },
        )

    def run(self, state: PipelineState) -> PipelineState:
        """Generate candidates for every assembled context."""
        for context in state.contexts:
            for i in range(self.n_candidates):
                state.candidates.append(self._generate_one(context, i))
        return state

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {
            "n_candidates": self.n_candidates,
            "prompt_version": self.prompt_version,
            "tau_t": self.tau_t,
            "max_revisions": self.max_revisions,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> Self:
        """Build from a config params block."""
        return cls(resources, **config)
