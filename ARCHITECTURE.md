# ragsynth architecture (one page)

## State flow

```
                     Pipeline.from_yaml(config)          <- ONLY deserialization entrypoint
                              |
              build_resources(config)  = composition root (pipeline/serialization.py)
              dataset -> embed -> ReferencePartition + movMF DemandArtifact (train split only,
              persisted w/ sha256 manifest) -> zoo -> adapters ==> frozen Resources
                              |
   PipelineState  ------------+------------------------------------------------------------
   .seeds         <- seed_sampler.{uniform|quota|spec}   what to generate from (A0/A1/A2)
   .contexts      <- context_assembler                   chunk texts + kNN style exemplars
   .candidates    <- generator                           answer-first jinja2 prompt, xN,
                                                         A2 target-check (cos >= tau_t) + 1 revision
   .gate_accepted <- gate                                dedup -> zero_context -> answerability
   .rejected      |                                      -> round_trip -> uniqueness (cheap->expensive,
   .metrics[gate_reject_reasons]                         short-circuit; promotions ride gen_meta)
   .accepted      <- qrel_builder                        AnnotationRecords (binary | relabel_nearest)
                  <- curator                             dedup, stratified mix, memorization audit
   .metrics[eval_report] <- validator                    runs comparison arms (arms/ presets),
                                                         fidelity/efficiency/validity/diversity per arm,
                                                         writes metrics.json + report.md + figures
```

Rules that keep this sane: `Resources` is frozen and injected once (steps
never construct adapters); all randomness derives from the config seed via
`Resources.rng(name)` substreams; embeddings live only in the
`EmbeddingStore` (domain objects hold `embedding_ref`); `metrics.json`
contains no wall-clock, so identical seeds give identical bytes.

## Registries

Every extensible concept is a `Registry` (`pipeline/registry.py`): `STEPS`,
`CHECKS`, `CHAT_MODELS`, `EMBEDDERS`, `RETRIEVERS`, `JUDGES`, `DATASETS`,
`ARMS`, `QREL_STRATEGIES`, `OPTIMIZERS`. Configs reference registry keys;
unknown keys raise errors listing the known ones. The contract test
(`tests/pipeline/test_contract.py`) parametrizes over the registries, so a
new entry is automatically held to the same lifecycle rules.

## Extension recipe: a new GateCheck in ~20 lines

```python
# src/ragsynth/gate/checks/max_length.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragsynth.gate.checks.base import CHECKS, CheckResult, GateCheck

if TYPE_CHECKING:
    from ragsynth.domain import SyntheticQuery
    from ragsynth.pipeline.base import PipelineState, Resources


@CHECKS.register("max_length")
class MaxLengthCheck(GateCheck):
    """Reject candidates longer than real queries ever get."""

    name = "max_length"

    def __init__(self, max_words: int = 40) -> None:
        self.max_words = max_words

    def check(
        self, candidate: SyntheticQuery, state: PipelineState, resources: Resources
    ) -> CheckResult:
        n = len(candidate.text.split())
        if n > self.max_words:
            return CheckResult(False, float(n), f"{n} words > {self.max_words}")
        return CheckResult(True, float(n), "length ok")

    def to_config(self) -> dict[str, Any]:
        return {"max_words": self.max_words}
```

Wire-up: import it from `gate/checks/__init__.py`, then add `max_length`
to the gate's `checks:` list in your config (optionally with a
`max_length: {max_words: 30}` params block). Rejections show up in
`state.metrics["gate_reject_reasons"]` and the report automatically.

## Where things live

`domain/` frozen pydantic objects (conversation- and visual-ready fields) ·
`sampling/` vMF/movMF/demand/partition/spec-sampler (ports of
`reference/synth_query_eval.py`) · `metrics/` fidelity, efficiency,
diversity, validity (τ_AP, RBO, bootstrap agreement, positive controls,
system zoo) · `steps/` the seven pipeline steps + jinja2 prompt templates ·
`arms/` thin A0/A1/A2/ORACLE presets · `adapters/` Protocol + offline-first
concretes · `datasets/` toy world, jsonl loader, bundled sample corpus ·
`optimization/` v1 contract (NoOp + FidelityObjective; MIPROv2 stub) ·
`io/` EmbeddingStore + sha256-manifested ArtifactStore · `cli.py` typer.
