# SPEC-V2-OPT — Prompt-optimizer execution: DSPy MIPROv2 & GEPA behind adapter design patterns

> **Audience:** Claude Fable 5 coding agent (with superpowers skills).
> **Parent:** SPEC.md §6.3, §6.4, §11, §16 (R2). v1 shipped the contract only: `BasePromptOptimizer` ABC, frozen `TrialRecord`/`OptimizationResult`, `OPTIMIZERS` registry, `NoOpOptimizer`, `FidelityObjective`, `mipro_v2` stub.
> **Status:** v2 execution spec. Expand with a PLAN.md phase before coding; new decisions land as D52–D61.

---

## 0. How to execute this spec (read first)

1. **Plan before coding.** Add a v2-opt phase to PLAN.md; record every §15 default you take.
2. **TDD.** Known-value fixtures FIRST (§12; SPEC 15.4 style): feedback-string rendering, template validation, RecordingMetric bookkeeping, NoOp invariant — before any backend code.
3. **Do not modify v1 behavior.** `NoOpOptimizer`, `FidelityObjective.__call__`, and all v1 tests stay green untouched. `TrialRecord`/`OptimizationResult` may only gain fields with backward-compatible defaults.
4. **§15.3 commands green at every commit**, plus the extras job (§12.3). Full core suite stays air-gapped.
5. **Success = §13 acceptance criteria.** Nothing else is the bar.

---

## 1. Context & trigger (R2)

The v1 gate tallies reject reasons into `state.metrics["gate_reject_reasons"]` (§6.4) precisely so a v2 optimizer can *route* what to fix; `FidelityObjective` already scores a candidate prompt by running a mini pipeline (`-(KL + α·max(0, C2ST_AUC − 0.5))`, hard-penalized below `min_pass_rate`). R2's trigger: gate pass-rate and τ plateau under manual prompt iteration. This spec turns the stub into two executing backends — MIPROv2 (Opsahl-Ong et al., EMNLP 2024) for instruction + few-shot demo search, GEPA (Agrawal et al., arXiv 2025) for reflective genetic-Pareto evolution consuming the reject-reason tallies as textual feedback — behind a strict adapter boundary so backend types never touch the pipeline.

---

## 2. Goals & non-goals

### 2.1 Goals

1. **Adapter pattern:** one module per backend translating `(current_prompt, samples, metric)` into the backend-native program and its output back into `OptimizationResult`. Registry keys: `noop` | `mipro_v2` | `gepa` | `textgrad` (stub).
2. **MIPROv2 execution** over `FidelityObjective` with an explicit budget config; full trial history in `TrialRecord`s.
3. **GEPA execution** with reflective feedback built from gate reject-reason tallies (the R2 closed loop).
4. **Exemplar selection as an optimizable component** (Promptagator, Dai et al., ICLR 2023; InPars, Bonifacio et al., SIGIR 2022) — with an explicit anti-gaming guard around the round-trip gate.
5. **`ragsynth optimize` CLI** subcommand; schema_version 2 config block; v1 configs load unchanged.

### 2.2 Non-goals (stub or omit)

- TextGrad execution → registered stub only (`optimize` raises `NotImplementedError`), mirroring the v1 `mipro_v2` stub pattern.
- Optimizing gate-check prompts, judge prompts, or multi-step programs — v2 optimizes the **generator** prompt only.
- Online/continuous optimization, mlflow tracking, human-in-the-loop review UI.
- Any change to metrics/, sampling/, or gate semantics.

---

## 3. Key decisions (do not relitigate)

- **Backends never leak.** `dspy`/`gepa` types exist only inside their adapter module; everything crossing the boundary is `str`/`float`/`dict`/frozen dataclass. No `dspy` import outside `optimization/` and no import at module scope without the v1 `try/except ImportError` + `require_optional` pattern.
- **GEPA via `dspy.GEPA`.** The standalone `gepa` package (github.com/gepa-ai/gepa) is MIT — verified 2026-07 — and is what `dspy.GEPA` wraps. We spec the dspy route as primary anyway: one bridge (§5.3), one LM configuration path, one extra; the `optimization` extra becomes `dspy>=3.0` (bundles GEPA support) + `textgrad`. All three licenses (dspy MIT, gepa MIT, textgrad MIT) documented in README's license table.
- **All backend LLM traffic through our adapters.** A `dspy.BaseLM` bridge wraps the `ChatModel` Protocol; `MockChatModel` therefore drives dspy fully offline. No backend ever receives an API key or URL directly.
- **Candidate = full jinja2 template text**, validated before scoring (§6). Optimized templates are run artifacts, never package files.
- **Anti-gaming split discipline** (§9): objective reference = *train* split; reporting = *anchor* split + disjoint seed substream. The round-trip gate is the Promptagator consistency filter — optimization may see its tallies, never its reporting split.
- **Improvement ≥ 0 by construction:** every adapter scores the baseline as trial 1 and returns `argmax` over trials ∪ {baseline}.
- **Cross-family reflection rule:** config validation WARNS if GEPA's reflection model shares the generator's model family (extends the §6.4 judge rule; Fröbe et al., SIGIR 2025).

---

## 4. Module layout (`src/ragsynth/optimization/`)

```
optimization/
├── base.py            # v1 contract (unchanged surface; TrialRecord gains `meta`, §5.1)
├── noop.py            # v1, untouched
├── objectives.py      # v1 FidelityObjective + new FidelityFeedbackObjective (§8.2)
├── recording.py       # RecordingMetric wrapper — backend-agnostic trial capture (§5.2)
├── candidates.py      # PromptCandidate validation + template materialization (§6)
├── feedback.py        # reject-reason → instruction-pressure routing (§8.1)
├── dspy_bridge.py     # AdapterLM(dspy.BaseLM) over ChatModel (§5.3)  [extra]
├── mipro_adapter.py   # v1 stub becomes executing adapter (§7)        [extra]
├── gepa_adapter.py    # dspy.GEPA adapter (§8)                        [extra]
├── textgrad_adapter.py# stub, registers "textgrad"                    [extra]
└── factory.py         # mini_pipeline_factory + demo-pool assembly (§9.1)
```

`recording.py`, `candidates.py`, `feedback.py`, `factory.py` are core (no extras) and fully unit-tested offline.

---

## 5. Adapter contract

### 5.1 Boundary types

`TrialRecord` gains one backward-compatible field (frozen dataclass, `field(default_factory=dict)`):

```python
@dataclass(frozen=True)
class TrialRecord:
    trial_id: int
    prompt: str            # full candidate template text
    score: float | None
    timestamp: str
    meta: dict[str, Any] = field(default_factory=dict)
    # meta keys (JSON-safe only): "demo_ids", "feedback", "minibatch_seed_ids", "phase"
```

Each adapter module: (a) `try: import dspy except ImportError: dspy = None` + `require_optional` in `__init__`; (b) translate inputs into the backend-native program; (c) run the search; (d) translate the winner back into `OptimizationResult` with `optimizer_name` = its registry key and `config=self.config.model_dump()`. `OptimizationResult` stays exactly the v1 shape.

### 5.2 Trial capture — `RecordingMetric`

Backend logs are not our source of truth. Adapters wrap the metric before handing it to the backend:

```python
class RecordingMetric:
    """Wraps an OptimizationMetric; appends one TrialRecord per call (1-based ids)."""
    def __init__(self, inner: OptimizationMetric) -> None: ...
    def __call__(self, prompt: str) -> float: ...
    @property
    def trials(self) -> list[TrialRecord]: ...
```

`num_trials == len(trials) == number of objective calls`, regardless of what the backend reports.

### 5.3 dspy bridge

```python
class AdapterLM(dspy.BaseLM):
    """Routes every dspy completion through a ragsynth ChatModel (SPEC §12)."""
    def __init__(self, chat: ChatModel, model_label: str) -> None: ...
```

Both executing adapters build their program over `AdapterLM(resources.generator_llm, ...)`; GEPA's reflection LM is `AdapterLM(resources.reflection_llm, ...)` (new resources key, defaults to the judge model — §15).

---

## 6. Candidate representation & prompt plumbing

- A candidate is the **full jinja2 template text** for the generator's user prompt (same contract as `steps/prompts/answer_first_v1.j2`). Adapters compose it from `(instruction, demo_block, fixed variable scaffold)`.
- **Validation before scoring** (`candidates.py`): parse with `jinja2.Environment().parse`; `meta.find_undeclared_variables` must be a subset of `{chunk_texts, style_exemplars, instruction, candidate_index, n_candidates}`. Invalid template ⇒ `HARD_PENALTY` recorded as a trial, pipeline never runs, no crash.
- **Materialization:** valid candidates are written to `<artifacts_dir>/optimization/prompts/answer_first_v1__opt_<sha8>.j2`. `QueryGenerator` gains a jinja2 `ChoiceLoader([FileSystemLoader(resources.prompt_dir), PackageLoader(...)])` where `resources.prompt_dir` is optional; the mini-pipeline factory points `prompt_version` at the materialized file. `prompt_version` in `gen_meta` therefore records optimized-prompt provenance for free — package templates under `steps/prompts/` are never written.

---

## 7. MIPROv2 adapter (`mipro_v2`)

Instruction + few-shot demo optimization (Opsahl-Ong et al., EMNLP 2024) via `dspy.MIPROv2` over a single-predictor program whose signature mirrors the generator contract (evidence + exemplars → query).

```python
class MIPROv2Config(BaseOptimizerConfig):
    num_trials: int = 20              # Bayesian-search trials (supersedes max_trials here)
    minibatch_size: int = 8           # seeds per objective call during search
    n_seeds_per_trial: int = 24       # mini-pipeline size for full evaluation of finalists
    num_instruction_candidates: int = 8
    max_bootstrapped_demos: int = 3   # Promptagator-style self-generated demos (§9)
    max_labeled_demos: int = 4        # real (chunk, query) pairs from the train split
```

- The objective is `FidelityObjective` built by `factory.mini_pipeline_factory` (§9.1) at `minibatch_size` seeds; finalists re-scored at `n_seeds_per_trial`. `phase` ("search" | "full_eval") goes in `TrialRecord.meta`.
- Cost note (document in docstring): one objective call ≈ `minibatch_size × n_candidates` generator calls + ≤ 2 judge calls per candidate; total ≈ `num_trials × minibatch` + demo bootstrap.
- Winner = argmax over full-eval scores ∪ {baseline full-eval score}; baseline is always trial 1.

---

## 8. GEPA adapter (`gepa`) — the closed loop

Reflective prompt evolution with genetic-Pareto candidate selection (Agrawal et al., arXiv 2025), via `dspy.GEPA` on the same bridged program.

```python
class GEPAConfig(BaseOptimizerConfig):
    max_metric_calls: int = 60
    reflection_minibatch_size: int = 4
    candidate_selection: Literal["pareto", "best"] = "pareto"
```

### 8.1 Reject-reason routing (`feedback.py`)

GEPA's reflection step consumes *textual* feedback. We render it deterministically from the mini pipeline's `state.metrics` (`gate_reject_reasons`, `gate_pass_rate`, KL/C2ST components) via a versioned jinja2 template (`feedback_v1.j2`), tallies sorted by count desc then name. Routing table (the R2 mapping — keep these exact check names, they are the tally keys):

| tally key      | failure meaning                        | instruction pressure injected |
|----------------|----------------------------------------|-------------------------------|
| `answerability`| not answerable from evidence           | grounding: answer-first; the claim must be extractable from the chunk |
| `zero_context` | answerable without evidence            | evidence-dependence: require facts unique to the chunk, not common knowledge |
| `dedup`        | candidates collapse to near-duplicates | diversity: vary intent, aspect, and phrasing across candidates |
| `round_trip`   | gold chunk not retrieved top-k         | specificity: include discriminative terms; avoid generic phrasing |
| `uniqueness`   | non-gold chunks also answer            | anchoring: target details that distinguish this chunk from neighbors |

### 8.2 Feedback objective

```python
@dataclass(frozen=True)
class ScoredFeedback:
    score: float
    feedback: str

class FidelityFeedbackObjective(FidelityObjective):
    """One mini-pipeline run per call; never runs the pipeline twice."""
    def with_feedback(self, prompt: str) -> ScoredFeedback: ...
    def __call__(self, prompt: str) -> float:  # == with_feedback(prompt).score
```

The GEPA adapter requires `with_feedback` (duck-checked at `optimize()` start; actionable `TypeError` otherwise); MIPROv2 keeps using plain `__call__`. Feedback text is stored per trial in `TrialRecord.meta["feedback"]`.

---

## 9. Exemplar selection & the anti-gaming guard (Promptagator/InPars lineage)

### 9.1 Demo pool (`factory.py`)

Few-shot demos are an optimizable component alongside instruction text: the pool = real `(chunk, query)` pairs from the **train** split (labeled demos) ∪ gate-accepted records from a baseline bootstrap run (bootstrapped demos — Promptagator's round-trip-filtered self-generation). `factory.mini_pipeline_factory(config, resources, prompt, n_seeds, stream)` builds the mini pipeline (seed_sampler → … → gate) with seeds drawn from `Resources.rng(stream)`. Demo ids go in `TrialRecord.meta["demo_ids"]`.

### 9.2 The guard (state it in code comments and docs)

The v1 `round_trip` gate check **is** the Promptagator consistency filter. An optimizer scored through the gate can learn to game it (keyword-stuffing chunk terms passes round-trip while wrecking realism). Guards, all mandatory:

1. **Reference firewall:** `FidelityObjective.reference_embs` during optimization = *train*-split production embeddings only. The *anchor* split (τ/τ_AP/fidelity reporting reference) is never visible to any objective call.
2. **Seed firewall:** optimization mini pipelines draw seeds from substream `rng("optimizer")`; the final reported run draws from the standard pipeline streams — disjoint by construction, asserted in tests.
3. **Held-out gate evaluation:** the number that goes in `metrics.json`/`report.md` comes from ONE fresh outer-pipeline run with the winning prompt on the untouched split — never from trial history.
4. **Overfit guard test** (§13): held-out objective of the winner ≥ held-out objective of the baseline − ε (ε = 0.05).
5. Downstream curator memorization check (cos ≥ 0.9 vs production) still applies to optimized output.

---

## 10. Determinism policy

- **Not guaranteed:** byte-determinism of trial scores/history when `generator_llm`/`reflection_llm` is a real endpoint (`OpenAICompatibleChat`). LLM sampling is outside our control; SPEC's same-seed rule cannot hold there.
- **Guaranteed instead:** (a) with mock adapters the entire loop — trials, scores, feedback strings, winner — is same-seed deterministic (this is what CI asserts); (b) complete audit trail: every candidate, score, demo-id set, and feedback string is recorded in `trial_history` and `trials.jsonl`; (c) `NoOpOptimizer` invariant baseline == optimized, improvement == 0.0 exactly; (d) all sampling we own (seeds, minibatches, demo selection, C2ST splits) flows from `Resources.rng` substreams; backend seeds are set from `config.seed` where the backend accepts one; (e) the final outer pipeline run with a *fixed* winning prompt keeps the v1 same-seed ⇒ identical `metrics.json` guarantee. `TrialRecord.timestamp` stays wall-clock (D31 precedent; excluded from the determinism surface).

---

## 11. Config & serialization (schema_version 2)

```yaml
# configs/v2_opt_toy.yaml (schema v2)
ragsynth: {schema_version: 2, name: v2-opt-toy, seed: 0}
resources:
  # ... identical to v1_toy.yaml ...
  reflection_llm: {type: toy_chat}      # optional; defaults to judge_llm (WARN on generator family)
optimization:
  optimizer: gepa                       # noop | mipro_v2 | gepa | textgrad (stub)
  objective: {kind: fidelity_feedback, alpha: 1.0, min_pass_rate: 0.5}
  params: {max_metric_calls: 60, reflection_minibatch_size: 4, candidate_selection: pareto}
pipeline:
  # ... v1 steps unchanged; generator.prompt_version is the optimization target ...
```

- schema_version-2 gating and the `Pipeline.from_yaml` loader relaxation are defined once in the canonical trigger list in [specs/v2/README.md](README.md) (single owner; the loader change is made once, by whichever spec merges first, citing that section). Of this spec's features, the `optimization:` block and `resources.reflection_llm` require `schema_version: 2`.
- Every new config class (`MIPROv2Config`, `GEPAConfig`, objective/feedback settings) is a frozen pydantic model reachable via `to_config()/from_config()`; adapters register in `OPTIMIZERS` exactly like v1.
- CLI: `ragsynth optimize --config <yaml>` writes `<artifacts_dir>/optimization/{optimization_result.json, trials.jsonl, prompts/*.j2}` and prints the winner + improvement; `ragsynth run` is unchanged.

---

## 12. Testing & CI (TDD — fixtures FIRST)

**12.1 Known-value fixtures (write before implementation):** tally `{"answerability": 3, "dedup": 1}` ⇒ exact feedback string (grounding pressure first, diversity second — assert full text); template with undeclared variable `foo` ⇒ `HARD_PENALTY`, no pipeline invocation (spy factory); `RecordingMetric` on 3 calls ⇒ trial_ids `[1,2,3]`, scores match; NoOp result unchanged vs v1 golden; v1 YAML fixture loads to identical config under the v2 loader.

**12.2 Core (air-gapped, no extras):** the full closed loop exercised with a `MockChatModel`-driven `FidelityFeedbackObjective` on the toy world — objective called directly and via `RecordingMetric`; split/seed firewall assertions (§9.2); `create_optimizer("gepa", ...)` without extras ⇒ actionable `ImportError` naming `uv sync --extra optimization`.

**12.3 Extras job (`uv sync --extra optimization`; still offline):** `dspy` present, all LLM traffic through `AdapterLM(MockChatModel)`. MIPROv2 and GEPA smoke runs on the toy world within tiny budgets (`num_trials=3` / `max_metric_calls=6`); tests `pytest.importorskip("dspy")` so the core job skips them cleanly. mypy overrides add `gepa.*` next to the existing `dspy.*`/`textgrad.*`.

---

## 13. Acceptance criteria (Definition of Done)

- [ ] `NoOpOptimizer` invariant and every v1 optimization test pass **unmodified**.
- [ ] Toy world, mock LLMs, extras installed: `mipro_v2` and `gepa` each complete within budget and return `optimized_score ≥ baseline_score` (guaranteed by construction — argmax includes baseline; asserted with tolerance 1e-9).
- [ ] GEPA trial history contains non-empty `meta["feedback"]` reflecting the run's actual reject tallies.
- [ ] Overfit guard: winner's held-out objective ≥ baseline's held-out objective − 0.05 on the toy world; anchor split provably untouched by objective calls (spy test).
- [ ] `num_trials == len(trial_history)` == objective-call count for every backend.
- [ ] v2 YAML round-trips byte-stably; committed v1 configs load and round-trip byte-identically under the v2 loader; unknown optimizer key ⇒ registry error listing `noop, mipro_v2, gepa, textgrad`.
- [ ] §15.3 commands green air-gapped; extras job green offline; `ragsynth optimize --config configs/v2_opt_toy.yaml` completes < 10 min CPU with mocks.

---

## 14. Decisions pre-made for PLAN.md (record as D52–D61)

Decision numbers D52–D61 allocated per specs/v2/README.md.

| # | Decision |
|---|---|
| D52 | Registry keys `noop`/`mipro_v2`/`gepa`/`textgrad`(stub); one adapter module per backend; backend types never cross the module boundary. |
| D53 | GEPA via `dspy.GEPA` (extra becomes `dspy>=3.0`); standalone `gepa` package confirmed MIT (2026-07) but the dspy route wins: one bridge, one LM path, one extra. |
| D54 | All backend LLM calls through `AdapterLM(ChatModel)`; extras CI job runs offline on `MockChatModel`. |
| D55 | Optimized prompts materialize as versioned `.j2` run artifacts via a `ChoiceLoader`; package templates immutable; provenance via existing `gen_meta.prompt_version`. |
| D56 | Candidate = full template text; jinja2-meta variable-set validation; invalid ⇒ `HARD_PENALTY` trial, no pipeline run. |
| D57 | Anti-gaming firewall: train-split reference, `rng("optimizer")` seed substream, held-out reporting run, overfit guard ε = 0.05. |
| D58 | Trial history owned by `RecordingMetric`, never backend logs. |
| D59 | schema_version-2 gating and the single-owner loader change follow the canonical trigger list in specs/v2/README.md; of this spec's features, the `optimization:` block and `resources.reflection_llm` require schema 2. |
| D60 | Baseline is always trial 1; winner = argmax incl. baseline ⇒ improvement ≥ 0 by construction. |
| D61 | `TrialRecord` gains `meta: dict` (default `{}`) — the only contract-type change, backward compatible. |

---

## 15. Open questions — defaults for the agent (decide, document, proceed)

1. Spec file location — no path was mandated; defaulted to `specs/v2/`. Move if the repo grows a different v2-spec convention.
2. `dspy` pin: `>=3.0,<4` assumed to provide `dspy.GEPA`; if the installed 3.x lacks it, fall back to direct `gepa` (MIT) behind the same adapter — the boundary makes this a one-module swap.
3. `reflection_llm` default = `judge_llm` config (already forced distinct from generator by §6.4 validation); WARN, don't error, on family collision.
4. Budget defaults (`num_trials=20`, `minibatch_size=8`, `max_metric_calls=60`, ε=0.05) — config-exposed, tuned on the toy world during implementation if smoke runs exceed the 10-min bar.
5. `textgrad` remains a registered stub in v2 (extra already ships it; execution has no trigger yet).
6. Demo-pool sizing (`max_labeled_demos=4`, `max_bootstrapped_demos=3`) follows MIPROv2 paper defaults scaled down for mini pipelines.

---

## 16. Reference table (feature → primary sources)

| Feature | References |
|---|---|
| MIPROv2 instruction+demo optimization | Opsahl-Ong et al., EMNLP 2024 |
| GEPA reflective genetic-Pareto evolution | Agrawal et al., arXiv 2025 |
| DSPy programming model | Khattab et al., ICLR 2024 |
| Few-shot exemplar selection, round-trip consistency filter | Dai et al. (Promptagator), ICLR 2023; Bonifacio et al. (InPars), SIGIR 2022 |
| Filtering beats generating more (gate as fitness terrain) | Gospodinov et al. (Doc2Query--), ECIR 2023 |
| Cross-family judge/reflection rule | Fröbe et al., SIGIR 2025 |
| Fidelity objective components | Lopez-Paz & Oquab, ICLR 2017 (C2ST); Chroma Generative Benchmarking, 2025 (KL) |
