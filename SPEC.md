# SPEC — `ragsynth`: a Python toolkit for synthetic query generation & validation for RAG evaluation

> **Audience:** Claude Fable 5 coding agent (with superpowers skills).
> **Author of record:** Max (ML Tech Lead) — this spec consolidates a long design conversation; treat it as the source of truth.
> **Status:** v1 scaffold spec. Expand with a written plan before coding.

---

## 0. How to execute this spec (read first)

1. **Plan before coding.** Produce `PLAN.md` (brainstorm → task breakdown → file-by-file plan) before the first line of implementation. Use your planning/subagent skills; fan out subagents only for independent modules (e.g., `metrics/` vs `adapters/`), never for tightly coupled ones.
2. **Interpret literally; scope explicitly.** Where this spec says v1 excludes something, do not implement it "while you're at it" — create a stub/ABC only where the spec asks for one.
3. **TDD-ish loop.** For every metric and statistical component, write the known-value unit test first (fixtures given in §15), then implement.
4. **Verify continuously.** The commands in §15.3 must pass at every commit. Iterate against them, not against your own judgment of "done".
5. **Ask vs. assume.** For the open questions in §19, choose the listed default, document the choice in `PLAN.md`, and proceed. Do not block.
6. **Reference prototype.** A working single-file prototype `synth_query_eval.py` is provided alongside this spec. Vendor it unmodified at `reference/synth_query_eval.py` and port its logic into the package per the mapping in Appendix A. If it is missing, reimplement from the formulas in this spec.
7. **Success = §15 acceptance criteria.** Nothing else is the bar.

---

## 1. Context & motivation (condensed from the design conversation)

We operate RAG retrieval over a **dynamic knowledge base** (documents added/modified/removed continuously — the RAGOps setting, arXiv 2506.03401). Real production queries exist but are expensive to annotate, and KB churn invalidates annotations and leaves new content uncovered. The toolkit generates **synthetic queries with relevance annotations (qrels)** from corpus chunks, *validated against production query distributions*, for:

- **Primary (v1):** retrieval evaluation & monitoring — a "self-refreshing benchmark".
- **Secondary (v2/v3):** fine-tuning data for embedders/rerankers; conversational agent evaluation.

**Key design decisions already made (do not relitigate):**

- **Chunk-first v1.** Seeds are chunks from a vector-DB-like store. Chunks carry optional `canonical_bbox` + `page_image_ref` metadata for future visual grounding (ColPali/ColQwen setting), but v1 generation conditions on **text only**.
- **Standalone queries first.** Conversational (context-dependent) queries are a future stratum; domain objects must be ready for them (§4).
- **Validity over vibes.** The central acceptance test of a synthetic set is **system-ranking agreement** with a real-query anchor set (Kendall τ / τ_AP), plus **positive controls** (does the benchmark detect injected regressions?), not just distributional similarity.
- **Two data products, one firewall.** Representative eval set (fidelity-gated) vs. stress/discovery suite (edge cases, perturbations) are separate outputs; edge-case injection must never contaminate the demand-weighted headline. Training-split firewall (document-level) is v3.
- **Air-gapped, permissive-license only.** Runtime deps limited to Apache-2.0/MIT/BSD; all LLM/embedding calls go through adapters targeting OpenAI-compatible endpoints (vLLM/LiteLLM) with fully offline mock implementations for tests/CI.

---

## 2. Goals & non-goals

### 2.1 v1 goals

1. A composable, **sklearn-style pipeline** (`SeedSampler → ContextAssembler → QueryGenerator → VerificationGate → QrelBuilder → Curator → Validator`) producing `AnnotationRecord`s and an `EvalReport`.
2. **Serialization:** every step implements `to_config()/from_config()`; the whole pipeline round-trips YAML/JSON byte-stably (§13).
3. **Evaluation harness** with the four experiment arms (A0 naive / A1 quota / A2 spec-first / ORACLE) runnable end-to-end on (a) the synthetic **toy world** and (b) a small local JSONL corpus.
4. **Metrics suite:** fidelity (KL, C2ST, MMD, within-cluster C2ST), efficiency (importance weights, ESS, coverage gap, demand-weighted coverage, post-stratified headline), validity (τ, τ_AP, RBO, bootstrap CI, positive controls), light diversity (distinct-n, dedup rate).
5. **Iteration scaffolding:** `experiments/v1/` with notebook, `figures/`, `report.md`, `metrics.json`, and the frozen `config.yaml` snapshot.

### 2.2 v1 non-goals (stub or omit; listed so you don't build them)

- Visual/bbox-conditioned generation, masking-ablation checks (CiteVQA), page rendering → **v3**; only carry the metadata fields.
- Conversational generation & MTRAG-style eval → **v2/v3**; only domain objects + stratum enum value.
- Graded/pooled qrels (UMBRELA), LLM-judge calibration tooling → **v2**; v1 qrels are binary.
- Prompt-optimizer *execution* (DSPy/TextGrad) → **v2**; v1 ships the ABC + `NoOpOptimizer` + `FidelityObjective` only (§11).
- Training-data export, hard-negative mining (Gecko/NV-Retriever) → **v3**.
- Annotation lifecycle under churn (recycle/regenerate) → **v2**; v1 stores `content_hash` on every record so v2 needs no migration.
- Vendi score, MAUVE, movMF *soft* demand map as default → optional/v2 (movMF class itself IS v1 because A2 needs it; the *reporting* partition stays hard KMeans).
- Distributed execution, DB backends, web UI. Everything is in-memory/npz/jsonl.

---

## 3. Architecture principles

### 3.1 sklearn-style pipeline, adapted

Steps are not pure `transform(X)->X'`; they consume/extend a typed `PipelineState`. Keep the sklearn ergonomics anyway:

```python
class PipelineStep(ABC):
    """One stage of the synthetic-annotation pipeline.

    Lifecycle: ``fit(resources)`` (optional, idempotent) learns anything
    data-dependent (demand map, thresholds); ``run(state)`` consumes and
    returns the state. Both must be side-effect-free outside ``state`` and
    the step's own artifacts directory.
    """
    name: ClassVar[str]                      # registry key, stable across versions
    version: ClassVar[str] = "1"

    def fit(self, resources: Resources) -> Self: ...          # default: no-op
    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState: ...
    @abstractmethod
    def to_config(self) -> dict[str, Any]: ...                # JSON-safe params only
    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> Self: ...
```

- `Resources` (frozen): chunk store, production queries, embedder/LLM/judge/retriever adapters, rng seed, artifacts dir. Injected at the **composition root** (`Pipeline.from_yaml` / CLI) — steps never construct adapters (DIP).
- `Pipeline(steps: list[PipelineStep])` exposes `fit`, `run`, `named_steps`, `to_yaml/from_yaml`, and `get_params()` (flat `step__param` style, read-only in v1).
- `PipelineState` (pydantic): `seeds`, `contexts`, `candidates`, `accepted: list[AnnotationRecord]`, `rejected: list[Rejection]`, `metrics: dict`, `provenance` (config hash, seed, timestamps).

### 3.2 SOLID mapping (enforced in review)

- **S:** one step = one responsibility; gate checks are separate classes, not branches.
- **O:** new samplers/checks/optimizers/adapters are added by subclassing + registry entry; no edits to the pipeline runner.
- **L:** every concrete step must pass the shared `PipelineStepContract` test parametrized over the registry (§15.2).
- **I:** small Protocols — `Embedder`, `ChatModel`, `RelevanceJudge`, `Retriever`, `OptimizationMetric` — instead of one god-interface.
- **D:** steps and metrics depend on Protocols; concretes live only in `adapters/` and the composition root.

### 3.3 Patterns to use (and where)

- **Registry** (`dict[str, type]`) with a `@register("seed_sampler.quota")` decorator — powers deserialization and the LSP contract test. Mirror the healthbench-agent-lab registry-factory style (`step_class.from_config(config, resources)`).
- **Strategy** via constructor-injected callables/Protocols (e.g., `SeedSampler(allocation=MixtureAllocation(lam=0.7))`).
- **Frozen dataclasses / frozen pydantic** for all result objects (`OptimizationResult`, `EvalReport`, `Rejection`) — matches the reference repo's `OptimizationResult`/`TrialRecord` pattern.
- **Lazy optional imports** with a `require_optional(module, feature_name)` helper raising `ImportError: install with uv sync --extra <name>` — copy the pattern from `healthbench_agent/prompt_optimization/optimizer.py`.

---

## 4. Domain model (`src/ragsynth/domain/`) — pydantic v2, all frozen

Conversation-ready from day one; v1 populates only the standalone path.

```python
class Chunk(BaseModel):                 # chunk.py
    chunk_id: str                       # content-addressed: sha256(norm_text|doc_id|page|bbox)[:16]
    doc_id: str
    text: str
    content_hash: str
    embedding_ref: str | None = None    # key into the EmbeddingStore, never inline vectors
    page: int | None = None
    canonical_bbox: tuple[float, float, float, float] | None = None   # v3 visual
    page_image_ref: str | None = None                                  # v3 visual
    metadata: dict[str, str] = {}

class ProductionQuery(BaseModel):       # query.py
    query_id: str; text: str; timestamp: datetime | None = None
    embedding_ref: str | None = None
    stratum: Stratum | None = None

class Stratum(BaseModel):               # stratum.py — Hamel-style dimension tuple
    dims: dict[str, str]                # v1: {"query_type": "factoid|howto|keyword"}; later + persona, difficulty
    def key(self) -> str: ...           # canonical "query_type=factoid|persona=broker"

class Seed(BaseModel):                  # seed.py
    seed_id: str
    chunk_ids: tuple[str, ...]          # 1 = single-chunk; >1 = chunk-group (multi-evidence, v1 supports both)
    cluster_id: int
    stratum: Stratum
    z: list[float] | None = None        # A2 target embedding (planning space)

class GenerationContext(BaseModel):     # context.py
    seed: Seed
    chunk_texts: tuple[str, ...]
    style_exemplars: tuple[str, ...]    # nearest production queries
    instruction: str                    # rendered stratum instruction

class SyntheticQuery(BaseModel):        # candidate.py
    query_id: str; text: str; seed: Seed
    embedding_ref: str | None
    gen_meta: dict[str, Any]            # model, prompt_version, cos_to_target, n_candidates, ...

class AnnotationRecord(BaseModel):      # annotation.py — THE canonical output
    record_id: str
    query: SyntheticQuery
    qrels: dict[str, int]               # chunk_id -> grade (v1: {0,1}); gate-promoted golds included
    crucial: tuple[str, ...] = ()       # v3 masking-ablation fills this; v1 = all gold
    supplemental: tuple[str, ...] = ()
    stratum: Stratum
    dialogue_context: tuple[Turn, ...] | None = None   # None in v1 (conversational-ready)
    gate_meta: dict[str, Any]           # per-check pass/scores
    content_hashes: dict[str, str]      # chunk_id -> hash at annotation time (v2 lifecycle)
    benchmark_version: str; created_at: datetime

class Turn(BaseModel):                  # conversation.py (v1: model only, unused)
    role: Literal["user", "assistant"]; text: str

class Rejection(BaseModel):             # rejection.py
    candidate: SyntheticQuery; check: str; reason: str; score: float | None
```

`EvalReport` (report.py): frozen; per-arm fidelity/efficiency/validity blocks + config snapshot + git-style provenance; `to_json()`.

**Rule:** embeddings never live inside domain objects — an `EmbeddingStore` (npz + id index, `io/embeddings.py`) owns them; objects hold `embedding_ref`.

---

## 5. Package layout & engineering standards

Mirror `msobroza/healthbench-agent-lab` conventions: src layout, `uv_build` backend, MIT license, `requires-python >= 3.11`, extras with lazy imports, console scripts, Google-style docstrings, `TYPE_CHECKING` imports, strict typing.

```
ragsynth/
├── pyproject.toml            # uv_build; deps: numpy, scipy, scikit-learn, pydantic>=2,
│                             #   pyyaml, typer, rich, joblib.  extras: dev, notebooks,
│                             #   optimization (dspy, textgrad), bm25 (bm25s), st (sentence-transformers)
├── README.md                 # quickstart: 10 lines to run v1 toy
├── SPEC.md                   # this file
├── PLAN.md                   # written by the agent before coding
├── configs/
│   ├── v1_toy.yaml
│   └── v1_local_corpus.yaml
├── reference/
│   └── synth_query_eval.py   # frozen prototype (do not import from package code)
├── src/ragsynth/
│   ├── __init__.py  py.typed
│   ├── domain/               # §4
│   ├── pipeline/             # base.py (PipelineStep, Resources, PipelineState),
│   │                         # pipeline.py, registry.py, serialization.py
│   ├── steps/                # seed_sampler.py, context_assembler.py, generator.py,
│   │                         # gate.py (orchestrator), qrel_builder.py, curator.py, validator.py
│   ├── sampling/             # vmf.py, movmf.py, demand.py (p_hat, tilt, time decay),
│   │                         # partition.py (frozen KMeans ref partition), spec_sampler.py (guarded z)
│   ├── gate/checks/          # base.py (GateCheck ABC), dedup.py, zero_context.py,
│   │                         # answerability.py, round_trip.py, uniqueness.py
│   ├── metrics/
│   │   ├── fidelity.py       # kl_similarity_distributions, c2st_auc, mmd_rbf, within_cluster_c2st
│   │   ├── efficiency.py     # cluster_importance_weights, ess, post_stratified_estimate,
│   │   │                     # demand_weighted_coverage, msc/zero_query_clusters
│   │   ├── diversity.py      # distinct_n, semantic_dedup_rate  (vendi behind extra, v2)
│   │   └── validity/         # agreement.py (tau, tau_ap, rbo, bootstrap CI),
│   │                         # controls.py (degradations, paired_bootstrap_pvalue)
│   ├── arms/                 # a0_naive.py, a1_quota.py, a2_spec.py, oracle.py (presets over steps)
│   ├── optimization/         # §11: base.py, noop.py, objectives.py, mipro_adapter.py(stub, extra)
│   ├── adapters/             # llm/ (ChatModel Protocol, openai_compatible.py, mock.py),
│   │                         # embedder/ (Embedder Protocol, mock.py, hashed.py, st.py[extra]),
│   │                         # retriever/ (Retriever Protocol, dense_inmemory.py, bm25s.py[extra]),
│   │                         # judge/ (RelevanceJudge Protocol, llm_judge.py, mock.py)
│   ├── datasets/             # toy_world.py (port of prototype), jsonl_loader.py
│   ├── io/                   # embeddings.py (EmbeddingStore), artifacts.py (joblib/npz + manifest)
│   └── cli.py                # typer: ragsynth run|validate|report --config ...
├── experiments/
│   └── v1/                   # notebook(s), figures/, report.md, metrics.json, config.yaml (frozen copy)
└── tests/                    # mirrors src/; contract tests in tests/pipeline/test_contract.py
```

**Standards (audit-grade — this repo will be run through `python-library-audit`):** ruff (line 100, full ruleset incl. D for docstrings), mypy strict with `scipy-stubs`/`types-pyyaml`/`pandas-stubs`, pytest + pytest-cov (target ≥ 85% on `metrics/`, `sampling/`, `pipeline/`; ≥ 70% overall), `py.typed`, no `utils.py`, deterministic seeds threaded through `Resources`, structured logging (`logging`, no prints outside CLI). Every public function: Google docstring with Args/Returns/Raises and, for metrics, the paper reference.

---

## 6. Pipeline steps — v1 contracts, methods, references

Every step: `name` registry key, config-serializable, tested against the shared contract. Inputs/outputs refer to `PipelineState` fields.

### 6.1 `seed_sampler` — *what to generate from*
- **In:** chunk store, frozen reference partition (§7.4), demand map `p_hat` (§7.2). **Out:** `state.seeds`.
- **v1 strategies (Strategy pattern, each a registry entry):**
  - `seed_sampler.uniform` — uniform chunks (arm A0).
  - `seed_sampler.quota` — mixture allocation `n_c ∝ λ·p_hat_c + (1−λ)/C` with per-cluster floor `n_min`; within a cluster sample chunks by that cluster's membership. Defaults `λ=0.7`, `n_min=3`. Chunk-groups: with prob `p_group=0.2`, pair a chunk with a same-doc neighbor (multi-evidence seed).
  - `seed_sampler.spec` — A2: sample `z` from demand-tilted movMF via `SpecSampler` (on-manifold guard), attach kNN chunks as seed `chunk_ids`, store `z`. (Coverage/Chroma weighting: Hong et al. 2025; allocation smoothing: Jelinek-Mercer analogy; coverage floors: BCG arXiv 2510.00001, "Coverage, Not Averages" arXiv 2604.20763.)
- **Stratum assignment:** round-robin over configured `Stratum` dims per cluster in v1 (demand-calibrated dimension weights = v2 research item R3).

### 6.2 `context_assembler` — *what the generator sees*
- **In:** seeds. **Out:** `state.contexts`.
- v1: chunk text(s) + `k_style=3` nearest production queries (by embedding, within-cluster preferred) + rendered stratum instruction. Two-step tuple→query phrasing decoupling (Hamel/Shankar) and blind-summary mode (ViDoRe V2) = config flags stubbed, default off.

### 6.3 `generator` — *overgenerate candidates*
- **In:** contexts; `ChatModel` adapter. **Out:** `state.candidates` (`n_candidates=3` per seed).
- v1 prompt: **answer-first** (extract a claim from the evidence, then write the question a user with that need would type; forbid "according to the document" phrasing; match exemplar length/register). References: Alberti et al., ACL 2019; Promptagator (Dai et al., ICLR 2023); InPars (Bonifacio et al., SIGIR 2022).
- Prompt text lives in `steps/prompts/` as versioned jinja2 templates; `prompt_version` recorded in `gen_meta`. `MockChatModel` returns deterministic templated text so the full pipeline runs offline.

### 6.4 `gate` — *verification (ordered, cheap→expensive)*
`GateCheck` ABC: `check(candidate, state, resources) -> CheckResult(passed, score, reason)`. Orchestrator short-circuits, tallies reject reasons (`state.metrics["gate_reject_reasons"]` — this tally is the v2 optimizer's routing signal).
1. `dedup` — exact + semantic (cosine ≥ 0.95 vs accepted set) with MMR selection (Carbonell & Goldstein 1998).
2. `zero_context` — judge answers WITHOUT evidence ⇒ common knowledge ⇒ reject (CiteVQA zero-document self-test, arXiv 2605.12882).
3. `answerability` — judge answers FROM evidence only, else reject (CiteVQA).
4. `round_trip` — gold chunk(s) in top-`k=10` of the configured `Retriever` (Promptagator consistency filter; Doc2Query--, Gospodinov et al., ECIR 2023: filtering beats generating more).
5. `uniqueness` — judge whether top non-gold retrieved chunks also answer; if yes: `promote` (add to qrels) or `reject` per config (anti-leakage; corrupted golds invalidate retrieval eval).
- **Cross-family judge rule:** config validation WARNS if `judge.model == generator.model` family (Fröbe et al., SIGIR 2025: LLM assessors overestimate same-family rerankers by 9–17 rank positions).

### 6.5 `qrel_builder` — binary v1
Gold = seed chunks + uniqueness-promotions, grade 1. Emits `AnnotationRecord` with `content_hashes` filled. Graded/pooled (UMBRELA, Upadhyay/Clarke/Lin 2024) behind ABC `QrelStrategy` = v2.

### 6.6 `curator`
Stratified subsample to target mix; final dedup; memorization check — flag records with cosine ≥ 0.9 to any production query (Chroma's verbatim-reproduction audit).

### 6.7 `validator`
Runs the metrics suite (§9) against the anchor set + system zoo from `Resources`; writes `EvalReport`, `metrics.json`, and figures (matplotlib, saved under the experiment dir; no seaborn dependency in core).

---

## 7. Sampling & statistics (`sampling/`) — port from prototype

1. **`vmf.py`** — Wood (1994) rejection sampler; Householder rotation. Property test: mean resultant direction → μ as κ grows.
2. **`movmf.py`** — mixture of von Mises–Fisher, EM with log-space responsibilities, Banerjee κ approximation `κ = r̄(d−r̄²)/(1−r̄²)`, κ clipped to [1e-2, 1e5], KMeans init (Banerjee et al., JMLR 2005). Serialization: `to_artifact()/from_artifact()` (npz) + config manifest with fitted-on hash — the demand map is a **versioned frozen artifact per benchmark epoch**.
3. **`demand.py`** — `demand_from_responsibilities` (exponential time-decay half-life), `tilt_weights(p_hat, lam)`, exploration pseudo-component hooks (v2 cold-start).
4. **`partition.py`** — frozen KMeans reference partition (default `C=8`… configurable) used for ALL reporting/ESS/quotas; versioned artifact; changing it is a benchmark-migration event.
5. **`spec_sampler.py`** — guarded ancestral sampling: `c ~ Cat(π′)`, `z ~ vMF(μ_c, κ_c)`, reject unless `max_j zᵀq_j ≥ τ_r` (τ_r = 5th percentile of production NN-cosines) or c is exploration.

---

## 8–9. Evaluation module (`metrics/`) — the non-negotiable core

**Fidelity** (real reference = anchor embeddings, equal-n subsample; per-stratum where n permits):
- `kl_similarity_distributions(real, synth, chunks)` — KL(real‖synth) of top-1 query→chunk cosine histograms, 50 bins, ε-smoothed (Chroma monitor; steered target ≈ ≤0.16 on real data — a *reference band*, not a hard gate).
- `c2st_auc` — 5-fold logistic AUC on embeddings (Lopez-Paz & Oquab, ICLR 2017). Track, don't hard-gate in v1; expose top coefficients for the "what does the discriminator use" diagnostic.
- `mmd_rbf` — unbiased quadratic, median-heuristic bandwidth, subsample cap (Gretton et al., JMLR 2012).
- `within_cluster_c2st` — mean + per-cluster AUC inside the reference partition (min 30/side). **This is the A2 mechanism meter.**

**Efficiency:** `cluster_importance_weights` (w_i = p̂_c/q_c over covered clusters + coverage_gap), `effective_sample_size` (Kong 1992 identity: variance inflation = n/ESS), `post_stratified_estimate` (demand-weighted headline), `demand_weighted_coverage = Σ_c p̂_c·1[covered]`, `minimum_semantic_coverage`, `zero_query_clusters` (BCG; "Coverage, Not Averages"). **Dual-view reporting rule:** the report always shows (a) demand-weighted headline + ESS, (b) unweighted per-cluster table + worst-k clusters. Never a single blended unweighted average.

**Validity:**
- `agreement.py` — Kendall τ (scipy), `tau_ap` (Yilmaz/Aslam/Robertson SIGIR 2008 — exact loop from prototype), `rbo_ext` p=0.9 (Webber/Moffat/Zobel TOIS 2010), bootstrap CI over query resampling (n_boot=1000). Gate targets to *report against*: τ ≥ 0.9 AND τ_AP ≥ 0.8 per stratum ⇒ usable for model selection; below ⇒ regression-detection only (calibration: real-vs-synthetic literature ceiling τ≈0.82–0.86, Rahmani et al. SIGIR 2024; SynDL 2025).
- `controls.py` — positive-control battery: degradation factory (`drop_index(frac=0.10)`, `noise_transform(σ)`, `truncate_topk`) + `paired_bootstrap_pvalue` one-sided (Sakai SIGIR 2006 discriminative-power tradition). A benchmark that misses an injected 10% index deletion fails regardless of τ.
- Regeneration stability (two seeds, τ between their own rankings) = v2, but design `validator` so arms accept a seed list.

**Diversity (v1 light):** distinct-1/2, semantic-dedup rate. Vendi (Friedman & Dieng, TMLR 2023) behind extra, v2.

---

## 10. Experiment arms & datasets

`arms/` are thin **presets** composing steps + configs (not new logic): A0 = uniform+no exemplars; A1 = quota+exemplars; A2 = spec-sampler; ORACLE = held-out real subsample (the τ ceiling — always run it). `datasets/toy_world.py` ports the prototype's 8-component × 2-hidden-sub-mode world with the gate-style nearest-chunk gold relabeling — this is the CI end-to-end fixture and must reproduce the qualitative demo table (A2 best wC2ST/MMD; A1 ESS≈oracle but wC2ST≈1; A0 fails everything; A1 misses the noise positive-control). `datasets/jsonl_loader.py`: `chunks.jsonl` + `queries.jsonl` (+ optional `anchor_qrels.jsonl`) for the first real run.

---

## 11. Prompt-optimization abstraction (v1 = contract only)

Mirror `healthbench_agent/prompt_optimization/optimizer.py` exactly in spirit:
- `OptimizationMetric(Protocol)`: `__call__(prompt: str) -> float`.
- Frozen `TrialRecord` / `OptimizationResult` dataclasses (prompt, baseline/optimized score, improvement, trial_history, optimizer_name, config).
- `BasePromptOptimizer(ABC)` with `optimize(current_prompt, samples, metric) -> OptimizationResult`; config-subclass per adapter; registry factory; `require_optional` for lazy `dspy`/`textgrad` extras.
- v1 ships: `NoOpOptimizer` (returns baseline; proves the plumbing) and `objectives.FidelityObjective` — runs a mini pipeline with the candidate prompt and returns `−(KL + α·C2ST_AUC)` subject to gate pass-rate ≥ threshold. This is the v2 closed loop's objective (MIPROv2, Opsahl-Ong et al., EMNLP 2024; reject-reason routing from §6.4).

---

## 12. Adapters (offline-first)

- `ChatModel` Protocol: `complete(system, user, **kw) -> str`. `OpenAICompatibleChat(base_url, model, api_key_env)` covers vLLM/LiteLLM/internal gateways; `MockChatModel` deterministic (hash-seeded) for CI.
- `Embedder` Protocol: `encode(texts) -> np.ndarray` (L2-normalized). v1 default for real runs = `HashedNGramEmbedder` (pure-numpy featurizer, no downloads) + optional `sentence-transformers` extra; toy world bypasses text entirely.
- `Retriever` Protocol: `search(query_emb, k) -> list[(chunk_id, score)]`. `DenseInMemoryRetriever` (matmul); `BM25sRetriever` extra.
- `RelevanceJudge` Protocol: `judge(query, evidence_texts) -> JudgeVerdict(answerable, answer, confidence)`; `LLMJudge` (own ChatModel instance — enforce distinct config key) + `MockJudge`.

---

## 13. Serialization spec

```yaml
# configs/v1_toy.yaml (schema v1)
ragsynth: {schema_version: 1, name: v1-toy, seed: 0}
resources:
  dataset: {type: toy_world, params: {d: 64, k_true: 8, n_prod: 5000}}
  embedder: {type: passthrough}            # toy world provides embeddings
  generator_llm: {type: mock}
  judge_llm: {type: mock}
  retriever: {type: dense_inmemory}
artifacts_dir: experiments/v1/artifacts
pipeline:
  - {type: seed_sampler.quota, params: {lam: 0.7, n_min: 3, n_seeds: 500, p_group: 0.2}}
  - {type: context_assembler, params: {k_style: 3}}
  - {type: generator, params: {n_candidates: 3, prompt_version: answer_first_v1}}
  - {type: gate, params: {checks: [dedup, zero_context, answerability, round_trip, uniqueness],
                          round_trip: {k: 10}, dedup: {cos_threshold: 0.95}}}
  - {type: qrel_builder, params: {strategy: binary}}
  - {type: curator, params: {memorization_cos: 0.9}}
  - {type: validator, params: {arms: [a0, a1, a2, oracle], n_boot: 1000,
                               gates: {tau: 0.9, tau_ap: 0.8}}}
```

Rules: `to_config` returns JSON-safe primitives only (artifacts referenced by relative path + sha256 in a manifest); `Pipeline.from_yaml(path)` is the ONLY deserialization entrypoint; **round-trip test:** `from_yaml(to_yaml(p))` yields identical configs AND identical outputs under fixed seed; unknown `type` ⇒ actionable registry error listing known keys.

---

## 14. Iteration protocol

`experiments/vN/` = one iteration: frozen `config.yaml` copy, `run.ipynb` (or `analysis.ipynb`), `figures/*.png`, `metrics.json`, `report.md` (findings + decision + next-iteration hypotheses). CLI writes everything except the notebook narrative. **v1 experiment content:** toy-world 4-arm table (reproduce prototype), one jsonl mini-corpus run with mock LLM, figures: per-arm fidelity bars, τ with CI, ESS/coverage, gate reject-reason breakdown. Decision gates to open v2 are the research triggers in §16. Do NOT pre-create v2/v3 dirs.

---

## 15. v1 acceptance criteria (Definition of Done)

**15.1 Functional:** `uv sync && uv run ragsynth run --config configs/v1_toy.yaml` completes offline in < 10 min on CPU, producing `experiments/v1/{metrics.json, report.md, figures/, artifacts/}` with the 4-arm table; `ragsynth run --config configs/v1_local_corpus.yaml` works on the bundled 200-chunk sample jsonl with mocks; YAML round-trip byte-stable; same seed ⇒ identical `metrics.json` twice.

**15.2 Contract tests:** parametrized over the registry: every step serializes/deserializes/runs; every GateCheck emits a `Rejection` with reason; every adapter Protocol satisfied by its mock.

**15.3 Verification commands (green at every commit):**
```
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -q --cov=ragsynth --cov-fail-under=70
uv run ragsynth run --config configs/v1_toy.yaml
```

**15.4 Known-value metric fixtures (write these tests FIRST):** `tau_ap`: reference [0,1,2,3], candidate [1,0,2,3] ⇒ 2/3·(0+1/2+2/3+... ) — compute by hand in the test (expected 0.555…= 2/3·(0/1+1/2+2/2)/…; derive exactly, don't trust this note); identical rankings ⇒ 1.0; reversed ⇒ −1.0 (n=4). `rbo_ext`: identical ⇒ 1.0. `ess`: equal weights ⇒ N; one-hot ⇒ 1. `kl`: identical samples ⇒ ≈0. `c2st`: same-distribution split ⇒ AUC∈[0.45,0.55]. `paired_bootstrap`: injected +δ shift ⇒ p<0.05. `movmf`: recover 2 well-separated components' means within cos ≥ 0.99. `cluster_importance_weights`: hand example with an empty cluster ⇒ correct coverage_gap.

**15.5 Docs:** README quickstart; `ARCHITECTURE.md` (1 page: state flow diagram + extension recipe "add a new GateCheck in 20 lines"); every metric docstring cites its paper.

---

## 16. Roadmap — research propositions (each: hypothesis / method / meter / trigger)

- **R1 · z-sampling A2 at scale (text).** H: within-cluster fidelity ↑ (wC2ST↓, ESS↑) at equal τ_AP vs tuned A1. Method: `spec_first_generate` loop (kNN style+content around z, target-check `cos(emb(q),z) ≥ τ_t`, one revision — vec2text lineage, Morris et al. EMNLP 2023). Meter: within-cluster C2ST in a **held-out embedder** (never the planning space). Trigger: per-stratum KL/C2ST stays high while cluster quotas are matched, or ESS sags.
- **R2 · Closed-loop prompt optimization.** MIPROv2/TextGrad over `FidelityObjective`, reject-reason tallies route what to fix. Trigger: gate pass-rate and τ plateau. (Opsahl-Ong et al., EMNLP 2024.)
- **R3 · Dimension-tuple strata with demand-calibrated weights** (Hamel × our λ-mixture): interpretable persona×query_type cells weighted by p̂ instead of uniform cross-product; two-step tuple→query generation vs one-step (C2ST + human preference meter).
- **R4 · Style realism:** roughening post-processor (typos/keyword-ification; Penha et al., ECIR 2022), generator ensemble, persona conditioning (Ge et al. 2024). Trigger: discriminator separates on surface features.
- **R5 · Graded/pooled qrels:** UMBRELA judge + zoo pooling; κ-calibration vs ≥200 human labels; circularity guard (Fröbe SIGIR '25). Trigger: τ_AP stalls below gate while τ passes (unjudged holes).
- **R6 · Lifecycle under churn:** hash-keyed recycle/re-gate/regenerate on ingest events; drift monitors (MMD/C2ST weekly) + alert rule (drift AND anchor nDCG −3–5pts).
- **R7 · A3 arm — generator fine-tuned on (chunk, query) attribution pairs;** honest head-to-head vs A2 (cost, drift-refresh latency, multi-intent chunk coverage via ORCAS-style clicks).
- **R8 · Conversational strata:** CONVERSER-style few-shot dialogue generation, per-turn evidence, MTRAG protocols; `dialogue_context` activates.
- **R9 · Visual grounding (v3):** crop/overlay conditioning (VISA, Ma et al. 2024), CiteVQA masking-ablation as generation-time crucial-evidence filter, ColPali patch-to-region cross-check; ViDoRe V3 multi-evidence scheme.
- **R10 · Training split:** Gecko relabeling, NV-Retriever TopK-MarginPos negatives, document-level contamination firewall; extrinsic meter = fine-tune delta on REAL held-out queries only.
- **Benchmark validation experiments** (before trusting any arm on real data): oracle ceiling, per-stratum τ with bootstrap CI, positive-control battery, regeneration stability, human annotation-precision audit ≥90–95% (the noise floor), ~150–200 stratified records.

**Theory anchors to cite in docs (design rationale):** |E_p m − E_q m| ≤ TV ≤ √(KL/2) (Pinsker) ⇒ measured KL bounds headline bias for every system simultaneously; decisions with margin > 2·TV are preserved; IS sample cost ≍ e^KL (Chatterjee & Diaconis 2018) + ESS identity (Kong 1992) ⇒ match at the source, don't reweight after; post-stratification residual = Σ_c p̂_c TV(p(·|c), q(·|c)) ⇒ quota methods leave exactly the within-cluster term, which is what A2 targets and wC2ST measures.

---

## 17. Reference table (feature → primary sources)

| Feature | References |
|---|---|
| Representativeness protocol, steering, memorization audit | Chroma Generative Benchmarking, 2025 |
| Ranking-agreement validation, expectations | Rahmani et al., SIGIR 2024; SynDL 2025 |
| τ_AP / RBO | Yilmaz et al., SIGIR 2008; Webber et al., TOIS 2010 |
| Positive controls / paired bootstrap | Sakai, SIGIR 2006 |
| Round-trip & answer-first generation | Alberti et al., ACL 2019; Dai et al. (Promptagator), ICLR 2023; Bonifacio et al. (InPars), SIGIR 2022 |
| Filtering > generating more | Gospodinov et al. (Doc2Query--), ECIR 2023 |
| Zero-context / answerability / masking-ablation | CiteVQA, arXiv 2605.12882 |
| C2ST / MMD | Lopez-Paz & Oquab, ICLR 2017; Gretton et al., JMLR 2012 |
| movMF / vMF sampling | Banerjee et al., JMLR 2005; Wood 1994 |
| Coverage metrics | BCG arXiv 2510.00001; "Coverage, Not Averages" arXiv 2604.20763 |
| ESS / IS cost | Kong 1992; Chatterjee & Diaconis 2018 |
| Judge circularity / graded qrels | Fröbe et al., SIGIR 2025; UMBRELA 2024 |
| Prompt optimization | Opsahl-Ong et al. (MIPROv2), EMNLP 2024 |
| Practitioner generation tips (dimensions, seeding, filtering) | Hamel Husain & Shreya Shankar, evals FAQ / evals.info |
| Visual grounding (v3) | VISA 2024; ViDoRe V2/V3; ColPali 2024 |
| Conversational (v2/v3) | MTRAG, TACL 2025; CONVERSER 2023 |
| Training split (v3) | Gecko 2024; NV-Retriever 2024 |
| RAGOps framing | arXiv 2506.03401 |

---

## 18. Open questions — defaults for the agent (decide, document, proceed)

1. Package name `ragsynth` — keep unless collision on PyPI matters (it doesn't for v1, private).
2. Reference partition C: default 8, config-exposed; document migration-event caveat.
3. Real-corpus embedder default: `HashedNGramEmbedder`; `st` extra documented for real quality.
4. Figures: matplotlib only in core; notebook may use seaborn from `notebooks` extra.
5. Experiment tracking: none in v1 (mlflow noted as v2 option to match reference repo).
6. λ, τ_r percentile, τ_t: 0.7 / 5th pct / 0.6 — all config, defaults as stated.

## Appendix A — prototype → package mapping
`l2_normalize, sphere_uniform → sampling/vmf.py` · `sample_vmf, vmf_log_norm_const → sampling/vmf.py` · `MovMF → sampling/movmf.py` · `demand_from_responsibilities, tilt_weights, nn_cos_threshold, SpecSampler → sampling/{demand,spec_sampler}.py` · `cluster_importance_weights, effective_sample_size, post_stratified_estimate → metrics/efficiency.py` · `kl_similarity_distributions, c2st_auc, mmd_rbf, within_cluster_c2st → metrics/fidelity.py` · `tau_ap, rbo_ext, ranking_agreement, paired_bootstrap_pvalue → metrics/validity/` · `GenerationContext/spec_first_generate → steps/ + sampling/spec_sampler.py (A2 arm)` · `make_world, arms, zoo, run_demo → datasets/toy_world.py + arms/ + experiments/v1 notebook`.
