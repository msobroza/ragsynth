# ragsynth v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Parallel-safe tasks are marked; everything else is sequential.

**Goal:** Ship `ragsynth` v1 — a composable, serializable pipeline that generates gate-verified synthetic queries + qrels from corpus chunks and validates them against production-query distributions (fidelity / efficiency / validity metrics, 4-arm harness), passing every §15 acceptance criterion of SPEC.md.

**Architecture:** sklearn-style `PipelineStep` chain over a typed `PipelineState`, with all concretes behind registries and Protocols (DIP), frozen pydantic domain objects, embeddings externalized to an `EmbeddingStore`, and a composition root (`pipeline/serialization.py`) that is the only place adapters/artifacts get constructed. The vendored prototype `reference/synth_query_eval.py` is the authoritative source for every statistical algorithm (Appendix A mapping); the package re-homes that logic behind the step/metric interfaces without changing the math.

**Tech Stack:** Python ≥3.11, uv + uv_build, numpy/scipy/scikit-learn, pydantic v2, PyYAML, typer+rich (CLI), joblib, matplotlib (figures), jinja2 (prompt templates). Dev: ruff (full ruleset), mypy strict, pytest+pytest-cov.

## Global Constraints

- `requires-python >= 3.11`; src layout; `uv_build` backend; MIT license; `py.typed`.
- Core runtime deps ONLY: `numpy, scipy, scikit-learn, pydantic>=2, pyyaml, typer, rich, joblib, matplotlib, jinja2` (all BSD/MIT/Apache — air-gap rule §1). Extras: `notebooks`, `optimization` (dspy, textgrad), `bm25` (bm25s), `st` (sentence-transformers); dev tooling in the `dev` dependency group.
- ruff line-length 100, full ruleset incl. `D` (Google convention); mypy `strict = true` on `src`; coverage ≥70% overall (`--cov-fail-under=70`), target ≥85% on `metrics/`, `sampling/`, `pipeline/`.
- No `utils.py`. No `print` outside `cli.py` (use `logging`). Every public function: Google docstring with Args/Returns/Raises; metrics cite their paper.
- Embeddings never inline in domain objects — `embedding_ref` into `EmbeddingStore` only.
- All randomness flows from the single config seed via `Resources.rng(name)` substreams. Same seed ⇒ byte-identical `metrics.json`.
- §15.3 commands green at every commit:
  `uv run ruff check . && uv run ruff format --check .` · `uv run mypy src` · `uv run pytest -q --cov=ragsynth --cov-fail-under=70` · `uv run ragsynth run --config configs/v1_toy.yaml`
- v1 non-goals (§2.2) stay out: no visual conditioning, no conversational generation, no graded qrels, no optimizer execution, no training export, no lifecycle, no DB/UI. Stubs only where §2.2/§11 name them.

---

## 1. Decisions log (open questions §18 + interpretation calls)

| # | Decision | Rationale |
|---|---|---|
| D1 | Package name `ragsynth`, version `0.1.0`, private — keep. | §18.1 default. |
| D2 | Reference partition `C=8` default, config key `resources.partition.n_clusters`; docstring documents the migration-event caveat. | §18.2 default. |
| D3 | Real-corpus default embedder `hashed_ngram` (pure numpy); `st` extra documented in README. | §18.3 default. |
| D4 | matplotlib only in core; seaborn allowed in `notebooks` extra only. | §18.4 default. |
| D5 | No experiment tracking in v1; mlflow noted in README roadmap. | §18.5 default. |
| D6 | `lam=0.7`, `tau_r` = 5th pct NN-cosine, `tau_t=0.6` — all config-exposed defaults. | §18.6 default. |
| D7 | **Prototype present** (pushed by Max, commit 59c2ffe): vendored unmodified at `reference/synth_query_eval.py`. All algorithm ports cite prototype line ranges; the math is not re-derived. | §0.6. |
| D8 | **jinja2 added to core deps.** §5's dep list omits it but §6.3 mandates jinja2 templates; healthbench (the convention source) ships jinja2 in core. BSD-3, air-gap safe. | Conflict resolution §5 vs §6.3. |
| D9 | **Validator runs the comparison arms.** `validator` params `arms: [a0, a1, a2, oracle]` cause it to build+run each arm's generation sub-pipeline (via `arms/` presets, shared `Resources`) and compute the full metric block per arm ⇒ the 4-arm table lives in one `EvalReport`. Param `reuse_pipeline_for: <arm>` (v1_toy: `a1`) makes the named arm reuse the outer pipeline's accepted records instead of regenerating — no double generation, no magic. | §2.1(3), §6.7, §10, §13 reconciled. |
| D10 | **Three-way production-query split** train/anchor/oracle = 0.60/0.25/0.15 (prototype L698-700). Demand map, partition, exemplars fit on *train only*; anchor is the validation reference; oracle arm draws from the oracle split. No leakage. | Prototype; Chroma protocol. |
| D11 | **Toy adapters are explicit registry types** `toy_chat`, `toy_judge`, `passthrough` (bank-lookup embedder) in `configs/v1_toy.yaml`, not the generic `mock` shown in §13's illustrative YAML. The generic `MockChatModel`/`MockJudge` (hash-seeded, text-only) serve `v1_local_corpus.yaml` and CI. Explicit beats implicit re-mapping. | §13 example vs §12 semantics. |
| D12 | **Toy world emission model** = prototype `_emit` (L715-723): per-arm `(style, noise)` magnitudes A0=(0.45,0.55), A1=(0.15,0.68), A2=(0.05,0.15) simulate prompting-quality differences. Arm presets pass an optional `llm_override` config block; the arm builder swaps `resources.generator_llm` via `Resources.with_overrides`. | §10 "reproduce the qualitative demo table". |
| D13 | Dedup uses greedy accept-in-order + cosine-threshold rejection vs the already-accepted set (documented simplification of MMR; Carbonell & Goldstein cited). Exact-string dedup first. | §6.4(1); prototype has no MMR either. |
| D14 | `metrics.json` contains no wall-clock values (timestamps live in `report.md` provenance + record `created_at` only) so the determinism criterion is byte-exact. | §15.1. |
| D15 | Zoo for text corpora = same matrix-distortion family as toy (`make_system_zoo`, prototype L779-790) applied in the corpus embedding space — systems are embedding-space transforms, uniform across datasets. | §10; keeps zoo dataset-agnostic. |
| D16 | Per-query retrieval score = nDCG@k generalized to binary multi-gold qrels (DCG over relevant retrieved / IDCG); reduces to prototype's single-gold `1/log2(1+rank)` when one gold. | Prototype `score_system` L793-814 + §6.5 multi-gold reality. |
| D17 | `anchor_qrels` absent (both bundled datasets): gold = nearest-chunk relabel (prototype `_nearest_chunk`), the gate-style relabeling §10 names. | §10. |
| D18 | Sample corpus: `data/sample/{chunks,queries}.jsonl` (200 chunks / 120 queries), committed AND regenerable via `python -m ragsynth.datasets.sample_corpus` (deterministic, seed 0). | §15.1 "bundled". |
| D19 | Composition root = `pipeline/serialization.py` (`load_config`, `build_resources`, `build_pipeline`, `config_hash`); `Pipeline.from_yaml` delegates. Partition + movMF demand are fit-or-loaded there each run (deterministic), persisted via `ArtifactStore` with sha256 manifest = the "versioned frozen artifact". | §3.1, §7.2/7.4, §13. |
| D20 | Optimizer module mirrors healthbench: `require_optional(module, feature, extra)` helper lives in `ragsynth/optional_deps.py` (no `utils.py` rule). | §3.3, §11. |

---

## 2. File map (create; responsibilities)

```
pyproject.toml, .python-version(3.11), .gitignore, README.md, ARCHITECTURE.md, LICENSE(MIT)
configs/v1_toy.yaml            # §13 schema; quota pipeline + validator arms
configs/v1_local_corpus.yaml   # jsonl dataset + hashed embedder + mock LLM/judge
data/sample/{chunks.jsonl,queries.jsonl}
src/ragsynth/__init__.py       # __version__; nothing else
src/ragsynth/py.typed
src/ragsynth/optional_deps.py  # require_optional()
src/ragsynth/domain/{__init__,chunk,query,stratum,seed,context,candidate,annotation,conversation,rejection,report}.py
src/ragsynth/pipeline/{__init__,base,registry,pipeline,serialization}.py
src/ragsynth/io/{__init__,embeddings,artifacts}.py
src/ragsynth/adapters/llm/{__init__,base,mock,openai_compatible}.py
src/ragsynth/adapters/embedder/{__init__,base,mock,hashed,st}.py
src/ragsynth/adapters/retriever/{__init__,base,dense_inmemory,bm25s}.py
src/ragsynth/adapters/judge/{__init__,base,mock,llm_judge}.py
src/ragsynth/sampling/{__init__,vmf,movmf,demand,partition,spec_sampler}.py
src/ragsynth/metrics/{__init__,fidelity,efficiency,diversity}.py
src/ragsynth/metrics/validity/{__init__,agreement,controls,systems}.py
src/ragsynth/gate/__init__.py
src/ragsynth/gate/checks/{__init__,base,dedup,zero_context,answerability,round_trip,uniqueness}.py
src/ragsynth/steps/{__init__,seed_sampler,context_assembler,generator,gate,qrel_builder,curator,validator}.py
src/ragsynth/steps/prompts/{answer_first_v1.j2,revise_v1.j2,judge_v1.j2}
src/ragsynth/arms/{__init__,base,a0_naive,a1_quota,a2_spec,oracle}.py
src/ragsynth/datasets/{__init__,base,toy_world,jsonl_loader,sample_corpus}.py
src/ragsynth/optimization/{__init__,base,noop,objectives,mipro_adapter}.py
src/ragsynth/cli.py
experiments/v1/{config.yaml,analysis.ipynb,report.md,metrics.json,figures/,artifacts/}   # outputs, Phase 4
tests/  # mirrors src/: tests/domain/, tests/pipeline/ (incl. test_contract.py), tests/io/,
        # tests/adapters/, tests/sampling/, tests/metrics/, tests/gate/, tests/steps/,
        # tests/arms/, tests/datasets/, tests/optimization/, tests/test_cli.py,
        # tests/e2e/{test_toy_world_table.py,test_local_corpus.py,test_determinism.py}
        # tests/conftest.py: small-world fixture, tmp artifacts dir, tiny configs
```

`metrics/validity/systems.py` is the one addition beyond §5's inventory: `RetrievalSystem` Protocol + `MatrixSystem` + `make_system_zoo` + `ndcg_at_k` + `evaluate_zoo` (the prototype's "System zoo and retrieval scoring" section needs a package home; controls.py holds only degradations + p-value per §8-9).

---

## 3. Shared interface contracts

Every task codes against these signatures verbatim. Cross-module drift = bug.

```python
# ---- pipeline/base.py -------------------------------------------------------
@dataclass(frozen=True)
class DemandArtifact:
    p_hat: np.ndarray                 # (C,) demand over the reference partition (hard labels)
    movmf: MovMF                      # fitted planning density (K components, K != C allowed)
    movmf_demand: np.ndarray          # (K,) demand_from_responsibilities over train queries
    tilted: np.ndarray                # (K,) tilt_weights(movmf_demand, lam)
    tau_r: float                      # on-manifold guard threshold
    lam: float

@dataclass(frozen=True)
class Resources:
    chunks: tuple[Chunk, ...]
    queries_train: tuple[ProductionQuery, ...]
    queries_anchor: tuple[ProductionQuery, ...]
    queries_oracle: tuple[ProductionQuery, ...]
    anchor_qrels: Mapping[str, Mapping[str, int]]   # query_id -> {chunk_id: grade}
    oracle_qrels: Mapping[str, Mapping[str, int]]
    embedder: Embedder
    generator_llm: ChatModel
    judge: RelevanceJudge
    retriever: Retriever
    embeddings: EmbeddingStore        # pre-populated: every chunk_id + query_id
    partition: ReferencePartition
    demand: DemandArtifact
    zoo: Mapping[str, RetrievalSystem]
    artifacts: ArtifactStore
    seed: int
    def rng(self, name: str) -> np.random.Generator      # default_rng([seed, sha256-int(name)])
    def with_overrides(self, **kwargs: Any) -> Resources  # dataclasses.replace wrapper
    def chunk_embs(self) -> np.ndarray                    # (N_chunks, d), row i = chunks[i]
    def query_embs(self, which: str) -> np.ndarray        # "train" | "anchor" | "oracle"

class PipelineState(BaseModel):       # mutable; model_config = ConfigDict(arbitrary_types_allowed=True)
    seeds: list[Seed] = []
    contexts: list[GenerationContext] = []
    candidates: list[SyntheticQuery] = []
    gate_accepted: list[SyntheticQuery] = []
    accepted: list[AnnotationRecord] = []
    rejected: list[Rejection] = []
    metrics: dict[str, Any] = {}
    provenance: dict[str, Any] = {}   # config_hash, seed, benchmark_version, prompt_version

class PipelineStep(ABC):              # exactly as SPEC §3.1: name, version, fit, run, to_config, from_config
```

```python
# ---- pipeline/registry.py ---------------------------------------------------
class RegistryError(KeyError): ...            # message lists known keys
class Registry(Generic[T]):
    def __init__(self, kind: str) -> None
    def register(self, key: str) -> Callable[[type[T]], type[T]]
    def get(self, key: str) -> type[T]        # raises RegistryError("unknown <kind> 'x'; known: [...]")
    def keys(self) -> list[str]
STEPS: Registry[PipelineStep]; CHECKS: Registry[GateCheck]
CHAT_MODELS, EMBEDDERS, RETRIEVERS, JUDGES, DATASETS, ARMS, OPTIMIZERS  # Registry instances
# Concretes self-register via decorator at import; ragsynth/steps/__init__.py etc. import all concretes.
```

```python
# ---- adapters (Protocols; each in adapters/<kind>/base.py) ------------------
class ChatModel(Protocol):
    def complete(self, system: str, user: str, **kwargs: Any) -> str: ...
class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...      # (n, d), L2-normalized
class Retriever(Protocol):
    def search(self, query_emb: np.ndarray, k: int) -> list[tuple[str, float]]: ...
@dataclass(frozen=True)
class JudgeVerdict: answerable: bool; answer: str; confidence: float
class RelevanceJudge(Protocol):
    def judge(self, query: str, evidence_texts: Sequence[str]) -> JudgeVerdict: ...
# Adapter factory contract (used by composition root):
# CLASS.from_config(params: dict[str, Any], bundle: DatasetBundle, rng: Generator) -> Self
```

```python
# ---- sampling (signatures = prototype, + artifact IO) -----------------------
l2_normalize(x, axis=-1, eps=1e-12) -> np.ndarray                       # proto L67
sphere_uniform(n, d, rng) -> np.ndarray                                 # proto L72
sample_vmf(mu, kappa, n, rng) -> np.ndarray                             # proto L87 (Wood 1994)
vmf_log_norm_const(d, kappa) -> np.ndarray                              # proto L124
class MovMF:                                                            # proto L145 + npz IO
    fit(x) -> Self; responsibilities(x) -> np.ndarray; log_prob(x) -> np.ndarray
    sample(n, rng, weights=None) -> tuple[np.ndarray, np.ndarray]
    to_artifact(path: Path) -> None; from_artifact(path: Path) -> MovMF   # npz + fitted_on_hash
demand_from_responsibilities(resp, timestamps=None, half_life=None) -> np.ndarray  # proto L252
tilt_weights(p_hat, lam) -> np.ndarray                                  # proto L270
nn_cos_threshold(prod_emb, pct=5.0) -> float                            # proto L277
class SpecSampler:                                                      # proto L289
    sample(n, rng) -> tuple[np.ndarray, np.ndarray]                     # (z (n,d), component ids)
class ReferencePartition:                                               # partition.py (new)
    @classmethod fit(query_embs, n_clusters=8, seed=0) -> Self          # KMeans(n_init=10, random_state=seed)
    assign(embs) -> np.ndarray; proportions(embs) -> np.ndarray         # p_hat = proportions(train)
    n_clusters: int; centers: np.ndarray; fitted_on_hash: str
    to_artifact(path) / from_artifact(path)
```

```python
# ---- metrics (signatures = prototype where it exists) ------------------------
# fidelity.py
kl_similarity_distributions(real_q, synth_q, chunk_emb, bins=50, eps=1e-6) -> float     # proto L375
c2st_auc(x_real, x_synth, n_splits=5, seed=0) -> float                                  # proto L395
c2st_auc_with_coefs(x_real, x_synth, seed=0) -> tuple[float, np.ndarray]                # §8 diagnostic
mmd_rbf(x, y, gamma=None, max_n=2000, seed=0) -> float                                  # proto L410
within_cluster_c2st(x_real, x_synth, labels_real, labels_synth,
                    min_per_side=30, seed=0) -> tuple[float, dict[int, float]]          # proto L439
# efficiency.py
cluster_importance_weights(labels_synth, p_hat) -> tuple[np.ndarray, float]             # proto L329
effective_sample_size(weights) -> float                                                 # proto L350
post_stratified_estimate(per_query_metric, labels, p_hat) -> float                      # proto L356
demand_weighted_coverage(labels_synth, p_hat) -> float           # sum p_hat[c] over covered c
zero_query_clusters(labels_synth, n_clusters) -> list[int]
minimum_semantic_coverage(labels_synth, p_hat, floor: int) -> float  # frac of clusters with >= floor samples, demand-weighted
# diversity.py
distinct_n(texts, n=1) -> float                                  # unique n-grams / total n-grams
semantic_dedup_rate(embs, threshold=0.95) -> float               # frac removed by greedy cos-dedup
# validity/agreement.py
system_ranking(mean_scores) -> list[int]                                                # proto L470
tau_ap(reference, candidate) -> float                                                   # proto L475
rbo_ext(s, t, p=0.9) -> float                                                           # proto L490
@dataclass(frozen=True) class RankingAgreement: tau, tau_ap_, rbo, tau_ci_low, tau_ci_high
ranking_agreement(anchor_scores, arm_scores, n_boot=1000, seed=0) -> RankingAgreement   # proto L521
# validity/controls.py
paired_bootstrap_pvalue(scores_base, scores_degraded, n_boot=2000, seed=0)
    -> tuple[float, float]                                                              # proto L543
drop_index_mask(n_chunks, frac, rng) -> np.ndarray               # bool mask
noise_transform(d, sigma, rng) -> np.ndarray                     # I + sigma*G/sqrt(d)
truncate_topk(scores_matrix_fn, k) ...                           # degradation via smaller k
# validity/systems.py
class RetrievalSystem(Protocol):
    def per_query_scores(self, query_embs: np.ndarray, qrels: Sequence[Mapping[str, int]],
                         k: int = 10, drop_mask: np.ndarray | None = None) -> np.ndarray: ...
@dataclass(frozen=True) class MatrixSystem:                      # matrix (d,d), chunk_ids, chunk_embs
    ...implements RetrievalSystem via ndcg_at_k                  # proto score_system L793 generalized (D16)
make_system_zoo(chunk_ids, chunk_embs, seed=0) -> dict[str, MatrixSystem]               # proto L779
evaluate_zoo(zoo, query_embs, qrels, k=10) -> np.ndarray         # (S, Q), insertion order  # proto L817
```

```python
# ---- gate/checks/base.py ----------------------------------------------------
@dataclass(frozen=True)
class CheckResult: passed: bool; score: float | None; reason: str; promoted: tuple[str, ...] = ()
class GateCheck(ABC):
    name: ClassVar[str]
    @abstractmethod def check(self, candidate: SyntheticQuery, state: PipelineState,
                              resources: Resources) -> CheckResult
    def to_config(self) -> dict[str, Any]; @classmethod def from_config(cls, config, resources) -> Self
# uniqueness promotions ride CheckResult.promoted -> qrel_builder adds them at grade 1.
```

```python
# ---- datasets/base.py --------------------------------------------------------
@dataclass(frozen=True)
class DatasetBundle:
    chunks: tuple[Chunk, ...]
    queries_train: tuple[ProductionQuery, ...]; queries_anchor: tuple[...]; queries_oracle: tuple[...]
    anchor_qrels: dict[str, dict[str, int]]; oracle_qrels: dict[str, dict[str, int]]
    embeddings: EmbeddingStore                       # pre-filled for toy; empty for text corpora
    bank: ToyEmbeddingBank | None                    # toy only: shared mutable text->vec bank
# DATASETS registry entries: build(params: dict, seed: int) -> DatasetBundle
```

```python
# ---- arms/base.py -------------------------------------------------------------
# ARMS registry entries are builder callables wrapped in a tiny class:
class ArmPreset(ABC):
    name: ClassVar[str]
    @abstractmethod def build_steps(self, resources: Resources, params: dict[str, Any])
        -> tuple[Resources, list[PipelineStep]]      # may override generator_llm (D12)
run_arm(name: str, resources: Resources, params: dict[str, Any]) -> list[AnnotationRecord]
# oracle.py: OraclePreset returns records built directly from queries_oracle + oracle_qrels
#            (no generation): AnnotationRecord with gen-free gen_meta {"arm": "oracle"}.
```

```python
# ---- io ----------------------------------------------------------------------
class EmbeddingStore:                # io/embeddings.py; float32, L2-normalized rows
    add(ids: Sequence[str], vectors: np.ndarray) -> None      # overwrite = error
    get(ids: Sequence[str]) -> np.ndarray
    __contains__(id: str) -> bool;  __len__() -> int;  dim: int | None
    save(path: Path) -> None;  @classmethod load(path: Path) -> EmbeddingStore   # npz(ids, matrix)
class ArtifactStore:                 # io/artifacts.py
    __init__(root: Path);  path_for(name: str) -> Path
    save_npz(name, **arrays) / save_json(name, obj) / save_joblib(name, obj)
    load_npz/load_json/load_joblib;  manifest: dict[str, str]  # name -> sha256, persisted manifest.json
sha256_hex(data: bytes) -> str; canonical_json(obj) -> str     # sorted keys, no whitespace drift
```

```python
# ---- steps: constructor params == config params (to_config round-trips them) --
UniformSeedSampler(n_seeds: int, p_group: float = 0.0, strata: list[str] = ["factoid","howto","keyword"])
QuotaSeedSampler(n_seeds: int, lam: float = 0.7, n_min: int = 3, p_group: float = 0.2, strata=[...])
SpecSeedSampler(n_seeds: int, n_chunks_per_seed: int = 5, strata=[...])       # z + kNN chunks
ContextAssembler(k_style: int = 3, two_step: bool = False, blind_summary: bool = False)  # flags stubbed off (§6.2)
QueryGenerator(n_candidates: int = 3, prompt_version: str = "answer_first_v1",
               tau_t: float = 0.6, max_revisions: int = 1)                    # target-check only when seed.z
VerificationGate(checks: list[str], **check_params: dict[str, Any])           # §13 shape: gate params nest per-check dicts
QrelBuilder(strategy: str = "binary")                                         # QrelStrategy ABC + BinaryQrels
Curator(memorization_cos: float = 0.9, target_mix: dict[str, float] | None = None,
        max_records: int | None = None)
Validator(arms: list[str], n_boot: int = 1000, gates: dict[str, float] = {"tau": 0.9, "tau_ap": 0.8},
          k: int = 10, n_per_arm: int = 500, reuse_pipeline_for: str | None = None,
          arm_params: dict[str, dict[str, Any]] = {})
```

```python
# ---- optimization/base.py (mirror healthbench optimizer.py) -------------------
class OptimizationMetric(Protocol): def __call__(self, prompt: str) -> float: ...
@dataclass(frozen=True) class TrialRecord: trial_id: int; prompt: str; score: float | None; timestamp: str
@dataclass(frozen=True) class OptimizationResult: optimized_prompt; baseline_score; optimized_score;
    improvement; num_trials; trial_history: list[TrialRecord]; optimizer_name: str; config: dict[str, Any]
class BasePromptOptimizer(ABC):
    def __init__(self, config: BaseOptimizerConfig) -> None
    @abstractmethod def optimize(self, current_prompt: str, samples: Sequence[Any] | None,
                                 metric: OptimizationMetric | None) -> OptimizationResult
# noop.py: NoOpOptimizer (1 trial = baseline). objectives.py: FidelityObjective(resources, base_steps, ...)
#   __call__(prompt) -> -(kl + alpha*max(0, c2st-0.5)); hard -1e9 if gate pass-rate < min_pass_rate.
# mipro_adapter.py: MIPROv2Optimizer stub -> require_optional(dspy, "MIPROv2Optimizer", "optimization").
```

**Config schema (§13, normative):** top-level keys `ragsynth {schema_version:1, name, seed}`, `resources {dataset, embedder, generator_llm, judge_llm, retriever, partition?, demand?}`, `artifacts_dir`, `pipeline: [{type, params}]`. `load_config` validates: schema_version==1, unknown `type` ⇒ `RegistryError` listing known keys, WARN (logging) when `judge_llm` and `generator_llm` resolve to the same model family (§6.4). `config_hash = sha256_hex(canonical_json(config))`. `Pipeline.to_yaml` = `yaml.safe_dump(config, sort_keys=True)` ⇒ byte-stable round-trip.

---

## 4. Execution strategy

- **Phase 0-1 (sequential, this session):** bootstrap + foundation (domain, pipeline core, io, protocols+mocks). Highest-coherence code; single author.
- **Phase 2 (PARALLEL fan-out, disjoint paths):** T2.1 sampling · T2.2 fidelity+diversity · T2.3 efficiency · T2.4 validity · T2.5 concrete adapters. Each agent gets: this PLAN (contracts §3), the prototype, its file list; must pass scoped `ruff`/`mypy`/`pytest` before returning. No agent touches another's paths or shared files.
- **Phase 3 (sequential):** gate checks → steps → arms/datasets → validator/report → optimization → CLI/configs. Tightly coupled (§0.1 rule: no fan-out here).
- **Phase 4 (sequential, iterative):** e2e toy table tuning, local-corpus run, determinism/round-trip, coverage top-up, docs, experiments/v1.
- **Phase 5:** full §15 acceptance sweep + adversarial review (formula-vs-paper verification fan-out) + fixes.
- Commit at the end of every task (conventional commits). §15.3 must be green at each commit from T0.2 onward (pytest green trivially before tests exist).

---

## 5. Tasks

### Phase 0 — Bootstrap

#### Task 0.1: Repo scaffolding & toolchain
**Files:** `pyproject.toml`, `.python-version`, `.gitignore`, `LICENSE`, `src/ragsynth/__init__.py`, `src/ragsynth/py.typed`, `tests/__init__.py` (absent — use pytest rootdir config instead), placeholder `README.md`.
- [ ] Write `pyproject.toml`: project metadata (D1), core deps (Global Constraints), extras `notebooks/optimization/bm25/st`, dev group (`pytest`, `pytest-cov`, `ruff`, `mypy`, `scipy-stubs`, `types-pyyaml`), `[project.scripts] ragsynth = "ragsynth.cli:app"`, uv_build backend, ruff config (line 100, `select=["ALL"]`, google docstrings, documented ignores: `COM812`,`ISC001` (formatter), `D203`,`D213` (convention pair), `ANN401`, `PLR0913`, `TC001-003` pragmatic; per-file-ignores `tests/**`: `S101`,`PLR2004`,`D`,`ANN`,`SLF001`; `cli.py`: `T201` not needed—no prints, rich console), mypy strict + pydantic plugin + `ignore_missing_imports` overrides (`sklearn.*`, `joblib.*`, `dspy.*`, `textgrad.*`, `bm25s.*`, `sentence_transformers.*`), pytest config (`testpaths=["tests"]`).
- [ ] `src/ragsynth/__init__.py` with `__version__ = "0.1.0"` + module docstring; empty `py.typed`.
- [ ] `uv sync` → lockfile; `uv run python -c "import ragsynth"` works.
- [ ] Run all four §15.3 commands (pytest passes with no tests via `--cov-fail-under=0`? NO — keep 70: add one smoke test `tests/test_version.py::test_version`); `ragsynth run` will fail until T3.8 — acceptable gap, note in commit; the §15.3-at-every-commit rule binds fully once the CLI exists.
- [ ] Commit `chore: bootstrap uv project, toolchain, strict lint/type config`.

#### Task 0.2: optional_deps + io
**Files:** `src/ragsynth/optional_deps.py`, `src/ragsynth/io/{__init__,embeddings,artifacts}.py`; tests `tests/io/test_embeddings.py`, `tests/io/test_artifacts.py`, `tests/test_optional_deps.py`.
**Interfaces:** §3 io block. **Produces:** `EmbeddingStore`, `ArtifactStore`, `sha256_hex`, `canonical_json`, `require_optional`.
- [ ] Failing tests: store add/get roundtrip preserves float32 + order; duplicate id ⇒ `ValueError`; missing id ⇒ `KeyError`; save/load npz identical; ArtifactStore manifest sha256 stable across two saves of same content; `canonical_json({"b":1,"a":[1.0]})` byte-stable; `require_optional(None, "X", "optimization")` raises ImportError containing `uv sync --extra optimization`.
- [ ] Implement; run; commit `feat(io): EmbeddingStore, ArtifactStore with sha256 manifest, optional-dep helper`.

### Phase 1 — Foundation

#### Task 1.1: Domain models
**Files:** `src/ragsynth/domain/*.py` (§2 map); test `tests/domain/test_models.py`.
**Interfaces:** SPEC §4 verbatim, all `model_config = ConfigDict(frozen=True)`. IDs: `chunk_id = sha256_hex(f"{norm(text)}|{doc_id}|{page}|{bbox}".encode())[:16]`, `content_hash = sha256_hex(norm(text).encode())[:16]`, `norm(t) = " ".join(t.split())` — provide `Chunk.create(text, doc_id, page=None, ...)` factory. `Stratum.key()` = `"|".join(f"{k}={v}" for k, v in sorted(dims.items()))`.
- [ ] Failing tests: frozen-ness (assignment raises), `Stratum.key` ordering (`{"b":"2","a":"1"}` ⇒ `"a=1|b=2"`), `Chunk.create` id determinism + 16-hex shape, `AnnotationRecord` JSON round-trip via pydantic, `Turn` role literal validation.
- [ ] Implement all 11 modules + `domain/__init__.py` re-exports; commit `feat(domain): frozen pydantic v2 domain model (conversation- and visual-ready fields)`.

#### Task 1.2: Registry + pipeline base
**Files:** `src/ragsynth/pipeline/{registry,base}.py`; tests `tests/pipeline/test_registry.py`, `tests/pipeline/test_base.py`.
**Interfaces:** §3 blocks. **Produces:** `Registry`, all registry instances, `Resources`, `DemandArtifact`, `PipelineState`, `PipelineStep`.
- [ ] Failing tests: register/get/duplicate-key error; unknown key error message contains `known` + every registered key; `Resources.rng("a")` deterministic across instances and ≠ `rng("b")`; `with_overrides` swaps one field, keeps rest; `PipelineState` field defaults independent between instances.
- [ ] Implement; note `Resources` uses `TYPE_CHECKING` imports + string annotations to avoid cycles (protocols imported from adapters bases; sampling imported lazily for types only).
- [ ] Commit `feat(pipeline): registries, Resources, PipelineState, PipelineStep ABC`.

#### Task 1.3: Adapter Protocols + mocks
**Files:** `src/ragsynth/adapters/{llm,embedder,retriever,judge}/{__init__,base,mock}.py`, `src/ragsynth/adapters/retriever/dense_inmemory.py`; tests `tests/adapters/test_mocks.py`, `tests/adapters/test_dense_retriever.py`.
**Produces:** the four Protocols + `JudgeVerdict`; `MockChatModel(seed=0)` (hash-seeded deterministic template: same (system,user) ⇒ same output); `MockEmbedder(dim=32)` (hash-of-text → rng vector, L2); `MockJudge(answerable=True, confidence=1.0)` + configurable rules; `DenseInMemoryRetriever(chunk_ids, matrix)` exact top-k via argpartition.
- [ ] Failing tests: mock determinism (two instances, same inputs ⇒ same outputs); `MockEmbedder` rows unit-norm; retriever returns exact top-k (hand fixture: 4 vectors, known cosines, k=2); Protocol conformance via `isinstance` checks with `runtime_checkable` OFF — assert structurally by passing mocks where Protocol is annotated (mypy covers it) + a `tests/pipeline/test_contract.py` parametrized adapter-satisfies-mock check (§15.2).
- [ ] Commit `feat(adapters): protocols, deterministic mocks, dense in-memory retriever`.

#### Task 1.4: Pipeline runner + serialization (composition root)
**Files:** `src/ragsynth/pipeline/{pipeline,serialization}.py`; tests `tests/pipeline/test_pipeline.py`, `tests/pipeline/test_serialization.py`.
**Consumes:** T1.1-1.3. **Produces:** `Pipeline(steps)` with `fit/run/named_steps/get_params/to_yaml`; `load_config(path) -> dict`, `config_hash(config) -> str`, `build_resources(config) -> Resources`, `build_pipeline(config, resources) -> Pipeline`, `Pipeline.from_yaml(path) -> tuple[Pipeline, Resources]`.
- [ ] Failing tests with two toy `PipelineStep` fakes registered under test-only keys: run threads state; named_steps; `get_params()` flat `step__param`; to_yaml→from_yaml→to_yaml byte-identical; unknown step type raises `RegistryError` naming known keys; cross-family judge/generator warning emitted via `caplog` (§6.4).
- [ ] `build_resources`: dataset via `DATASETS` → embed any missing chunk/query vectors with configured embedder → split already in bundle (D10) → fit-or-load `ReferencePartition` + movMF `DemandArtifact` (keyed on `sha256(train query ids)+params`, stored via ArtifactStore) → build zoo (D15) → assemble frozen `Resources`. *Partition/demand imports come from `sampling/` (T2.1); to keep Phase 1 self-contained, `build_resources` lands in this task but its test that exercises partition/demand is marked `pytest.importorskip`-free and simply deferred to T2.1's completion — write it now, expect green after T2.1 merges (documented in test docstring). CI stays green because Phase 2 merges before Phase 3 uses it.*
- [ ] Commit `feat(pipeline): runner + YAML composition root with artifact-backed partition/demand`.

### Phase 2 — PARALLEL fan-out (independent modules; contracts in §3 are law)

#### Task 2.1: `sampling/` (agent A)
**Files:** `src/ragsynth/sampling/*.py`; tests `tests/sampling/test_{vmf,movmf,demand,partition,spec_sampler}.py`.
**Port:** prototype L62-321 per Appendix A; add artifact IO + `ReferencePartition`.
- [ ] Known-value/property tests FIRST (§15.4): vMF mean resultant direction → μ as κ grows (`cos(mean_dir(sample_vmf(mu, 200, 2000)), mu) > 0.99`; κ=0 ⇒ ~uniform, mean resultant length < 0.1); `vmf_log_norm_const` finite for κ ∈ {1e-9, 1, 1e5}, d ∈ {2, 64}; **movMF recovers 2 well-separated components** (μ₁·μ₂ ≈ −1, κ=100, n=1000/side): fitted means match within cos ≥ 0.99, weights ≈ [.5,.5] ± .05; movMF `to_artifact/from_artifact` round-trip: identical params + `log_prob`; `demand_from_responsibilities` with half-life: newer queries dominate (hand fixture, 2 components, 2 timestamps); `tilt_weights([1,0], lam=.7) == [.85,.15]`; `nn_cos_threshold` on a hand grid; `SpecSampler` all-rejected ⇒ RuntimeError, guard keeps only on-manifold z (fixture: 2 clusters, τ_r between); `ReferencePartition` deterministic assign, artifact round-trip, `proportions` sums to 1.
- [ ] Implement (`MovMF` becomes a plain class w/ explicit `__init__`, not field-default dataclass — mypy strict); run scoped `ruff/mypy/pytest`; commit `feat(sampling): vMF/movMF/demand/partition/spec-sampler with artifact IO`.

#### Task 2.2: `metrics/fidelity.py` + `metrics/diversity.py` (agent B)
**Files + tests:** `tests/metrics/test_fidelity.py`, `tests/metrics/test_diversity.py`. **Port:** proto L375-462.
- [ ] Tests FIRST: `kl` identical samples ⇒ < 0.02 (§15.4 "≈0" with ε-smoothing); disjoint similarity profiles ⇒ > 1; `c2st_auc` same-distribution split (one gaussian cloud halved, n=400, d=8) ⇒ AUC ∈ [0.45, 0.55]; shifted clouds ⇒ > 0.9; `c2st_auc_with_coefs` returns d coefficients, top-coef dim = the shifted dim; `mmd_rbf(x, x) == 0` exactly (unbiased, same set) and small for iid halves, large for shifted; `within_cluster_c2st` skips clusters with < min_per_side (fixture asserts `skipped` behavior: cluster absent from dict), mean over included only, `{}` ⇒ NaN mean; `distinct_n(["a b", "a b"], 1) == 0.5`; `semantic_dedup_rate` hand fixture (3 vectors, 2 identical, thr .95 ⇒ 1/3).
- [ ] Implement; scoped checks; commit `feat(metrics): fidelity (KL/C2ST/MMD/within-cluster) + light diversity`.

#### Task 2.3: `metrics/efficiency.py` (agent C)
**Files + tests:** `tests/metrics/test_efficiency.py`. **Port:** proto L329-367 + 3 new functions (§3 contract).
- [ ] Tests FIRST (§15.4): `effective_sample_size(np.ones(7)) == 7`; one-hot ⇒ 1; **`cluster_importance_weights` hand example w/ empty cluster:** `p_hat=[.5,.3,.2]`, `labels_synth=[0,0,1,1]` (cluster 2 empty) ⇒ `coverage_gap=0.2`, renormalized `p_cov=[.625,.375,0]`, `q=[.5,.5]` ⇒ per-sample weights `[1.25,1.25,.75,.75]`, ESS/N = (4²)/(2·1.25²+2·0.75²)/4 = 16/(3.125+1.125)/4 = 0.941…— assert to 1e-9; `post_stratified_estimate` hand fixture incl. renormalization over covered mass; `demand_weighted_coverage` same fixture ⇒ 0.8; `zero_query_clusters` ⇒ [2]; `minimum_semantic_coverage` floor logic hand fixture.
- [ ] Implement; scoped checks; commit `feat(metrics): efficiency layer (IS weights, ESS, post-stratification, coverage)`.

#### Task 2.4: `metrics/validity/` (agent D)
**Files + tests:** `tests/metrics/validity/test_{agreement,controls,systems}.py`. **Port:** proto L470-554 + L779-821.
- [ ] Tests FIRST (§15.4 — hand-derived): `tau_ap([0,1,2,3],[1,0,2,3])` — derivation: candidate positions (1-idx) i=2..4, C(2)=0 (item 0 has item 1 above it, reference orders 0 first ⇒ discordant), C(3)=2, C(4)=3 ⇒ τ_AP = (2/3)·(0/1 + 2/2 + 3/3) − 1 = **1/3** (the spec's 0.555 note is wrong; assert `pytest.approx(1/3)`); identical ⇒ 1.0; reversed (n=4) ⇒ −1.0; `rbo_ext` identical ⇒ 1.0, disjoint ⇒ 0.0, p→ weight sanity (`rbo_ext(x,y,p=.9) > rbo_ext(x,y_tail_swapped,...)` top-weighted); `ranking_agreement` perfect-correlation fixture ⇒ tau=1, CI within [−1,1], `taus` bootstrap deterministic under seed; `paired_bootstrap_pvalue` injected +δ shift (base = degraded + 0.1) ⇒ p < 0.05, and null (identical) ⇒ p ≈ 1 side; `ndcg_at_k` single gold at rank r ⇒ 1/log2(1+r) (D16 reduction), multi-gold hand example, gold dropped ⇒ 0; `make_system_zoo` returns 12 deterministic systems, `exact` = identity scores; `evaluate_zoo` shape (S, Q) insertion-ordered; degradations: `drop_index_mask` frac exact count, `noise_transform` shape/determinism.
- [ ] Implement; scoped checks; commit `feat(metrics): validity layer (tau_ap/RBO/bootstrap agreement, controls, system zoo)`.

#### Task 2.5: Concrete adapters (agent E)
**Files:** `adapters/llm/openai_compatible.py`, `adapters/embedder/{hashed,st}.py`, `adapters/retriever/bm25s.py`, `adapters/judge/llm_judge.py` (+ registry registration of ALL adapters incl. Phase-1 mocks in each subpackage `__init__`); tests `tests/adapters/test_{hashed_embedder,openai_compatible,llm_judge,extras_stubs}.py`.
- [ ] Tests FIRST: `HashedNGramEmbedder(dim=256, ngram_range=(3,5))` — deterministic, unit rows, `encode(["abc"]) == encode(["abc"])`, different texts differ, empty-string safe; `OpenAICompatibleChat` — offline test via injected `httpx`-free transport: NO network in tests — implement over `urllib.request` with a `_post_json` seam monkeypatched in tests (asserts request shape: `{model, messages:[{role:system},{role:user}]}`, parses `choices[0].message.content`, api key from env var name, raises actionable error when env missing); `LLMJudge(chat, prompt_version="judge_v1")` parses strict-JSON verdict from its ChatModel, malformed ⇒ `JudgeVerdict(False, "", 0.0)` + warning log; `BM25sRetriever`/`SentenceTransformerEmbedder` import without extras ⇒ ImportError from `require_optional` (assert message), classes registered but lazily-imported.
- [ ] Implement; scoped checks; commit `feat(adapters): openai-compatible chat, hashed n-gram embedder, LLM judge, extras stubs`.

### Phase 3 — Steps, arms, datasets, validator, optimization, CLI (sequential)

#### Task 3.1: Gate checks
**Files:** `src/ragsynth/gate/checks/*.py`; tests `tests/gate/test_checks.py`.
**Consumes:** GateCheck ABC contract (§3), mocks (T1.3). Order/semantics = SPEC §6.4 exactly; every check registered in `CHECKS`.
- [ ] Tests FIRST (each check: pass fixture + reject fixture + reason string):
  `DedupCheck(cos_threshold=.95)` exact-text dup ⇒ reject "exact duplicate"; cos ≥ thr vs `state.gate_accepted` ⇒ reject; below ⇒ pass (D13). `ZeroContextCheck` — judge answerable *without* evidence ⇒ reject (MockJudge rule: answerable iff evidence non-empty ⇒ pass path; inverse mock ⇒ reject path). `AnswerabilityCheck` — judge with evidence not answerable ⇒ reject. `RoundTripCheck(k=10)` — gold chunk in retriever top-k ⇒ pass with score = 1/rank; absent ⇒ reject. `UniquenessCheck(mode="promote"|"reject", top_m=5)` — non-gold top-m chunk judged answerable: promote-mode ⇒ pass + `promoted=(chunk_id,)`; reject-mode ⇒ reject "leaky gold".
- [ ] Implement + `gate/checks/__init__.py` imports all; commit `feat(gate): five verification checks (dedup, zero-context, answerability, round-trip, uniqueness)`.

#### Task 3.2: Steps — samplers, assembler, generator, gate orchestrator
**Files:** `steps/{seed_sampler,context_assembler,generator,gate}.py`, `steps/prompts/{answer_first_v1,revise_v1,judge_v1}.j2`, `steps/__init__.py`; tests `tests/steps/test_{seed_sampler,context_assembler,generator,gate}.py`.
**Consumes:** sampling (T2.1), checks (T3.1). Small-world conftest fixture: 2 clusters × 20 chunks, 60 train queries, mocks everywhere, C=2 partition.
- [ ] Tests FIRST: quota allocation `n_c` matches `λ·p̂+(1−λ)/C` with `n_min` floor (hand p̂); `p_group=1` ⇒ every seed has 2 same-doc chunk_ids; round-robin strata cycle within cluster; spec sampler seeds carry `z` (len d) + kNN chunk_ids; assembler: k_style nearest *train* queries by cosine, within-cluster preferred (fixture where nearest global ≠ nearest in-cluster), instruction contains stratum value; generator: n_candidates per seed, `gen_meta` records model/prompt_version/`cos_to_target` (A2 fixture: MockChatModel + MockEmbedder, cos_to_target computed, revision path triggered when < tau_t exactly once); gate orchestrator: check order respected (spy), short-circuit on first fail, `state.metrics["gate_reject_reasons"]` tallies per check, promotions accumulate on candidate via `gate_meta["promoted"]`, accepted → `state.gate_accepted`; every step `to_config/from_config` round-trip.
- [ ] Implement (answer-first template per §6.3: extract-claim-then-question, forbid "according to the document", match exemplar register); commit `feat(steps): seed samplers (uniform/quota/spec), context assembler, generator, gate orchestrator`.

#### Task 3.3: QrelBuilder + Curator
**Files:** `steps/{qrel_builder,curator}.py`; tests `tests/steps/test_{qrel_builder,curator}.py`.
- [ ] Tests FIRST: binary qrels = seed chunks + promotions at grade 1; `content_hashes` filled for every qrel chunk; `record_id` deterministic; `QrelStrategy` ABC + registry (`binary` only, unknown ⇒ RegistryError); curator: stratified subsample hits `target_mix` (fixture 2:1), memorization flag in `gate_meta["memorization_cos"]` when cos ≥ 0.9 to any train query (record kept, flagged — Chroma audit semantics), final dedup drops exact-text repeats, `max_records` cap.
- [ ] Implement; commit `feat(steps): binary qrel builder with promotions, curator with memorization audit`.

#### Task 3.4: Datasets — toy world, jsonl loader, sample corpus
**Files:** `datasets/{base,toy_world,jsonl_loader,sample_corpus}.py`; `data/sample/*.jsonl`; tests `tests/datasets/test_{toy_world,jsonl_loader,sample_corpus}.py`.
**Port:** proto L652-712 (`make_world`), L715-723 (`_emit`), L779-790 (zoo wiring via T2.4).
- [ ] Tests FIRST: `build_toy_world(d=16, k_true=4, n_chunks=80, n_prod=400, seed=0)` bundle invariants — split sizes 60/25/15, chunks per component, all embeddings in store, bank populated, `ToyEmbeddingBank` text↔vector lookup; `ToyChatModel(bank, style=.45, noise=.55, seed=…)` emission math = `_emit` (unit norm, deterministic per (seed_id, cand_idx, attempt), revision attempt ⇒ noise/2 toward z); `ToyJudge` geometric rules (no evidence ⇒ answerable only for hash-tail 2% "common knowledge"; with evidence ⇒ max-cos ≥ τ_ans); `PassthroughEmbedder(bank)` lookup + KeyError on unknown; toy `oracle_qrels`/`anchor_qrels` = nearest-chunk gold (D17); `load_chunks/load_queries/load_anchor_qrels` jsonl round-trip incl. optional fields; sample corpus: regeneration is byte-identical to committed files (the test that D18 promises).
- [ ] Implement; generate + commit `data/sample/`; commit `feat(datasets): toy world (geometric emission adapters), jsonl loader, bundled sample corpus`.

#### Task 3.5: Arms
**Files:** `arms/{base,a0_naive,a1_quota,a2_spec,oracle}.py`; tests `tests/arms/test_arms.py`.
- [ ] Tests FIRST: each preset builds the documented step list (a0: uniform+k_style=0; a1: quota+exemplars; a2: spec sampler+target check) with params merged over defaults; `llm_override` swaps generator_llm (D12) via `with_overrides`; `run_arm` on small toy world returns non-empty `AnnotationRecord`s for all four arms; oracle records: queries from oracle split verbatim, qrels = oracle_qrels, no LLM calls (spy asserts zero).
- [ ] Implement; commit `feat(arms): A0/A1/A2/ORACLE presets over pipeline steps`.

#### Task 3.6: Validator + EvalReport + figures
**Files:** `steps/validator.py`, extend `domain/report.py` usage; tests `tests/steps/test_validator.py`.
**Consumes:** all metrics (T2.2-2.4), arms (T3.5).
- [ ] Tests FIRST (small toy world, n_per_arm=60, n_boot=50): report has one block per arm; `reuse_pipeline_for="a1"` ⇒ a1 block computed from `state.accepted` (spy: a1 preset never invoked); each block contains fidelity {kl, c2st, wc2st_mean, wc2st_per_cluster, mmd}, efficiency {ess_ratio, coverage_gap, demand_weighted_coverage, zero_query_clusters, post_stratified_ndcg, per_cluster_table (dual-view D14/§8-9 rule), worst_k}, validity {tau, tau_ci, tau_ap, rbo, controls: {drop: p, noise: p}}, diversity {distinct_1, distinct_2, dedup_rate}, gate reject-reason tally, `gates_passed` flags (τ≥0.9 ∧ τ_AP≥0.8); `metrics.json` written, deterministic (no timestamps — D14), `report.md` contains the 4-arm table; figures dir gets `fidelity_bars.png`, `tau_ci.png`, `ess_coverage.png`, `gate_rejects.png` (matplotlib Agg; existence + non-empty only).
- [ ] Implement (fidelity real-reference = equal-n anchor subsample via `rng("validator")`; labels via `resources.partition`; per-stratum τ where n permits — else recorded as null); commit `feat(validator): 4-arm metric harness, EvalReport, metrics.json, figures`.

#### Task 3.7: Optimization contract
**Files:** `optimization/{base,noop,objectives,mipro_adapter}.py`; tests `tests/optimization/test_optimization.py`.
- [ ] Tests FIRST: `NoOpOptimizer.optimize("p", None, metric)` ⇒ result with optimized==baseline prompt, improvement 0.0, 1 trial, config serialized; `FidelityObjective` on small toy world returns finite float, better prompt (toy: same) reproducible; gate pass-rate below `min_pass_rate` ⇒ −1e9 penalty (mock a gate-hostile judge); `MIPROv2Optimizer()` without dspy ⇒ ImportError mentioning `uv sync --extra optimization`; frozen dataclasses reject mutation.
- [ ] Implement (mirror healthbench: frozen `TrialRecord`/`OptimizationResult`, `BaseOptimizerConfig` pydantic frozen, `OPTIMIZERS` registry); commit `feat(optimization): optimizer ABC, NoOp, FidelityObjective, MIPROv2 stub`.

#### Task 3.8: CLI + configs
**Files:** `src/ragsynth/cli.py`, `configs/v1_toy.yaml`, `configs/v1_local_corpus.yaml`; tests `tests/test_cli.py` (typer `CliRunner`).
- [ ] Tests FIRST: `ragsynth validate --config configs/v1_toy.yaml` exit 0 + summary table; broken type ⇒ exit 1, message lists known keys; `ragsynth run --config <tiny toy config>` (conftest-written, n small) produces `metrics.json`/`report.md`/`figures/`/`artifacts/` + records jsonl; `ragsynth report --config <same>` re-renders report.md from stored metrics.json (delete report.md first, rerun, identical bytes).
- [ ] Write both real configs (v1_toy: §13 shape + D11 adapter types, `reuse_pipeline_for: a1`, validator `arm_params` carrying D12 emission overrides; v1_local_corpus: jsonl dataset, hashed embedder dim 256, mock chat/judge, arms [a0,a1,oracle] — a2 included too: hashed space is a valid planning space); rich console output, logging config (INFO default, `--verbose`).
- [ ] Commit `feat(cli): run/validate/report commands + v1 configs`.

### Phase 4 — Integration, acceptance, docs

#### Task 4.1: Contract tests over the full registries (§15.2)
**Files:** `tests/pipeline/test_contract.py`.
- [ ] Parametrize over `STEPS.keys()`: instantiate from that step's params in `configs/v1_toy.yaml` (or defaults), `to_config→from_config→to_config` fixed-point, `fit(resources)` idempotent (twice == once), `run(state)` returns a `PipelineState` and never mutates `Resources`; over `CHECKS.keys()`: failing fixture ⇒ gate emits `Rejection` with that check's name + non-empty reason; over adapter registries: every mock satisfies its Protocol (assignment to Protocol-typed variable + behavioral smoke). Commit `test(contract): LSP contract tests across all registries`.

#### Task 4.2: E2E toy-world table + tuning loop (the §10 gate)
**Files:** `tests/e2e/test_toy_world_table.py`; tuning touches only `configs/v1_toy.yaml` arm/emission/judge params.
- [ ] Write the qualitative assertions (CI world: d=32, n_prod=1500, n_per_arm=200, n_boot=200, fixed seed): `wc2st(a2) < wc2st(a1) − 0.10` and `wc2st(a1) > 0.80`; `mmd(a2) < mmd(a1) < mmd(a0)`; `kl(a0) > kl(a1) ≥ kl(a2)`; `ess(a1) ≥ ess(oracle) − 0.15` and `ess(a1) > ess(a0)`; `tau(a2) > tau(a1)`, `tau(oracle) ≥ tau(a2) − 0.05`, gates_passed(a0) is False; controls: `p_drop < 0.05` for all arms, `p_noise < 0.05` for a2 & oracle, `p_noise > 0.05` for a1 (the "A1 misses the noise control" claim).
- [ ] Iterate emission/judge/κ params (documented knobs: per-arm style/noise D12, `kappa_chunk=150`, `kappa_query=400`, zoo σ set, noise-control σ=0.5, τ_ans) until stable at the fixed seed **and** at seeds {0,1,2} (run thrice locally; assert only seed 0 in CI). Record the tuning trace in `experiments/v1/report.md` findings.
- [ ] Commit `test(e2e): toy-world 4-arm qualitative table locked`.

#### Task 4.3: Determinism, round-trip, local-corpus e2e (§15.1)
**Files:** `tests/e2e/test_determinism.py`, `tests/e2e/test_local_corpus.py`, `tests/pipeline/test_roundtrip_outputs.py`.
- [ ] Determinism: run tiny toy config twice into two tmp dirs ⇒ `metrics.json` byte-identical; round-trip: `from_yaml(to_yaml)` config-identical AND run ⇒ identical metrics.json; local corpus: tiny variant of v1_local_corpus over `data/sample/` completes, ≥1 accepted record per arm, report exists. Commit `test(e2e): determinism, YAML round-trip, local-corpus run`.

#### Task 4.4: Acceptance runs + experiments/v1 + coverage top-up
- [ ] `uv run ragsynth run --config configs/v1_toy.yaml` (full: 500 seeds, 1000 boot) — assert < 10 min, outputs under `experiments/v1/`; freeze `experiments/v1/config.yaml` copy; run local-corpus config; write `experiments/v1/analysis.ipynb` (load metrics.json, render table + 4 figures, findings narrative: what the table shows, decision, next-iteration hypotheses per §14) — notebook executes top-to-bottom via `uv run jupyter nbconvert --execute` (notebooks extra) but is NOT in CI.
- [ ] Coverage: `pytest --cov` per-package report; add tests until metrics/sampling/pipeline ≥ 85%, overall ≥ 70%.
- [ ] Commit `chore(experiments): v1 acceptance run artifacts + notebook`.

#### Task 4.5: Docs
**Files:** `README.md` (what/why, 10-line quickstart, extras table, roadmap pointer), `ARCHITECTURE.md` (1 page: state-flow diagram `SeedSampler → … → Validator` with PipelineState fields annotated, extension recipe "add a GateCheck in 20 lines" — real compilable example, registry + config wiring), docstring audit (every metric cites its paper — §15.5).
- [ ] Commit `docs: README quickstart + ARCHITECTURE with extension recipe`.

### Phase 5 — Final verification & adversarial review
- [ ] Full §15.3 sweep from clean checkout (`git stash -u` sanity): all four commands.
- [ ] Adversarial review fan-out (independent verifiers): (1) formula-vs-prototype diff audit (every Appendix-A port semantically identical); (2) spec-compliance sweep §§4-13 (field names, defaults, config shape); (3) determinism hunter (any `datetime.now`/unseeded rng/dict-order leak into metrics.json); (4) v1-scope creep check (§2.2). Fix confirmed findings; re-run §15.3.
- [ ] Final commit `chore: v1 acceptance` + summary against §15 checklist.

---

## 6. Acceptance matrix (SPEC §15 → where it's proven)

| Criterion | Proof |
|---|---|
| §15.1 toy run offline < 10 min, 4-arm table | T4.4 + `tests/e2e/test_toy_world_table.py` |
| §15.1 local corpus w/ mocks | `tests/e2e/test_local_corpus.py` |
| §15.1 YAML round-trip byte-stable | `tests/pipeline/test_serialization.py` + `test_roundtrip_outputs.py` |
| §15.1 same seed ⇒ identical metrics.json | `tests/e2e/test_determinism.py` |
| §15.2 step/check/adapter contracts | `tests/pipeline/test_contract.py` |
| §15.3 four commands | every commit; final sweep T5 |
| §15.4 known-value fixtures | T2.1-2.4 test lists (τ_AP=1/3 hand-derived, ESS, KL, C2ST, paired bootstrap, movMF, importance weights) |
| §15.5 README/ARCHITECTURE/docstring citations | T4.5 |
