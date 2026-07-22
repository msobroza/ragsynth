# SPEC-V2-QRELS — graded/pooled qrels (UMBRELA) + LLM-judge calibration tooling

> **Audience:** Claude coding agent (with superpowers skills).
> **Parent:** `SPEC.md` (v1 law) §6.5, §16 R5. **Status:** v2 feature spec — plan before coding.
> **Prime directive:** v1 behavior is frozen. `binary` stays the default strategy; schema v1 configs load byte-for-byte unchanged; the D16 binary nDCG path must be regression-locked before any graded code lands.

---

## 0. How to execute this spec

1. Read parent SPEC §0 and obey it. TDD: known-value fixtures (§14.3 here) FIRST.
2. New decisions go to `PLAN.md`; this spec pre-makes D42–D51 (§13). Decision numbers D42–D51 allocated per specs/v2/README.md.
3. §15.3 verification commands of the parent SPEC stay green at every commit, plus the new ones in §14.
4. Do not touch v3 items (masking-ablation crucial refinement, training splits) "while you're at it".

---

## 1. Context & motivation

v1 qrels are binary: gold = seed chunks + uniqueness promotions, grade 1 (`steps/qrel_builder.py`). This is honest but blind in two ways R5 predicts will bite:

- **No partial credit.** A chunk that *discusses* the topic without answering it scores identically to noise; graded relevance (UMBRELA 0–3; Upadhyay, Clarke & Lin, arXiv 2024, TREC-DL-validated) separates "related" from "answers".
- **Unjudged holes.** Only seed golds + promotions are ever labeled. Every other chunk a zoo system retrieves is *assumed* non-relevant. Holes corrupt top-heavy agreement first — the R5 trigger is exactly "τ_AP stalls below gate while τ passes" (§16). Pooling over the system zoo (TREC tradition) + judging the pool closes the holes at scoring depth.

Both need an LLM judge we can *trust*, hence: calibration tooling against ≥200 human labels (κ_w, Krippendorff α, confusion/bias report) with an acceptance band anchored to the ~90–95% human annotation-precision noise floor (parent SPEC §16), and a circularity guard hardening the v1 same-family warning (Fröbe et al., SIGIR 2025).

**Key decisions already made (do not relitigate):**

- **Grades live in the existing field.** `AnnotationRecord.qrels: dict[str, int]` already holds ints; graded uses 0–3 in place. No domain migration. Grade-0 entries ARE stored — "judged non-relevant" must be distinguishable from "unjudged hole".
- **`binary` stays default and byte-identical.** Graded is opt-in via `strategy: graded_umbrela` under `schema_version: 2`.
- **One judge call per (query, pooled chunk).** No batching cleverness in v2; the judgment cache (§5.3) is the cost lever.
- **The qrel judge is its own resource block** (`resources.qrel_judge`), never shared with the gate judge or the generator (v1 distinct-config-key rule, extended).
- **Pool at scoring depth.** `pool_depth` default = 10 = the D16 nDCG `k`. Judging exactly to the depth you score at makes condensed lists hole-free at k (§6).
- **Air-gapped CI forever.** `MockGradedJudge` is deterministic; the full suite runs offline.

---

## 2. Goals & non-goals

### 2.1 Goals

1. `QrelStrategy` registry gains `graded_umbrela`; `binary` and `relabel_nearest` untouched.
2. UMBRELA judge adapter (`GradedRelevanceJudge` Protocol, `UmbrelaJudge`, `MockGradedJudge`) through the adapter/registry machinery, in its own `GRADED_JUDGES` registry (§5.1).
3. Zoo-pooled judging with a per-query budget; graded nDCG in `metrics/validity/systems.py` with an exact binary reduction (regression fixture).
4. `ragsynth calibrate-judge` CLI: κ_w + Krippendorff α + per-grade confusion matrix + bias report + PASS/MARGINAL/FAIL verdict.
5. Circularity guard: judge family ≠ generator family AND ≠ any declared candidate-system family; ERROR (not warn) for the qrel judge.
6. Unjudged-holes diagnostics: hole rate, condensed-list τ_AP (Sakai), computed R5-trigger flag in `metrics.json`.

### 2.2 Non-goals (stub or omit)

- Multi-judge ensembles / self-consistency voting → later; single judge + calibration only.
- Human-label collection UI/workflow — we consume a jsonl, we do not produce it.
- Re-grading v1 archives; migration tooling → churn lifecycle (R6).
- Graded gain in *fidelity/efficiency* metrics — graded touches validity scoring only.
- τ_AP variants for ties/partial lists beyond condensed lists.

---

## 3. Domain changes (`src/ragsynth/domain/annotation.py`)

One additive, defaulted field — old records and v1 code paths untouched:

```python
class AnnotationRecord(BaseModel):
    ...  # all v1 fields unchanged; qrels: dict[str, int] now documented as 0-3 under graded
    qrel_meta: dict[str, Any] = Field(default_factory=dict)
    # graded fills: {"strategy": "graded_umbrela", "scale": "umbrela_0_3",
    #   "pool_depth": 10, "judge_prompt_version": "umbrela_v1",
    #   "judged": <n>, "pooled": <n>, "capped": <bool>}
```

`crucial` under graded = chunks with `grade >= min_grade_for_gold` (default 2, the TREC-DL binarization cutoff). Binary strategy semantics unchanged.

## 4. QrelStrategy — registry entry (`steps/qrel_builder.py`)

The v1 ABC signature is already sufficient — do NOT change it:

```python
class QrelStrategy(ABC):
    name: str
    @abstractmethod
    def build(self, candidate: SyntheticQuery, resources: Resources) -> dict[str, int]: ...

@QREL_STRATEGIES.register("graded_umbrela")
class GradedUmbrelaQrels(QrelStrategy):
    name = "graded_umbrela"
    def __init__(self, pool_depth: int = 10, max_judged: int = 50,
                 min_grade_for_gold: int = 2) -> None: ...
    def build(self, candidate, resources) -> dict[str, int]:
        # pool (§6) -> one judge call per pooled chunk via resources.qrel_judge (§5)
        # -> {chunk_id: grade 0..3}, grade-0 entries retained
```

`QrelBuilder.__init__` gains `strategy_params: dict[str, Any] | None = None`, forwarded to the strategy constructor. `to_config()` **omits** `strategy_params` when empty so v1 round-trips stay byte-stable (D50). `Resources` gains `qrel_judge: GradedRelevanceJudge | None = None`; `graded_umbrela` raises an actionable `ValueError` at construction if it is missing.

## 5. UMBRELA judge adapter (`adapters/judge/`)

### 5.1 Protocol + verdict (`adapters/judge/base.py`, additive)

```python
@dataclass(frozen=True)
class GradeVerdict:
    grade: int          # 0..3 (clamped)
    rationale: str      # judge's one-line justification (may be empty)
    confidence: float   # [0, 1]

class GradedRelevanceJudge(Protocol):
    def grade(self, query: str, passage: str) -> GradeVerdict: ...
```

Registered in a NEW `GRADED_JUDGES: Registry` in `adapters/judge/base.py` (keys `"umbrela"`, `"mock_graded"`) — **not** in the existing `JUDGES` registry: the two Protocols are incompatible (`judge(query, evidence_texts)` vs `grade(query, passage)`), and mixing them behind one registry would let a config point `judge_llm` at `mock_graded`, validate, then die with `AttributeError` deep in the gate — and the §15.2 contract-test sweep could not know which protocol a key implements. `resources.qrel_judge` is resolved from `GRADED_JUDGES` only; `build_resources` rejects a graded key under `judge_llm` (and a gate-judge key under `qrel_judge`) with the standard actionable `RegistryError`.

### 5.2 `UmbrelaJudge` (`adapters/judge/umbrela.py`)

- Owns its OWN `ChatModel` (nested `params["chat"]`, exactly the `LLMJudge` pattern), never shared.
- Prompt = the UMBRELA DNA prompt (Upadhyay, Clarke & Lin 2024, itself adapted from Thomas et al., SIGIR 2024) adapted to chunks: intent-match (M) and trust (T) sub-judgments, then final 0–3; scale wording verbatim from the paper (0 fails to serve, 1 related but does not answer, 2 partial/unclear answer, 3 dedicated to the query, contains the answer). Lives in `steps/prompts/umbrela_v1.j2`; `prompt_version` recorded in `qrel_meta`.
- Strict-JSON reply `{"m": int, "t": int, "grade": int, "rationale": str}`; unparseable ⇒ `GradeVerdict(0, "", 0.0)` + warning log (flaky judge deflates, never inflates — mirrors `LLMJudge` fallback).

### 5.3 Judgment cache

`artifacts_dir/qrel_judgments.jsonl`, append-only, key = `sha256(prompt_version | sha256(query_text) | content_hash(chunk))`. Cache hit ⇒ no LLM call. This is the determinism story for real judges (§12) and the cost lever for re-runs.

### 5.4 `MockGradedJudge` (`adapters/judge/mock.py`, additive)

Deterministic, hash-seeded (the `MockChatModel` idiom): `grade = _stable_hash(query, passage) % 4`, overridable with `fixed_grades: dict[str, int]` (keyed by passage sha prefix) and `gold_grade: int = 3` applied when the passage text contains a configured marker. `to_config()/from_config()` like every adapter. CI never leaves the machine.

## 6. Pooling

- **Pool(q)** = union of top-`pool_depth` chunks over every system in `resources.zoo` ∪ seed golds ∪ uniqueness promotions. `resources.zoo` is the zoo the composition root already builds ONCE with the global config seed (`pipeline/serialization.py`: `make_system_zoo(chunk_ids, chunk_embs, seed=seed)`) and that the validator scores — the pool zoo and the scoring zoo are **the same object** by construction. The strategy builds no zoo of its own (no `zoo_seed` parameter exists); a second construction under a different seed would silently pool systems the validator never scores.
- **Default `pool_depth=10`, rationale:** it equals the D16 scoring cutoff `k`, so — because pool zoo and scoring zoo are the same object — every chunk that can contribute to any zoo system's nDCG@10 is judged; condensed and full lists coincide at k, holes → 0 by construction. Deeper pools buy nothing for the meter and cost |zoo| more judge calls.
- **Budget:** `max_judged=50` per query. Overflow is trimmed deterministically by descending max-cosine to the query over the exact system, ties by chunk_id (D44). Golds + promotions are never trimmed. `qrel_meta.capped` records truncation; capped-rate is reported.
- **Cost note (document in README):** expected calls/query on the 12-system zoo ≈ |union| ≤ 120, typically 15–35 after overlap; toy world with mocks is free.

## 7. Graded nDCG (upgrade of PLAN D16, `metrics/validity/systems.py`)

`MatrixSystem.per_query_scores` gain becomes `(2**grade - 1) / log2(1 + rank)`; IDCG places grades sorted descending at ranks 1… (Järvelin & Kekäläinen, TOIS 2002 exponential-gain form).

- **Binary reduction (regression-locked):** with grades ∈ {0,1}, `2**1 - 1 = 1` ⇒ formula is *algebraically identical* to the shipped code. Required fixture: run the existing D16 test corpus through old and new paths ⇒ `np.array_equal` (not approx) on scores; the §15.4 single-gold `1/log2(1+rank)` fixtures pass unchanged.
- Grade-0 entries: already excluded by the shipped `grade > 0` relevance test — judged-0 rides through with zero code change, but now *exists* in qrels for hole accounting (§10).
- `drop_mask` / tie semantics / ValueError-on-unknown-chunk unchanged.

## 8. Calibration tooling (`metrics/validity/calibration.py` + CLI)

```python
def weighted_cohen_kappa(a: NDArray[np.int_], b: NDArray[np.int_],
                         n_grades: int = 4, weights: str = "quadratic") -> float: ...
def krippendorff_alpha_ordinal(a: NDArray[np.int_], b: NDArray[np.int_],
                               n_grades: int = 4) -> float: ...
def confusion_matrix(human: NDArray[np.int_], judge: NDArray[np.int_],
                     n_grades: int = 4) -> NDArray[np.int_]: ...
def bias_report(human, judge) -> BiasReport  # frozen: mean_delta, per_grade_over/under,
                                             # binarized_{accuracy,precision,recall} at >=2
def calibration_verdict(report: CalibrationReport) -> Literal["PASS", "MARGINAL", "FAIL"]: ...
```

Pure numpy — no new runtime deps (Cohen 1968; Krippendorff 2004, ordinal difference function).

**CLI:** `ragsynth calibrate-judge --config <yaml> --human-labels <jsonl> [--out <dir>]`
- Labels jsonl: `{"query_id": str, "chunk_id": str, "grade": int}`; texts resolved from the run's records/corpus; judge grades come from the cache or fresh calls through the configured `qrel_judge`.
- n < 50 ⇒ hard error; 50 ≤ n < 200 ⇒ result stamped `"provisional": true` with a warning; **calibrated status requires n ≥ 200** (parent SPEC §16: "κ-calibration vs ≥200 human labels").
- Writes `calibration_report.json` (deterministic, no timestamps — D14 discipline) + `calibration_report.md` (confusion matrix table, bias narrative, verdict).
- **Acceptance band (D47):** PASS ⇔ κ_w ≥ 0.6 AND binarized (grade ≥ 2) judge-vs-human accuracy ≥ 0.90 — the floor is the ~90–95% *human* annotation-precision noise band of parent SPEC §16: a judge inside the band is indistinguishable from a second human, demanding more is asking the judge to out-agree humans with themselves. MARGINAL ⇔ κ_w ∈ [0.4, 0.6) (report-only use); FAIL below. α ≥ 0.667 reported against Krippendorff's own tentative-conclusion cutoff, informational.

## 9. Circularity guard (`pipeline/serialization.py`, extends v1 warning)

Fröbe et al. (SIGIR 2025): LLM assessors overestimate same-family systems by 9–17 rank positions. The v1 WARN covers gate-judge vs generator. v2:

- New optional `resources.candidate_systems: [{name: str, family: str}]` — declares the LLM/embedding families of real systems under evaluation (the matrix zoo has no family; real rerankers do).
- Guard: `_llm_family(qrel_judge) != _llm_family(generator_llm)` AND `!= family` of every declared candidate. Violation for the **qrel judge is a `ValueError`** (it mints the labels; circular labels poison every downstream τ), downgradeable to WARN only via explicit `qrel_judge: {allow_circular: true}`. The gate-judge check stays a WARN exactly as shipped (D29 token comparison reused — one `_llm_family`, no second implementation).

## 10. Unjudged-holes diagnostics (`metrics/validity/holes.py`)

- `hole_rate(zoo_runs, qrels, k)` — fraction of top-k retrieved chunks with NO qrel entry (judged-0 is not a hole), per system + mean. Binary strategy on real corpora will show high hole rates — that is the point of the diagnostic.
- `condensed_scores(...)` — nDCG on condensed lists: unjudged chunks removed from each ranking before scoring (Sakai, "Alternatives to Bpref", SIGIR 2007; Sakai & Kando, IRJ 2008). Report `tau_ap_condensed` = τ_AP between zoo rankings under full vs condensed scoring, and `condensed_divergence` = mean |nDCG_full − nDCG_condensed| per system.
- **R5 trigger, computed:** validator writes `validity.holes = {hole_rate_mean, hole_rate_per_system, tau_ap_condensed, condensed_divergence, r5_trigger}` with `r5_trigger = (tau >= gates.tau) and (tau_ap < gates.tau_ap)` — the §16 R5 trigger becomes a reported boolean instead of folklore. `report.md` renders it with the remediation line: "switch qrel strategy to graded_umbrela (pooled)".

## 11. Serialization (schema_version 2)

```yaml
# configs/v2_graded_toy.yaml (schema v2)
ragsynth: {schema_version: 2, name: v2-graded-toy, seed: 0}
resources:
  dataset: {type: toy_world, params: {d: 64, k_true: 8, n_prod: 5000}}
  embedder: {type: passthrough}
  generator_llm: {type: mock}
  judge_llm: {type: mock}                       # gate judge, unchanged
  qrel_judge: {type: mock_graded, params: {gold_grade: 3}}
  candidate_systems: []                          # real runs: [{name: rerank-x, family: llama}]
artifacts_dir: experiments/v2/artifacts
pipeline:
  # seed_sampler / context_assembler / generator / gate / curator blocks unchanged from v1
  - {type: qrel_builder, params: {strategy: graded_umbrela,
      strategy_params: {pool_depth: 10, max_judged: 50, min_grade_for_gold: 2}}}
  - {type: validator, params: {arms: [a0, a1, a2, oracle], n_boot: 1000,
                               gates: {tau: 0.9, tau_ap: 0.8}}}
```

Rules: the schema_version-2 trigger list and the one-time `from_yaml` loader relaxation are owned by the canonical section in `specs/v2/README.md` (single-owner loader-change semantics); of this spec's features, `qrel_builder.strategy: graded_umbrela` and `resources.qrel_judge` are the ones that require `schema_version: 2`. `qrel_judge` is required iff a graded strategy appears; `graded_umbrela` under `schema_version: 1` ⇒ actionable error ("declare schema_version: 2"). `from_yaml(to_yaml(p))` byte-stable for both versions.

## 12. Determinism & offline guarantees

- **Guaranteed:** same seed, same process, mock judges ⇒ identical `metrics.json` and identical `qrel_judgments.jsonl` content. The pool, trimming order, and zoo are seed-deterministic (D44); the pooled zoo is `resources.zoo`, built once from the global config seed.
- **Not guaranteed and stated plainly:** a real `UmbrelaJudge` over an OpenAI-compatible endpoint is not reproducible across providers/model updates. What IS guaranteed instead: every verdict is cached (§5.3) keyed by (prompt_version, query sha, content_hash); a re-run against a warm cache issues zero LLM calls and is byte-identical; `qrel_meta.judge_prompt_version` + the cache file make any past run auditable and replayable.
- Runtime deps: none added (numpy-only κ/α). No new extras. CI stays air-gapped.

## 13. Decisions this spec pre-makes (append to PLAN.md)

Decision numbers D42–D51 allocated per specs/v2/README.md.

| # | Decision |
|---|---|
| D42 | Grades 0–3 live in the existing `qrels: dict[str,int]`; grade-0 stored (judged-0 ≠ hole); additive `qrel_meta` field, default `{}` — no domain migration. |
| D43 | Graded `crucial` = grades ≥ `min_grade_for_gold` (default 2, TREC-DL binarization); binary strategy semantics untouched. |
| D44 | Pool = union@`pool_depth` (default 10 = D16 k) over `resources.zoo` — the same zoo object the validator scores, built once from the global config seed — ∪ golds ∪ promotions; the strategy builds no zoo of its own. `max_judged=50` cap trimmed by exact-system cosine desc, chunk_id tiebreak; golds never trimmed. |
| D45 | Graded nDCG gain `2^g − 1`; binary reduction is algebraic identity, locked by an `np.array_equal` regression fixture against the shipped scorer. |
| D46 | `resources.qrel_judge` is its own block, resolved from a separate `GRADED_JUDGES` registry (never `JUDGES`); circularity with generator or any declared candidate family is an ERROR for the qrel judge (`allow_circular` opt-out), WARN retained for the gate judge. |
| D47 | Calibration acceptance: PASS ⇔ κ_w(quadratic) ≥ 0.6 ∧ binarized accuracy ≥ 0.90 (§16 noise floor); MARGINAL κ_w ∈ [0.4, 0.6); calibrated status needs n ≥ 200 (50–199 = provisional). |
| D48 | Judgment cache: append-only jsonl in `artifacts_dir`, key `sha256(prompt_version | query_sha | content_hash)`; warm-cache re-runs issue zero LLM calls. |
| D49 | R5 trigger is computed and reported: `validity.holes.r5_trigger = (τ ≥ gate) ∧ (τ_AP < gate)`, alongside hole_rate and condensed-list diagnostics. |
| D50 | `QrelBuilder.to_config()` omits empty `strategy_params`; schema-v1 byte-stable round-trip preserved. |
| D51 | schema_version-2 gating follows the canonical trigger list and single-owner loader change in `specs/v2/README.md`; this spec's triggers are `graded_umbrela` / `resources.qrel_judge`, and graded features under schema 1 are a hard error naming the fix. |

## 14. Acceptance criteria (Definition of Done)

**14.1 Functional**
- [ ] `uv run ragsynth run --config configs/v2_graded_toy.yaml` completes offline; the records output contains graded qrels with at least one grade-0 entry; `metrics.json` contains the new `validity.holes` block (`hole_rate`, `tau_ap_condensed`, `r5_trigger`).
- [ ] `uv run ragsynth run --config configs/v1_toy.yaml` output is bit-identical to the pre-change baseline (binary untouched).
- [ ] `uv run ragsynth calibrate-judge --config configs/v2_graded_toy.yaml --human-labels tests/fixtures/human_labels_200.jsonl` writes both report files offline with a PASS/MARGINAL/FAIL verdict.

**14.2 Contract tests**
- [ ] `graded_umbrela` in the parametrized registry sweep: serializes, deserializes, runs; missing `qrel_judge` ⇒ actionable error.
- [ ] `MockGradedJudge` satisfies `GradedRelevanceJudge`; verdict deterministic across processes.
- [ ] Registry separation: `judge_llm` pointing at a `GRADED_JUDGES` key (e.g. `mock_graded`) — and `qrel_judge` at a `JUDGES` key — is rejected by `build_resources` with an actionable `RegistryError`, at config time, not deep in the gate.
- [ ] Circularity: same-family qrel judge ⇒ `ValueError`; `allow_circular: true` ⇒ WARN; gate-judge behavior unchanged.

**14.3 Known-value fixtures (write FIRST)**
- [ ] `weighted_cohen_kappa`: identical vectors ⇒ 1.0; a 4-item hand example (e.g. human `[0,1,2,3]`, judge `[0,2,2,3]`) with κ_w derived by hand in the test — derive exactly, do not trust a note; quadratic vs linear differ on it.
- [ ] `krippendorff_alpha_ordinal`: identical ⇒ 1.0; hand-derived small example; degenerate single-category input handled (defined value or documented NaN, asserted).
- [ ] `confusion_matrix` + `bias_report`: hand example with a known +1 over-grading bias ⇒ `mean_delta == +0.5`, binarized accuracy hand-checked.
- [ ] Graded nDCG: single gold grade 3 at rank r ⇒ `1/log2(1+r)` (equals binary — assert it, it is the reduction made visible); multi-grade `{a: 3, b: 1}` hand example; **binary regression:** old-vs-new `np.array_equal` on the D16 test corpus.
- [ ] Pooling: toy zoo hand example ⇒ exact pool union; cap trimming order asserted; golds survive the cap.
- [ ] Holes: constructed run with known unjudged docs ⇒ exact `hole_rate`; condensed vs full divergence on a hand example; `r5_trigger` true/false both exercised.
- [ ] Cache: second `build()` on the same candidate issues zero judge calls (spy).

**14.4 Config & hygiene**
- [ ] Byte-stable YAML round-trip for v1 AND v2 example configs; v1 config with `strategy_params` absent re-serializes without the key.
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy src` green; coverage ≥ 70 maintained; full suite passes with the network disabled.
- [ ] Every new public function's docstring cites its paper (§16 table).

## 15. Open questions — defaults for the agent (decide, document, proceed)

1. κ weighting: quadratic vs linear — default **quadratic** (standard for ordinal scales), `--weights linear` flag exposed.
2. Binarization cutoff for the noise-floor check: grade ≥ 2 (TREC-DL convention) — keep, config-exposed as `min_grade_for_gold` reuse.
3. Judge rationale retention: cache stores it, `AnnotationRecord` does not (size) — keep out of the record; revisit if the bias report needs qualitative drill-down.
4. Pool over real configured `Retriever` in addition to the matrix zoo: **not in v2 default** (zoo is the ranked population; adding the production retriever is one config key later) — stub the config field, leave it off.
5. Spec file location `specs/v2/` — adopted as the v2 spec convention; parent SPEC.md remains the v1 law.

## 16. Reference table

| Feature | References |
|---|---|
| Graded 0–3 scale, DNA prompt, TREC-DL validation | Upadhyay, Clarke & Lin (UMBRELA), arXiv 2024; Thomas et al., SIGIR 2024 |
| Graded nDCG (exponential gain) | Järvelin & Kekäläinen, TOIS 2002 |
| Pooling / incomplete judgments | TREC pooling tradition; Buckley & Voorhees, SIGIR 2004 (bpref context) |
| Condensed lists / holes | Sakai, SIGIR 2007 ("Alternatives to Bpref"); Sakai & Kando, IRJ 2008 |
| τ_AP | Yilmaz, Aslam & Robertson, SIGIR 2008 |
| Judge circularity | Fröbe et al., SIGIR 2025 |
| Weighted κ / ordinal α | Cohen, Psych. Bull. 1968; Krippendorff, Content Analysis 2004 |
| Human noise floor, ≥200 labels | parent SPEC §16 (benchmark-validation bullet) |
