# SPEC — `ragsynth` v2: real-benchmark validation (FiQA-2018 · NFCorpus · LegalBench-RAG)

> **Audience:** Claude Fable 5 coding agent (with superpowers skills).
> **Author of record:** Max (ML Tech Lead). Child spec of `SPEC.md` (v1 law) — where silent, v1 rules apply verbatim.
> **Status:** v2 execution spec for the §16 "Benchmark validation experiments" bullet. Expand into PLAN.md tasks before coding; new decisions land as PLAN D32+.

---

## 0. How to execute this spec (read first)

1. **v1 is law.** Do not modify v1 behavior. `schema_version: 1` configs MUST keep loading and producing byte-identical outputs. Everything here is additive.
2. **TDD.** Known-value fixtures in §13.4 are written FIRST (SPEC §15.4 style), then implementation.
3. **Air-gapped CI stays air-gapped.** Real datasets and real endpoints never enter CI. Converters are tested against bundled mini-fixtures; LLM paths against the v1 mocks.
4. **Ask vs. assume.** §15 defaults: decide, document in PLAN.md, proceed.
5. **Success = §13 acceptance criteria.** Nothing else is the bar.

---

## 1. Context & motivation

v1 shipped: the 4-arm harness (A0 naive / A1 quota / A2 spec-first / ORACLE) reproduces the prototype's toy-world table, and the jsonl mini-corpus run correctly self-reports as regression-detection-only under weak components (`experiments/v1/report.md`). The toy world proves the *mechanism*; nothing yet proves the harness on **real text, real embeddings, real LLMs, real qrels**. SPEC §16 pre-registered the exit test: oracle ceiling, per-stratum τ with bootstrap CI, positive-control battery, and a ~150–200-record stratified human precision audit — *before trusting any arm on real data*. This spec executes that test on three public domain-specific IR datasets spanning three demand regimes:

- **FiQA-2018** (finance; ~57.6K StackExchange/news passages, ~6.6K real user questions) — the *matched* case: queries were typed by actual users with actual needs; closest public proxy to production demand.
- **NFCorpus** (medical/nutrition; 3.6K PubMed-derived docs, ~3.2K queries harvested from NutritionFacts.org titles/topics) — *shifted*: queries are editorial artifacts, not user demand.
- **LegalBench-RAG 2024** (legal; CUAD, ContractNLI, MAUD, PrivacyQA sub-corpora; expert-written questions with **span-level** ground truth) — *shifted + long documents*: exercises the span→chunk qrel path and small-corpus stratification.

---

## 2. Goals & non-goals

### 2.1 v2 goals
- Converters producing the existing `chunks.jsonl` / `queries.jsonl` / `anchor_qrels.jsonl` schema (`src/ragsynth/datasets/jsonl_loader.py` is NOT modified).
- One 4-arm run + report per dataset with real MiniLM embeddings and real cross-family LLM generator/judge.
- Pre-registered expectations evaluated against the §16 R-triggers (R1/R2/R5) — this experiment *decides what v2 research opens next*.
- Stratified human precision-audit export per dataset (SPEC §16 noise floor: ≥90–95%).

### 2.2 v2 non-goals (do not build "while you're at it")
- No graded qrels, no pooling, no UMBRELA (R5 — this spec may *trigger* it, not build it).
- No prompt optimization beyond `NoOpOptimizer` (R2 likewise).
- No BEIR-wide sweep, no MTEB, no training split, no new metrics. No leaderboard claims — this validates *our harness*, not the datasets.
- No streaming/DB ingestion; converted jsonl on local disk only.

---

## 3. Key decisions (do not relitigate) — recorded as PLAN D32–D41

Decision numbers D32–D41 allocated per specs/v2/README.md.

- **D32 · Data layout & fetch.** `scripts/fetch_benchmarks.py <name>` downloads upstream releases into `data/benchmarks/<name>/raw/`; `scripts/convert_benchmark.py <name>` (thin wrapper over importable `ragsynth.datasets.converters.<name>`) emits the three jsonl files into `data/benchmarks/<name>/`. The whole `data/benchmarks/` tree is gitignored. The fetch script writes `data/benchmarks/README.md` recording, per dataset: source URL, release version/sha256, and the upstream license text it found. Datasets are **data, not runtime deps** — runtime dependency policy (Apache-2.0/MIT/BSD only) is untouched.
- **D33 · FiQA/NFCorpus chunking.** Native unit = chunk, 1:1 (FiQA passage / NFCorpus abstract); `doc_id` = upstream id. No re-chunking.
- **D34 · LegalBench-RAG chunking.** Contracts are chunked with a deterministic pure-python splitter: 1,000-character windows, 200-character overlap, window end snapped back to the last whitespace. Char offsets of every chunk are kept for D35.
- **D35 · Span→chunk gold rule.** A chunk is gold for a query iff `|span ∩ chunk| ≥ min(0.5 · |span|, 200)` characters for at least one gold span. Rationale: a long span split across two chunks credits both halves (200-char absolute branch); a chunk grazing a short span is excluded (50%-of-span branch). Grade 1 (binary, v1 qrel semantics).
- **D36 · Legal sub-corpora merged.** CUAD + ContractNLI + MAUD + PrivacyQA form ONE corpus (`metadata.subcorpus` on every chunk/query); the 60/25/15 split is stratified per sub-corpus; the report carries a per-sub-corpus τ appendix. One dataset ⇒ one report (three reports total).
- **D37 · Cluster-count ladder.** Default `C=8`. Before arms run, pick the largest `C ∈ {8, 6, 4, 2}` such that (a) every cluster holds ≥ 30 anchor queries AND (b) expected **seed** allocation `n_seeds · q_c ≥ 30` per cluster, where `n_seeds = 240` (the ladder reads it from the §8 sampler config) and `q_c = λ·p̂_c + (1−λ)/C` (λ=0.7) — evaluated *a priori* and deterministically. Arithmetic, so the rule is auditable: (b) ⇔ `q_c ≥ 30/240 = 0.125` ⇔ `min_c p̂_c ≥ (0.125 − 0.3/C)/0.7`; C=8 needs `min_c p̂_c ≥ 0.125` (exactly uniform 1/8 — any demand skew fails); C=6 needs `≥ 0.1071` (64% of uniform 1/6); C=4 needs `≥ 0.0714` (29% of uniform 1/4); C=2 has a negative bound (0.3/2 = 0.15 ≥ 0.125 even at `p̂_c = 0`, so (b) always passes; only (a) can fail it). Basis is deliberately seeds, not `n_per_arm = 160`: `160 · q_c ≥ 30` would need `q_c ≥ 0.1875 > 1/6 ≥ min_c q_c`, making every `C ≥ 6` impossible even under perfectly uniform demand (160/6 ≈ 26.7 < 30). (b) is a necessary a-priori proxy, not a guarantee: a stratum may still end with < 30 curated records at runtime, in which case the v1 SPEC §8 wC2ST 30/side floor rule applies as shipped (stratum flagged and skipped; run continues). Chosen `C` is recorded in `metrics.json` and the report header. Expected outcome under this rule: FiQA C=4 (genuine user demand is skewed — a smallest cluster below 64% of uniform mass kills C=6, while 29% of uniform is plausible), NFCorpus C=4–6 (editorial queries authored to cover the corpus ⇒ flatter p̂ may clear the C=6 bar), legal C=2–4 (heterogeneous sub-corpora ⇒ strong cluster imbalance).
- **D38 · n_per_arm = 160** (within the 150–200 §16 band). The curated A2 set doubles as the human precision-audit sample (§11).
- **D39 · Models.** Embedder: `sentence-transformers/all-MiniLM-L6-v2` via the existing `sentence_transformer` registry key (`st` extra; model download is data, Apache-2.0, documented). Generator: `Qwen/Qwen2.5-7B-Instruct`; judge: `meta-llama/Llama-3.1-8B-Instruct` — different families, so the §6.4 same-family warning stays quiet *by construction*, verified by a config test. Both via the existing `openai_compatible` adapter.
- **D40 · Determinism boundary = transcript replay.** New `cached` ChatModel wrapper (§7): first run records request-hash→response transcripts; replay runs never touch the network. Same seed + same transcripts ⇒ identical `metrics.json`. Live first-run responses are NOT deterministic — this is the explicit, documented exception; everything downstream of the transcripts is.
- **D41 · No time-decay.** None of the three benchmarks has query timestamps ⇒ demand half-life decay is off (uniform recency weights). Stated as a threat in §9.

---

## 4. Datasets, licenses, provenance

| Dataset | Corpus | Queries | Qrels | Upstream | License (verify at fetch; script records exact text) |
|---|---|---|---|---|---|
| FiQA-2018 | ~57,638 passages | ~6,648 real StackExchange questions | official binary qrels (shallow) | BEIR distribution (Thakur et al., NeurIPS D&B 2021) | no formal license from the WWW'18 challenge — research use only, documented |
| NFCorpus | 3,633 docs | ~3,237 queries | official graded qrels → binarized (grade ≥ 1) | BEIR / original (Boteva et al., ECIR 2016) | derived from NutritionFacts.org content — non-commercial research use |
| LegalBench-RAG | 4 sub-corpora, hundreds of long contracts | ~6.9K expert questions | **span-level** → D35 rule | Pipitone & Houir Alami, 2024 release | per sub-corpus: CUAD CC-BY-4.0, MAUD CC-BY-4.0 (Atticus), ContractNLI research release, PrivacyQA research release |

Non-negotiables: nothing under `data/benchmarks/` is ever committed; the license column above is *best-known* and the fetch script's recorded upstream text is authoritative; any dataset whose recorded license forbids our use is dropped from the run (report notes it) rather than argued around.

---

## 5. Converters (`src/ragsynth/datasets/converters/`)

One module per dataset behind a shared contract; scripts are argument-parsing shells only, so mypy/ruff/pytest cover all logic.

```python
@dataclass(frozen=True)
class ConversionManifest:
    dataset: str
    n_chunks: int
    n_queries: int
    n_qrel_entries: int
    source_version: str          # upstream release tag or archive sha256
    license_note: str
    output_sha256: dict[str, str]  # filename -> sha256 of emitted jsonl

class BenchmarkConverter(Protocol):
    name: str
    def convert(self, raw_dir: Path, out_dir: Path) -> ConversionManifest: ...

def spans_to_qrels(
    spans: Mapping[str, Sequence[tuple[str, int, int]]],  # query_id -> [(doc_id, start, end)]
    chunk_offsets: Mapping[str, tuple[str, int, int]],     # chunk_id -> (doc_id, start, end)
    *, min_frac: float = 0.5, min_chars: int = 200,        # D35
) -> dict[str, dict[str, int]]: ...
```

Rules: output is *exactly* the jsonl_loader schema (`chunks.jsonl`: text/doc_id/metadata; `queries.jsonl`: query_id/text; `anchor_qrels.jsonl`: query_id/qrels) — `Chunk.create` content-hash ids are recomputed by the loader, so converters emit a `metadata.upstream_id` field and qrels are re-keyed to content ids by the converter (it constructs the same `Chunk` objects to obtain ids). Emission order is upstream order; two conversions of the same raw archive are byte-identical (manifest sha256s prove it). Official qrels naming queries that fall outside the anchor split are dropped by the loader (existing behavior, logged) — converters do not pre-split.

Per-dataset notes: **fiqa** — passages 1:1 (D33), all splits' queries concatenated in upstream order. **nfcorpus** — abstracts 1:1, graded qrels binarized at grade ≥ 1. **legalbench_rag** — D34 chunking, D35 `spans_to_qrels`, D36 merge with `metadata.subcorpus`.

---

## 6. Splits, partition, C-ladder

- Split = the existing seeded 60/25/15 train(demand)/anchor/oracle permutation (PLAN D10) — no new split code except D36's per-sub-corpus stratification for legal (implemented as a converter-emitted `metadata.subcorpus` + a `split_stratify_by` param on the `jsonl` dataset, schema v2 only, absent ⇒ v1 behavior).
- Demand map `p̂` fit on `queries_train` embeddings only (existing movMF path); exemplars from train only; anchor is the τ reference; ORACLE arm draws from `queries_oracle` (`n_records = n_per_arm`).
- Reference partition: KMeans on MiniLM chunk embeddings at the D37-chosen `C`; frozen versioned artifact per dataset (existing partition machinery). The ladder decision runs once, before any arm, and its inputs (per-cluster anchor counts, expected seed allocations `n_seeds · q_c`) are logged in `metrics.json["partition_ladder"]`.

---

## 7. Adapters — one new wrapper, everything else exists

```python
@CHAT_MODELS.register("cached")
class CachedChatModel:
    """Transcript-record/replay decorator around any ChatModel (D40).

    Key = sha256 over (system, user, sorted kwargs). Modes:
    record (miss -> call backend, append jsonl, return), replay (miss -> raise
    actionable error naming the transcript path). to_config/from_config
    serialize the backend block + transcript_path + mode.
    """
    def complete(self, system: str, user: str, **kw: Any) -> str: ...
```

- Generator and judge each get their own transcript file. CI/tests wrap `mock` backends to prove the contract without network.
- Embeddings: the existing `EmbeddingStore` artifact caches the MiniLM matrix per (model, corpus-hash); embedding runs once per dataset.
- The §6.4 cross-family warning must NOT fire on any v2 config (D39); a test loads all three configs and asserts zero warnings.

---

## 8. Config — schema_version 2

This spec's features that require `schema_version: 2` are: `generator_llm.type: cached` / `judge_llm.type: cached` (transcript replay), `partition.ladder`, `split_stratify_by`, and `validator.audit_export`. The canonical schema-2 trigger list and the single-owner `from_yaml` loader-change semantics live in specs/v2/README.md ("schema_version 2 — canonical trigger list"); this spec does not redefine them. `schema_version: 1` configs load unchanged (regression-tested).

```yaml
# configs/v2_fiqa.yaml (schema v2) — nfcorpus/legalbench differ only where noted in the table below
ragsynth: {schema_version: 2, name: v2-fiqa, seed: 0}
resources:
  dataset:
    type: jsonl
    params:
      chunks_path: data/benchmarks/fiqa/chunks.jsonl
      queries_path: data/benchmarks/fiqa/queries.jsonl
      anchor_qrels_path: data/benchmarks/fiqa/anchor_qrels.jsonl
  embedder: {type: sentence_transformer, params: {model: sentence-transformers/all-MiniLM-L6-v2, device: cpu}}
  generator_llm:
    type: cached
    params:
      mode: record            # replay for reruns
      transcript_path: data/benchmarks/fiqa/transcripts/generator.jsonl
      backend: {type: openai_compatible,
                params: {base_url: "${RAGSYNTH_LLM_BASE_URL}", model: Qwen/Qwen2.5-7B-Instruct,
                         api_key_env: RAGSYNTH_LLM_API_KEY, temperature: 0.7}}
  judge_llm:
    type: cached
    params:
      mode: record
      transcript_path: data/benchmarks/fiqa/transcripts/judge.jsonl
      backend: {type: openai_compatible,
                params: {base_url: "${RAGSYNTH_LLM_BASE_URL}", model: meta-llama/Llama-3.1-8B-Instruct,
                         api_key_env: RAGSYNTH_LLM_API_KEY, temperature: 0.0}}
  retriever: {type: dense_inmemory}
  partition: {n_clusters: 8, ladder: {candidates: [8, 6, 4, 2], min_per_side: 30}}   # D37 — (b) evaluated on n_seeds from the sampler
  demand: {n_components: 8, lam: 0.7, tau_r_pct: 5.0}
artifacts_dir: experiments/v2_fiqa/artifacts
pipeline:
  - {type: seed_sampler.quota, params: {lam: 0.7, n_min: 3, n_seeds: 240, p_group: 0.2}}
  - {type: context_assembler, params: {k_style: 3}}
  - {type: generator, params: {n_candidates: 2, prompt_version: answer_first_v1}}
  - {type: gate, params: {checks: [dedup, zero_context, answerability, round_trip, uniqueness],
                          dedup: {cos_threshold: 0.95}, round_trip: {k: 10},
                          uniqueness: {mode: promote, top_m: 3}}}
  - {type: qrel_builder, params: {strategy: binary}}
  - {type: curator, params: {memorization_cos: 0.9}}
  - {type: validator, params: {arms: [a0, a1, a2, oracle], n_boot: 1000, n_per_arm: 160,
                               reuse_pipeline_for: a1, gates: {tau: 0.9, tau_ap: 0.8},
                               audit_export: {n: 160, arm: a2, stratify: [cluster, stratum]},
                               arm_params: {oracle: {n_records: 160}}}}
```

Per-dataset deltas: **nfcorpus** — paths; expect ladder to settle at C=4–6 (D37 arithmetic). **legalbench_rag** — paths; `split_stratify_by: subcorpus` on the dataset params; `n_seeds: 240` unchanged; expect C=2–4; per-sub-corpus τ appendix flag on the validator report.

---

## 9. Pre-registered expectations & threats to validity

Registered here, *before* any run; the report must quote this table and fill the outcome column.

**Primary threat (state it in every report): arm separation may compress vs the toy world.** Public query sets lack real demand skew — FiQA queries are genuine user questions (matched regime), but NFCorpus/legal queries were authored to cover the corpus (shifted regime). A flat true demand makes `p̂` ≈ uniform, which collapses the A0↔A1 quota gap by construction. The matched-vs-shifted contrast is therefore itself a pre-registered comparison: if A1/A2 separation is visible on FiQA but not on NFCorpus/legal, the harness is behaving as the theory predicts, not failing. Secondary threats: shallow FiQA qrels (unjudged holes deflate τ_AP first — exactly the R5 signature); D41 no time-decay; MiniLM as both pipeline and reporting embedder (the R1 held-out-embedder rule is *not* yet in force — note it).

| # | Expectation (per dataset unless noted) | If it holds | If it fails → trigger |
|---|---|---|---|
| E1 | ORACLE τ ≥ all synthetic arms (ceiling, SPEC §10) | harness sane | STOP: debug harness before any research trigger |
| E2 | FiQA: wC2ST(A2) < wC2ST(A1) ≈ wC2ST(A0); ESS/N(A1), ESS/N(A2) > ESS/N(A0) | toy mechanism transfers | per-stratum KL/C2ST high at matched quotas or ESS sags ⇒ **R1** (scaled A2, held-out embedder) |
| E3 | τ(A2) within oracle CI on FiQA; regression-only band acceptable on NFCorpus/legal | matched-regime validity | gate pass-rate AND τ plateau below gates across datasets ⇒ **R2** (closed-loop prompt optimization) |
| E4 | τ_AP tracks τ (gap < 0.1) | qrel depth adequate | τ_AP stalls < 0.8 while τ ≥ 0.9 (expected first on FiQA) ⇒ **R5** (graded/pooled qrels) |
| E5 | Positive controls: every arm detects drop_index(0.10); A1 may miss noise_transform (toy signature) | metric stack transfers | any arm missing drop_index ⇒ report that arm as regression-detection-unsafe; no research trigger |
| E6 | NFCorpus/legal: A0↔A1 compression per the primary threat | threat model correct | strong separation on shifted data ⇒ revisit the demand-skew assumption in PLAN |

The positive-control battery reuses the v1 degradation factory and `MatrixSystem` zoo (`metrics/validity/systems.py` is dimension-agnostic; d=384 works unchanged).

---

## 10. Runtime budget & call-count estimate (n_candidates = 2)

Per dataset, per generated arm (A0/A1/A2; ORACLE makes zero LLM calls): generation = 240 seeds × 2 = **480 calls**; judge ≤ 480 (zero_context) + 480 (answerability) + 3 × 480 = 1,440 (uniqueness: the shipped `UniquenessCheck` issues one judge call per non-gold retrieved chunk, so `top_m: 3` ⇒ up to 3 per candidate) = **≤ 2,400 calls**; arm ≤ 2,880 ⇒ **≤ 8,640 LLM calls per dataset**, ≤ ~25.9K for all three. These are upper bounds; gate short-circuiting (a candidate failing an earlier check never reaches uniqueness) only lowers actuals. Budgets (report actuals vs these):

- Embedding: CPU-feasible — FiQA worst case ~65K texts through MiniLM ≤ 45 min CPU, once (EmbeddingStore artifact); NFCorpus/legal ≤ 10 min.
- First (record) run: ≤ 4.5 h wall per dataset at ~1.5 s/call sequential (8,640 × 1.5 s ≈ 3.6 h + overhead).
- Replay run: ≤ 20 min per dataset, fully offline, deterministic (D40).
- CI unchanged: mocks + mini-fixtures only, existing < 10 min bar.

---

## 11. Human precision-audit export

`validator.audit_export` writes `experiments/v2_<name>/audit/audit_sample.csv`: the 160 curated A2 records (D38), stratified by (cluster, stratum), columns = `record_id, query_text, evidence_texts, gold_chunk_ids, cluster, stratum, gate_scores` + blank `annotator_verdict, annotator_note` columns. Randomized row order (seeded), no arm/metric columns that could bias annotators. Target: annotation precision ≥ 90–95% (SPEC §16 noise floor); the audit itself is human work outside this spec — the export just has to make it a spreadsheet task.

---

## 12. Determinism statement

Guaranteed: same seed + same converted jsonl + same transcripts ⇒ byte-identical `metrics.json` (the v1 guarantee, extended through `cached` replay). NOT guaranteed and explicitly out of scope: live-endpoint first-run responses (D40), upstream archive stability (mitigated: manifest sha256s fail loudly on drift), MiniLM weights immutability (mitigated: model name + revision pinned in config, embedding artifact hashed).

---

## 13. v2 acceptance criteria (Definition of Done)

**13.1 Functional:** `fetch → convert → run(record) → run(replay)` documented in README and executed for all three datasets; each produces `experiments/v2_<name>/{metrics.json, report.md, figures/, audit/audit_sample.csv}` with the 4-arm table, dual-view efficiency block, positive-control table, chosen-C ladder log, per-sub-corpus τ appendix (legal), and the §9 expectations table with outcomes filled; replay twice ⇒ identical `metrics.json`.

**13.2 Compatibility:** all v1 configs load and run unchanged (byte-identical toy `metrics.json`); `schema_version: 2` required iff a v2 feature is used (canonical trigger list: specs/v2/README.md); unknown-type registry errors stay actionable. The byte-identical claim is absolute across v2: spec 05's per-item RNG execution path is opt-in (schema-2 execution configs only) and does not touch schema-1 byte-stability.

**13.3 Verification commands (green at every commit):**
```
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -q --cov=ragsynth --cov-fail-under=70   # air-gapped; converters via mini-fixtures
uv run ragsynth run --config configs/v1_toy.yaml       # v1 regression
```

**13.4 Known-value fixtures (write FIRST):** `spans_to_qrels`: span (100, 400) with chunks (0, 250)/(200, 450)/(380, 600) ⇒ overlaps 150/200/20, threshold min(150, 200)=150 ⇒ gold/gold/not — hand-derive in the test; short span (100, 130) fully inside one chunk ⇒ only that chunk. D34 splitter: crafted 2,300-char doc ⇒ exact window offsets, whitespace snap verified, overlap region duplicated. D37 ladder: synthetic anchor counts [40,35,31,30,29,…] at C=8 fail the floor ⇒ ladder descends; hand-built case where (b) binds but (a) passes. `cached` chat: record then replay ⇒ 1 backend call, identical outputs; replay miss ⇒ ImportError-grade actionable message naming the path. Converter mini-fixtures: ≤ 10-doc bundled raw samples per dataset under `tests/fixtures/benchmarks/` ⇒ exact expected jsonl bytes + manifest. Config: all three v2 configs round-trip byte-stably; §6.4 warning silent on all three (D39).

**13.5 Docs:** README v2 quickstart (fetch/convert/run/replay); `data/benchmarks/README.md` generated with licenses; PLAN.md gains D32–D41; each report ends with findings/decision/next-hypotheses (SPEC §14).

---

## 14. Reference table (additions to SPEC §17)

| Feature | References |
|---|---|
| FiQA-2018 | Maia et al., WWW 2018 (challenge); Thakur et al. (BEIR), NeurIPS D&B 2021 |
| NFCorpus | Boteva et al., ECIR 2016 |
| LegalBench-RAG + sub-corpora | Pipitone & Houir Alami 2024; Hendrycks et al. (CUAD), NeurIPS 2021; Koreeda & Manning (ContractNLI), EMNLP Findings 2021; Wang et al. (MAUD), 2023; Ravichander et al. (PrivacyQA), EMNLP 2019 |
| MiniLM embedder | Wang et al. (MiniLM), NeurIPS 2020; Reimers & Gurevych (SBERT), EMNLP 2019 |
| Pre-registration discipline | Sakai, SIGIR 2006 (discriminative power); Rahmani et al., SIGIR 2024 (τ expectations) |

---

## 15. Open questions — defaults for the agent (decide, document, proceed)

1. BEIR download route: pull the BEIR-hosted zips directly (no `beir` package — it would be a needless runtime dep in a script). Default: plain `urllib` + sha256 pin in the fetch script.
2. NFCorpus binarization grade: default ≥ 1 (any relevance); document; revisit only if R5 triggers.
3. FiQA query set: use all splits' queries as the production pool (our 60/25/15 re-split supersedes upstream splits; upstream test-split identity is irrelevant to our design). Default: yes, concatenate.
4. `${RAGSYNTH_LLM_BASE_URL}` env interpolation in configs: only inside `openai_compatible` params, resolved at adapter construction (never serialized resolved). Default: yes, minimal.
5. Legal `n_seeds` if a sub-corpus is too small to fill quotas: let the existing `n_min` floor handle it; do not special-case. Default: yes.
6. Judge temperature 0.0 / generator 0.7 (shown in §8): keep; both recorded in transcripts so replay is exact either way.
