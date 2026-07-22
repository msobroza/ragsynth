# v2 spec 04 — soft demand & diversity metrics (movMF soft demand, Vendi, MAUVE)

> **Audience:** Claude Fable 5 coding agent (with superpowers skills).
> **Author of record:** Max (ML Tech Lead). Child spec of `SPEC.md` §2.2 ("Vendi score, MAUVE, movMF *soft* demand map as default → optional/v2"). SPEC.md remains law where this document is silent.
> **Status:** v2 feature spec — ready to plan against. New decisions land in `PLAN.md` as D62–D68.
> Decision numbers D62–D68 allocated per `specs/v2/README.md`.

---

## 0. How to execute this spec (read first)

1. **Plan before coding.** Extend `PLAN.md` (D62–D68 recorded, file-by-file plan) before implementation.
2. **TDD.** Every fixture in §9 is written FIRST (SPEC §15.4 style), then implemented.
3. **Do not touch v1 behavior.** The bar for every commit: SPEC §15.3 commands green AND a schema-1 config (`configs/v1_toy.yaml`) produces **byte-identical** `metrics.json` + `report.md` versus the v1-accepted baseline.
4. **Success = §10 acceptance checklist.** Nothing else is the bar.

---

## 1. Scope & the constraint carved in stone

Three additions, all **report-only** (no new gates), all offline-deterministic:

1. A **soft demand estimator** (`soft_movmf`) as an opt-in alternative to the v1 hard-histogram demand `p_hat` — changes demand **estimation only**.
2. **Vendi score** (Friedman & Dieng, TMLR 2023) in the diversity block.
3. **MAUVE** (Pillutla et al., NeurIPS 2021) in the fidelity block, as a pure-sklearn reimplementation of the quantized divergence frontier.

**Carved in stone by v1 (SPEC §7.4, do not relitigate):** the REPORTING partition stays the frozen, versioned hard-KMeans `ReferencePartition`. ESS, quotas, coverage, importance weights, post-stratification, per-cluster tables — all keep operating on its `C` cluster ids. Changing the partition is a benchmark-migration event. The soft map is only a different way of putting probability mass **onto that same frozen partition**.

---

## 2. Key decisions (do not relitigate)

| # | Decision | Rationale |
|---|---|---|
| D62 | `resources.demand.estimator: hard_kmeans \| soft_movmf`; default **stays `hard_kmeans`** in v2.0. `soft_movmf` is the promotion *candidate*; promotion to default is a separate future PLAN decision, triggered only by evidence (§11 Q1). | "Strictly opt-in until promoted" (acceptance §10). |
| D63 | Soft→hard aggregation = frozen **transport matrix** `T (K×C)` built by responsibility-weighted assignment of each fit-time production query to its hard cluster (§3.2). `component_map: empirical` default; `density` (seeded MC) optional. | Components and clusters need not align; `T` pins the mapping to the benchmark epoch. |
| D64 | Vendi = pure-numpy cosine-kernel eigenvalue entropy, order `q=1` only, per-side subsample cap `max_n=2000` (same convention as `mmd_rbf`). No `vendi-score` dep. | Trivial to implement; zero new runtime deps. |
| D65 | MAUVE = **reimplement, not depend**. `mauve-text` is Apache-2.0 but drags torch/transformers/faiss-shaped weight; we ship `mauve_frontier` on numpy+sklearn with documented deviations (§5.2). No `mauve` extra. | Air-gapped CI, lean runtime deps (SPEC §1). |
| D66 | Report-only everywhere: reference bands (like the KL ≤0.16 band), **no gates**. New `metrics.json` keys are **absent when the feature is disabled** — never emitted as `null` — so schema-1 outputs stay byte-identical. | v1 byte-stability is the regression contract. |
| D67 | schema_version-2 triggers and the `Pipeline.from_yaml` loader relaxation are owned by the canonical section in `specs/v2/README.md` ("schema_version 2 — canonical trigger list"); of this spec's features, `resources.demand.estimator: soft_movmf` requires `schema_version: 2`. | SPEC §13 discipline. |
| D68 | Zero new runtime dependencies (numpy/scipy/sklearn already present); everything seeds through named `resources.rng(...)` substreams. | Determinism §8. |

---

## 3. Soft demand estimator (`sampling/demand.py` + composition root)

### 3.1 What changes and what does not

- **Unchanged:** `ReferencePartition` (frozen artifact), `DemandArtifact.p_hat` as the single field every downstream consumer reads (`cluster_importance_weights`, ESS, quotas, coverage trio, post-stratification), `movmf_demand`/`tilted`/`tau_r` used by A2's `SpecSampler`.
- **Changed (opt-in):** *how* `p_hat` over the C hard clusters is estimated. `hard_kmeans` = `partition.proportions(train_embs)` (v1, unchanged, default). `soft_movmf` = §3.2.

### 3.2 The mapping: soft movMF component mass → frozen hard clusters

Let `γ ∈ R^{n×K}` = `MovMF.responsibilities(train_embs)` (K components, default 16), `h_i = partition.assign(q_i) ∈ [0, C)` the frozen hard label of fit-time production query `i` (C clusters, default 8; `K ≠ C` in general).

**Step 1 — component demand (existing code):** `m = demand_from_responsibilities(γ, timestamps, half_life) ∈ Δ^K`. Time decay, when configured, applies **here**, at component level.

**Step 2 — frozen transport matrix (new):** responsibility-weighted assignment of each production query to its hard cluster, *undecayed* (the structure map is part of the benchmark epoch):

```
T[k, c] = Σ_i γ_k(q_i) · 1[h_i = c]  /  Σ_i γ_k(q_i)
```

Guard: if `Σ_i γ_k(q_i) < 1e-12` (starved component), set row `k` one-hot at `partition.assign(μ_k)`. With `component_map: density`, replace the empirical row by a seeded Monte-Carlo estimate `T[k, c] ≈ P_{z~vMF(μ_k, κ_k)}(assign(z) = c)` (`n_mc=4096` draws per component, rng substream `"demand.soft_map"`) — model smoothing that can put mass on clusters no fit query occupies (cold-start coverage).

**Step 3 — projection:** `p_hat = normalize(Tᵀ m) ∈ Δ^C`. ESS, quotas, and coverage consume this exactly as before — same partition, same shapes, same downstream code.

### 3.3 Invariants (each is a §9 test)

- **I1 (required invariant):** with **one-hot responsibilities** (and no decay), `soft_movmf` reproduces the `hard_kmeans` `p_hat` **exactly** (`atol=1e-12`). Proof sketch: one-hot γ makes `m_k = n_k/n` and `T[k,c]` the within-component hard histogram; `Tᵀm` telescopes to the global hard histogram.
- **I2 (continuity/no-op property):** for *any* soft γ, empirical map, same fit queries, no decay: `Tᵀm` telescopes to the hard histogram exactly (`m_k` cancels `T`'s denominator). Switching the estimator on a timestamp-free config is a provable no-op on `p_hat` — soft estimates differ only via (a) component-level time decay, (b) `component_map: density`, (c) future refreshed-traffic `m` (roadmap R6 lifecycle, out of scope here).
- **I3:** `p_hat` is a probability vector: nonnegative, sums to 1 (`atol=1e-12`), length `C`.
- **I4 (density map):** every cluster containing at least one component mean gets `p_hat > 0`.

### 3.4 Contracts

```python
# sampling/demand.py (new)
def component_cluster_transport(
    movmf: MovMF,
    partition: ReferencePartition,
    fit_embs: NDArray[np.float64],
    *,
    mode: Literal["empirical", "density"] = "empirical",
    n_mc: int = 4096,
    rng: Generator | None = None,   # required iff mode == "density"
) -> NDArray[np.float64]: ...       # (K, C), rows sum to 1

def soft_demand_on_partition(
    transport: NDArray[np.float64],       # (K, C)
    movmf_demand: NDArray[np.float64],    # (K,) — demand_from_responsibilities output
) -> NDArray[np.float64]: ...             # (C,), sums to 1
```

`DemandArtifact` gains defaulted fields `estimator: str = "hard_kmeans"` and `transport: NDArray | None = None` (provenance). When `soft_movmf` is active, the composition root persists `demand-transport.npz` (transport + `component_map` + `fitted_on_hash`) via the artifact store, next to `demand-movmf.npz`. Dispatch lives in `pipeline/serialization.py::build_resources` only — steps see an unchanged `Resources`.

---

## 4. Vendi score (`metrics/diversity.py`)

Effective number of distinct samples: `VS(X) = exp(−Σ_i p_i ln p_i)` where `p` are the eigenvalues of the trace-normalized cosine kernel `K = X Xᵀ / n` (rows L2-normalized; `K_ii = 1/n` so trace is 1). Friedman & Dieng, TMLR 2023.

```python
def vendi_score(
    embs: NDArray[np.float64], *, max_n: int = 2000, seed: int = 0
) -> float: ...
```

- Subsample cap identical in spirit to `mmd_rbf`: if `n > max_n`, seeded uniform subsample without replacement.
- `np.linalg.eigvalsh` on the symmetric kernel; clip eigenvalues at 0 (float noise); renormalize to sum 1 defensively; `0·ln 0 = 0`.
- Joins the diversity block **report-only** — no gate. Report both `vendi` and the ratio `vendi / n_used` (effective-diversity fraction) so the number is comparable across arms of different size.
- Known-value fixtures (§9): `n` identical points ⇒ `1.0`; `n` orthonormal points ⇒ `n`.

---

## 5. MAUVE (`metrics/fidelity.py`)

### 5.1 Algorithm — `mauve_frontier` (pure numpy + sklearn)

Quantized divergence-frontier area (Pillutla et al., NeurIPS 2021), on the pipeline's own embeddings:

1. **Equal-n subsample** each side to `min(len(real), len(synth), max_n=2000)` (seeded; same equal-n fidelity convention as the v1 audit fix).
2. **Quantize:** `KMeans(n_buckets, n_init=10, random_state=seed)` on the pooled sample; `n_buckets = min(128, max(4, n_side // 10))`.
3. **Histograms** `p` (real), `q` (synth) over buckets; additive smoothing `ε=1e-6`, renormalized.
4. **Frontier:** for `λ` on a 25-point open grid in (0, 1): `r = λp + (1−λ)q`; point `(exp(−c·KL(q‖r)), exp(−c·KL(p‖r)))` with `c=5`; append endpoints `(0, 1)` and `(1, 0)`; sort by x.
5. **MAUVE = trapezoid AUC** of the frontier, in (0, 1]; identical distributions ⇒ 1.

```python
def mauve_frontier(
    real: NDArray[np.float64],
    synth: NDArray[np.float64],
    *,
    n_buckets: int | None = None,   # None => the formula above
    scale_c: float = 5.0,
    grid_size: int = 25,
    max_n: int = 2000,
    seed: int = 0,
) -> float: ...
```

### 5.2 Documented deviations from the reference implementation

State these verbatim in the docstring; numbers are **not comparable to published MAUVE values**, only within-repo across arms and epochs (which is the report-only intent):

- Features are the pipeline embedder's vectors, not GPT-2 activations; no PCA/whitening before quantization.
- Plain seeded sklearn KMeans (no faiss, no GPU path); fixed defaults `c=5`, 25-point grid.
- Per-side `max_n` cap with equal-n subsampling.

Joins the fidelity block **report-only**. Reference band (not a gate): same-family sets land ≥ 0.90; a same-sample self-test sits at ≈ 1.0.

---

## 6. Config & serialization (schema v2)

```yaml
# configs/v2_toy_soft.yaml (schema v2) — deltas vs v1_toy.yaml only
ragsynth: {schema_version: 2, name: v2-toy-soft, seed: 0}
resources:
  demand:
    estimator: soft_movmf        # default: hard_kmeans (v1 behavior, unchanged)
    component_map: empirical     # or: density
    n_mc: 4096                   # density map only
    n_components: 16
    lam: 0.7
    tau_r_pct: 5.0
    half_life: null
pipeline:
  # ... unchanged through curator ...
  - {type: validator, params: {arms: [a0, a1, a2, oracle], n_boot: 1000,
                               gates: {tau: 0.9, tau_ap: 0.8},
                               v2_metrics: {vendi: true, mauve: true}}}
```

Rules (SPEC §13 still governs): schema_version-2 triggers and the `Pipeline.from_yaml` loader change follow the canonical single-owner section in `specs/v2/README.md` ("schema_version 2 — canonical trigger list"); of this spec's features, `resources.demand.estimator: soft_movmf` requires `schema_version: 2`. Round-trip byte-stable under schema 2; schema-1 configs load **unchanged** (defaults: `estimator: hard_kmeans`, `v2_metrics` absent ⇒ both off). `validator.to_config()` emits `v2_metrics` only when at least one flag is true (round-trip stability without polluting v1 configs).

---

## 7. Reporting — where the numbers land

`metrics.json` (per arm, keys **present only when enabled**, D66):

- `arms.<arm>.fidelity.mauve: float`
- `arms.<arm>.diversity.vendi: float`, `arms.<arm>.diversity.vendi_ratio: float`, `arms.<arm>.diversity.vendi_n_used: int`

`report.md`: the headline table gains `MAUVE*` and `Vendi/n` columns **only when enabled** (footnote: `*` = frontier reimplementation, §5.2; not comparable to published MAUVE). Reference bands printed alongside, styled like the KL band line: `MAUVE* ≥ 0.90 (same-family band)`; Vendi has no band — it is contextualized by `n_used`. When `soft_movmf` is active, the report's demand section states `demand estimator: soft_movmf (component_map=<mode>)` and shows hard-vs-soft `p_hat` side by side (the delta is the smoothing effect).

**Explicitly NO new gates in this iteration.** `gates_passed` logic is untouched.

Figures (optional, not acceptance-blocking): divergence-frontier curve per arm; hard-vs-soft `p_hat` bar pair.

---

## 8. Determinism & licensing

- All three features are pure computation over in-memory arrays — **no LLM, embedding-service, or external-DB calls**, so the v1 guarantee holds in full: same seed, same process ⇒ identical `metrics.json`. Named substreams: `"validator.vendi.<arm>"`, `"validator.mauve.<arm>"`, `"demand.soft_map"`. There is no feature in this spec that cannot honor determinism.
- Zero new runtime deps (numpy/scipy/sklearn, all BSD). `mauve-text` (Apache-2.0) and `vendi-score` (MIT) are **not** added, not even as extras (D65/D64); their licenses are noted here for the record since we reimplement their published algorithms with citation.
- mypy strict + `ruff select=ALL` stay green; full suite runs air-gapped.

---

## 9. Test plan — known-value fixtures FIRST (SPEC §15.4 style)

- `vendi_score`: n identical points ⇒ `1.0` (`atol 1e-9`); n orthonormal rows ⇒ `n`; one duplicated point among orthonormal rows ⇒ strictly inside `(n−1, n)`; permutation-invariant.
- `mauve_frontier`: same sample both sides ⇒ ≥ 0.99; two far-separated vMF blobs ⇒ ≤ 0.10; output in (0, 1]; deterministic across two calls with the same seed.
- Soft demand: **I1** one-hot-responsibility fixture ⇒ soft `p_hat` == hard `p_hat` exactly (`atol 1e-12`); **I2** soft γ, empirical map, no decay ⇒ equality again (telescoping); **I3** simplex property (hypothesis-style random γ); **I4** density-map positive-mass property; transport rows sum to 1; starved-component guard hits the `assign(μ_k)` fallback.
- Serialization: schema-2 round-trip byte-stable; v2 key under schema 1 ⇒ `ValueError`; schema-1 config still loads with `estimator=hard_kmeans` and no v2 metric keys in output.
- **Byte-stability regression:** run `configs/v1_toy.yaml` on this branch ⇒ `metrics.json` and `report.md` byte-identical to the v1-accepted baseline.
- Contract tests: any new/changed step params keep passing the registry-parametrized `PipelineStepContract`.

---

## 10. Acceptance (Definition of Done)

- [ ] Schema-1 configs produce **byte-identical** `metrics.json` + `report.md` vs the v1 baseline (soft map and v2 metrics strictly opt-in).
- [ ] `configs/v2_toy_soft.yaml` runs offline end-to-end < 10 min CPU; same seed twice ⇒ identical `metrics.json`; YAML round-trip byte-stable under schema 2.
- [ ] All §9 fixtures pass, written before implementation; SPEC §15.3 commands green at every commit (ruff, mypy, pytest cov ≥ 70, toy run).
- [ ] `to_config()/from_config()` implemented for every touched step/param; unknown keys still raise the actionable registry error.
- [ ] Report-only confirmed: no change to any `gates_passed` computation; new numbers carry reference bands only.
- [ ] `PLAN.md` updated with D62–D68 verbatim from §2; docstrings cite (Friedman & Dieng, TMLR 2023), (Pillutla et al., NeurIPS 2021), (Banerjee et al., JMLR 2005), (Kong, 1992).
- [ ] Figures optional; if added, matplotlib-only in core.

---

## 11. Open questions — defaults for the agent (decide, document, proceed)

1. **Promote `soft_movmf` to default?** No — deferred. Trigger to revisit: on ≥ 2 real corpora, soft `p_hat` reduces month-over-month demand-estimate churn (L1 distance between consecutive epochs) without moving τ/τ_AP beyond bootstrap noise. Record outcome as a future PLAN decision.
2. **MAUVE constants** (`c=5`, 25-point grid, bucket formula): keep reference-style defaults, config-exposed; do not sweep in v2.
3. **Vendi Rényi order:** fix `q=1` (Shannon). Order-grid `q ∈ {0.5, 1, 2, ∞}` is out of scope.
4. **Cross-check vs `mauve-text`:** no test dependency; a one-off offline parity note (direction-of-ranking agreement on the toy world) goes in the v2 experiment `report.md`, not in CI.
5. **Traffic window for soft `m`:** v2 estimates from the train split, same as v1. Refresh-window plumbing belongs to roadmap R6 (lifecycle), not here.
6. **Metric naming:** JSON key stays `mauve` (report column `MAUVE*` + footnote); renaming to `mauve_star` would churn downstream readers for no benefit.

---

## 12. References

- Friedman & Dieng, "The Vendi Score: A Diversity Evaluation Metric for Machine Learning" (TMLR 2023).
- Pillutla et al., "MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers" (NeurIPS 2021).
- Banerjee et al., "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions" (JMLR 2005).
- Kong, "A Note on Importance Sampling using Standardized Weights" (1992); Holt & Smith (JRSS A 1979) — post-stratification consumers of `p_hat`.
- SPEC.md §2.2, §7.2–7.4, §8–9, §13, §16 — parent contracts this spec extends.
