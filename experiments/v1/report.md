# EvalReport - v1-toy

Config hash: `4e6595110976` - seed 0

Reading guide: KL/C2ST/wC2ST/MMD lower is better (0.5 = indistinguishable
for AUCs); ESS/N higher is better; tau/tau_AP/RBO vs the anchor ranking
higher is better; control p-values < 0.05 mean the arm detects the
injected regression.

| arm | n | KL | C2ST | wC2ST | MMD | ESS/N | gap | tau | tau 95% CI | tau_AP | RBO | PC drop | PC noise | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a0 | 500 | 0.198 | 1.000 | 1.000 | 0.1121 | 0.55 | 0.00 | 0.716 | [0.68, 0.75] | 0.883 | 0.97 | 0.000 | 0.139 | fail |
| a1 | 500 | 0.382 | 0.916 | 1.000 | 0.0338 | 0.90 | 0.00 | 0.822 | [0.71, 0.88] | 0.830 | 0.94 | 0.000 | 1.000 | fail |
| a2 | 500 | 0.383 | 0.716 | 0.680 | 0.0047 | 0.91 | 0.00 | 0.909 | [0.85, 1.00] | 0.892 | 0.96 | 0.000 | 0.000 | PASS |
| oracle | 500 | 0.318 | 0.489 | 0.466 | 0.0000 | 0.99 | 0.00 | 1.000 | [0.91, 1.00] | 1.000 | 1.00 | 0.000 | 0.000 | PASS |

## a0

Dual view (SPEC §8-9): demand-weighted headline nDCG@10 = 1.000 (unweighted 1.000, ESS/N 0.55); per-cluster table below; worst clusters: [0, 1, 2]; zero-query clusters: [].

| cluster | p_hat | n_synth | mean nDCG |
|---|---|---|---|
| 0 | 0.404 | 63 | 1.000 |
| 1 | 0.181 | 64 | 1.000 |
| 2 | 0.122 | 59 | 1.000 |
| 3 | 0.087 | 60 | 1.000 |
| 4 | 0.057 | 70 | 1.000 |
| 5 | 0.067 | 62 | 1.000 |
| 6 | 0.039 | 59 | 1.000 |
| 7 | 0.043 | 63 | 1.000 |

## a1

Dual view (SPEC §8-9): demand-weighted headline nDCG@10 = 1.000 (unweighted 1.000, ESS/N 0.90); per-cluster table below; worst clusters: [0, 1, 2]; zero-query clusters: [].

| cluster | p_hat | n_synth | mean nDCG |
|---|---|---|---|
| 0 | 0.404 | 135 | 1.000 |
| 1 | 0.181 | 90 | 1.000 |
| 2 | 0.122 | 62 | 1.000 |
| 3 | 0.087 | 54 | 1.000 |
| 4 | 0.057 | 40 | 1.000 |
| 5 | 0.067 | 44 | 1.000 |
| 6 | 0.039 | 38 | 1.000 |
| 7 | 0.043 | 37 | 1.000 |

Gate reject reasons: {'dedup': 227, 'zero_context': 34}

## a2

Dual view (SPEC §8-9): demand-weighted headline nDCG@10 = 1.000 (unweighted 1.000, ESS/N 0.91); per-cluster table below; worst clusters: [0, 1, 2]; zero-query clusters: [].

| cluster | p_hat | n_synth | mean nDCG |
|---|---|---|---|
| 0 | 0.404 | 169 | 1.000 |
| 1 | 0.181 | 83 | 1.000 |
| 2 | 0.122 | 49 | 1.000 |
| 3 | 0.087 | 71 | 1.000 |
| 4 | 0.057 | 45 | 1.000 |
| 5 | 0.067 | 18 | 1.000 |
| 6 | 0.039 | 27 | 1.000 |
| 7 | 0.043 | 38 | 1.000 |

## oracle

Dual view (SPEC §8-9): demand-weighted headline nDCG@10 = 1.000 (unweighted 1.000, ESS/N 0.99); per-cluster table below; worst clusters: [0, 1, 2]; zero-query clusters: [].

| cluster | p_hat | n_synth | mean nDCG |
|---|---|---|---|
| 0 | 0.404 | 210 | 1.000 |
| 1 | 0.181 | 93 | 1.000 |
| 2 | 0.122 | 54 | 1.000 |
| 3 | 0.087 | 38 | 1.000 |
| 4 | 0.057 | 27 | 1.000 |
| 5 | 0.067 | 34 | 1.000 |
| 6 | 0.039 | 27 | 1.000 |
| 7 | 0.043 | 17 | 1.000 |

---

## Findings, decision, next-iteration hypotheses (SPEC §14)

**Findings.** The packaged pipeline reproduces the vendored prototype's
4-arm table at seed 0: (1) A2 is the only arm matching within-cluster shape
(wC2ST 0.680 vs ~1.0 for the chunk-first arms; MMD an order of magnitude
lower) — quota matching fixes marginals, not the within-cluster term, which
is exactly the post-stratification residual the theory predicts (SPEC §16).
(2) A1's ESS/N (0.90) is near the oracle's (0.99) while A0 sits at 0.55 —
demand quotas work. (3) A0 and A1 fail the tau usability gate (0.716 /
0.822 < 0.9); A2 passes (0.909); the oracle ceiling is 1.0 with its own CI.
(4) The positive-control battery separates the arms: every arm detects a
10% index deletion, but A1 completely misses the embedding-noise regression
(p = 1.00; A0 = 0.14) while A2 and the oracle detect it (p < 0.001) — a
distributionally-off benchmark can be insensitive to real regressions.

**Tuning trace** (what it took to reproduce the table; PLAN D21-D26): cap
every arm at n_per_arm; relabel_nearest qrels (the §10 gate-style
relabeling); A2 toy emission noise 0.42 (the vMF kappa=400 dispersion the
exemplar-mean base loses); p_group 0 for toy quota arms (chunk-group
midpoint emissions are noise-fragile by construction); post-review, the
§8-compliant wC2ST floor (30/side) and equal-n shared-reference fidelity
(PLAN D27) sharpened the same qualitative table.

**Decision.** The 4-arm harness and metric stack are trustworthy on the toy
world; v1 ships. The jsonl mini-corpus run (experiments/v1_local_corpus)
correctly reports itself as regression-detection-only (all gates fail with
hashed n-grams + mock LLM) — the gates do their job on weak setups.

**Next-iteration hypotheses (SPEC §16 triggers).** R1 (scaled A2) when a
real corpus shows per-stratum KL/C2ST staying high at matched quotas; R2
(closed-loop prompt optimization over FidelityObjective) when gate
pass-rate and tau plateau; R5 (graded/pooled qrels) if tau_AP stalls below
gate while tau passes. First real-data step: swap the dataset block for a
production chunks/queries export, keep the validator, and run the
benchmark-validation battery (oracle ceiling, per-stratum tau CIs, positive
controls, ~150-200 stratified records for the human precision audit).
