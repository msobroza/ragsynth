# EvalReport - v1-toy

Config hash: `4e6595110976` - seed 0

Reading guide: KL/C2ST/wC2ST/MMD lower is better (0.5 = indistinguishable
for AUCs); ESS/N higher is better; tau/tau_AP/RBO vs the anchor ranking
higher is better; control p-values < 0.05 mean the arm detects the
injected regression.

| arm | n | KL | C2ST | wC2ST | MMD | ESS/N | gap | tau | tau 95% CI | tau_AP | RBO | PC drop | PC noise | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a0 | 500 | 0.190 | 1.000 | 1.000 | 0.1107 | 0.55 | 0.00 | 0.716 | [0.68, 0.75] | 0.883 | 0.97 | 0.000 | 0.139 | fail |
| a1 | 500 | 0.421 | 0.918 | 1.000 | 0.0319 | 0.89 | 0.00 | 0.921 | [0.79, 0.92] | 0.955 | 0.98 | 0.000 | 0.376 | PASS |
| a2 | 500 | 0.301 | 0.730 | 0.674 | 0.0049 | 0.91 | 0.00 | 0.909 | [0.85, 1.00] | 0.892 | 0.96 | 0.000 | 0.000 | PASS |
| oracle | 500 | 0.362 | 0.486 | 0.501 | 0.0000 | 0.99 | 0.00 | 1.000 | [0.91, 1.00] | 1.000 | 1.00 | 0.000 | 0.000 | PASS |

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

Dual view (SPEC §8-9): demand-weighted headline nDCG@10 = 1.000 (unweighted 1.000, ESS/N 0.89); per-cluster table below; worst clusters: [0, 1, 2]; zero-query clusters: [].

| cluster | p_hat | n_synth | mean nDCG |
|---|---|---|---|
| 0 | 0.404 | 132 | 1.000 |
| 1 | 0.181 | 87 | 1.000 |
| 2 | 0.122 | 65 | 1.000 |
| 3 | 0.087 | 56 | 1.000 |
| 4 | 0.057 | 39 | 1.000 |
| 5 | 0.067 | 45 | 1.000 |
| 6 | 0.039 | 37 | 1.000 |
| 7 | 0.043 | 39 | 1.000 |

Gate reject reasons: {'dedup': 224, 'zero_context': 37}

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
