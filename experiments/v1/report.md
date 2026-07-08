# EvalReport - v1-toy

Config hash: `d011ee4a578d` - seed 0

Reading guide: KL/C2ST/wC2ST/MMD lower is better (0.5 = indistinguishable
for AUCs); ESS/N higher is better; tau/tau_AP/RBO vs the anchor ranking
higher is better; control p-values < 0.05 mean the arm detects the
injected regression.

| arm | n | KL | C2ST | wC2ST | MMD | ESS/N | gap | tau | tau 95% CI | tau_AP | RBO | PC drop | PC noise | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a0 | 1309 | 0.169 | 1.000 | 1.000 | 0.1191 | 0.51 | 0.00 | 0.939 | [0.91, 0.94] | 0.927 | 0.97 | 0.000 | 0.000 | PASS |
| a1 | 1345 | 0.227 | 0.928 | 1.000 | 0.0310 | 0.93 | 0.00 | 0.970 | [0.94, 0.97] | 0.955 | 0.98 | 0.000 | 0.000 | PASS |
| a2 | 301 | 2.449 | 0.754 | 0.709 | 0.0015 | 0.93 | 0.00 | 0.909 | [0.88, 0.97] | 0.868 | 0.95 | 0.000 | 0.000 | PASS |
| oracle | 500 | 0.362 | 0.486 | 0.501 | 0.0000 | 0.99 | 0.00 | 1.000 | [0.91, 1.00] | 1.000 | 1.00 | 0.000 | 0.000 | PASS |

## a0

Dual view (SPEC §8-9): demand-weighted headline nDCG@10 = 1.000 (unweighted 1.000, ESS/N 0.51); per-cluster table below; worst clusters: [0, 1, 2]; zero-query clusters: [].

| cluster | p_hat | n_synth | mean nDCG |
|---|---|---|---|
| 0 | 0.404 | 151 | 1.000 |
| 1 | 0.181 | 170 | 1.000 |
| 2 | 0.122 | 148 | 1.000 |
| 3 | 0.087 | 166 | 1.000 |
| 4 | 0.057 | 183 | 1.000 |
| 5 | 0.067 | 158 | 1.000 |
| 6 | 0.039 | 163 | 1.000 |
| 7 | 0.043 | 170 | 1.000 |

## a1

Dual view (SPEC §8-9): demand-weighted headline nDCG@10 = 0.999 (unweighted 0.999, ESS/N 0.93); per-cluster table below; worst clusters: [2, 1, 3]; zero-query clusters: [].

| cluster | p_hat | n_synth | mean nDCG |
|---|---|---|---|
| 0 | 0.404 | 400 | 1.000 |
| 1 | 0.181 | 221 | 0.999 |
| 2 | 0.122 | 168 | 0.998 |
| 3 | 0.087 | 137 | 0.999 |
| 4 | 0.057 | 110 | 1.000 |
| 5 | 0.067 | 116 | 0.999 |
| 6 | 0.039 | 97 | 1.000 |
| 7 | 0.043 | 96 | 1.000 |

Gate reject reasons: {'dedup': 117, 'zero_context': 38}

## a2

Dual view (SPEC §8-9): demand-weighted headline nDCG@10 = 0.998 (unweighted 0.998, ESS/N 0.93); per-cluster table below; worst clusters: [6, 2, 0]; zero-query clusters: [].

| cluster | p_hat | n_synth | mean nDCG |
|---|---|---|---|
| 0 | 0.404 | 112 | 0.997 |
| 1 | 0.181 | 50 | 0.997 |
| 2 | 0.122 | 31 | 0.997 |
| 3 | 0.087 | 27 | 1.000 |
| 4 | 0.057 | 33 | 0.998 |
| 5 | 0.067 | 11 | 0.999 |
| 6 | 0.039 | 15 | 0.996 |
| 7 | 0.043 | 22 | 0.999 |

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
