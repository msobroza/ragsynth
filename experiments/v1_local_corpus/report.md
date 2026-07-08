# EvalReport - v1-local-corpus

Config hash: `3e3c4cd8fd82` - seed 0

Reading guide: KL/C2ST/wC2ST/MMD lower is better (0.5 = indistinguishable
for AUCs); ESS/N higher is better; tau/tau_AP/RBO vs the anchor ranking
higher is better; control p-values < 0.05 mean the arm detects the
injected regression.

| arm | n | KL | C2ST | wC2ST | MMD | ESS/N | gap | tau | tau 95% CI | tau_AP | RBO | PC drop | PC noise | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a0 | 100 | 10.770 | 1.000 | 1.000 | 0.0753 | 0.54 | 0.07 | 0.687 | [0.63, 0.75] | 0.686 | 0.88 | 0.000 | 0.000 | fail |
| a1 | 100 | 9.196 | 1.000 | 1.000 | 0.0805 | 0.71 | 0.14 | 0.748 | [0.66, 0.78] | 0.752 | 0.90 | 0.000 | 0.000 | fail |
| a2 | 100 | 12.947 | 1.000 | 1.000 | 0.0897 | 0.50 | 0.14 | 0.687 | [0.60, 0.75] | 0.686 | 0.88 | 0.000 | 0.000 | fail |
| oracle | 18 | 10.199 | 0.431 | - | 0.0000 | 0.92 | 0.11 | 0.750 | [0.48, 0.82] | 0.787 | 0.94 | 1.000 | 0.117 | fail |

## a0

Dual view (SPEC §8-9): demand-weighted headline nDCG@10 = 1.000 (unweighted 1.000, ESS/N 0.54); per-cluster table below; worst clusters: [7, 3, 0]; zero-query clusters: [6].

| cluster | p_hat | n_synth | mean nDCG |
|---|---|---|---|
| 0 | 0.167 | 10 | 1.000 |
| 1 | 0.111 | 2 | 1.000 |
| 2 | 0.153 | 13 | 1.000 |
| 3 | 0.222 | 28 | 1.000 |
| 4 | 0.111 | 30 | 1.000 |
| 5 | 0.069 | 2 | 1.000 |
| 6 | 0.069 | 0 | - |
| 7 | 0.097 | 15 | 0.999 |

## a1

Dual view (SPEC §8-9): demand-weighted headline nDCG@10 = 0.979 (unweighted 0.978, ESS/N 0.71); per-cluster table below; worst clusters: [0, 4, 3]; zero-query clusters: [5, 6].

| cluster | p_hat | n_synth | mean nDCG |
|---|---|---|---|
| 0 | 0.167 | 16 | 0.963 |
| 1 | 0.111 | 3 | 0.989 |
| 2 | 0.153 | 18 | 0.993 |
| 3 | 0.222 | 23 | 0.976 |
| 4 | 0.111 | 25 | 0.972 |
| 5 | 0.069 | 0 | - |
| 6 | 0.069 | 0 | - |
| 7 | 0.097 | 15 | 0.987 |

Gate reject reasons: {'dedup': 19, 'round_trip': 5}

## a2

Dual view (SPEC §8-9): demand-weighted headline nDCG@10 = 0.881 (unweighted 0.880, ESS/N 0.50); per-cluster table below; worst clusters: [4, 0, 7]; zero-query clusters: [5, 6].

| cluster | p_hat | n_synth | mean nDCG |
|---|---|---|---|
| 0 | 0.167 | 3 | 0.864 |
| 1 | 0.111 | 10 | 0.892 |
| 2 | 0.153 | 18 | 0.910 |
| 3 | 0.222 | 27 | 0.881 |
| 4 | 0.111 | 31 | 0.860 |
| 5 | 0.069 | 0 | - |
| 6 | 0.069 | 0 | - |
| 7 | 0.097 | 11 | 0.879 |

## oracle

Dual view (SPEC §8-9): demand-weighted headline nDCG@10 = 1.000 (unweighted 1.000, ESS/N 0.92); per-cluster table below; worst clusters: [0, 1, 2]; zero-query clusters: [4].

| cluster | p_hat | n_synth | mean nDCG |
|---|---|---|---|
| 0 | 0.167 | 3 | 1.000 |
| 1 | 0.111 | 3 | 1.000 |
| 2 | 0.153 | 2 | 1.000 |
| 3 | 0.222 | 5 | 1.000 |
| 4 | 0.111 | 0 | - |
| 5 | 0.069 | 1 | 1.000 |
| 6 | 0.069 | 1 | 1.000 |
| 7 | 0.097 | 3 | 1.000 |
