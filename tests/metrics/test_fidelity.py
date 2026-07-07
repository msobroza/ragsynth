"""Known-value fixtures for fidelity metrics (SPEC §15.4, prototype L375-462)."""

import math

import numpy as np
import pytest

from ragsynth.metrics.fidelity import (
    c2st_auc,
    c2st_auc_with_coefs,
    kl_similarity_distributions,
    mmd_rbf,
    within_cluster_c2st,
)


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class TestKLSimilarityDistributions:
    def test_identical_samples_near_zero(self):
        rng = np.random.default_rng(0)
        q = _unit(rng.normal(size=(500, 8)))
        chunks = _unit(rng.normal(size=(60, 8)))
        assert kl_similarity_distributions(q, q, chunks) < 0.02

    def test_disjoint_similarity_profiles_large(self):
        rng = np.random.default_rng(1)
        # Chunks hug e0; real queries hug e0 (top-1 cos ~ 1); synth queries hug
        # e1, near orthogonal to every chunk (top-1 cos ~ 0) => sharply
        # different top-1 cosine profiles.
        chunks = _unit(np.eye(8)[0] + 0.05 * rng.normal(size=(40, 8)))
        real = _unit(np.eye(8)[0] + 0.05 * rng.normal(size=(200, 8)))
        synth = _unit(np.eye(8)[1] + 0.05 * rng.normal(size=(200, 8)))
        assert kl_similarity_distributions(real, synth, chunks) > 1.0


class TestC2stAuc:
    def test_same_distribution_split_near_half(self):
        rng = np.random.default_rng(0)
        cloud = rng.normal(size=(400, 8))
        auc = c2st_auc(cloud[:200], cloud[200:])
        assert 0.45 <= auc <= 0.55

    def test_shifted_cloud_detectable(self):
        rng = np.random.default_rng(0)
        real = rng.normal(size=(200, 8))
        synth = rng.normal(size=(200, 8))
        synth[:, 0] += 2.0
        assert c2st_auc(real, synth) > 0.9


class TestC2stAucWithCoefs:
    def test_top_coef_is_shifted_dim_and_auc_matches(self):
        rng = np.random.default_rng(0)
        real = rng.normal(size=(200, 8))
        synth = rng.normal(size=(200, 8))
        synth[:, 3] += 2.0
        auc, coefs = c2st_auc_with_coefs(real, synth, seed=0)
        assert coefs.shape == (8,)
        assert int(np.argmax(np.abs(coefs))) == 3
        assert auc == pytest.approx(c2st_auc(real, synth, seed=0))


class TestMmdRbf:
    def test_same_set_exactly_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(150, 8))
        assert mmd_rbf(x, x) == 0.0

    def test_iid_halves_below_shifted_clouds(self):
        rng = np.random.default_rng(0)
        cloud = rng.normal(size=(400, 8))
        shifted = rng.normal(size=(200, 8))
        shifted[:, 0] += 2.0
        near = mmd_rbf(cloud[:200], cloud[200:])
        far = mmd_rbf(cloud[:200], shifted)
        assert near < far

    def test_deterministic_under_seed(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(300, 8))
        y = rng.normal(size=(300, 8)) + 0.5
        # max_n below n forces the seeded subsample path.
        assert mmd_rbf(x, y, max_n=100, seed=7) == mmd_rbf(x, y, max_n=100, seed=7)


class TestWithinClusterC2st:
    def test_cluster_below_min_per_side_absent(self):
        rng = np.random.default_rng(0)
        x_real = rng.normal(size=(200, 8))
        labels_real = np.array([0] * 100 + [1] * 100)
        x_synth = rng.normal(size=(110, 8))
        labels_synth = np.array([0] * 100 + [1] * 10)  # synth side of cluster 1 < 30
        mean, per = within_cluster_c2st(x_real, x_synth, labels_real, labels_synth)
        assert set(per) == {0}
        assert mean == pytest.approx(per[0])

    def test_matched_near_half_mismatched_higher(self):
        rng = np.random.default_rng(1)
        mode = np.eye(8)[0] * 5.0
        # Real is bimodal inside the (single) cluster: modes at 0 and +5 e0.
        x_real = np.vstack([rng.normal(size=(60, 8)), rng.normal(size=(60, 8)) + mode])
        labels = np.zeros(120, dtype=int)
        # Matched synth reproduces both modes; collapsed synth is unimodal at 0.
        x_synth_matched = np.vstack([rng.normal(size=(60, 8)), rng.normal(size=(60, 8)) + mode])
        x_synth_collapsed = rng.normal(size=(120, 8))
        _, per_matched = within_cluster_c2st(x_real, x_synth_matched, labels, labels)
        _, per_collapsed = within_cluster_c2st(x_real, x_synth_collapsed, labels, labels)
        assert 0.35 <= per_matched[0] <= 0.65
        assert per_collapsed[0] > per_matched[0]
        assert per_collapsed[0] > 0.6

    def test_empty_dict_gives_nan_mean(self):
        rng = np.random.default_rng(2)
        x_real = rng.normal(size=(20, 8))
        x_synth = rng.normal(size=(20, 8))
        labels = np.zeros(20, dtype=int)
        mean, per = within_cluster_c2st(x_real, x_synth, labels, labels)
        assert per == {}
        assert math.isnan(mean)
