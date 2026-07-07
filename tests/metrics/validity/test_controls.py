"""Tests for positive-control degradations and the paired bootstrap (PLAN Task 2.4)."""

import numpy as np
import pytest

from ragsynth.metrics.validity import (
    drop_index_mask,
    noise_transform,
    paired_bootstrap_pvalue,
)


class TestPairedBootstrapPvalue:
    def test_detects_injected_shift(self):
        rng = np.random.default_rng(0)
        degraded = rng.uniform(0.2, 0.8, size=50)
        base = degraded + 0.1
        delta, p = paired_bootstrap_pvalue(base, degraded, n_boot=500, seed=0)
        assert delta == pytest.approx(0.1)
        assert p < 0.05

    def test_no_false_alarm_under_null(self):
        rng = np.random.default_rng(1)
        scores = rng.uniform(0.2, 0.8, size=50)
        delta, p = paired_bootstrap_pvalue(scores, scores.copy(), n_boot=500, seed=0)
        assert delta == pytest.approx(0.0)
        assert p >= 0.5


class TestDropIndexMask:
    def test_exact_count(self):
        mask = drop_index_mask(100, 0.1, np.random.default_rng(7))
        assert mask.shape == (100,)
        assert mask.dtype == np.bool_
        assert int(mask.sum()) == 10

    def test_rounds_fraction(self):
        mask = drop_index_mask(30, 0.25, np.random.default_rng(3))
        assert int(mask.sum()) == round(0.25 * 30)

    def test_zero_fraction_drops_nothing(self):
        mask = drop_index_mask(20, 0.0, np.random.default_rng(0))
        assert not mask.any()

    def test_deterministic_under_seed(self):
        m1 = drop_index_mask(100, 0.1, np.random.default_rng(7))
        m2 = drop_index_mask(100, 0.1, np.random.default_rng(7))
        assert np.array_equal(m1, m2)


class TestNoiseTransform:
    def test_shape(self):
        m = noise_transform(16, 0.5, np.random.default_rng(3))
        assert m.shape == (16, 16)

    def test_deterministic_under_seed(self):
        m1 = noise_transform(16, 0.5, np.random.default_rng(3))
        m2 = noise_transform(16, 0.5, np.random.default_rng(3))
        assert np.array_equal(m1, m2)

    def test_sigma_zero_is_identity(self):
        m = noise_transform(8, 0.0, np.random.default_rng(0))
        assert np.array_equal(m, np.eye(8))

    def test_perturbation_scales_with_sigma(self):
        small = noise_transform(16, 0.1, np.random.default_rng(5))
        large = noise_transform(16, 1.0, np.random.default_rng(5))
        eye = np.eye(16)
        assert np.abs(small - eye).sum() < np.abs(large - eye).sum()
