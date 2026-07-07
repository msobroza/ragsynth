"""Known-value fixtures for the efficiency metric layer (SPEC section 15.4).

Hand-derived example used throughout: reference partition with C=3 clusters,
production demand p_hat = [.5, .3, .2], synthetic labels [0, 0, 1, 1]
(cluster 2 empty). Derivation:
  counts = [2, 2, 0]  =>  q = [.5, .5, 0],  covered = {0, 1}
  coverage_gap = p_hat[2] = 0.2
  p_cov = [.5, .3, 0] / 0.8 = [.625, .375, 0]
  weights = p_cov[labels] / q[labels] = [1.25, 1.25, .75, .75]
  ESS = (sum w)^2 / sum w^2 = 4^2 / (2*1.25^2 + 2*0.75^2) = 16 / 4.25
"""

import numpy as np
import pytest

from ragsynth.metrics.efficiency import (
    cluster_importance_weights,
    demand_weighted_coverage,
    effective_sample_size,
    minimum_semantic_coverage,
    post_stratified_estimate,
    zero_query_clusters,
)

TOL = 1e-9


@pytest.fixture
def p_hat() -> np.ndarray:
    return np.array([0.5, 0.3, 0.2])


@pytest.fixture
def labels_synth() -> np.ndarray:
    return np.array([0, 0, 1, 1])


class TestClusterImportanceWeights:
    def test_coverage_gap_is_demand_of_empty_cluster(
        self, labels_synth: np.ndarray, p_hat: np.ndarray
    ) -> None:
        _, coverage_gap = cluster_importance_weights(labels_synth, p_hat)
        assert coverage_gap == 0.2

    def test_per_sample_weights_renormalized_over_covered(
        self, labels_synth: np.ndarray, p_hat: np.ndarray
    ) -> None:
        weights, _ = cluster_importance_weights(labels_synth, p_hat)
        np.testing.assert_allclose(weights, [1.25, 1.25, 0.75, 0.75], rtol=0, atol=TOL)

    def test_all_clusters_covered_gives_zero_gap(self, p_hat: np.ndarray) -> None:
        labels = np.array([0, 0, 1, 2])
        weights, coverage_gap = cluster_importance_weights(labels, p_hat)
        assert coverage_gap == 0.0
        # q = [.5, .25, .25] => w = [.5/.5, .5/.5, .3/.25, .2/.25]
        np.testing.assert_allclose(weights, [1.0, 1.0, 1.2, 0.8], rtol=0, atol=TOL)


class TestEffectiveSampleSize:
    def test_equal_weights_give_n(self) -> None:
        assert effective_sample_size(np.ones(7)) == 7.0

    def test_one_hot_gives_one(self) -> None:
        assert effective_sample_size(np.array([1.0, 0.0, 0.0, 0.0])) == 1.0

    def test_fixture_weights(self, labels_synth: np.ndarray, p_hat: np.ndarray) -> None:
        weights, _ = cluster_importance_weights(labels_synth, p_hat)
        ess = effective_sample_size(weights)
        assert ess == pytest.approx(16.0 / 4.25, abs=TOL)
        assert ess == pytest.approx(3.7647058823529411, abs=TOL)


class TestPostStratifiedEstimate:
    def test_renormalizes_over_covered_mass(
        self, labels_synth: np.ndarray, p_hat: np.ndarray
    ) -> None:
        metric = np.array([1.0, 1.0, 0.0, 0.0])
        # (.5*1 + .3*0) / (.5 + .3) = .5 / .8 = .625
        estimate = post_stratified_estimate(metric, labels_synth, p_hat)
        assert estimate == pytest.approx(0.625, abs=TOL)

    def test_all_covered_equals_plain_demand_weighted_mean(self, p_hat: np.ndarray) -> None:
        labels = np.array([0, 0, 1, 2])
        metric = np.array([1.0, 0.0, 0.5, 0.25])
        # cluster means [.5, .5, .25] => .5*.5 + .3*.5 + .2*.25 = .45, mass = 1
        estimate = post_stratified_estimate(metric, labels, p_hat)
        assert estimate == pytest.approx(0.45, abs=TOL)


class TestDemandWeightedCoverage:
    def test_fixture(self, labels_synth: np.ndarray, p_hat: np.ndarray) -> None:
        assert demand_weighted_coverage(labels_synth, p_hat) == pytest.approx(0.8, abs=TOL)

    def test_full_coverage(self, p_hat: np.ndarray) -> None:
        labels = np.array([0, 1, 2])
        assert demand_weighted_coverage(labels, p_hat) == pytest.approx(1.0, abs=TOL)


class TestZeroQueryClusters:
    def test_fixture(self, labels_synth: np.ndarray) -> None:
        assert zero_query_clusters(labels_synth, n_clusters=3) == [2]

    def test_no_empty_clusters(self) -> None:
        assert zero_query_clusters(np.array([0, 1, 2]), n_clusters=3) == []

    def test_multiple_empties_sorted(self) -> None:
        assert zero_query_clusters(np.array([1, 1]), n_clusters=4) == [0, 2, 3]


class TestMinimumSemanticCoverage:
    def test_floor_two_counts_both_covered_clusters(
        self, labels_synth: np.ndarray, p_hat: np.ndarray
    ) -> None:
        cov = minimum_semantic_coverage(labels_synth, p_hat, floor=2)
        assert cov == pytest.approx(0.8, abs=TOL)

    def test_floor_three_covers_nothing(self, labels_synth: np.ndarray, p_hat: np.ndarray) -> None:
        assert minimum_semantic_coverage(labels_synth, p_hat, floor=3) == 0.0

    def test_floor_one_equals_demand_weighted_coverage(
        self, labels_synth: np.ndarray, p_hat: np.ndarray
    ) -> None:
        cov = minimum_semantic_coverage(labels_synth, p_hat, floor=1)
        assert cov == pytest.approx(demand_weighted_coverage(labels_synth, p_hat), abs=TOL)
