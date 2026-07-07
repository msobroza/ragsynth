"""Known-value tests for ranking-agreement metrics (SPEC §15.4, PLAN Task 2.4)."""

import numpy as np
import pytest

from ragsynth.metrics.validity import (
    RankingAgreement,
    ranking_agreement,
    rbo_ext,
    system_ranking,
    tau_ap,
)


class TestTauAP:
    def test_hand_derived_single_top_swap(self):
        # Reference [0,1,2,3], candidate [1,0,2,3].
        # tau_AP (Yilmaz/Aslam/Robertson 2008) sums, for each candidate item at
        # 1-indexed position i = 2..n, the fraction of items ranked above it in
        # the candidate that the reference also ranks above it (concordant):
        #   i=2 (item 0): item 1 is above it, but reference orders 0 first
        #                 -> discordant -> C(2) = 0
        #   i=3 (item 2): items {1, 0} above, both before 2 in reference -> C(3) = 2
        #   i=4 (item 3): items {1, 0, 2} above, all before 3            -> C(4) = 3
        # tau_AP = (2/(n-1)) * (C(2)/1 + C(3)/2 + C(4)/3) - 1
        #        = (2/3) * (0/1 + 2/2 + 3/3) - 1 = 4/3 - 1 = 1/3.
        # (The SPEC §15.4 inline guess of 0.555 is wrong; PLAN D-log confirms 1/3.)
        assert tau_ap([0, 1, 2, 3], [1, 0, 2, 3]) == pytest.approx(1 / 3)

    def test_identical_rankings(self):
        assert tau_ap([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)

    def test_reversed_ranking_n4(self):
        assert tau_ap([0, 1, 2, 3], [3, 2, 1, 0]) == pytest.approx(-1.0)

    def test_single_element(self):
        assert tau_ap([0], [0]) == pytest.approx(1.0)


class TestRboExt:
    def test_identical_lists(self):
        assert rbo_ext([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == pytest.approx(1.0)

    def test_fully_disjoint_lists(self):
        assert rbo_ext([0, 1, 2], [3, 4, 5]) == pytest.approx(0.0)

    def test_top_weightedness(self):
        # Swapping the top pair must hurt more than swapping the bottom pair.
        base = [0, 1, 2, 3, 4]
        top_swapped = [1, 0, 2, 3, 4]
        bottom_swapped = [0, 1, 2, 4, 3]
        assert rbo_ext(base, bottom_swapped, p=0.9) > rbo_ext(base, top_swapped, p=0.9)


class TestSystemRanking:
    def test_best_first_order(self):
        assert system_ranking(np.array([0.2, 0.9, 0.5])) == [1, 2, 0]


class TestRankingAgreement:
    @staticmethod
    def _perfect_fixture():
        # 3 systems x 40 queries; system separation (0.4) dwarfs per-query
        # noise (0.01), so every bootstrap resample preserves the ranking.
        rng = np.random.default_rng(42)
        base = np.array([0.9, 0.5, 0.1])
        return base[:, None] + 0.01 * rng.standard_normal((3, 40))

    def test_perfect_correlation(self):
        scores = self._perfect_fixture()
        result = ranking_agreement(scores, scores, n_boot=200, seed=0)
        assert isinstance(result, RankingAgreement)
        assert result.tau == pytest.approx(1.0)
        assert result.tau_ap_ == pytest.approx(1.0)
        assert result.rbo == pytest.approx(1.0)
        assert -1.0 <= result.tau_ci_low <= result.tau_ci_high <= 1.0
        assert result.tau_ci_high == pytest.approx(1.0)

    def test_deterministic_under_seed(self):
        scores = self._perfect_fixture()
        r1 = ranking_agreement(scores, scores, n_boot=100, seed=7)
        r2 = ranking_agreement(scores, scores, n_boot=100, seed=7)
        assert r1 == r2
