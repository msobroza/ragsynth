"""Hand fixtures for the light diversity metrics (SPEC §8-9, v1)."""

import numpy as np
import pytest

from ragsynth.metrics.diversity import distinct_n, semantic_dedup_rate


class TestDistinctN:
    def test_repeated_unigrams(self):
        assert distinct_n(["a b", "a b"], 1) == 0.5

    def test_bigram_hand_fixture(self):
        # bigrams: ("a","b"), ("b","c") from "a b c"; ("a","b") from "a b"
        # => 2 unique / 3 total
        assert distinct_n(["a b c", "a b"], 2) == pytest.approx(2 / 3)

    def test_texts_shorter_than_n_contribute_nothing(self):
        # "a" yields no bigram; only ("b","c") is counted => 1 unique / 1 total
        assert distinct_n(["a", "b c"], 2) == 1.0

    def test_empty_list_zero(self):
        assert distinct_n([], 1) == 0.0

    def test_all_texts_too_short_zero(self):
        assert distinct_n(["a", "b"], 2) == 0.0


class TestSemanticDedupRate:
    def test_two_identical_of_three(self):
        embs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert semantic_dedup_rate(embs, threshold=0.95) == pytest.approx(1 / 3)

    def test_all_orthogonal_zero(self):
        assert semantic_dedup_rate(np.eye(4)) == 0.0

    def test_empty_zero(self):
        assert semantic_dedup_rate(np.empty((0, 4))) == 0.0
