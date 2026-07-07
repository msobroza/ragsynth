"""Tests for the retrieval system zoo and nDCG@k scoring (PLAN Task 2.4, D16)."""

import math

import numpy as np
import pytest

from ragsynth.metrics.validity import MatrixSystem, evaluate_zoo, make_system_zoo


def _unit_rows(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _identity_system():
    # 6 chunks embedded as the first 6 basis vectors of R^8: the query below
    # has strictly decreasing cosine to c0 > c1 > ... > c5, so ranks are known.
    d = 8
    chunk_embs = np.eye(d)[:6]
    chunk_ids = tuple(f"c{i}" for i in range(6))
    return MatrixSystem(name="exact", matrix=np.eye(d), chunk_ids=chunk_ids, chunk_embs=chunk_embs)


def _query():
    return _unit_rows(np.array([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0]]))


class TestMatrixSystemNdcg:
    def test_single_gold_at_known_rank_reduces_to_prototype(self):
        # Gold c2 sits at rank 3 => nDCG = (1/log2(1+3)) / (1/log2(2)) = 1/log2(4).
        system = _identity_system()
        scores = system.per_query_scores(_query(), [{"c2": 1}], k=10)
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(1.0 / math.log2(4.0))

    def test_gold_at_rank_one(self):
        system = _identity_system()
        scores = system.per_query_scores(_query(), [{"c0": 1}], k=10)
        assert scores[0] == pytest.approx(1.0)

    def test_gold_outside_top_k_scores_zero(self):
        system = _identity_system()
        scores = system.per_query_scores(_query(), [{"c2": 1}], k=2)
        assert scores[0] == pytest.approx(0.0)

    def test_multi_gold_matches_hand_computed_ndcg(self):
        # Golds c0 (rank 1) and c3 (rank 4), k=10:
        #   DCG  = 1/log2(2) + 1/log2(5)
        #   IDCG = 1/log2(2) + 1/log2(3)   (two relevant chunks, ideal ranks 1, 2)
        system = _identity_system()
        scores = system.per_query_scores(_query(), [{"c0": 1, "c3": 1}], k=10)
        expected = (1.0 / math.log2(2.0) + 1.0 / math.log2(5.0)) / (
            1.0 / math.log2(2.0) + 1.0 / math.log2(3.0)
        )
        assert scores[0] == pytest.approx(expected)

    def test_drop_mask_removing_gold_scores_zero(self):
        system = _identity_system()
        drop = np.zeros(6, dtype=bool)
        drop[2] = True  # kill the gold chunk c2
        scores = system.per_query_scores(_query(), [{"c2": 1}], k=10, drop_mask=drop)
        assert scores[0] == pytest.approx(0.0)

    def test_drop_mask_promotes_surviving_gold(self):
        # Dropping c0 lifts gold c3 from rank 4 to rank 3.
        system = _identity_system()
        drop = np.zeros(6, dtype=bool)
        drop[0] = True
        scores = system.per_query_scores(_query(), [{"c3": 1}], k=10, drop_mask=drop)
        assert scores[0] == pytest.approx(1.0 / math.log2(4.0))

    def test_no_relevant_chunk_scores_zero(self):
        system = _identity_system()
        scores = system.per_query_scores(_query(), [{}], k=10)
        assert scores[0] == pytest.approx(0.0)

    def test_unknown_qrel_chunk_id_raises(self):
        system = _identity_system()
        with pytest.raises(ValueError, match="unknown chunk_id"):
            system.per_query_scores(_query(), [{"missing": 1}], k=10)


class TestSystemZoo:
    @staticmethod
    def _corpus(d=16, n=20, seed=0):
        rng = np.random.default_rng(seed)
        embs = _unit_rows(rng.standard_normal((n, d)))
        ids = tuple(f"c{i}" for i in range(n))
        return ids, embs

    def test_zoo_has_twelve_named_systems(self):
        ids, embs = self._corpus()
        zoo = make_system_zoo(ids, embs, seed=0)
        names = list(zoo)
        assert len(names) == 12
        assert names[0] == "exact"
        assert sum(name.startswith("distort-") for name in names) == 7
        assert sum(name.startswith("rank-") for name in names) == 4
        # r = max(2, round(d * f)) for f in (0.75, 0.5, 0.4375, 0.375) at d=16.
        assert [n for n in names if n.startswith("rank-")] == [
            "rank-12",
            "rank-8",
            "rank-7",
            "rank-6",
        ]

    def test_exact_matrix_is_identity(self):
        ids, embs = self._corpus()
        zoo = make_system_zoo(ids, embs, seed=0)
        assert np.array_equal(zoo["exact"].matrix, np.eye(16))

    def test_zoo_deterministic_under_seed(self):
        ids, embs = self._corpus()
        zoo1 = make_system_zoo(ids, embs, seed=0)
        zoo2 = make_system_zoo(ids, embs, seed=0)
        assert list(zoo1) == list(zoo2)
        for name in zoo1:
            assert np.array_equal(zoo1[name].matrix, zoo2[name].matrix)

    def test_zoo_systems_share_corpus(self):
        ids, embs = self._corpus()
        zoo = make_system_zoo(ids, embs, seed=0)
        for system in zoo.values():
            assert system.chunk_ids == ids
            assert np.array_equal(system.chunk_embs, embs)


class TestEvaluateZoo:
    def test_shape_and_insertion_order(self):
        ids, embs = TestSystemZoo._corpus()
        zoo = make_system_zoo(ids, embs, seed=0)
        # Queries are the first 5 chunk embeddings; each query's gold is itself,
        # so the 'exact' system retrieves every gold at rank 1 (score 1.0).
        query_embs = embs[:5]
        qrels = [{ids[i]: 1} for i in range(5)]
        scores = evaluate_zoo(zoo, query_embs, qrels, k=10)
        assert scores.shape == (12, 5)
        assert np.allclose(scores[0], 1.0)
        for row, name in zip(scores, zoo, strict=True):
            assert np.allclose(row, zoo[name].per_query_scores(query_embs, qrels, k=10))
