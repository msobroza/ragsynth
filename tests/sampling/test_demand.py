import numpy as np
import pytest

from ragsynth.sampling import demand_from_responsibilities, nn_cos_threshold, tilt_weights


def test_no_decay_is_normalized_column_sums():
    # Column sums [0.8, 1.2] normalize to [0.4, 0.6] exactly.
    resp = np.array([[0.2, 0.8], [0.6, 0.4]])
    p_hat = demand_from_responsibilities(resp)
    np.testing.assert_allclose(p_hat, [0.4, 0.6], atol=1e-15)
    assert p_hat.sum() == pytest.approx(1.0)


def test_half_life_makes_newer_query_dominate():
    # resp = eye(2): query 0 (t=0) owns component 0, query 1 (t=10) owns
    # component 1. half_life=1 (same unit as timestamps) decays query 0 by
    # 2**-10, so p_hat[1] = 1 / (1 + 2**-10) > 0.99.
    resp = np.eye(2)
    p_hat = demand_from_responsibilities(resp, timestamps=np.array([0.0, 10.0]), half_life=1.0)
    assert p_hat[1] > 0.99
    assert p_hat[1] == pytest.approx(1.0 / (1.0 + 2.0**-10))
    assert p_hat.sum() == pytest.approx(1.0)


def test_tilt_weights_hand_value():
    # 0.7*[1,0] + 0.3/2 = [0.85, 0.15].
    out = tilt_weights(np.array([1.0, 0.0]), lam=0.7)
    np.testing.assert_allclose(out, [0.85, 0.15], atol=1e-12)
    assert out.sum() == pytest.approx(1.0)


def test_nn_cos_threshold_hand_grid():
    # Four unit vectors in the plane at 0, 20, 90, 120 degrees.
    # Nearest-neighbour angles: p0<->p1 = 20deg (both), p2<->p3 = 30deg (both),
    # so NN cosines are [cos20, cos20, cos30, cos30] and the 50th percentile
    # (linear interpolation) is (cos30 + cos20) / 2.
    angles = np.deg2rad([0.0, 20.0, 90.0, 120.0])
    pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    expected = (np.cos(np.deg2rad(30.0)) + np.cos(np.deg2rad(20.0))) / 2.0
    assert nn_cos_threshold(pts, pct=50.0) == pytest.approx(expected, abs=1e-9)
