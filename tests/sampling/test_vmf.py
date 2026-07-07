import numpy as np
import pytest

from ragsynth.sampling import (
    l2_normalize,
    log_sphere_area,
    sample_vmf,
    sphere_uniform,
    vmf_log_norm_const,
)


def test_l2_normalize_rows_unit():
    rng = np.random.default_rng(0)
    x = 3.0 * rng.standard_normal((10, 5))
    normed = l2_normalize(x)
    np.testing.assert_allclose(np.linalg.norm(normed, axis=-1), 1.0, atol=1e-12)


def test_sphere_uniform_shape_and_unit_rows():
    rng = np.random.default_rng(0)
    x = sphere_uniform(50, 7, rng)
    assert x.shape == (50, 7)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-12)


def test_sample_vmf_mean_resultant_direction_matches_mu():
    rng = np.random.default_rng(42)
    mu = l2_normalize(rng.standard_normal(16))
    x = sample_vmf(mu, kappa=200.0, n=2000, rng=rng)
    assert x.shape == (2000, 16)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-9)
    mean_dir = l2_normalize(x.mean(axis=0))
    assert float(mean_dir @ mu) > 0.99


def test_sample_vmf_kappa_zero_is_near_uniform():
    rng = np.random.default_rng(1)
    mu = np.zeros(16)
    mu[0] = 1.0
    x = sample_vmf(mu, kappa=0.0, n=2000, rng=rng)
    assert float(np.linalg.norm(x.mean(axis=0))) < 0.1


@pytest.mark.parametrize("d", [2, 64])
@pytest.mark.parametrize("kappa", [1e-9, 1.0, 1e5])
def test_vmf_log_norm_const_finite(d, kappa):
    out = vmf_log_norm_const(d, kappa)
    assert np.all(np.isfinite(out))


def test_vmf_log_norm_const_vectorized():
    out = vmf_log_norm_const(8, np.array([1e-9, 1.0, 1e5]))
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_log_sphere_area_circle():
    # Surface "area" of S^1 is the circumference 2*pi.
    assert log_sphere_area(2) == pytest.approx(np.log(2.0 * np.pi))


def test_determinism_under_same_generator_seed():
    mu = np.zeros(8)
    mu[0] = 1.0
    a = sample_vmf(mu, 50.0, 100, np.random.default_rng(7))
    b = sample_vmf(mu, 50.0, 100, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)

    u1 = sphere_uniform(20, 4, np.random.default_rng(3))
    u2 = sphere_uniform(20, 4, np.random.default_rng(3))
    np.testing.assert_array_equal(u1, u2)
