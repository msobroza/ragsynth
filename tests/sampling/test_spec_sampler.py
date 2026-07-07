import numpy as np
import pytest

from ragsynth.sampling import MovMF, SpecSampler, sample_vmf


def _two_cluster_model(d=8):
    """Manually parameterized movMF: cluster A at +e1, cluster B at -e1."""
    mu = np.zeros(d)
    mu[0] = 1.0
    model = MovMF(n_components=2)
    model.means_ = np.vstack([mu, -mu])
    model.kappas_ = np.array([100.0, 100.0])
    model.weights_ = np.array([0.5, 0.5])
    return model, mu


def test_guard_keeps_only_on_manifold_z():
    model, mu = _two_cluster_model()
    # Production queries live only near cluster A.
    prod = sample_vmf(mu, 100.0, 200, np.random.default_rng(0))
    tau_r = 0.5  # cluster-B draws have max cos to prod ~ -0.8 => rejected
    sampler = SpecSampler(
        model=model,
        tilted_weights=np.array([0.5, 0.5]),
        prod_emb=prod,
        tau_r=tau_r,
    )
    z, comps = sampler.sample(100, np.random.default_rng(1))
    assert z.shape == (100, 8)
    assert comps.shape == (100,)
    max_cos = (z @ prod.T).max(axis=1)
    assert np.all(max_cos >= tau_r - 1e-9)
    assert np.all(comps == 0)


def test_exploration_mask_bypasses_guard():
    model, mu = _two_cluster_model()
    prod = sample_vmf(mu, 100.0, 200, np.random.default_rng(0))
    sampler = SpecSampler(
        model=model,
        tilted_weights=np.array([0.0, 1.0]),  # force cluster-B draws
        prod_emb=prod,
        tau_r=0.5,
        exploration=np.array([False, True]),
    )
    z, comps = sampler.sample(50, np.random.default_rng(2))
    assert z.shape == (50, 8)
    assert np.all(comps == 1)
    # Off-manifold (guard would reject) but kept via the exploration mask.
    max_cos = (z @ prod.T).max(axis=1)
    assert np.all(max_cos < 0.5)


def test_impossible_guard_raises_runtime_error():
    model, mu = _two_cluster_model()
    prod = sample_vmf(mu, 100.0, 50, np.random.default_rng(0))
    sampler = SpecSampler(
        model=model,
        tilted_weights=np.array([0.5, 0.5]),
        prod_emb=prod,
        tau_r=0.999,
        max_tries=2,
    )
    with pytest.raises(RuntimeError, match="lower tau_r"):
        sampler.sample(5, np.random.default_rng(3))


def test_determinism_under_same_generator_seed():
    model, mu = _two_cluster_model()
    prod = sample_vmf(mu, 100.0, 200, np.random.default_rng(0))
    sampler = SpecSampler(
        model=model,
        tilted_weights=np.array([0.5, 0.5]),
        prod_emb=prod,
        tau_r=0.5,
    )
    z1, c1 = sampler.sample(30, np.random.default_rng(11))
    z2, c2 = sampler.sample(30, np.random.default_rng(11))
    np.testing.assert_array_equal(z1, z2)
    np.testing.assert_array_equal(c1, c2)
