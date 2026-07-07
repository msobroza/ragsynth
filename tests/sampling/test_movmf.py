import numpy as np
import pytest

from ragsynth.sampling import MovMF, l2_normalize, sample_vmf


@pytest.fixture(scope="module")
def two_modes():
    """SPEC §15.4 fixture: two antipodal vMF components (d=8, kappa=100)."""
    rng = np.random.default_rng(0)
    mu = l2_normalize(rng.standard_normal(8))
    x = np.vstack(
        [
            sample_vmf(mu, 100.0, 1000, rng),
            sample_vmf(-mu, 100.0, 1000, rng),
        ]
    )
    return x, mu


def test_fit_recovers_two_antipodal_components(two_modes):
    x, mu = two_modes
    model = MovMF(n_components=2, seed=0).fit(x)
    targets = np.vstack([mu, -mu])
    cos = model.means_ @ targets.T  # (2 fitted, 2 targets)
    best = cos.argmax(axis=1)
    # Best assignment is a bijection and every match is tight.
    assert sorted(best.tolist()) == [0, 1]
    assert cos[0, best[0]] >= 0.99
    assert cos[1, best[1]] >= 0.99
    np.testing.assert_allclose(model.weights_, [0.5, 0.5], atol=0.05)
    assert np.all(model.kappas_ > 0)
    assert np.all(np.isfinite(model.kappas_))


def test_responsibilities_rows_sum_to_one(two_modes):
    x, _ = two_modes
    model = MovMF(n_components=2, seed=0).fit(x)
    resp = model.responsibilities(x)
    assert resp.shape == (x.shape[0], 2)
    np.testing.assert_allclose(resp.sum(axis=1), 1.0, atol=1e-12)


def test_artifact_round_trip(tmp_path, two_modes):
    x, _ = two_modes
    model = MovMF(n_components=2, max_iter=150, tol=1e-5, kappa_min=0.5, kappa_max=1e4, seed=3).fit(
        x
    )
    assert model.fitted_on_hash != ""

    path = tmp_path / "movmf.npz"
    model.to_artifact(path)
    loaded = MovMF.from_artifact(path)

    assert loaded.n_components == 2
    assert loaded.max_iter == 150
    assert loaded.tol == 1e-5
    assert loaded.kappa_min == 0.5
    assert loaded.kappa_max == 1e4
    assert loaded.seed == 3
    assert loaded.fitted_on_hash == model.fitted_on_hash
    np.testing.assert_array_equal(loaded.weights_, model.weights_)
    np.testing.assert_array_equal(loaded.means_, model.means_)
    np.testing.assert_array_equal(loaded.kappas_, model.kappas_)

    probe = l2_normalize(np.random.default_rng(9).standard_normal((16, 8)))
    np.testing.assert_array_equal(loaded.log_prob(probe), model.log_prob(probe))


def test_sample_respects_weight_override(two_modes):
    x, _ = two_modes
    model = MovMF(n_components=2, seed=0).fit(x)
    z, comps = model.sample(400, np.random.default_rng(5), weights=np.array([0.0, 1.0]))
    assert z.shape == (400, 8)
    counts = np.bincount(comps, minlength=2)
    assert counts[0] == 0
    assert counts[1] == 400
    mean_dir = z.mean(axis=0)
    mean_dir = mean_dir / np.linalg.norm(mean_dir)
    assert float(mean_dir @ model.means_[1]) > 0.9


def test_unfitted_model_raises():
    model = MovMF(n_components=2)
    probe = np.eye(3)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.log_prob(probe)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.sample(3, np.random.default_rng(0))
