import numpy as np
import pytest

from ragsynth.sampling import ReferencePartition


def _blobs(seed=0, n=120, d=6, k=3):
    rng = np.random.default_rng(seed)
    centers = 5.0 * rng.standard_normal((k, d))
    x = np.vstack([centers[i] + 0.1 * rng.standard_normal((n // k, d)) for i in range(k)])
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_deterministic_assign_across_two_fits_same_seed():
    x = _blobs()
    p1 = ReferencePartition.fit(x, n_clusters=3, seed=0)
    p2 = ReferencePartition.fit(x, n_clusters=3, seed=0)
    np.testing.assert_allclose(p1.centers, p2.centers)
    np.testing.assert_array_equal(p1.assign(x), p2.assign(x))
    assert p1.fitted_on_hash == p2.fitted_on_hash


def test_artifact_round_trip_preserves_centers_and_assign(tmp_path):
    x = _blobs()
    part = ReferencePartition.fit(x, n_clusters=3, seed=0)
    path = tmp_path / "partition.npz"
    part.to_artifact(path)
    loaded = ReferencePartition.from_artifact(path)
    np.testing.assert_array_equal(loaded.centers, part.centers)
    np.testing.assert_array_equal(loaded.assign(x), part.assign(x))
    assert loaded.n_clusters == part.n_clusters
    assert loaded.seed == part.seed
    assert loaded.fitted_on_hash == part.fitted_on_hash


def test_proportions_sums_to_one():
    x = _blobs()
    part = ReferencePartition.fit(x, n_clusters=3, seed=0)
    props = part.proportions(x)
    assert props.shape == (3,)
    assert props.sum() == pytest.approx(1.0)
    assert np.all(props >= 0.0)
    # Three balanced blobs => each cluster holds a third of the points.
    np.testing.assert_allclose(np.sort(props), [1 / 3, 1 / 3, 1 / 3], atol=1e-12)


def test_n_clusters_respected():
    x = _blobs()
    part = ReferencePartition.fit(x, n_clusters=5, seed=0)
    assert part.n_clusters == 5
    assert part.centers.shape == (5, x.shape[1])
    labels = part.assign(x)
    assert labels.min() >= 0
    assert labels.max() < 5
