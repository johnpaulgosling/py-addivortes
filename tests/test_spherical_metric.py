"""Tests for the spherical (great-circle) distance metric.

These tests verify that the spherical metric is applied consistently in both
prediction paths:

* **In-sample** predictions are produced inside the C++ ``run_mcmc`` routine,
  which reduces ``metric``/``members`` via the C++ ``make_reduced_metric``.
* **Out-of-sample** predictions go through the :meth:`AddiVortesRegressor.predict`
  class method, which calls ``_core.cell_indices`` with the reduced metric and
  member arrays computed in Python by
  :func:`addivortes.preprocessing.reduced_metric_and_members`.

Both paths must resolve observations to the same Voronoi cells. Particular
attention is paid to the coordinate convention where the azimuthal (longitude)
variable is the *last* column in each spherical membership group and the
latitude-type variables come before it.
"""

import numpy as np
import pytest

from conftest import fast_model

from addivortes import _core
from addivortes.preprocessing import reduced_metric_and_members


# ---------------------------------------------------------------------------
# Pure-Python reference implementation mirroring the C++ backend.
# ---------------------------------------------------------------------------

def _spherical_sq(a, b):
    """Squared great-circle distance for one spherical group.

    Mirrors ``spherical_distance`` in ``python_src/addivortes_python.cpp``. The
    *last* entry of each group is the azimuthal (longitude) coordinate; earlier
    entries are polar (latitude) coordinates.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    size = a.size
    if size == 1:
        a1 = abs(float(a[0] - b[0]))
        a2 = 2.0 * np.pi - a1
        return min(a1, a2) ** 2

    angle_diff = np.cos(a[size - 1] - b[size - 1])
    for idx in range(size - 2, -1, -1):
        internal = np.sin(a[idx]) * np.sin(b[idx]) + np.cos(a[idx]) * np.cos(b[idx]) * angle_diff
        internal = float(np.clip(internal, -1.0, 1.0))
        angle_diff = np.arccos(internal) if idx == 0 else internal
    return float(angle_diff) ** 2


def _calc_distance(a, b, metric_red, member_red):
    """Mirror of the C++ ``calc_distance`` group dispatch."""

    offset = 0
    total = 0.0
    for metric_value, size in zip(metric_red, member_red):
        seg_a = a[offset : offset + size]
        seg_b = b[offset : offset + size]
        if metric_value == 0:
            total += float(np.sum((seg_a - seg_b) ** 2))
        elif metric_value == 1:
            total += _spherical_sq(seg_a, seg_b)
        offset += size
    return total


def _reference_cell_indices(query, centres, dim, metric_red, member_red):
    """Reference nearest-centre assignment mirroring ``knn1_internal``."""

    query = np.asarray(query, dtype=float)
    centres = np.asarray(centres, dtype=float)
    dim = np.asarray(dim, dtype=int)
    n_obs, n_features = query.shape
    n_centres, n_dims = centres.shape
    if n_centres == 1:
        return np.zeros(n_obs, dtype=int)

    out = np.zeros(n_obs, dtype=int)
    for obs in range(n_obs):
        best_centre = 0
        best_distance = np.inf
        for centre in range(n_centres):
            tess_point = query[obs].copy()
            for local_dim in range(n_dims):
                tess_point[dim[local_dim]] = centres[centre, local_dim]
            distance = _calc_distance(query[obs], tess_point, metric_red, member_red)
            if distance < best_distance:
                best_distance = distance
                best_centre = centre
        out[obs] = best_centre
    return out


# ---------------------------------------------------------------------------
# Direct ``cell_indices`` tests against the reference metric.
# ---------------------------------------------------------------------------

def test_cell_indices_matches_reference_single_sphere_azimuthal_last():
    rng = np.random.default_rng(0)
    n = 60
    lat = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    lon = rng.uniform(-np.pi, np.pi, size=n)
    query = np.column_stack([lat, lon])

    centres = np.array(
        [
            [-1.0, -2.5],
            [0.0, 0.0],
            [1.2, 3.0],
            [-0.7, 2.9],
        ]
    )
    dims = np.array([0, 1], dtype=np.int32)
    metric_red = np.array([1], dtype=np.int32)
    member_red = np.array([2], dtype=np.int32)

    indices = _core.cell_indices(query, centres, dims, metric_red, member_red)
    expected = _reference_cell_indices(query, centres, dims, metric_red, member_red)

    np.testing.assert_array_equal(indices, expected)


def test_cell_indices_matches_reference_partially_active_spherical_dim():
    # A tessellation may split on only one column of a spherical group. The
    # inactive column is taken from the query, so it must cancel in the
    # great-circle distance identically in the C++ backend and the reference.
    rng = np.random.default_rng(1)
    n = 50
    lat = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    lon = rng.uniform(-np.pi, np.pi, size=n)
    query = np.column_stack([lat, lon])
    metric_red = np.array([1], dtype=np.int32)
    member_red = np.array([2], dtype=np.int32)

    for active in ([0], [1]):
        dims = np.array(active, dtype=np.int32)
        centres = np.array([[-1.0], [0.5], [2.8]])
        indices = _core.cell_indices(query, centres, dims, metric_red, member_red)
        expected = _reference_cell_indices(query, centres, dims, metric_red, member_red)
        np.testing.assert_array_equal(indices, expected)


def test_cell_indices_matches_reference_multi_sphere_azimuthal_last():
    # Two independent spherical surfaces laid out as [lat, lon | lat, lat, lon].
    rng = np.random.default_rng(2)
    n = 80
    lat1 = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    lon1 = rng.uniform(-np.pi, np.pi, size=n)
    lat2a = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    lat2b = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    lon2 = rng.uniform(-np.pi, np.pi, size=n)
    query = np.column_stack([lat1, lon1, lat2a, lat2b, lon2])

    centres = rng.uniform(-1.0, 1.0, size=(6, 5))
    dims = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    metric_red = np.array([1, 1], dtype=np.int32)
    member_red = np.array([2, 3], dtype=np.int32)

    indices = _core.cell_indices(query, centres, dims, metric_red, member_red)
    expected = _reference_cell_indices(query, centres, dims, metric_red, member_red)

    np.testing.assert_array_equal(indices, expected)


def test_cell_indices_matches_reference_mixed_euclidean_and_spherical():
    rng = np.random.default_rng(3)
    n = 70
    euc = rng.normal(size=n)
    lat = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    lon = rng.uniform(-np.pi, np.pi, size=n)
    query = np.column_stack([euc, lat, lon])

    centres = np.array(
        [
            [-1.5, -1.0, -2.8],
            [0.0, 0.0, 0.0],
            [1.5, 1.0, 2.8],
        ]
    )
    dims = np.array([0, 1, 2], dtype=np.int32)
    metric_red = np.array([0, 1], dtype=np.int32)
    member_red = np.array([1, 2], dtype=np.int32)

    indices = _core.cell_indices(query, centres, dims, metric_red, member_red)
    expected = _reference_cell_indices(query, centres, dims, metric_red, member_red)

    np.testing.assert_array_equal(indices, expected)


def test_spherical_metric_respects_longitude_wraparound():
    # A query just west of -pi is closest (on the globe) to a centre just east
    # of +pi. Euclidean distance would instead pick the centre at longitude 0.
    query = np.array([[0.0, -np.pi + 0.05]])
    centres = np.array([[0.0, 0.0], [0.0, np.pi - 0.05]])
    dims = np.array([0, 1], dtype=np.int32)
    member_red = np.array([2], dtype=np.int32)

    spherical = _core.cell_indices(query, centres, dims, np.array([1], dtype=np.int32), member_red)
    euclidean = _core.cell_indices(query, centres, dims, np.array([0], dtype=np.int32), member_red)

    assert spherical[0] == 1  # wrap-around neighbour across the date line
    assert euclidean[0] == 0  # naive nearest ignores wrap-around


def test_azimuthal_dimension_is_the_last_column():
    # Build data where treating the last column as azimuthal gives different
    # cell assignments than treating the first column as azimuthal. The C++
    # backend must follow the "azimuthal is last" convention.
    rng = np.random.default_rng(4)
    n = 60
    col0 = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    col1 = rng.uniform(-np.pi, np.pi, size=n)
    query = np.column_stack([col0, col1])

    centres = np.array([[-1.0, -3.0], [1.0, 3.0], [0.0, 0.2]])
    dims = np.array([0, 1], dtype=np.int32)
    metric_red = np.array([1], dtype=np.int32)
    member_red = np.array([2], dtype=np.int32)

    indices = _core.cell_indices(query, centres, dims, metric_red, member_red)

    # Reference with the correct convention (azimuthal = last column).
    expected_last = _reference_cell_indices(query, centres, dims, metric_red, member_red)
    # Reference with the columns reversed, i.e. azimuthal = first column.
    swapped_query = query[:, ::-1].copy()
    swapped_centres = centres[:, ::-1].copy()
    expected_first = _reference_cell_indices(
        swapped_query, swapped_centres, dims, metric_red, member_red
    )

    np.testing.assert_array_equal(indices, expected_last)
    # The dataset must be sensitive to the ordering, otherwise this test would
    # pass even if the azimuthal column were mishandled.
    assert np.any(expected_last != expected_first)


# ---------------------------------------------------------------------------
# In-sample vs. out-of-sample consistency through the full estimator.
# ---------------------------------------------------------------------------

def _in_sample_response(model):
    matrix = model.posterior_.prediction_matrix
    return matrix.mean(axis=1) * model.y_range_ + model.y_centre_


def _spherical_dataset(seed):
    rng = np.random.default_rng(seed)
    n = 160
    lat = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    lon = rng.uniform(-np.pi, np.pi, size=n)
    y = 20 * np.cos(lat) + 5 * np.sin(lon) + rng.normal(scale=1.0, size=n)
    return np.column_stack([lat, lon]), y


def test_in_sample_matches_out_of_sample_single_sphere():
    X, y = _spherical_dataset(seed=10)
    model = fast_model(
        n_tessellations=20,
        total_mcmc_iter=200,
        burn_in=60,
        thinning=1,
        random_state=10,
        metric="spherical",
    ).fit(X, y)

    np.testing.assert_allclose(model.predict(X), _in_sample_response(model), atol=1e-9)


def test_in_sample_matches_out_of_sample_multi_sphere_azimuthal_last():
    rng = np.random.default_rng(11)
    n = 160
    lat = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    lon = rng.uniform(-np.pi, np.pi, size=n)
    lat_prime1 = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    lat_prime2 = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    lon_prime = rng.uniform(-np.pi, np.pi, size=n)
    y = (
        20 * np.cos(lat)
        + 5 * np.sin(lon)
        - 5 * np.cos(lat_prime1) * np.sin(lat_prime2)
        + 10 * np.sin(lon_prime) ** 2 * np.sin(np.abs(lat_prime1 - lat_prime2))
        + rng.normal(scale=1.0, size=n)
    )
    X = np.column_stack([lat, lon, lat_prime1, lat_prime2, lon_prime])

    model = fast_model(
        n_tessellations=20,
        total_mcmc_iter=200,
        burn_in=60,
        thinning=1,
        random_state=11,
        metric="spherical",
        members=[1, 1, 2, 2, 2],
    ).fit(X, y)

    # The membership grouping must place the azimuthal column last in each group.
    np.testing.assert_array_equal(model.metric_red_, [1, 1])
    np.testing.assert_array_equal(model.member_red_, [2, 3])
    np.testing.assert_allclose(model.predict(X), _in_sample_response(model), atol=1e-9)


def test_in_sample_matches_out_of_sample_mixed_metrics():
    rng = np.random.default_rng(12)
    n = 160
    euc = rng.normal(size=n)
    lat = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    lon = rng.uniform(-np.pi, np.pi, size=n)
    y = 3 * euc + 20 * np.cos(lat) + 5 * np.sin(lon) + rng.normal(scale=1.0, size=n)
    X = np.column_stack([euc, lat, lon])

    model = fast_model(
        n_tessellations=20,
        total_mcmc_iter=200,
        burn_in=60,
        thinning=1,
        random_state=12,
        metric=["euclidean", "spherical", "spherical"],
    ).fit(X, y)

    np.testing.assert_array_equal(model.metric_red_, [0, 1])
    np.testing.assert_array_equal(model.member_red_, [1, 2])
    np.testing.assert_allclose(model.predict(X), _in_sample_response(model), atol=1e-9)


def test_out_of_sample_prediction_is_reproducible_for_new_points():
    X, y = _spherical_dataset(seed=13)
    model = fast_model(
        n_tessellations=15,
        total_mcmc_iter=150,
        burn_in=50,
        thinning=1,
        random_state=13,
        metric="spherical",
    ).fit(X, y)

    rng = np.random.default_rng(99)
    X_new = np.column_stack(
        [
            rng.uniform(-np.pi / 2, np.pi / 2, size=40),
            rng.uniform(-np.pi, np.pi, size=40),
        ]
    )

    first = model.predict(X_new)
    second = model.predict(X_new)
    np.testing.assert_allclose(first, second)
    assert np.all(np.isfinite(first))


# ---------------------------------------------------------------------------
# Reduced metric/member grouping used by the out-of-sample path.
# ---------------------------------------------------------------------------

def test_reduced_metric_and_members_groups_azimuthal_last_layout():
    metric = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    members = np.array([1, 1, 2, 2, 2], dtype=np.int32)

    metric_red, member_red = reduced_metric_and_members(metric, members)

    np.testing.assert_array_equal(metric_red, [1, 1])
    np.testing.assert_array_equal(member_red, [2, 3])


def test_reduced_metric_and_members_default_single_sphere():
    metric = np.array([1, 1], dtype=np.int32)
    members = np.array([2, 2], dtype=np.int32)

    metric_red, member_red = reduced_metric_and_members(metric, members)

    np.testing.assert_array_equal(metric_red, [1])
    np.testing.assert_array_equal(member_red, [2])


def test_reduced_metric_and_members_length_mismatch_raises():
    with pytest.raises(ValueError):
        reduced_metric_and_members(np.array([1, 1]), np.array([1]))
