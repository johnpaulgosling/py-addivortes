import numpy as np
import pandas as pd
import pytest

from addivortes import _core
from addivortes.preprocessing import category_counts, prepare_design, reduced_metric_and_members
from conftest import fast_model


def _eskin_distance(first: np.ndarray, second: np.ndarray, ncats: np.ndarray) -> float:
    total = 0.0
    for left, right, n_cat in zip(first, second, ncats, strict=True):
        if left != right:
            total += 2.0 / (float(n_cat) ** 2)
    return total


def test_prepare_design_integer_codes_when_cat_onehot_false():
    frame = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0],
            "group": ["b", "a", "c"],
        }
    )

    design = prepare_design(frame, metric="euclidean", cat_onehot=False)

    assert design.columns == ("x1", "group")
    assert design.encoding is not None
    assert design.encoding.one_hot is False
    assert design.encoding.encoded_binary_cols == ()
    assert design.encoding.levels[1] == ("a", "b", "c")
    np.testing.assert_array_equal(design.metric, [0, 2])
    # Alphabetical levels: a=1, b=2, c=3
    np.testing.assert_allclose(design.values[:, 1], [2.0, 1.0, 3.0])
    np.testing.assert_array_equal(category_counts(design.metric, design.values), [3])


def test_prepare_design_reuses_integer_encoding_and_maps_unseen_to_reference():
    train = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "group": ["a", "b", "c"]})
    design = prepare_design(train, metric="euclidean", cat_onehot=False)
    new = pd.DataFrame({"x1": [3.0, 4.0], "group": ["b", "unknown"]})

    encoded = prepare_design(new, metric="euclidean", encoding=design.encoding)

    assert encoded.encoding is not None
    assert encoded.encoding.one_hot is False
    np.testing.assert_allclose(encoded.values[:, 1], [2.0, 1.0])


def test_cell_indices_uses_eskin_distance_for_categorical_metric():
    # Three observations with a single categorical covariate (3 levels).
    query = np.array([[1.0], [2.0], [3.0]], dtype=float)
    # Centres at codes 1 and 3; query code 2 should be tied on Eskin distance
    # (both mismatches contribute 2/9) and stable on the first centre.
    centres = np.array([[1.0], [3.0]], dtype=float)
    dims = np.array([0], dtype=np.int32)
    metric_red = np.array([2], dtype=np.int32)
    member_red = np.array([1], dtype=np.int32)
    ncats = np.array([3], dtype=np.int32)

    indices = _core.cell_indices(query, centres, dims, metric_red, member_red, ncats)
    np.testing.assert_array_equal(indices, [0, 0, 1])

    # Explicit Eskin checks for the middle query against both centres.
    assert _eskin_distance(np.array([2.0]), np.array([1.0]), ncats) == pytest.approx(2.0 / 9.0)
    assert _eskin_distance(np.array([2.0]), np.array([3.0]), ncats) == pytest.approx(2.0 / 9.0)


def test_eskin_weight_decreases_with_more_categories():
    ncats_small = np.array([2], dtype=np.int32)
    ncats_large = np.array([10], dtype=np.int32)
    mismatch_small = _eskin_distance(np.array([1.0]), np.array([2.0]), ncats_small)
    mismatch_large = _eskin_distance(np.array([1.0]), np.array([2.0]), ncats_large)
    assert mismatch_small == pytest.approx(0.5)
    assert mismatch_large == pytest.approx(0.02)
    assert mismatch_small > mismatch_large


def test_model_fits_and_predicts_with_cat_onehot_false():
    rng = np.random.default_rng(321)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=36),
            "group": pd.Categorical(rng.choice(["a", "b", "c"], size=36), categories=["a", "b", "c"]),
            "flag": rng.choice(["yes", "no"], size=36),
        }
    )
    y = frame["x1"].to_numpy() + (frame["group"].astype(str) == "b").to_numpy(dtype=float)

    model = fast_model(
        n_tessellations=4,
        total_mcmc_iter=20,
        burn_in=6,
        cat_onehot=False,
        random_state=7,
    ).fit(frame, y)

    assert model.cat_encoding_ is not None
    assert model.cat_encoding_.one_hot is False
    assert model.cat_encoding_.encoded_binary_cols == ()
    assert model.feature_names_in_ == ("x1", "group", "flag")
    np.testing.assert_array_equal(model.metric_, [0, 2, 2])
    np.testing.assert_array_equal(model.ncats_, [3, 2])
    assert np.isfinite(model.in_sample_rmse_)

    # Categorical centres should be integer codes in the observed ranges.
    cat_cols = np.where(model.metric_ == 2)[0]
    col_to_ncats = {int(col): int(model.ncats_[idx]) for idx, col in enumerate(cat_cols)}
    observed_cat_dim = False
    for sample_tess, sample_dims in zip(model.posterior_.tessellations, model.posterior_.dimensions, strict=True):
        for tess, dims in zip(sample_tess, sample_dims, strict=True):
            for local_idx, global_dim in enumerate(dims):
                global_dim = int(global_dim)
                if global_dim not in col_to_ncats:
                    continue
                observed_cat_dim = True
                values = np.asarray(tess)[:, local_idx]
                assert np.all(values == np.floor(values))
                assert np.all(values >= 1.0)
                assert np.all(values <= float(col_to_ncats[global_dim]))
    assert observed_cat_dim

    new_frame = pd.DataFrame(
        {
            "x1": [0.0, 1.0],
            "group": pd.Categorical(["b", "unknown"], categories=["a", "b", "c", "unknown"]),
            "flag": ["yes", "maybe"],
        }
    )
    pred = model.predict(new_frame)
    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))


def test_in_sample_predictions_align_for_eskin_path():
    rng = np.random.default_rng(99)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=40),
            "group": rng.choice(["low", "mid", "high"], size=40),
        }
    )
    y = frame["x1"].to_numpy() - (frame["group"] == "high").to_numpy(dtype=float)

    model = fast_model(
        n_tessellations=5,
        total_mcmc_iter=24,
        burn_in=8,
        cat_onehot=False,
        random_state=3,
    ).fit(frame, y)

    in_sample = model.posterior_.prediction_matrix.mean(axis=1) * model.y_range_ + model.y_centre_
    oos = model.predict(frame)
    np.testing.assert_allclose(oos, in_sample, rtol=1e-10, atol=1e-10)


def test_reduced_metric_keeps_native_categorical_type():
    metric = np.array([0, 2, 2], dtype=np.int32)
    members = np.array([1, 3, 3], dtype=np.int32)
    metric_red, member_red = reduced_metric_and_members(metric, members)
    np.testing.assert_array_equal(metric_red, [0, 2])
    np.testing.assert_array_equal(member_red, [1, 2])
