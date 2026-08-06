"""Fixed-seed Friedman benchmark guarding MCMC/predict hot-path changes."""

from __future__ import annotations

import time

import numpy as np

from addivortes import AddiVortesRegressor
from addivortes import _core


def friedman_response(x: np.ndarray) -> np.ndarray:
    return (
        10.0 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20.0 * (x[:, 2] - 0.5) ** 2
        + 10.0 * x[:, 3]
        + 5.0 * x[:, 4]
    )


def friedman_data(n: int, p: int = 10, sigma: float = 1.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, p))
    y = friedman_response(x) + rng.normal(scale=sigma, size=n)
    return x, y


FRIEDMAN_BENCH = {
    "seed_train": 20260806,
    "seed_test": 20260807,
    "n_train": 200,
    "n_test": 80,
    "p": 10,
    "sigma": 1.0,
    "n_tessellations": 50,
    "total_mcmc_iter": 200,
    "burn_in": 50,
    "thinning": 1,
    # Locked after the 0.7.1 hot-path port; regenerate deliberately if the
    # sampler's RNG contract changes.
    "expected_in_sample_rmse": 0.59,
    "expected_test_rmse": 1.928,
    "rmse_digits": 3,
}


def test_friedman_fixed_seed_fit_and_predict_stay_stable():
    train_x, train_y = friedman_data(
        FRIEDMAN_BENCH["n_train"],
        FRIEDMAN_BENCH["p"],
        FRIEDMAN_BENCH["sigma"],
        FRIEDMAN_BENCH["seed_train"],
    )
    test_x, test_y = friedman_data(
        FRIEDMAN_BENCH["n_test"],
        FRIEDMAN_BENCH["p"],
        FRIEDMAN_BENCH["sigma"],
        FRIEDMAN_BENCH["seed_test"],
    )

    model = AddiVortesRegressor(
        n_tessellations=FRIEDMAN_BENCH["n_tessellations"],
        total_mcmc_iter=FRIEDMAN_BENCH["total_mcmc_iter"],
        burn_in=FRIEDMAN_BENCH["burn_in"],
        thinning=FRIEDMAN_BENCH["thinning"],
        random_state=FRIEDMAN_BENCH["seed_train"],
        verbose=False,
    )

    fit_started = time.perf_counter()
    model.fit(train_x, train_y)
    fit_secs = time.perf_counter() - fit_started

    in_sample = model.posterior_.prediction_matrix.mean(axis=1) * model.y_range_ + model.y_centre_
    in_sample_rmse = float(np.sqrt(np.mean((train_y - in_sample) ** 2)))

    pred_started = time.perf_counter()
    preds = model.predict(test_x)
    pred_secs = time.perf_counter() - pred_started
    test_rmse = float(np.sqrt(np.mean((test_y - preds) ** 2)))

    digits = FRIEDMAN_BENCH["rmse_digits"]
    rounded_in = round(in_sample_rmse, digits)
    rounded_test = round(test_rmse, digits)

    assert rounded_in == FRIEDMAN_BENCH["expected_in_sample_rmse"]
    assert rounded_test == FRIEDMAN_BENCH["expected_test_rmse"]

    print(
        f"Friedman benchmark (n={FRIEDMAN_BENCH['n_train']}, "
        f"m={FRIEDMAN_BENCH['n_tessellations']}, "
        f"iter={FRIEDMAN_BENCH['total_mcmc_iter']}): "
        f"fit={fit_secs:.3f}s predict={pred_secs:.3f}s "
        f"in={in_sample_rmse:.3f} test={test_rmse:.3f} "
        f"(rounded in={rounded_in} test={rounded_test})"
    )


def test_friedman_ensemble_predict_matches_cell_indices_reference():
    train_x, train_y = friedman_data(60, 10, 1.0, 4242)
    model = AddiVortesRegressor(
        n_tessellations=8,
        total_mcmc_iter=40,
        burn_in=10,
        thinning=1,
        random_state=4242,
        verbose=False,
    )
    model.fit(train_x, train_y)

    test_x, _ = friedman_data(30, 10, 1.0, 4243)
    preds_cpp = model.predict(test_x)

    from addivortes.preprocessing import apply_scaling, prepare_design, category_counts

    design = prepare_design(
        test_x,
        metric=model.metric_labels_,
        members=model.original_members_,
        cat_scaling=float(model.cat_scaling),
        encoding=model.cat_encoding_,
        cat_onehot=bool(model.cat_onehot),
    )
    x_scaled = apply_scaling(design.values, model.x_centres_, model.x_ranges_)
    x_scaled = model._restore_unscaled_columns(x_scaled, design.values, model.metric_, model.cat_encoding_)
    ncats = getattr(model, "ncats_", None)
    if ncats is None:
        ncats = category_counts(model.metric_, design.values)

    n_obs = x_scaled.shape[0]
    n_samples = len(model.posterior_.tessellations)
    pred_mat = np.zeros((n_obs, n_samples), dtype=float)
    for sample_idx in range(n_samples):
        sample_pred = np.zeros(n_obs, dtype=float)
        for tess, dims, cell_pred in zip(
            model.posterior_.tessellations[sample_idx],
            model.posterior_.dimensions[sample_idx],
            model.posterior_.predictions[sample_idx],
            strict=True,
        ):
            indices = _core.cell_indices(
                x_scaled,
                np.asarray(tess, dtype=float),
                np.asarray(dims, dtype=np.int32),
                model.metric_red_,
                model.member_red_,
                ncats,
            )
            sample_pred += np.asarray(cell_pred, dtype=float)[indices]
        pred_mat[:, sample_idx] = sample_pred

    preds_ref = pred_mat.mean(axis=1) * model.y_range_ + model.y_centre_
    np.testing.assert_allclose(preds_cpp, preds_ref, atol=1e-12)
