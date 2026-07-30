"""Generate a small deterministic MCMC parity fixture for regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent


def main() -> None:
    rng = np.random.default_rng(2024)
    n_obs, n_features = 12, 3
    x = rng.normal(size=(n_obs, n_features))
    y = x[:, 0] - 0.4 * x[:, 1] + rng.normal(scale=0.1, size=n_obs)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FIXTURE_DIR / "x_scaled.npy", x)
    np.save(FIXTURE_DIR / "y_scaled.npy", y)
    np.save(FIXTURE_DIR / "metric.npy", np.zeros(n_features, dtype=np.int32))
    np.save(FIXTURE_DIR / "members.npy", np.arange(n_features, dtype=np.int32))
    np.save(FIXTURE_DIR / "proposal_sd.npy", np.full(n_features, 0.2))
    np.save(FIXTURE_DIR / "proposal_mu.npy", np.zeros(n_features))
    np.save(FIXTURE_DIR / "binary_cols.npy", np.empty(0, dtype=np.int32))

    init_tess = [np.array([[0.0]], dtype=float), np.array([[0.5]], dtype=float)]
    init_dim = [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)]
    init_pred = [np.array([0.0], dtype=float), np.array([0.1], dtype=float)]
    np.save(FIXTURE_DIR / "init_tess.npy", np.array(init_tess, dtype=object))
    np.save(FIXTURE_DIR / "init_dim.npy", np.array(init_dim, dtype=object))
    np.save(FIXTURE_DIR / "init_pred.npy", np.array(init_pred, dtype=object))

    meta = {
        "m": 2,
        "total_iter": 16,
        "burn_in": 6,
        "thinning": 2,
        "nu": 6.0,
        "lambda_value": 0.05,
        "sigma_squared_mu": 0.01,
        "omega": 2.0,
        "lambda_rate": 5.0,
        "cat_scaling": 1.0,
        "seed": 424242,
    }
    np.save(FIXTURE_DIR / "meta.npy", meta)


if __name__ == "__main__":
    main()
