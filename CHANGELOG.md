# Changelog

## 0.6.9

- Added native categorical covariates via `cat_onehot=False`, using Eskin
  distance (Eskin et al., 2002) instead of one-hot encoding.
- Threaded categorical level counts through the C++ MCMC sampler, proposals,
  and nearest-cell assignment.
- Categorical centre proposals now draw integer category codes uniformly.
- Released previously unreleased structural-move Metropolis-Hastings fixes so
  cell-count and covariate-count priors match the published model.
- Fixed proposal draws for add-centre, change, and swap so centre coordinates
  use each selected covariate's own proposal settings.
- Kept the default `lambda_rate` at 5 so typical cell counts stay aligned with
  the corrected cell-count prior.

## 0.6.6

- Added in more tests to check that in-sample and out-of-sample predictions align (especially in the spherical case with permuted variables).

## 0.6.5

- Aligned MCMC proposal and acceptance logic with the updated R package implementation.
- Added per-iteration `trace_stats` output from the C++ backend during fitting.
- Added `traceplots()` for burn-in-aware MCMC trace diagnostics.

## 0.6.1

- Migrated AddiVortes to a Python-only package named `addivortes`.
- Added a Pythonic `AddiVortesRegressor` estimator API.
- Added a C++20 pybind11 backend for fitting and prediction support.
- Added numpy and pandas preprocessing, including categorical covariate encoding.
- Added Python packaging metadata, tests, wheel build support, and CI.
