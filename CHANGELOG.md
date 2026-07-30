# Changelog

## Unreleased

- Fixed structural-move acceptance ratios so cell-count and covariate-count
  priors match the published model (extra birth penalty, doubled covariate pick
  ratio, and missing boundary selection adjustments).
- Fixed proposal draws for add-centre, change, and swap so centre coordinates
  use each selected covariate's own proposal settings.
- Changed the default `lambda_rate` from 25 to 5 so typical cell counts stay
  aligned with the corrected cell-count prior.

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
