# API reference

## `AddiVortesRegressor`

```python
from addivortes import AddiVortesRegressor
```

`AddiVortesRegressor` is the primary estimator class.

### Constructor

```python
AddiVortesRegressor(
    *,
    n_tessellations=200,
    total_mcmc_iter=1200,
    burn_in=200,
    thinning=1,
    nu=6.0,
    q=0.85,
    k=3,
    omega=None,
    lambda_rate=25.0,
    initial_sigma="linear",
    metric="euclidean",
    members=None,
    cat_scaling=1.0,
    random_state=None,
    verbose=False,
)
```

Set `verbose=True` to show terminal progress bars while fitting the MCMC
sampler and while generating predictions.

### Main methods

#### `fit(X, y)`

Fit the model to covariates `X` and response `y`.

- `X` may be a numpy array or pandas DataFrame.
- pandas object, string, and categorical columns are treated as categorical.
- `y` must be one-dimensional and numeric.

Returns the fitted estimator.

#### `predict(X, *, kind="response", quantiles=(0.025, 0.975), interval="credible", random_state=None)`

Predict from a fitted model.

- `kind="response"` returns posterior mean predictions with shape `(n_obs,)`.
- `kind="quantile"` returns posterior quantiles with shape
  `(n_obs, len(quantiles))`.
- `interval="credible"` summarizes uncertainty in the posterior mean.
- `interval="prediction"` includes Gaussian observation noise using posterior
  sigma samples.

#### Distance metrics

The `metric` argument controls how covariate distances are computed:

- `"euclidean"` (default): standard Euclidean distance.
- `"spherical"`: great-circle distance for geographic or other spherical
  coordinates in radians. See [Modelling spherical data](tutorials/spherical.md).

Pass a list with one value per column to mix metrics, for example
`metric=["spherical", "spherical", "euclidean"]`.

When covariates lie on multiple spherical surfaces, pass `members` with one
integer per column indicating surface membership, for example
`members=[1, 1, 2, 2, 2]`.

#### Categorical covariates

String, object, and pandas `Categorical` columns are one-hot encoded
automatically. Use `cat_scaling` to control the weight of categorical
differences relative to continuous covariates. See
[Using categorical covariates](tutorials/categorical.md).

#### `fit_predict(X, y, **predict_kwargs)`

Fit the model and return predictions for the training covariates.

#### `score(X, y)`

Return the coefficient of determination on test data.

#### `plot(x_train, y_train, *, sigma_trace=None, which=(1, 2, 3), ask=False, axes=None, show=False, **kwargs)`

Create diagnostic plots for a fitted model. Plotting uses matplotlib, which can
be installed with:

```bash
python -m pip install "addivortes[plot]"
```

The `which` argument selects diagnostics:

- `1`: residuals vs fitted values with a smoothed trend line,
- `2`: posterior sigma trace,
- `3`: average tessellation complexity trace,
- `4`: predicted vs observed values with 95% credible intervals.

When `sigma_trace` is omitted, the method uses the fitted posterior variance
samples and plots their square roots as sigma values. The method returns the
matplotlib axes used for the requested plots.

Note that `which=3` traces the average centre count in **posterior thinned
samples** after burn-in. For full-chain MCMC trace diagnostics, use
`traceplots()` instead.

#### `traceplots(*, ask=False, axes=None, show=False, **kwargs)`

Create four MCMC trace plots recorded at every iteration:

- average centres per tessellation,
- standard deviation of centre counts across tessellations,
- average active dimensions per tessellation,
- retained-state log-likelihood component.

This method does not require training data. It uses `trace_stats_`, which is
populated during `fit`.

#### `trace_diagnostics(*, plot_types=None, stats=None, lag_k=250, ask=False, axes=None, show=False, **kwargs)`

Create detailed MCMC diagnostics for a fitted model.

- `plot_types` may include any combination of `"trace"`, `"histogram"`, and
  `"autocorrelation"`.
- `stats` may include `"average_centres_per_tessellation"`,
  `"average_dimensions_per_tessellation"`, and `"sigma"`.
- `lag_k` controls the maximum lag for autocorrelation summaries and defaults to
  250.
- When `show=True`, the diagnostics display sequentially and prompt before
  showing each plot when `ask=True`.

This method prints an estimate of effective sample size for each statistic and
returns the matplotlib axes used for the requested plots.

#### `summary()`

Return a dictionary with fitted-model summary information, including:

- number of features,
- number of tessellations,
- number of posterior samples,
- in-sample RMSE,
- estimated lambda,
- average cells per tessellation.

### Fitted attributes

After `fit`, common fitted attributes include:

- `posterior_`
- `trace_stats_`
- `x_centres_`
- `x_ranges_`
- `y_centre_`
- `y_range_`
- `in_sample_rmse_`
- `cat_encoding_`
- `feature_names_in_`
- `n_features_in_`

## Alias

`AddiVortes` is an alias for `AddiVortesRegressor`:

```python
from addivortes import AddiVortes
```

The package also exposes a functional plotting wrapper:

```python
from addivortes import plot

plot(model, X, y, which=(1, 4))
```

MCMC trace diagnostics are also available without training data:

```python
from addivortes import traceplots

traceplots(model, show=True)
```
