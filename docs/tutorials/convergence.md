# Convergence diagnostics and trace analysis

AddiVortes records MCMC trace statistics during fitting and exposes diagnostics
for assessing convergence, effective sample size, and posterior stability.

This tutorial shows how to inspect trace plots, density curves, and
autocorrelation summaries for the fitted model.

## 1. Fit a model

```python
import numpy as np
from addivortes import AddiVortesRegressor

rng = np.random.default_rng(2026)
X = rng.normal(size=(80, 3))
y = X[:, 0] * 0.8 - 0.5 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(scale=0.2, size=80)

model = AddiVortesRegressor(
    n_tessellations=50,
    total_mcmc_iter=400,
    burn_in=100,
    thinning=2,
    random_state=2026,
)
model.fit(X, y)
```

## 2. Trace plots and diagnostics

After fitting, use `model.trace_diagnostics()` to inspect MCMC behavior for
key trace statistics.

```python
axes = model.trace_diagnostics(
    show=True,
    plot_types=("trace", "histogram", "autocorrelation"),
)
```

This creates three types of plots for each diagnostic statistic:

- `trace`: the full MCMC trace and burn-in boundary
- `histogram`: a kernel density estimate of post-burn-in samples
- `autocorrelation`: the lag autocorrelation sequence up to 250

### What to look for

- **Stable trace**: after burn-in, the trace should fluctuate around a constant
  range rather than drifting.
- **Reasonable effective sample size**: AddiVortes prints the estimated
  effective sample size (ESS) for each trace statistic.
- **Low autocorrelation**: autocorrelation should decay toward zero by lag 250.

## 3. Understanding the output

`trace_diagnostics()` evaluates the following statistics by default:

- `average_centres_per_tessellation`
- `average_dimensions_per_tessellation`
- `sigma`

These are the same trace statistics used by `model.traceplots()` plus the
log-likelihood trace, and they help diagnose whether the posterior sampler has
mixed adequately.

## 4. Interactive and sequential display

When `show=True`, diagnostics are displayed sequentially with optional prompts,
so each plot is shown one at a time. This is helpful when reviewing multiple
statistics in a notebook or interactive session.

```python
model.trace_diagnostics(show=True, ask=True)
```

## 5. Customizing the diagnostics

You can restrict diagnostics to a subset of plot types or statistics:

```python
model.trace_diagnostics(
    plot_types=("autocorrelation",),
    stats=("sigma",),
    lag_k=250,
    show=True,
)
```

This only draws the selected diagnostics and uses a maximum lag of 250 for the
autocorrelation summary.

## 6. Link to the API reference

For the full parameter list and additional details, see
[the API reference](../api.md#addivortesregressor).