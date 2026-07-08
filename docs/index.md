# AddiVortes

AddiVortes is a Python package for Bayesian Additive Voronoi Tessellation
regression. It provides a machine-learning style estimator for non-parametric
regression, spatial modeling, uncertainty quantification, and complex function
approximation.

The package is implemented as a Python API backed by a C++20 extension. It
supports numpy arrays and pandas data frames, including categorical covariates
through one-hot encoding.

## Highlights

- Python estimator API with `fit`, `predict`, `fit_predict`, `score`, and
  sklearn-style parameter access.
- Posterior mean predictions, credible intervals, and prediction intervals.
- Numeric and categorical covariate preprocessing.
- C++20 backend for the sampler and nearest-cell assignment.
- PyPI-oriented packaging with wheel and source distribution checks.

## Tutorials

Step-by-step walkthroughs are available in the documentation:

- [Machine learning with AddiVortes](tutorials/introduction.md): Boston Housing
  regression with train/test evaluation.
- [Bayesian regression and prediction](tutorials/prediction.md): synthetic data
  with known structure and credible intervals.
- [Convergence diagnostics and trace analysis](tutorials/convergence.md): effective
  sample size, autocorrelation, and density-based trace diagnostics.
- [Modelling spherical data](tutorials/spherical.md): great-circle distance,
  mixed metrics, and multiple spherical surfaces.
- [Using categorical covariates](tutorials/categorical.md): one-hot encoding,
  `cat_scaling`, and unseen levels at prediction time.

## Minimal example

```python
import numpy as np
from addivortes import AddiVortesRegressor

rng = np.random.default_rng(123)
X = rng.normal(size=(100, 4))
y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.2, size=100)

model = AddiVortesRegressor(
    n_tessellations=25,
    total_mcmc_iter=300,
    burn_in=100,
    random_state=123,
)
model.fit(X, y)

predictions = model.predict(X[:5])
intervals = model.predict(X[:5], kind="quantile", quantiles=(0.025, 0.975))
```
