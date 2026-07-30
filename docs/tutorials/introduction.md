# Machine learning with AddiVortes

AddiVortes offers a Bayesian alternative to BART (Bayesian Additive Regression
Trees), using Voronoi tessellations for spatial partitioning. This tutorial
walks through loading data, training a regression model, and evaluating
predictions on a held-out test set.

## 1. Loading data

We use the Boston Housing dataset. The predictors are the first thirteen numeric
columns and the response is median house value.

```python
import numpy as np
import pandas as pd
from addivortes import AddiVortesRegressor

url = (
    "https://raw.githubusercontent.com/anonymous2738/"
    "AddiVortesAlgorithm/DataSets/BostonHousing_Data.csv"
)
boston = pd.read_csv(url)

X = boston.iloc[:, 1:14].to_numpy(dtype=float)
y = boston.iloc[:, 14].to_numpy(dtype=float)
```

## 2. Train and test split

We hold out one sixth of the observations for testing.

```python
rng = np.random.default_rng(1025)
n = len(y)
train_size = int(5 * n / 6)
train_idx = np.sort(rng.choice(n, size=train_size, replace=False))
test_idx = np.setdiff1d(np.arange(n), train_idx)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
```

## 3. Training the model

Fit `AddiVortesRegressor` on the training data. The parameters below mirror a
typical production configuration; reduce `total_mcmc_iter` for a quicker
demonstration.

```python
model = AddiVortesRegressor(
    n_tessellations=200,
    total_mcmc_iter=2000,
    burn_in=200,
    nu=6.0,
    q=0.85,
    k=3,
    omega=3.0,
    lambda_rate=5.0,
    initial_sigma="linear",
    random_state=1025,
    verbose=False,
)
model.fit(X_train, y_train)
```

## 4. Predictions and evaluation

```python
preds = model.predict(X_test)

print("In-sample RMSE:", round(model.in_sample_rmse_, 3))

test_rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
print("Test RMSE:", round(test_rmse, 3))
```

## 5. Visualising results

Install the optional plotting dependency first:

```bash
python -m pip install "addivortes[plot]"
```

A predicted-versus-observed plot with credible intervals is a useful sanity
check. Points should cluster around the equality line `y = x`.

```python
import matplotlib.pyplot as plt

pred_quantiles = model.predict(X_test, kind="quantile", quantiles=(0.025, 0.975))

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, preds, color="darkblue", s=25, zorder=3)
ax.vlines(y_test, pred_quantiles[:, 0], pred_quantiles[:, 1], color="darkblue", linewidth=1)
ax.axline((0, 0), slope=1, color="darkred", linewidth=2)
ax.set(xlabel="True values", ylabel="Predicted values", title="AddiVortes predictions vs true values")
ax.legend(["Prediction", "y = x line", "95% interval"], loc="upper left")
plt.tight_layout()
plt.show()
```

The fitted model also exposes diagnostic plots via `model.plot()` and MCMC trace
plots via `model.traceplots()`; see the [API reference](../api.md) for details.
