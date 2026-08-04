# Using categorical covariates

This tutorial explains how AddiVortes handles **categorical covariates**: variables
that take a discrete set of named levels, such as region, product type, or
treatment group. Voronoi tessellations require numerical distances between
points. By default, categorical variables are converted using **one-hot
encoding**. Set `cat_onehot=False` to keep native categorical columns and use
**Eskin distance** instead.

## 1. What is one-hot encoding?

A categorical variable with *d* distinct levels cannot be treated as a single
number because there is no natural ordering or magnitude between categories.
Assigning "North" = 1, "South" = 2, "East" = 3, "West" = 4 would incorrectly
imply that "West" is four times "North".

**One-hot encoding** converts a categorical variable with *d* levels into *d − 1*
binary (0/1) indicator columns. One level is chosen as the **reference level**
(by convention, the first level in alphabetical order), and the remaining
*d − 1* levels each receive their own column:

| Level | `region_North` | `region_South` | `region_West` |
|-------|:--------------:|:--------------:|:-------------:|
| East  | 0              | 0              | 0             |
| North | 1              | 0              | 0             |
| South | 0              | 1              | 0             |
| West  | 0              | 0              | 1             |

The reference level ("East" here, as the alphabetically first) is represented by
all zeros. Using *d − 1* rather than *d* columns avoids perfect collinearity
while retaining full information about group membership.

AddiVortes applies this encoding automatically to pandas columns with `object`,
`string`, or `Categorical` dtype. You do not need to pre-process your data.

## 2. The `cat_scaling` parameter

After one-hot encoding, each indicator column takes values 0 or `cat_scaling`,
while continuous covariates are normalised to the range [−0.5, 0.5]. With
`cat_scaling=1.0` (the default), the binary jump from 0 to 1 has a magnitude
comparable to the full range of a normalised continuous covariate, giving
categorical and continuous covariates roughly equal influence on tessellation
distances.

You can adjust this with the `cat_scaling` argument:

- **`cat_scaling > 1`**: gives categorical differences *more* weight than
  continuous differences.
- **`cat_scaling < 1`**: gives categorical differences *less* weight, so the
  model relies more heavily on continuous covariates.

Each binary indicator column is named `<original_column>_<level>`. For example,
a column `region` with levels `"East"`, `"North"`, `"South"`, `"West"` produces
`region_North`, `region_South`, and `region_West` (with `"East"` as reference).

## 3. A synthetic example

We create 400 observations with two continuous covariates and two categorical
covariates. The response depends on all four.

```python
import numpy as np
import pandas as pd
from addivortes import AddiVortesRegressor

rng = np.random.default_rng(123)
n = 400

x = pd.DataFrame(
    {
        "age": rng.normal(40, 10, size=n),
        "income": rng.uniform(20, 120, size=n),
        "region": rng.choice(["East", "North", "South", "West"], size=n),
        "product": rng.choice(["Basic", "Premium", "Deluxe"], size=n),
    }
)

region_effect = np.select(
    [x["region"] == "North", x["region"] == "South"],
    [5.0, -5.0],
    default=0.0,
)
product_effect = np.select(
    [x["product"] == "Premium", x["product"] == "Deluxe"],
    [10.0, 20.0],
    default=0.0,
)

y = (
    0.3 * x["age"]
    + 0.1 * x["income"]
    + region_effect
    + product_effect
    + rng.normal(scale=3, size=n)
)
```

`region` has four levels and `product` has three. When passed as string columns,
they are encoded as three and two binary columns respectively, alongside the two
continuous covariates.

## 4. Inspecting the encoding

Use `prepare_design` to see the encoded matrix before fitting:

```python
from addivortes.preprocessing import prepare_design

print(x.head())

design = prepare_design(x, metric="euclidean", cat_scaling=1.0)
encoded = pd.DataFrame(design.values, columns=design.columns)
print(encoded.head())
```

The columns produced are:

- `age` and `income` (unchanged continuous columns)
- `region_North`, `region_South`, `region_West` (three indicators; `"East"` is
  the reference)
- `product_Deluxe`, `product_Premium` (two indicators; `"Basic"` is the
  reference)

Binary columns take values 0 or `cat_scaling` (here 1.0). With
`cat_scaling=1.0`, indicator columns and scaled continuous columns span a
comparable range inside the model.

## 5. Fitting the model

Pass the data frame with string or categorical columns directly.
`AddiVortesRegressor` handles encoding internally.

```python
train_rng = np.random.default_rng(42)
train_idx = train_rng.choice(n, size=300, replace=False)
test_mask = np.ones(n, dtype=bool)
test_mask[train_idx] = False

x_train = x.iloc[train_idx]
y_train = y[train_idx]
x_test = x.iloc[test_mask]
y_test = y[test_mask]

model = AddiVortesRegressor(
    n_tessellations=50,
    total_mcmc_iter=500,
    burn_in=100,
    cat_scaling=1.0,
    random_state=42,
    verbose=False,
)
model.fit(x_train, y_train)

print("In-sample RMSE:", round(model.in_sample_rmse_, 3))

print("\nReference levels used:")
enc = model.cat_encoding_
for col_idx in enc.cat_col_indices:
    col_name = enc.original_columns[col_idx]
    levels = enc.levels[col_idx]
    print(
        f"  {col_name}: reference = {levels[0]}",
        f"| all levels: {', '.join(levels)}",
    )
```

Encoding metadata is stored in `model.cat_encoding_` and applied automatically
when predicting, so new data uses the same reference levels as the training set.

## 6. Making predictions

```python
preds = model.predict(x_test)

test_rmse = float(np.sqrt(np.mean((np.asarray(y_test) - preds) ** 2)))
print("Test RMSE:", round(test_rmse, 3))
```

Plot predicted versus observed values, coloured by product category:

```python
import matplotlib.pyplot as plt

prod_cols = {
    "Basic": "steelblue",
    "Premium": "darkorange",
    "Deluxe": "darkgreen",
}
point_cols = x_test["product"].map(prod_cols)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, preds, c=point_cols, s=40)
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="grey",
    linestyle="--",
    linewidth=2,
)
ax.set(
    xlabel="Observed values",
    ylabel="Predicted values",
    title="Predicted vs observed (coloured by product category)",
)
ax.legend(
    [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8)
     for color in prod_cols.values()],
    prod_cols.keys(),
    title="Product",
    loc="upper left",
    frameon=False,
)
plt.tight_layout()
plt.show()
```

## 7. Handling unseen category levels

At prediction time, a category level not seen during training is treated as the
**reference level** (all binary indicators set to zero). The model cannot infer
anything about an unseen level and falls back to the baseline.

```python
x_new = pd.DataFrame(
    {
        "age": [45.0],
        "income": [80.0],
        "region": ["North"],
        "product": ["Luxury"],  # unseen level
    }
)

pred_new = model.predict(x_new)
print(
    "Prediction for unseen category 'Luxury' (treated as 'Basic'):",
    round(float(pred_new[0]), 3),
)
```

## 8. Effect of `cat_scaling`

`cat_scaling` controls how much influence categorical differences have in
distance calculations. Compare models with `cat_scaling=1.0` and
`cat_scaling=2.0`:

```python
model_cs2 = AddiVortesRegressor(
    n_tessellations=50,
    total_mcmc_iter=500,
    burn_in=100,
    cat_scaling=2.0,
    random_state=42,
    verbose=False,
)
model_cs2.fit(x_train, y_train)

preds_cs2 = model_cs2.predict(x_test)
rmse_cs2 = float(np.sqrt(np.mean((np.asarray(y_test) - preds_cs2) ** 2)))

print("Test RMSE (cat_scaling = 1.0):", round(test_rmse, 3))
print("Test RMSE (cat_scaling = 2.0):", round(rmse_cs2, 3))
```

In this example the true response has substantial category effects (up to ±20
units for product type) relative to the continuous effects, so increasing
`cat_scaling` may help the model focus more on categorical group membership.

## 9. Native categorical covariates with Eskin distance

Set `cat_onehot=False` to skip one-hot encoding. Each categorical column is
stored as a single integer-coded covariate (codes `1, …, d`) and distances use
the Eskin measure: a mismatch on a variable with *d* levels contributes
`2 / d²`.

```python
model_eskin = AddiVortesRegressor(
    n_tessellations=50,
    total_mcmc_iter=500,
    burn_in=100,
    cat_onehot=False,
    random_state=42,
    verbose=False,
)
model_eskin.fit(x_train, y_train)
preds_eskin = model_eskin.predict(x_test)
```

With this option:

- feature names stay as the original column names (no `column_level` expansion);
- `model.metric_` marks categorical columns with code `2`;
- `model.ncats_` stores the number of levels per categorical column;
- unseen levels at prediction time still map to the reference level (code `1`).

## 10. Summary

- Pass string or categorical columns directly in a pandas DataFrame;
  AddiVortes encodes them automatically.
- **One-hot encoding** (`cat_onehot=True`, default): a categorical variable with
  *d* levels becomes *d − 1* binary indicator columns.
- **Eskin path** (`cat_onehot=False`): keep one integer-coded column per
  categorical variable and use Eskin distance.
- The **reference level** is the alphabetically first level; all its one-hot
  indicators are zero, and unseen levels map to this reference.
- Indicator columns are named `<original_column>_<level>` (for example
  `product_Premium`) when one-hot encoding is used.
- **`cat_scaling`** (default 1.0) controls the weight of one-hot categorical
  differences relative to continuous differences.
- Encoding metadata is stored in `model.cat_encoding_` and applied automatically
  in `predict()`.
