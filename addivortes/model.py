from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any
import warnings

import numpy as np
from scipy import optimize, stats

from . import _core
from .preprocessing import (
    CategoryEncoding,
    apply_scaling,
    normalize_members,
    normalize_metric,
    prepare_design,
    reduced_metric_and_members,
    scale_matrix,
    scale_vector,
)


@dataclass(frozen=True)
class TraceStats:
    iteration: np.ndarray
    is_burn_in: np.ndarray
    average_centres_per_tessellation: np.ndarray
    sd_centres_per_tessellation: np.ndarray
    average_dimensions_per_tessellation: np.ndarray
    log_likelihood: np.ndarray


@dataclass(frozen=True)
class PosteriorSamples:
    tessellations: list[list[np.ndarray]]
    dimensions: list[list[np.ndarray]]
    predictions: list[list[np.ndarray]]
    sigma: np.ndarray
    prediction_matrix: np.ndarray
    trace_stats: TraceStats | None = None


class AddiVortesRegressor:
    """Bayesian Additive Voronoi Tessellation regressor.

    The estimator follows the usual Python machine-learning pattern:
    instantiate with model hyperparameters, call :meth:`fit`, then call
    :meth:`predict`.
    """

    def __init__(
        self,
        *,
        n_tessellations: int = 200,
        total_mcmc_iter: int = 1200,
        burn_in: int = 200,
        thinning: int = 1,
        nu: float = 6.0,
        q: float = 0.85,
        k: int = 3,
        omega: float | None = None,
        lambda_rate: float = 25.0,
        initial_sigma: str = "linear",
        metric: str | list[str] = "euclidean",
        members: list[int] | np.ndarray | None = None,
        cat_scaling: float = 1.0,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.n_tessellations = n_tessellations
        self.total_mcmc_iter = total_mcmc_iter
        self.burn_in = burn_in
        self.thinning = thinning
        self.nu = nu
        self.q = q
        self.k = k
        self.omega = omega
        self.lambda_rate = lambda_rate
        self.initial_sigma = initial_sigma
        self.metric = metric
        self.members = members
        self.cat_scaling = cat_scaling
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y) -> "AddiVortesRegressor":
        """Fit the AddiVortes regression model."""

        self._validate_hyperparameters()

        raw_columns = np.asarray(X).shape[1] if np.asarray(X).ndim == 2 else 1
        metric_labels = normalize_metric(self.metric, raw_columns)
        original_members = normalize_members(self.members, metric_labels)
        design = prepare_design(
            X,
            metric=metric_labels,
            members=original_members,
            cat_scaling=float(self.cat_scaling),
        )
        y_scaled, y_centre, y_range = scale_vector(y)

        if design.values.shape[0] != y_scaled.size:
            raise ValueError("X and y must contain the same number of observations.")

        x_scaled, x_centres, x_ranges = scale_matrix(design.values)
        x_scaled = self._restore_unscaled_columns(x_scaled, design.values, design.metric, design.encoding)

        n_obs, n_features = x_scaled.shape
        if n_features > n_obs:
            warnings.warn(
                "Number of covariates exceeds number of observations; model results may not be stable.",
                RuntimeWarning,
                stacklevel=2,
            )
        omega = min(3.0, float(raw_columns)) if self.omega is None else float(self.omega)
        sigma_squared_mu = (0.5 / (float(self.k) * np.sqrt(float(self.n_tessellations)))) ** 2
        sigma_squared_hat = self._initial_sigma_squared(x_scaled, y_scaled)
        lambda_value = self._estimate_lambda(sigma_squared_hat)

        proposal_sd = self._proposal_sd(x_ranges, design.metric)
        proposal_mu = np.zeros(n_features, dtype=float)
        proposal_mu[design.metric != 0] = x_centres[design.metric != 0]

        rng = np.random.default_rng(self.random_state)
        init_tess, init_dim, init_pred = self._initial_state(
            rng,
            n_features=n_features,
            y_scaled=y_scaled,
            proposal_sd=proposal_sd,
            metric=design.metric,
            members=design.members,
            encoding=design.encoding,
        )

        binary_cols = (
            np.asarray(design.encoding.encoded_binary_cols, dtype=np.int32)
            if design.encoding is not None
            else np.empty(0, dtype=np.int32)
        )
        seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))

        result = _core.run_mcmc(
            x_scaled,
            y_scaled,
            design.metric,
            design.members,
            int(self.n_tessellations),
            int(self.total_mcmc_iter),
            int(self.burn_in),
            int(self.thinning),
            float(self.nu),
            float(lambda_value),
            float(sigma_squared_mu),
            float(omega),
            float(self.lambda_rate),
            proposal_sd,
            proposal_mu,
            init_tess,
            init_dim,
            init_pred,
            binary_cols,
            float(self.cat_scaling),
            seed,
            bool(self.verbose),
        )

        posterior = PosteriorSamples(
            tessellations=result["posterior_tess"],
            dimensions=result["posterior_dim"],
            predictions=result["posterior_pred"],
            sigma=np.asarray(result["posterior_sigma"], dtype=float),
            prediction_matrix=np.asarray(result["prediction_matrix"], dtype=float),
            trace_stats=_trace_stats_from_result(result["trace_stats"]),
        )

        if posterior.prediction_matrix.shape[1] > 0:
            mean_yhat = posterior.prediction_matrix.mean(axis=1) * y_range + y_centre
            y_arr = np.asarray(y, dtype=float)
            in_sample_rmse = float(np.sqrt(np.mean((y_arr - mean_yhat) ** 2)))
        else:
            in_sample_rmse = float("nan")

        self.posterior_ = posterior
        self.trace_stats_ = posterior.trace_stats
        self.x_centres_ = x_centres
        self.x_ranges_ = x_ranges
        self.y_centre_ = y_centre
        self.y_range_ = y_range
        self.in_sample_rmse_ = in_sample_rmse
        self.metric_ = design.metric
        self.members_ = design.members
        self.metric_labels_ = metric_labels
        self.original_members_ = original_members
        self.cat_encoding_: CategoryEncoding | None = design.encoding
        self.feature_names_in_ = tuple(design.columns)
        self.n_original_features_in_ = raw_columns
        self.n_features_in_ = n_features
        self.metric_red_, self.member_red_ = reduced_metric_and_members(design.metric, design.members)
        self.lambda_ = float(lambda_value)
        self.sigma_squared_mu_ = float(sigma_squared_mu)
        return self

    def predict(
        self,
        X,
        *,
        kind: str = "response",
        quantiles: tuple[float, ...] | list[float] = (0.025, 0.975),
        interval: str = "credible",
        random_state: int | None = None,
    ) -> np.ndarray:
        """Predict responses or posterior quantiles for new observations."""

        self._check_is_fitted()
        kind = _match_choice(kind, {"response", "quantile"}, "kind")
        interval = _match_choice(interval, {"credible", "prediction"}, "interval")
        quantiles_arr = np.asarray(quantiles, dtype=float)
        if np.any((quantiles_arr < 0) | (quantiles_arr > 1)):
            raise ValueError("quantiles must be probabilities in [0, 1].")

        raw_columns = np.asarray(X).shape[1] if np.asarray(X).ndim == 2 else 1
        if raw_columns != self.n_original_features_in_:
            raise ValueError("X has a different number of columns than the fitted model.")

        design = prepare_design(
            X,
            metric=self.metric_labels_,
            members=self.original_members_,
            cat_scaling=float(self.cat_scaling),
            encoding=self.cat_encoding_,
        )
        if design.values.shape[1] != self.n_features_in_:
            raise ValueError("X has a different encoded feature count than the fitted model.")

        x_scaled = apply_scaling(design.values, self.x_centres_, self.x_ranges_)
        x_scaled = self._restore_unscaled_columns(x_scaled, design.values, self.metric_, self.cat_encoding_)

        n_obs = x_scaled.shape[0]
        n_samples = len(self.posterior_.tessellations)
        if n_samples == 0:
            raise RuntimeError("The fitted model contains no posterior samples.")

        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        prediction_matrix = np.empty((n_obs, n_samples), dtype=float)

        progress_steps = sum(len(tess_sample) for tess_sample in self.posterior_.tessellations)
        with _ProgressBar("Predict", progress_steps, bool(self.verbose)) as progress:
            for sample_idx in range(n_samples):
                sample_pred = np.zeros(n_obs, dtype=float)
                tess_sample = self.posterior_.tessellations[sample_idx]
                dim_sample = self.posterior_.dimensions[sample_idx]
                pred_sample = self.posterior_.predictions[sample_idx]

                for tess, dims, cell_pred in zip(tess_sample, dim_sample, pred_sample, strict=True):
                    indices = _core.cell_indices(
                        x_scaled,
                        np.asarray(tess, dtype=float),
                        np.asarray(dims, dtype=np.int32),
                        self.metric_red_,
                        self.member_red_,
                    )
                    sample_pred += np.asarray(cell_pred, dtype=float)[indices]
                    progress.advance()

                if kind == "quantile" and interval == "prediction":
                    sample_pred += rng.normal(
                        loc=0.0,
                        scale=float(np.sqrt(self.posterior_.sigma[sample_idx])),
                        size=n_obs,
                    )

                prediction_matrix[:, sample_idx] = sample_pred

        if kind == "response":
            return prediction_matrix.mean(axis=1) * self.y_range_ + self.y_centre_

        return np.quantile(prediction_matrix, quantiles_arr, axis=1).T * self.y_range_ + self.y_centre_

    def fit_predict(self, X, y, **predict_kwargs) -> np.ndarray:
        """Fit the model and return predictions for the training data."""

        return self.fit(X, y).predict(X, **predict_kwargs)

    def score(self, X, y) -> float:
        """Return the coefficient of determination on test data."""

        y_arr = np.asarray(y, dtype=float)
        pred = self.predict(X)
        ss_res = float(np.sum((y_arr - pred) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def plot(
        self,
        x_train,
        y_train,
        *,
        sigma_trace=None,
        which=(1, 2, 3),
        ask: bool = False,
        axes=None,
        show: bool = False,
        **kwargs,
    ):
        """Create diagnostic plots for a fitted AddiVortes model.

        Parameters
        ----------
        x_train, y_train
            Training covariates and responses used for fitted-value diagnostics.
        sigma_trace
            Optional standard-deviation trace. When omitted, the stored
            posterior variance samples are converted to standard deviations.
        which
            Plot identifiers to draw: 1 = residuals, 2 = sigma trace,
            3 = tessellation complexity, 4 = predicted vs observed.
        ask
            If true, wait for Enter before drawing each plot when multiple
            diagnostics are requested.
        axes
            Optional matplotlib axes to draw into.
        show
            If true, call ``matplotlib.pyplot.show()`` before returning.
        **kwargs
            Additional keyword arguments passed to the primary matplotlib
            plotting calls.

        Returns
        -------
        list
            The matplotlib axes used for the requested plots.
        """

        return plot(
            self,
            x_train,
            y_train,
            sigma_trace=sigma_trace,
            which=which,
            ask=ask,
            axes=axes,
            show=show,
            **kwargs,
        )

    def traceplots(
        self,
        *,
        ask: bool = False,
        axes=None,
        show: bool = False,
        **kwargs,
    ):
        """Create MCMC trace diagnostic plots for a fitted model.

        Displays four trace plots recorded at every MCMC iteration:
        average centres per tessellation, the standard deviation of centre
        counts, average active dimensions per tessellation, and the retained-
        state log-likelihood component. This differs from :meth:`plot` with
        ``which=3``, which traces posterior thinned samples only.
        """
        return traceplots(self, ask=ask, axes=axes, show=show, **kwargs)

    def trace_diagnostics(
        self,
        *,
        plot_types: tuple[str, ...] | None = None,
        stats: tuple[str, ...] | None = None,
        lag_k: int = 50,
        ask: bool = False,
        axes=None,
        show: bool = False,
        **kwargs,
    ):
        """Create MCMC trace diagnostics for a fitted model.

        The diagnostics include trace plots, histograms, and autocorrelation
        summaries for selected trace statistics.
        """
        return trace_diagnostics(
            self,
            plot_types=plot_types,
            stats=stats,
            lag_k=lag_k,
            ask=ask,
            axes=axes,
            show=show,
            **kwargs,
        )

    def summary(self) -> dict[str, Any]:
        """Return a dictionary of fitted model summary information."""

        self._check_is_fitted()
        tess_sizes = [
            [np.asarray(tess).shape[0] for tess in sample]
            for sample in self.posterior_.tessellations[: min(5, len(self.posterior_.tessellations))]
        ]
        flat_sizes = [size for sample in tess_sizes for size in sample]
        return {
            "n_features": self.n_features_in_,
            "n_tessellations": self.n_tessellations,
            "posterior_samples": len(self.posterior_.tessellations),
            "in_sample_rmse": self.in_sample_rmse_,
            "lambda": self.lambda_,
            "average_cells_per_tessellation": float(np.mean(flat_sizes)) if flat_sizes else float("nan"),
        }

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "n_tessellations": self.n_tessellations,
            "total_mcmc_iter": self.total_mcmc_iter,
            "burn_in": self.burn_in,
            "thinning": self.thinning,
            "nu": self.nu,
            "q": self.q,
            "k": self.k,
            "omega": self.omega,
            "lambda_rate": self.lambda_rate,
            "initial_sigma": self.initial_sigma,
            "metric": self.metric,
            "members": self.members,
            "cat_scaling": self.cat_scaling,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "AddiVortesRegressor":
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter {key!r}.")
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        params = self.get_params()
        compact = ", ".join(f"{key}={value!r}" for key, value in params.items())
        return f"AddiVortesRegressor({compact})"

    def _validate_hyperparameters(self) -> None:
        if int(self.n_tessellations) <= 0:
            raise ValueError("n_tessellations must be positive.")
        if int(self.total_mcmc_iter) <= 0:
            raise ValueError("total_mcmc_iter must be positive.")
        if int(self.burn_in) < 0:
            raise ValueError("burn_in must be non-negative.")
        if int(self.thinning) <= 0:
            raise ValueError("thinning must be positive.")
        if int(self.burn_in) >= int(self.total_mcmc_iter):
            raise ValueError("burn_in must be smaller than total_mcmc_iter.")
        if float(self.nu) <= 0:
            raise ValueError("nu must be positive.")
        if not (0 < float(self.q) < 1):
            raise ValueError("q must be in (0, 1).")
        if int(self.k) <= 0:
            raise ValueError("k must be positive.")
        if float(self.lambda_rate) <= 0:
            raise ValueError("lambda_rate must be positive.")
        if float(self.cat_scaling) <= 0:
            raise ValueError("cat_scaling must be positive.")

    def _initial_sigma_squared(self, x_scaled: np.ndarray, y_scaled: np.ndarray) -> float:
        if str(self.initial_sigma).lower() == "naive":
            return float(np.var(y_scaled, ddof=1))

        design = np.column_stack([np.ones(x_scaled.shape[0]), x_scaled])
        coef, *_ = np.linalg.lstsq(design, y_scaled, rcond=None)
        resid = y_scaled - design @ coef
        dof = max(1, x_scaled.shape[0] - x_scaled.shape[1] - 1)
        return float(np.sum(resid**2) / dof)

    def _estimate_lambda(self, sigma_squared_hat: float) -> float:
        shape = float(self.nu) / 2.0

        def objective(lambda_value: float) -> float:
            rate = float(self.nu) * lambda_value / 2.0
            quantile = stats.invgamma.ppf(float(self.q), a=shape, scale=rate)
            return float((sigma_squared_hat - quantile) ** 2)

        result = optimize.minimize_scalar(objective, bounds=(0.001, 100.0), method="bounded")
        return float(result.x)

    @staticmethod
    def _proposal_sd(ranges: np.ndarray, metric: np.ndarray) -> np.ndarray:
        sd = (ranges / 2.0) / stats.norm.ppf(0.75)
        sd = np.asarray(sd, dtype=float)
        sd[metric == 0] = 0.8
        return sd

    def _initial_state(
        self,
        rng: np.random.Generator,
        *,
        n_features: int,
        y_scaled: np.ndarray,
        proposal_sd: np.ndarray,
        metric: np.ndarray,
        members: np.ndarray,
        encoding: CategoryEncoding | None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        init_tess: list[np.ndarray] = []
        init_dim: list[np.ndarray] = []
        init_pred: list[np.ndarray] = []
        binary_cols = set(encoding.encoded_binary_cols if encoding is not None else ())

        for _ in range(int(self.n_tessellations)):
            dim = np.asarray([int(rng.integers(0, n_features))], dtype=np.int32)
            col = dim[0]
            value = rng.normal(loc=0.0, scale=proposal_sd[col])
            if metric[col] == 1 and _is_last_member_column(col, members):
                value = _period_shift(value, np.pi)
            if col in binary_cols:
                value = rng.uniform(0.0, float(self.cat_scaling))

            init_dim.append(dim)
            init_tess.append(np.asarray([[value]], dtype=float))
            init_pred.append(np.asarray([float(np.mean(y_scaled) / float(self.n_tessellations))], dtype=float))

        return init_tess, init_dim, init_pred

    @staticmethod
    def _restore_unscaled_columns(
        scaled: np.ndarray,
        original: np.ndarray,
        metric: np.ndarray,
        encoding: CategoryEncoding | None,
    ) -> np.ndarray:
        out = np.asarray(scaled, dtype=float).copy()
        out[:, metric != 0] = original[:, metric != 0]
        if encoding is not None and encoding.encoded_binary_cols:
            binary_cols = np.asarray(encoding.encoded_binary_cols, dtype=int)
            out[:, binary_cols] = original[:, binary_cols]
        return np.ascontiguousarray(out, dtype=float)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "posterior_"):
            raise RuntimeError("This AddiVortesRegressor instance is not fitted yet.")


def _period_shift(value: float, limit: float) -> float:
    while value >= limit:
        value -= 2 * limit
    while value < -limit:
        value += 2 * limit
    return value


def _is_last_member_column(index: int, members: np.ndarray) -> bool:
    return index == members.size - 1 or members[index + 1] != members[index]


def _match_choice(value: str, choices: set[str], name: str) -> str:
    normalized = str(value).lower()
    if normalized not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}.")
    return normalized


class _ProgressBar:
    def __init__(self, label: str, total: int, enabled: bool, *, width: int = 30) -> None:
        self.label = label
        self.total = int(total)
        self.enabled = bool(enabled) and self.total > 0
        self.width = int(width)
        self.current = 0
        self._last_rendered = -1
        self._update_step = max(1, self.total // 100) if self.total > 0 else 1
        self._finished = False

    def __enter__(self) -> "_ProgressBar":
        if self.enabled:
            self._render(0)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc, traceback
        if not self.enabled or self._finished:
            return
        if exc_type is None:
            self._render(self.total)
        else:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def advance(self, step: int = 1) -> None:
        if not self.enabled:
            return
        self.current = min(self.total, self.current + int(step))
        should_render = self.current == self.total or self.current - self._last_rendered >= self._update_step
        if should_render:
            self._render(self.current)

    def _render(self, current: int) -> None:
        self._last_rendered = current
        fraction = current / self.total
        filled = int(round(fraction * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        percent = int(round(fraction * 100))
        sys.stderr.write(f"\r{self.label} [{bar}] {percent:3d}% ({current}/{self.total})")
        if current >= self.total:
            sys.stderr.write("\n")
            self._finished = True
        sys.stderr.flush()


def plot(
    model: AddiVortesRegressor,
    x_train,
    y_train,
    *,
    sigma_trace=None,
    which=(1, 2, 3),
    ask: bool = False,
    axes=None,
    show: bool = False,
    **kwargs,
):
    """Create diagnostic plots for a fitted :class:`AddiVortesRegressor`.

    This is a functional wrapper around :meth:`AddiVortesRegressor.plot`.
    """

    if not isinstance(model, AddiVortesRegressor):
        raise TypeError("model must be an AddiVortesRegressor instance.")

    model._check_is_fitted()
    y_arr = _validate_plot_training_data(x_train, y_train)
    which_values = _normalize_plot_selection(which)
    if len(model.posterior_.tessellations) == 0:
        raise RuntimeError("The fitted model contains no posterior samples.")

    sigma_values = _sigma_trace(model, sigma_trace) if 2 in which_values else None
    plt = _matplotlib_pyplot()
    axes_list = _plot_axes(plt, axes, len(which_values))

    needs_predictions = any(value in {1, 4} for value in which_values)
    if needs_predictions:
        y_pred_mean = model.predict(x_train)
        residuals = y_arr - y_pred_mean
    else:
        y_pred_mean = None
        residuals = None

    if 3 in which_values:
        tess_complexity = _tessellation_complexity(model.posterior_.tessellations)
    else:
        tess_complexity = None

    for axis, plot_id in zip(axes_list, which_values, strict=True):
        if ask and len(which_values) > 1:
            input(f"Press [Enter] to see {_PLOT_NAMES[plot_id]}: ")

        if plot_id == 1:
            _plot_residuals(axis, y_pred_mean, residuals, model.in_sample_rmse_, kwargs)
        elif plot_id == 2:
            _plot_sigma_trace(axis, sigma_values, kwargs)
        elif plot_id == 3:
            _plot_tessellation_complexity(axis, tess_complexity, kwargs)
        elif plot_id == 4:
            quantiles = model.predict(x_train, kind="quantile", quantiles=(0.025, 0.975))
            _plot_predicted_observed(axis, y_arr, y_pred_mean, residuals, quantiles, kwargs)

    figure = axes_list[0].figure if axes_list else None
    if figure is not None and hasattr(figure, "tight_layout"):
        figure.tight_layout()
    if show:
        plt.show()

    return axes_list


_TRACEPLOT_SPECS = (
    ("average_centres_per_tessellation", "Average Number of Tessellation Centres", "MCMC Trace: Average Centres", "purple", 1),
    ("sd_centres_per_tessellation", "SD of Tessellation Centres", "MCMC Trace: Centre Count SD", "darkorange", 2),
    ("average_dimensions_per_tessellation", "Average Number of Dimensions", "MCMC Trace: Average Dimensions", "darkblue", 1),
    ("log_likelihood", "Log Likelihood", "MCMC Trace: Log Likelihood", "darkgreen", 4),
)

_TRACEPLOT_PROMPTS = (
    "average centres trace plot",
    "centre-count standard deviation trace plot",
    "average dimensions trace plot",
    "log-likelihood trace plot",
)


def traceplots(
    model: AddiVortesRegressor,
    *,
    ask: bool = False,
    axes=None,
    show: bool = False,
    **kwargs,
):
    """Create MCMC trace diagnostic plots for a fitted :class:`AddiVortesRegressor`."""

    if not isinstance(model, AddiVortesRegressor):
        raise TypeError("model must be an AddiVortesRegressor instance.")

    model._check_is_fitted()
    if model.trace_stats_ is None:
        raise RuntimeError(
            "The fitted model does not contain trace statistics. Refit with a current package version."
        )

    trace_stats = model.trace_stats_
    plt = _matplotlib_pyplot()
    axes_list = _plot_axes(plt, axes, 4)

    for axis, (field, ylab, main, color, legend_digits), prompt in zip(
        axes_list, _TRACEPLOT_SPECS, _TRACEPLOT_PROMPTS, strict=True
    ):
        if ask:
            input(f"Press [Enter] to see {prompt}: ")
        values = getattr(trace_stats, field)
        _plot_burn_in_trace(
            axis,
            trace_stats,
            values,
            ylab=ylab,
            main=main,
            color=color,
            legend_digits=legend_digits,
            **kwargs,
        )

    figure = axes_list[0].figure if axes_list else None
    if figure is not None and hasattr(figure, "tight_layout"):
        figure.tight_layout()
    if show:
        plt.show()

    return axes_list


_TRACE_DIAGNOSTIC_STATS = (
    ("average_centres_per_tessellation", "Average Number of Tessellation Centres"),
    ("average_dimensions_per_tessellation", "Average Number of Dimensions"),
    ("log_likelihood", "Log Likelihood"),
)


def trace_diagnostics(
    model: AddiVortesRegressor,
    *,
    plot_types: tuple[str, ...] | None = None,
    stats: tuple[str, ...] | None = None,
    lag_k: int = 50,
    ask: bool = False,
    axes=None,
    show: bool = False,
    **kwargs,
):
    """Create MCMC trace diagnostics for a fitted :class:`AddiVortesRegressor`."""

    if not isinstance(model, AddiVortesRegressor):
        raise TypeError("model must be an AddiVortesRegressor instance.")

    model._check_is_fitted()
    if model.trace_stats_ is None:
        raise RuntimeError(
            "The fitted model does not contain trace statistics. Refit with a current package version."
        )

    plot_types = tuple(plot_types) if plot_types is not None else ("trace", "histogram", "autocorrelation")
    if not plot_types:
        raise ValueError("plot_types must contain at least one diagnostic type.")

    normalized_plot_types: list[str] = []
    for plot_type in plot_types:
        plot_type = str(plot_type).lower()
        if plot_type not in {"trace", "histogram", "autocorrelation"}:
            raise ValueError(
                "plot_types may only contain 'trace', 'histogram', or 'autocorrelation'."
            )
        if plot_type not in normalized_plot_types:
            normalized_plot_types.append(plot_type)

    stats = tuple(stats) if stats is not None else tuple(item[0] for item in _TRACE_DIAGNOSTIC_STATS)
    normalized_stats: list[tuple[str, str]] = []
    valid_stats = dict(_TRACE_DIAGNOSTIC_STATS)
    for stat in stats:
        stat = str(stat)
        if stat not in valid_stats:
            raise ValueError(
                f"stats may only contain {sorted(valid_stats)}. Got {stat!r}."
            )
        if stat not in [field for field, _ in normalized_stats]:
            normalized_stats.append((stat, valid_stats[stat]))

    trace_stats = model.trace_stats_
    plt = _matplotlib_pyplot()
    sequential_show = bool(show)
    axes_list: list = []
    if not sequential_show:
        axes_list = _plot_axes(plt, axes, len(normalized_stats) * len(normalized_plot_types))

    plot_index = 0
    for stat, label in normalized_stats:
        values = getattr(trace_stats, stat)
        post_burn_in_values = _post_burn_in_values(trace_stats, values)

        # Compute and print effective sample size for this statistic
        try:
            ess = _effective_sample_size(post_burn_in_values, lag_k)
            print(f"Effective sample size ({label}): {ess:.1f}")
        except Exception:
            # If ESS computation fails for any reason, continue without
            # interrupting plotting.
            pass

        for plot_type in normalized_plot_types:
            if ask:
                input(f"Press [Enter] to see {plot_type} for {label}: ")

            if sequential_show:
                # plt.subplots may return (fig, ax) where ax can be an
                # Axes instance or an array/list. Normalize to a single
                # Axes object for consistent downstream plotting.
                sub_result = plt.subplots(1, 1)
                if isinstance(sub_result, tuple) and len(sub_result) == 2:
                    figure, raw_axis = sub_result
                else:
                    # Some backends or mocks may return only the axes
                    figure = None
                    raw_axis = sub_result

                if isinstance(raw_axis, (list, tuple, np.ndarray)):
                    raw_axis_arr = np.asarray(raw_axis, dtype=object).ravel()
                    axis = raw_axis_arr[0]
                else:
                    axis = raw_axis
            else:
                axis = axes_list[plot_index]

            if plot_type == "trace":
                _plot_burn_in_trace(
                    axis,
                    trace_stats,
                    values,
                    ylab=label,
                    main=f"MCMC Trace: {label}",
                    color="darkblue",
                    legend_digits=2,
                    **kwargs,
                )
            elif plot_type == "histogram":
                _plot_histogram(
                    axis,
                    post_burn_in_values,
                    xlab=label,
                    main=f"Density: {label}",
                    color="darkblue",
                    **kwargs,
                )
            else:
                _plot_autocorrelation(
                    axis,
                    post_burn_in_values,
                    lag_k=lag_k,
                    xlab="Lag",
                    ylab="Autocorrelation",
                    main=f"Autocorrelation (lag {lag_k}): {label}",
                    color="darkblue",
                    **kwargs,
                )

            if sequential_show:
                # If the caller requested interactive prompting via `ask`, the
                # earlier `input()` call has already waited. Otherwise prompt
                # the user before showing each plot so graphs appear one at a
                # time.
                if not ask:
                    try:
                        input(f"Press [Enter] to see {plot_type} for {label}: ")
                    except Exception:
                        pass
                if figure is not None and hasattr(figure, "tight_layout"):
                    figure.tight_layout()
                try:
                    plt.show(block=True)
                except TypeError:
                    try:
                        plt.show()
                    except Exception:
                        pass
                axes_list.append(axis)

            plot_index += 1

    figure = axes_list[0].figure if axes_list else None
    if not sequential_show:
        if figure is not None and hasattr(figure, "tight_layout"):
            figure.tight_layout()
        if show:
            plt.show()

    return axes_list


def _post_burn_in_values(trace_stats: TraceStats, values: np.ndarray) -> np.ndarray:
    burn_in_mask = np.asarray(trace_stats.is_burn_in, dtype=bool)
    post_burn_in = values[~burn_in_mask]
    return post_burn_in if post_burn_in.size > 0 else values


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    if lag == 0:
        return 1.0

    n = values.size
    if lag >= n:
        return float("nan")

    centered = values - values.mean()
    front = centered[:-lag]
    back = centered[lag:]
    denom = np.std(front, ddof=0) * np.std(back, ddof=0)
    if denom == 0.0:
        return 0.0
    return float(np.dot(front, back) / ((n - lag) * denom))


def _effective_sample_size(values: np.ndarray, max_lag: int) -> float:
    """Estimate effective sample size (ESS) for a univariate MCMC trace.

    Uses the standard estimator ESS = n / (1 + 2 * sum_{k=1..K} rho_k) where
    rho_k are autocorrelations up to the first non-positive lag or up to
    ``max_lag`` (whichever comes first).
    """
    vals = np.asarray(values, dtype=float)
    n = vals.size
    if n <= 1:
        return float(n)

    L = min(int(max_lag), n - 1)
    rhos = []
    for k in range(1, L + 1):
        rho = _autocorrelation(vals, k)
        if np.isnan(rho):
            break
        rhos.append(rho)
        if rho <= 0.0:
            break

    sum_rho = float(np.sum(rhos)) if rhos else 0.0
    denom = 1.0 + 2.0 * sum_rho
    if denom <= 0:
        return float(n)
    return float(n) / denom


def _plot_histogram(axis, values: np.ndarray, *, xlab: str, main: str, color: str, **kwargs) -> None:
    values = np.asarray(values, dtype=float)
    plot_kwargs = {"color": color, "linewidth": 1.5}
    plot_kwargs.update(kwargs)

    if values.size <= 1 or np.all(values == values[0]):
        center = float(values[0]) if values.size == 1 else 0.0
        xs = np.asarray([center - 0.5, center, center + 0.5], dtype=float)
        ys = np.asarray([0.0, 1.0, 0.0], dtype=float)
    else:
        kde = stats.gaussian_kde(values)
        xs = np.linspace(np.min(values), np.max(values), 200)
        ys = kde(xs)

    axis.plot(xs, ys, **plot_kwargs)
    if hasattr(axis, "fill_between"):
        axis.fill_between(xs, ys, color=color, alpha=0.3)
    axis.set(xlabel=xlab, ylabel="Density", title=main)
    axis.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)


def _plot_autocorrelation(
    axis,
    values: np.ndarray,
    *,
    lag_k: int,
    xlab: str,
    ylab: str,
    main: str,
    color: str,
    **kwargs,
) -> None:
    # Always show autocorrelation for lags 0..50
    lags = np.arange(0, 51)
    corr = np.asarray([_autocorrelation(values, int(lag)) for lag in lags], dtype=float)
    line_kwargs = {"color": color, "marker": "o", "linewidth": 1.5}
    line_kwargs.update(kwargs)
    axis.plot(lags, corr, **line_kwargs)
    axis.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    axis.set(xlabel=xlab, ylabel=ylab, title=main)

    # Prefer integer tick labels at 0,10,20,30,40,50. Use set_xticks/set_xticklabels
    # when available; fall back to axis.set(...) so unit tests with FakeAxis
    # continue to record the intended ticks.
    ticks = np.arange(0, 51, 10)
    labels = [str(int(t)) for t in ticks]
    try:
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels)
    except Exception:
        try:
            axis.set(xticks=list(ticks), xticklabels=labels)
        except Exception:
            pass


def _trace_stats_from_result(trace_stats: dict[str, Any]) -> TraceStats:
    return TraceStats(
        iteration=np.asarray(trace_stats["iteration"], dtype=int),
        is_burn_in=np.asarray(trace_stats["is_burn_in"], dtype=bool),
        average_centres_per_tessellation=np.asarray(
            trace_stats["average_centres_per_tessellation"], dtype=float
        ),
        sd_centres_per_tessellation=np.asarray(trace_stats["sd_centres_per_tessellation"], dtype=float),
        average_dimensions_per_tessellation=np.asarray(
            trace_stats["average_dimensions_per_tessellation"], dtype=float
        ),
        log_likelihood=np.asarray(trace_stats["log_likelihood"], dtype=float),
    )


def _plot_burn_in_trace(
    axis,
    trace_stats: TraceStats,
    values: np.ndarray,
    *,
    ylab: str,
    main: str,
    color: str,
    legend_digits: int = 2,
    **kwargs,
) -> None:
    line_kwargs = {"color": color, "linewidth": 1.5}
    line_kwargs.update(kwargs)
    iterations = np.asarray(trace_stats.iteration, dtype=int)
    axis.plot(iterations, values, **line_kwargs)

    burn_in_mask = np.asarray(trace_stats.is_burn_in, dtype=bool)
    if np.any(burn_in_mask):
        burn_in_end = int(iterations[burn_in_mask][-1])
        axis.axvline(burn_in_end, color="gray", linestyle=":", linewidth=1.5, label="Burn-in end")

    post_burn_in = values[~burn_in_mask]
    if post_burn_in.size == 0:
        post_burn_in = values

    post_mean = float(np.mean(post_burn_in))
    post_sd = float(np.std(post_burn_in, ddof=1)) if post_burn_in.size > 1 else 0.0
    axis.axhline(post_mean, color="red", linestyle="--", linewidth=1.5)
    axis.text(
        0.98,
        0.95,
        f"Mean: {post_mean:.{legend_digits}f}\nSD: {post_sd:.{legend_digits}f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    axis.set(xlabel="MCMC Iteration", ylabel=ylab, title=main)


_PLOT_NAMES = {
    1: "residuals plot",
    2: "sigma trace plot",
    3: "tessellation complexity trace",
    4: "predicted vs observed plot",
}


def _validate_plot_training_data(x_train, y_train) -> np.ndarray:
    y_arr = np.asarray(y_train, dtype=float)
    if y_arr.ndim != 1:
        raise ValueError("y_train must be a one-dimensional numeric array.")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y_train contains missing or non-finite values.")

    n_rows = _num_observations(x_train)
    if n_rows != y_arr.size:
        raise ValueError("x_train and y_train must contain the same number of observations.")
    return y_arr


def _num_observations(x_train) -> int:
    shape = getattr(x_train, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])

    arr = np.asarray(x_train)
    if arr.ndim == 0:
        raise ValueError("x_train must be a one- or two-dimensional array-like object.")
    return int(arr.shape[0])


def _normalize_plot_selection(which) -> tuple[int, ...]:
    try:
        requested = [int(value) for value in np.atleast_1d(which)]
    except (TypeError, ValueError) as exc:
        raise ValueError("which must contain integer plot identifiers between 1 and 4.") from exc

    selected: list[int] = []
    for value in requested:
        if value in {1, 2, 3, 4} and value not in selected:
            selected.append(value)

    if not selected:
        raise ValueError("which must contain at least one value between 1 and 4.")
    return tuple(selected)


def _matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised when matplotlib is absent
        raise ImportError(
            "AddiVortesRegressor.plot requires matplotlib. Install it with "
            "`pip install addivortes[plot]` or `pip install matplotlib`."
        ) from exc
    return plt


def _plot_axes(plt, axes, n_plots: int) -> list:
    if axes is not None:
        axes_list = list(np.asarray(axes, dtype=object).ravel())
        if len(axes_list) < n_plots:
            raise ValueError("axes must provide at least one matplotlib axis for each requested plot.")
        return axes_list[:n_plots]

    if n_plots == 1:
        rows, cols = 1, 1
    elif n_plots == 2:
        rows, cols = 1, 2
    else:
        cols = 2
        rows = (n_plots + cols - 1) // cols

    _, axes_array = plt.subplots(rows, cols, squeeze=False)
    axes_list = list(np.asarray(axes_array, dtype=object).ravel())
    for unused_axis in axes_list[n_plots:]:
        if hasattr(unused_axis, "set_visible"):
            unused_axis.set_visible(False)
    return axes_list[:n_plots]


def _tessellation_complexity(tessellations: list[list[np.ndarray]]) -> np.ndarray:
    return np.asarray(
        [
            float(np.mean([np.asarray(tess).shape[0] for tess in sample]))
            if len(sample) > 0
            else float("nan")
            for sample in tessellations
        ],
        dtype=float,
    )


def _sigma_trace(model: AddiVortesRegressor, sigma_trace) -> np.ndarray:
    if sigma_trace is None:
        sigma_values = np.sqrt(np.asarray(model.posterior_.sigma, dtype=float))
    else:
        sigma_values = np.asarray(sigma_trace, dtype=float)

    if sigma_values.ndim != 1 or sigma_values.size == 0:
        raise ValueError("sigma_trace must be a non-empty one-dimensional numeric array.")
    if not np.all(np.isfinite(sigma_values)):
        raise ValueError("sigma_trace contains missing or non-finite values.")
    if sigma_values.size != len(model.posterior_.tessellations):
        warnings.warn(
            "Length of sigma_trace does not match the number of posterior samples.",
            RuntimeWarning,
            stacklevel=2,
        )
    return sigma_values


def _plot_residuals(axis, y_pred_mean, residuals, rmse: float, kwargs: dict[str, Any]) -> None:
    scatter_kwargs = {"s": 30, "color": "darkblue", "alpha": 0.8}
    scatter_kwargs.update(kwargs)
    axis.scatter(y_pred_mean, residuals, **scatter_kwargs)
    axis.axhline(0.0, color="red", linestyle="--", linewidth=2)
    if y_pred_mean.size > 3:
        smooth_x, smooth_y = _smooth_line(y_pred_mean, residuals)
        axis.plot(smooth_x, smooth_y, color="orange", linewidth=2)
    axis.text(0.98, 0.95, f"RMSE = {rmse:.4f}", transform=axis.transAxes, ha="right", va="top")
    axis.set(xlabel="Fitted Values", ylabel="Residuals", title="Residuals vs Fitted")


def _plot_sigma_trace(axis, sigma_values: np.ndarray, kwargs: dict[str, Any]) -> None:
    line_kwargs = {"color": "darkgreen", "linewidth": 1.5}
    line_kwargs.update(kwargs)
    iterations = np.arange(1, sigma_values.size + 1)
    axis.plot(iterations, sigma_values, **line_kwargs)
    axis.axhline(float(np.mean(sigma_values)), color="red", linestyle="--")
    sigma_sd = float(np.std(sigma_values, ddof=1)) if sigma_values.size > 1 else 0.0
    axis.text(
        0.98,
        0.95,
        f"Mean: {np.mean(sigma_values):.4f}\nSD: {sigma_sd:.4f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    axis.set(xlabel="MCMC Iteration", ylabel="sigma", title="MCMC Trace: Error Standard Deviation")


def _plot_tessellation_complexity(axis, tess_complexity: np.ndarray, kwargs: dict[str, Any]) -> None:
    line_kwargs = {"color": "purple", "linewidth": 1.5}
    line_kwargs.update(kwargs)
    iterations = np.arange(1, tess_complexity.size + 1)
    axis.plot(iterations, tess_complexity, **line_kwargs)
    axis.axhline(float(np.mean(tess_complexity)), color="red", linestyle="--")
    complexity_range = np.nanmin(tess_complexity), np.nanmax(tess_complexity)
    axis.text(
        0.98,
        0.95,
        f"Mean: {np.mean(tess_complexity):.1f}\nRange: [{complexity_range[0]:.1f}, {complexity_range[1]:.1f}]",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    axis.set(
        xlabel="MCMC Iteration",
        ylabel="Average Number of Tessellation Centers",
        title="MCMC Trace: Tessellation Complexity",
    )


def _plot_predicted_observed(
    axis,
    y_train: np.ndarray,
    y_pred_mean: np.ndarray,
    residuals: np.ndarray,
    quantiles: np.ndarray,
    kwargs: dict[str, Any],
) -> None:
    scatter_kwargs = {"s": 30, "color": "darkblue", "alpha": 0.8}
    scatter_kwargs.update(kwargs)
    data_range = (
        float(np.min([np.min(y_train), np.min(y_pred_mean)])),
        float(np.max([np.max(y_train), np.max(y_pred_mean)])),
    )
    axis.scatter(y_train, y_pred_mean, **scatter_kwargs)
    axis.set_xlim(data_range)
    axis.set_ylim(data_range)
    axis.plot(data_range, data_range, color="red", linewidth=2, linestyle="--", label="Perfect Prediction")
    axis.vlines(
        y_train,
        quantiles[:, 0],
        quantiles[:, 1],
        color="lightblue",
        linewidth=1,
        label="95% Credible Intervals",
    )

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_train - np.mean(y_train)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    axis.text(0.98, 0.05, f"R^2 = {r_squared:.3f}", transform=axis.transAxes, ha="right", va="bottom")
    axis.legend(loc="upper left", frameon=False)
    axis.set(xlabel="Observed Values", ylabel="Predicted Values", title="Predicted vs Observed")


def _smooth_line(x_values: np.ndarray, y_values: np.ndarray, frac: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x_values)
    x_sorted = np.asarray(x_values, dtype=float)[order]
    y_sorted = np.asarray(y_values, dtype=float)[order]
    window = max(3, int(np.ceil(frac * x_sorted.size)))
    smoothed = np.empty_like(y_sorted)

    for idx, x_val in enumerate(x_sorted):
        distances = np.abs(x_sorted - x_val)
        neighbours = np.argsort(distances)[:window]
        max_distance = float(np.max(distances[neighbours]))
        if max_distance == 0:
            smoothed[idx] = float(np.mean(y_sorted[neighbours]))
            continue

        scaled = distances[neighbours] / max_distance
        weights = (1.0 - scaled**3) ** 3
        design = np.column_stack([np.ones(neighbours.size), x_sorted[neighbours] - x_val])
        weighted_design = design * weights[:, None]
        try:
            coef, *_ = np.linalg.lstsq(weighted_design, y_sorted[neighbours] * weights, rcond=None)
            smoothed[idx] = coef[0]
        except np.linalg.LinAlgError:
            smoothed[idx] = float(np.average(y_sorted[neighbours], weights=weights))

    return x_sorted, smoothed


AddiVortes = AddiVortesRegressor
