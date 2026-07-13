import types

import numpy as np
import pytest

from addivortes import AddiVortesRegressor, traceplots, trace_diagnostics
from conftest import fast_model, regression_data


class FakeAxis(types.SimpleNamespace):
    def __init__(self):
        super().__init__(
            calls=[],
            figure=types.SimpleNamespace(tight_layout=lambda: None),
            transAxes=object(),
        )

    def plot(self, *args, **kwargs):
        self.calls.append(("plot", args, kwargs))

    def hist(self, *args, **kwargs):
        self.calls.append(("hist", args, kwargs))

    def axvline(self, *args, **kwargs):
        self.calls.append(("axvline", args, kwargs))

    def axhline(self, *args, **kwargs):
        self.calls.append(("axhline", args, kwargs))

    def fill_between(self, *args, **kwargs):
        self.calls.append(("fill_between", args, kwargs))

    def text(self, *args, **kwargs):
        self.calls.append(("text", args, kwargs))

    def set(self, **kwargs):
        self.calls.append(("set", (), kwargs))


class FakePyplot(types.ModuleType):
    def __init__(self):
        super().__init__("matplotlib.pyplot")
        self.show_called = False

    def subplots(self, rows, cols, squeeze=False):
        axes = [[FakeAxis() for _ in range(cols)] for _ in range(rows)]
        return None, axes

    def show(self):
        self.show_called = True


@pytest.fixture
def fake_pyplot(monkeypatch):
    matplotlib = types.ModuleType("matplotlib")
    pyplot = FakePyplot()
    monkeypatch.setitem(__import__("sys").modules, "matplotlib", matplotlib)
    monkeypatch.setitem(__import__("sys").modules, "matplotlib.pyplot", pyplot)
    return pyplot


def test_traceplots_draws_four_diagnostics(fake_pyplot, monkeypatch):
    x, y = regression_data(seed=3, n_obs=10, n_features=2)
    model = fast_model(total_mcmc_iter=10, burn_in=4, thinning=1)
    model.fit(x, y)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    axes = model.traceplots(show=True)

    assert len(axes) == 4
    assert fake_pyplot.show_called
    assert all(any(call[0] == "plot" for call in axis.calls) for axis in axes)
    set_calls = [call for call in axes[0].calls if call[0] == "set"]
    assert set_calls[-1][2]["title"] == "MCMC Trace: Average Centres"
    set_calls = [call for call in axes[3].calls if call[0] == "set"]
    assert set_calls[-1][2]["title"] == "MCMC Trace: Sigma"
    sigma_values = np.sqrt(model.posterior_.sigma)
    assert np.allclose(axes[3].calls[0][1][1], sigma_values)


def test_traceplots_requires_fitted_model(fake_pyplot):
    model = AddiVortesRegressor()
    with pytest.raises(RuntimeError, match="fitted"):
        traceplots(model)


def test_traceplots_prompts_when_ask_true(fake_pyplot, monkeypatch):
    x, y = regression_data(seed=5, n_obs=10, n_features=2)
    model = fast_model(total_mcmc_iter=8, burn_in=2, thinning=1)
    model.fit(x, y)

    prompts: list[str] = []

    def capture_input(prompt=""):
        prompts.append(prompt)

    monkeypatch.setattr("builtins.input", capture_input)
    model.traceplots(ask=True)

    assert len(prompts) == 4
    assert "average centres" in prompts[0]


def test_trace_diagnostics_creates_expected_plots(fake_pyplot):
    x, y = regression_data(seed=7, n_obs=10, n_features=2)
    model = fast_model(total_mcmc_iter=10, burn_in=3, thinning=1)
    model.fit(x, y)

    axes = model.trace_diagnostics(show=True, plot_types=("trace", "histogram", "autocorrelation"))

    assert len(axes) == 9
    assert fake_pyplot.show_called
    assert any(call[0] == "plot" for axis in axes for call in axis.calls)
    assert any(call[0] == "fill_between" for axis in axes for call in axis.calls)
    assert not any(call[0] == "hist" for axis in axes for call in axis.calls)
    assert any(call[0] == "set" for axis in axes for call in axis.calls)
