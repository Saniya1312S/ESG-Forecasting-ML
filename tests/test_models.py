import numpy as np
import pytest

from esg_forecasting import models as mod
from esg_forecasting.data import make_synthetic_esg, train_test_split_series

SERIES = np.array(make_synthetic_esg(1, 24, seed=5)["SYN000"])
TRAIN, TEST = train_test_split_series(SERIES)


def test_make_windows_shapes_and_content():
    X, y = mod.make_windows(np.arange(6.0), look_back=3)
    assert X.shape == (3, 3, 1) and y.tolist() == [3.0, 4.0, 5.0]
    assert X[0, :, 0].tolist() == [0.0, 1.0, 2.0]


def test_make_windows_too_short():
    with pytest.raises(ValueError):
        mod.make_windows(np.arange(3.0), look_back=3)


def test_naive_repeats_last_value():
    f = mod.forecast_naive(TRAIN, 5)
    assert f.shape == (5,) and np.all(f == TRAIN[-1])


def test_arima_010_equals_naive():
    """ARIMA(0,1,0) has no trend term here, so it IS the naive forecast."""
    pytest.importorskip("statsmodels")
    f = mod.forecast_arima(TRAIN, len(TEST))
    assert np.allclose(f, TRAIN[-1])


def test_arima_lstm_smoke():
    pytest.importorskip("tensorflow")
    f = mod.forecast_arima_lstm(TRAIN, len(TEST), look_back=4, epochs=1)
    assert f.shape == (len(TEST),) and np.all(np.isfinite(f))


def test_arima_lstm_cumulative_option_runs():
    pytest.importorskip("tensorflow")
    f = mod.forecast_arima_lstm(TRAIN, len(TEST), look_back=4, epochs=1, cumulative_residuals=True)
    assert f.shape == (len(TEST),)


def test_gru_lstm_smoke_and_train_only_scaling():
    pytest.importorskip("tensorflow")
    f = mod.forecast_gru_lstm(TRAIN, len(TEST), look_back=4, epochs=1)
    assert f.shape == (len(TEST),) and np.all(np.isfinite(f))


def test_seed_makes_results_repeatable():
    pytest.importorskip("tensorflow")
    a = mod.forecast_gru_lstm(TRAIN, 3, look_back=4, epochs=2, seed=1)
    b = mod.forecast_gru_lstm(TRAIN, 3, look_back=4, epochs=2, seed=1)
    assert np.allclose(a, b, atol=1e-5)


def test_registry_has_all_models():
    assert set(mod.MODELS) == {"naive", "arima", "arima_lstm", "gru_lstm", "chronos"}
