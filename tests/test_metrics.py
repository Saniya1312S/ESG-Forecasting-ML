import math

import numpy as np
import pytest

from esg_forecasting import metrics as m


def test_perfect_forecast_has_zero_error():
    a = [10.0, 20.0, 30.0]
    assert m.mse(a, a) == 0 and m.mae(a, a) == 0 and m.rmse(a, a) == 0
    assert m.bias(a, a) == 0 and m.theils_u(a, a) == 0 and m.mape(a, a) == 0


def test_known_values():
    actual, pred = [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]
    assert m.mse(actual, pred) == 1.0
    assert m.mae(actual, pred) == 1.0
    assert m.rmse(actual, pred) == 1.0
    assert m.bias(actual, pred) == 1.0  # over-prediction is positive
    assert math.isclose(m.mape(actual, pred), np.mean([100, 50, 100 / 3]))
    expected_u = np.sqrt(3) / (np.sqrt(29) + np.sqrt(14))
    assert math.isclose(m.theils_u(actual, pred), expected_u)


def test_mape_ignores_zero_actuals():
    assert math.isclose(m.mape([0.0, 10.0], [5.0, 12.0]), 20.0)
    assert math.isnan(m.mape([0.0, 0.0], [1.0, 2.0]))


def test_pearson_nan_when_constant():
    assert math.isnan(m.pearson([5, 5, 5], [1, 2, 3]))
    assert math.isclose(m.pearson([1, 2, 3], [2, 4, 6]), 1.0)


def test_mse_over_variance_blows_up_guard():
    # near-constant actuals would give an astronomically large ratio; we return NaN instead
    assert math.isnan(m.mse_over_variance([50.0, 50.0, 50.0], [51.0, 51.0, 51.0]))
    assert math.isclose(m.mse_over_variance([1, 2, 3], [1, 2, 4]), (1 / 3) / np.var([1, 2, 3]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        m.mse([1, 2, 3], [1, 2])
