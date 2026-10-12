"""Forecast error metrics used in this project.

Every function takes two 1-D arrays, ``actual`` and ``pred`` (same length),
and returns one number for ONE company. Averaging across companies is done
in ``benchmark.py``.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def _as_arrays(actual, pred):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"actual and pred must have the same shape, got {a.shape} and {p.shape}")
    return a, p


def mse(actual, pred) -> float:
    """Mean squared error. Big mistakes are punished heavily."""
    a, p = _as_arrays(actual, pred)
    return float(np.mean((a - p) ** 2))


def mae(actual, pred) -> float:
    """Mean absolute error: the typical size of a miss, in ESG-score points."""
    a, p = _as_arrays(actual, pred)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, pred) -> float:
    """Square root of MSE: like MAE, but punishes big misses more."""
    return float(np.sqrt(mse(actual, pred)))


def mape(actual, pred) -> float:
    """Mean absolute percentage error (%). Points where actual == 0 are ignored."""
    a, p = _as_arrays(actual, pred)
    ok = a != 0
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs((a[ok] - p[ok]) / np.abs(a[ok]))) * 100)


def bias(actual, pred) -> float:
    """Average (prediction - actual). Positive = the model over-predicts."""
    a, p = _as_arrays(actual, pred)
    return float(np.mean(p - a))


def theils_u(actual, pred) -> float:
    """Theil's U1 as used in the original notebooks: ||pred-actual|| / (||pred|| + ||actual||).

    0 = perfect, 1 = worst possible. (This is U1, not the "vs naive forecast" U2.)
    """
    a, p = _as_arrays(actual, pred)
    denom = np.sqrt(np.sum(p**2)) + np.sqrt(np.sum(a**2))
    if denom == 0:
        return float("nan")
    return float(np.sqrt(np.sum((p - a) ** 2)) / denom)


def pearson(actual, pred) -> float:
    """Correlation between actual and predicted. NaN if either one is constant."""
    a, p = _as_arrays(actual, pred)
    if len(a) < 2 or np.ptp(a) < 1e-12 or np.ptp(p) < 1e-12:
        return float("nan")
    return float(pearsonr(a, p)[0])


def mse_over_variance(actual, pred, var_threshold: float = 1e-6) -> float:
    """MSE divided by the variance of the actual values.

    The original notebooks call this ``WQL``. It is NOT the standard weighted
    quantile loss. It explodes when the actual values are (almost) constant,
    so we return NaN when the variance is below ``var_threshold``.
    """
    a, p = _as_arrays(actual, pred)
    var = float(np.var(a))
    if var < var_threshold:
        return float("nan")
    return mse(a, p) / var


def all_metrics(actual, pred) -> dict:
    """All metrics for one company, as a dict."""
    return {
        "MSE": mse(actual, pred),
        "MAE": mae(actual, pred),
        "RMSE": rmse(actual, pred),
        "MAPE": mape(actual, pred),
        "MSE_over_variance": mse_over_variance(actual, pred),
        "Theils_U": theils_u(actual, pred),
        "Pearson": pearson(actual, pred),
        "Bias": bias(actual, pred),
    }
