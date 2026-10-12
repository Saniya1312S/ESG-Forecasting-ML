"""The forecasting models compared in this project.

Every model has the same simple interface::

    forecast = model(train, horizon, **options)

``train`` is the company's ESG history (1-D array, oldest first), ``horizon`` is
how many future steps to predict, and the result is a 1-D array of ``horizon`` numbers.

Differences from the original notebooks (on purpose, see docs/METHODOLOGY.md):
  * scalers are fitted on TRAINING data only (the original GRU+LSTM notebook scaled
    the whole series, which leaks test information);
  * random seeds are fixed;
  * Chronos forecasts use the median over samples along the HORIZON axis.
"""
from __future__ import annotations

import functools
import warnings
from typing import Callable, Dict

import numpy as np
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------------------------------- helpers
def make_windows(arr: np.ndarray, look_back: int):
    """Turn a series into (inputs, targets) for a sequence model.

    Example with look_back=3 on [a, b, c, d, e]:
        inputs [a,b,c] -> target d      inputs [b,c,d] -> target e
    """
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if len(arr) <= look_back:
        raise ValueError(f"Need more than look_back={look_back} points, got {len(arr)}")
    X = np.stack([arr[i : i + look_back] for i in range(len(arr) - look_back)])
    y = arr[look_back:]
    return X.reshape(len(X), look_back, 1), y


def _keras():
    try:
        import keras
    except ImportError as exc:  # pragma: no cover
        raise ImportError("This model needs TensorFlow/Keras: pip install tensorflow") from exc
    return keras


def _build_net(kind: str, look_back: int, units: int):
    keras = _keras()
    keras.backend.clear_session()
    layers = keras.layers
    net = keras.Sequential([layers.Input(shape=(look_back, 1))])
    if kind == "lstm":
        net.add(layers.LSTM(units))
    elif kind == "gru_lstm":
        net.add(layers.GRU(units, return_sequences=True))
        net.add(layers.LSTM(units))
    else:
        raise ValueError(kind)
    net.add(layers.Dense(1))
    net.compile(loss="mean_squared_error", optimizer="adam")
    return net


def _fit_and_roll(kind, scaled, horizon, look_back, units, epochs, batch_size, seed):
    """Train on ``scaled`` (1-D, already scaled) and predict ``horizon`` steps recursively:
    each prediction is fed back in as the newest input for the next step."""
    keras = _keras()
    keras.utils.set_random_seed(seed)
    X, y = make_windows(scaled, look_back)
    net = _build_net(kind, look_back, units)
    net.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    history = list(scaled[-look_back:])
    out = []
    for _ in range(horizon):
        x = np.asarray(history[-look_back:], dtype="float32").reshape(1, look_back, 1)
        nxt = float(net.predict(x, verbose=0)[0, 0])
        history.append(nxt)
        out.append(nxt)
    return np.asarray(out)


# --------------------------------------------------------------------------- models
def forecast_naive(train, horizon: int) -> np.ndarray:
    """'Tomorrow looks like today': repeat the last observed value."""
    train = np.asarray(train, dtype=float)
    return np.full(horizon, train[-1])


def forecast_arima(train, horizon: int, order=(0, 1, 0)) -> np.ndarray:
    """Statistical baseline. With order (0,1,0) and no trend term, statsmodels'
    forecast is a flat line at the last training value (identical to ``forecast_naive``)."""
    from statsmodels.tsa.arima.model import ARIMA

    train = np.asarray(train, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(train, order=order).fit()
        return np.asarray(fit.forecast(steps=horizon), dtype=float)


def forecast_arima_lstm(
    train,
    horizon: int,
    look_back: int = 8,
    epochs: int = 20,
    units: int = 50,
    batch_size: int = 1,
    seed: int = 42,
    order=(0, 1, 0),
    cumulative_residuals: bool = False,
) -> np.ndarray:
    """Hybrid: ARIMA forecast + an LSTM that predicts what ARIMA gets wrong.

    1. Fit ARIMA on the training data and keep its residuals (the leftovers).
    2. Scale the residuals (training data only) and train an LSTM on sliding windows.
    3. Roll the LSTM forward ``horizon`` steps and add its output to the ARIMA forecast.

    ``cumulative_residuals=False`` reproduces the original notebook (the LSTM output is
    added step by step). With order (0,1,0) the residuals are one-step CHANGES, so
    ``True`` adds their running total instead, which treats them as changes.
    """
    from statsmodels.tsa.arima.model import ARIMA

    train = np.asarray(train, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(train, order=order).fit()
    base = np.asarray(fit.forecast(steps=horizon), dtype=float)
    resid = np.asarray(fit.resid, dtype=float)[1:]  # first residual is a start-up artefact

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(resid.reshape(-1, 1)).reshape(-1)
    pred_scaled = _fit_and_roll("lstm", scaled, horizon, look_back, units, epochs, batch_size, seed)
    pred_resid = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    if cumulative_residuals:
        pred_resid = np.cumsum(pred_resid)
    return base + pred_resid


def forecast_gru_lstm(
    train,
    horizon: int,
    look_back: int = 8,
    epochs: int = 20,
    units: int = 50,
    batch_size: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """Pure deep-learning model: GRU(50) -> LSTM(50) -> Dense(1), trained on the ESG series.

    The scaler is fitted on the TRAINING part only (no peeking at the test period).
    """
    train = np.asarray(train, dtype=float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train.reshape(-1, 1)).reshape(-1)
    pred_scaled = _fit_and_roll("gru_lstm", scaled, horizon, look_back, units, epochs, batch_size, seed)
    return scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)


@functools.lru_cache(maxsize=2)
def _chronos_pipeline(model_name: str):
    try:
        from chronos import ChronosPipeline
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Chronos needs: pip install -r requirements-chronos.txt") from exc
    return ChronosPipeline.from_pretrained(model_name, device_map="auto")


def forecast_chronos(
    train,
    horizon: int,
    window: int = 8,
    model_name: str = "amazon/chronos-t5-small",
    num_samples: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Pretrained foundation model (Amazon Chronos). No training on ESG data: the last
    ``window`` points are given as context and the model forecasts ``horizon`` steps.

    ``predict`` returns an array shaped (series, samples, horizon). We take the MEDIAN
    over the samples axis (axis 0 after dropping the series axis) to get one path of
    length ``horizon``.  NOTE: not executed in this repo's CI (needs torch + a model download).
    """
    import torch

    pipeline = _chronos_pipeline(model_name)
    torch.manual_seed(seed)
    context = torch.tensor(np.asarray(train, dtype=float)[-window:], dtype=torch.float32)
    samples = pipeline.predict(context, prediction_length=horizon, num_samples=num_samples)
    path = np.median(samples[0].detach().cpu().numpy(), axis=0)
    assert path.shape == (horizon,), path.shape
    return path


MODELS: Dict[str, Callable[..., np.ndarray]] = {
    "naive": forecast_naive,
    "arima": forecast_arima,
    "arima_lstm": forecast_arima_lstm,
    "gru_lstm": forecast_gru_lstm,
    "chronos": forecast_chronos,
}
