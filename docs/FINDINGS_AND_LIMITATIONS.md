# Findings and limitations

A review of the original notebooks and report. Cell numbers refer to the notebooks in `notebooks/` (code cells counted from 0).
Nothing in the original files was changed.

## A. Things that affect how far the numbers can be trusted

### 1. Chronos forecast is sliced along the wrong axis (notebook 03)
- Notebook 03 installs `chronos-forecasting` 1.5.2 (commit `6a9c8da`, cell 1 install log). In that version `ChronosPipeline.predict()` returns an array shaped **(batch, num_samples, prediction_length)** (its docstring).
- Cell 3 does `forecast[0].mean(dim=-1)` (averages over the horizon, leaving one number per sample).
- Cells 5, 7 and 8 do `forecast[0, :, 0]` (the **first future step** across all samples).
- Both give a vector of length *num_samples*, which is then cut to the test length and compared with the test values. That is a set of random draws for the next step, not a forecast over the test period.
- **Fix** (used in `src/esg_forecasting/models.py`): `np.median(samples[0], axis=0)`, which gives one path of length *horizon*.
- **Status:** the fixed Chronos code could not be run here (needs PyTorch and a model download), so it is untested on real data. Treat all Chronos numbers as provisional.

### 2. Data leakage in the GRU+LSTM notebook (notebook 02)
`MinMaxScaler` is fitted on the entire series (train + test) before splitting ("Scale the entire series" comment in cell 1). The scaler therefore sees the test range. Effect is probably small, but it is avoidable. The hybrid notebook fits its scaler on training residuals only.

### 3. "WQL" values around 10²⁶–10²⁷
- Notebook 02: `wql = mse / variance if variance != 0 else nan`. If a test series is nearly (but not exactly) constant, the variance is tiny and the ratio explodes. Its printed average WQL is 7.65e+26.
- The Chronos notebooks guard with `variance < 1e-6 → NaN`.
- The hybrid notebook supplied here does not compute WQL, RMSE or error rate at all, so the 1.28e+27 and 1.48e+27 in the report come from a version that was not supplied. Same cause is likely but unverified.
- WQL here is not the standard *weighted quantile loss* either.

### 4. Report vs notebook outputs do not match for Chronos

| Source | MSE | MAE | MAPE % |
|---|---:|---:|---:|
| Report, window 8 | 294.96 | 9.11 | 48.35 |
| Notebook 03 cell 5 | 734.46 | 9.34 | 48.95 |
| Notebook 03 cell 7 | 739.69 | 9.61 | 51.59 |
| Notebook 03 cell 8 (seed 42) | 699.71 | 9.52 | 49.92 |

MAE and MAPE are close, MSE is not. MSE is driven by a few huge errors (for example one company shows MSE ≈ 32,273 in cell 7). Chronos sampling is random and only cell 8 sets a seed. Use median error and fixed seeds.

### 5. The models are not scored on the same companies
Hybrid and GRU+LSTM skip series with fewer than 20 values; Chronos skips fewer than 16 (cell 5: 411 processed, 84 skipped). The progress bar "495/495" counts the loop, not the companies scored. The clean benchmark applies one rule to every model.

### 6. The ESG dataset and the ARIMA notebook are missing
`PaperReady_numeric_esg.txt` was not available, and there is no ARIMA notebook; the ARIMA numbers exist only in the report.

## B. Modelling notes

- **ARIMA(0,1,0) is the naive "last value" forecast.** In statsmodels, an integrated model (d=1) has no trend term by default, so the forecast is flat (tested in `tests/test_models.py`). The report calls it a "random walk with drift"; without a drift term it is a random walk without drift.
- **Hybrid design:** for ARIMA(0,1,0) the residuals are one-step *changes*. The notebook adds the LSTM's predicted changes to the flat forecast step by step, rather than accumulating them. `cumulative_residuals=True` in `src/` offers the accumulated version. Neither has been run on real data.
- **Frequency unknown:** the notebooks do not say whether the ESG series are yearly, quarterly or monthly, or how many points a typical company has.
- **Small test windows:** with a short history, the test part has very few points, so scores per company are noisy.
- **Single split:** one 75:25 split gives one estimate per company. Walk-forward validation would be more reliable.

## C. How the clean code addresses these

| Issue | In `src/` |
|---|---|
| 1 | Median over samples axis, assertion on output shape |
| 2 | Scalers fitted on training data only |
| 3 | `MSE_over_variance` returns NaN under variance 1e-6 |
| 4 | Mean **and** median reported, seeds fixed |
| 5 | One `--min-obs` rule for all models; failures recorded with the error message |
