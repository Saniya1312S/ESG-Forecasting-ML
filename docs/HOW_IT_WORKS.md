# How it works (simple version)

## 1. The data
Each company has a list of ESG scores in time order, for example `[61.2, 62.0, 62.4, ...]`.
We forecast the **future part** of that list.

## 2. Split in time (never shuffle!)
For every company: first **75%** of the scores = *training*, last **25%** = *test*.
Time series must be split by time. Shuffling would let the model "see the future".

```
|-------- train (75%) --------|--- test (25%) ---|
 model learns/fits here         hidden, used for scoring
```

## 3. The models

**Naive / ARIMA(0,1,0)**: predicts the last training value for every future step.
(We checked: with statsmodels' default settings, `ARIMA(0,1,0)` gives exactly this flat line,
and a unit test enforces it.)

**ARIMA + LSTM hybrid**
1. Fit ARIMA, collect its leftover errors (*residuals*).
2. Scale them to 0-1 (using training data only).
3. Cut them into windows of 8 (or 7) values: "these 8 leftovers → the next leftover".
4. Train an LSTM (50 units, 20 epochs) on those windows.
5. Predict future leftovers one step at a time, feeding each prediction back in.
6. Final forecast = ARIMA forecast + predicted leftovers.

**GRU + LSTM**: scale the ESG scores to 0-1, build windows of 8, train `GRU(50) → LSTM(50) → Dense(1)`, predict step by step, scale back.

**Chronos**: a pretrained transformer (`amazon/chronos-t5-small`). Give it the last 8 scores; it returns many possible futures (samples). We take the **median** at each future step.

## 4. Recursive forecasting
The neural models predict one step, then use that prediction as input to predict the next step, and so on.
Small errors can pile up, which is one reason long forecasts get harder.

## 5. Scoring
Compare predictions with the hidden 25% using MAE, RMSE, MAPE, Theil's U, bias, Pearson (see METRICS_EXPLAINED.md).
Do this per company, then average. The clean benchmark also reports the **median**, which ignores a few extreme companies.

## Original notebooks vs clean `src/` code

| Topic | Original notebooks | `src/esg_forecasting` |
|---|---|---|
| Scaling (GRU+LSTM) | fitted on train **and** test | fitted on **train only** |
| Companies scored | each notebook has its own skip rule | **one** rule (`--min-obs`, default 20) for every model |
| Chronos output | `forecast[0, :, 0]` / `.mean(dim=-1)` | median over samples → one path of length *horizon* |
| Random seeds | mostly not fixed | fixed (`seed=42`) |
| Failures | skipped or printed | recorded in the results file with the error message |
| Hybrid residuals | added step by step | same by default; `cumulative_residuals=True` adds their running total |
| Window builder | `range(len - look_back - 1)` drops the last window | uses every window |
| Tests | none | 24 pytest tests |
