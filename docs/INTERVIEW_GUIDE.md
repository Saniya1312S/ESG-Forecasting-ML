# Interview guide

## 60-second pitch

"I compared four ways of forecasting company ESG scores: ARIMA, a hybrid of ARIMA and an LSTM, a GRU+LSTM neural network, and Amazon's pretrained Chronos model. I used 495 companies and, for each one, trained on the first 75% of its history and tested on the last 25%.
The surprise was that the simplest model, ARIMA(0,1,0), which is basically 'next value equals last value', was the most accurate on every error measure. The hybrid and Chronos were worse.
Then I reviewed my own notebooks and found issues: a Chronos output-slicing mistake, a small scaling leak in the GRU notebook, an unstable WQL metric, and models scored on slightly different sets of companies. I documented them and built a clean, tested pipeline that fixes them. So the main lesson is: always beat a simple baseline first, and check your evaluation before trusting a result."

## Likely questions

**Why does the simple model win?**
Probably because ESG scores change slowly, histories are short, and neural networks need a lot of data to learn from. That is a hypothesis; I would test it by checking how much each series actually changes and by trying longer histories.

**What is ARIMA(0,1,0)?**
No autoregressive part, one round of differencing, no moving-average part. In practice it forecasts the last value as a flat line.

**What does the LSTM add in the hybrid?**
ARIMA's leftover errors might still contain a pattern. The LSTM learns that pattern from windows of past leftovers and its output is added to ARIMA's forecast. In this data it did not help.

**What is Chronos?**
A pretrained transformer model for time series from Amazon. No ESG training needed, because it forecasts from the last few values.

**Why not trust the Chronos numbers?**
I found that the notebook took the wrong slice of the model output, so those results are provisional. The fixed code is in the repo; I still need to re-run it on the real data.

**What is data leakage? Did you have any?**
Using information from the test period when preparing training. The GRU+LSTM notebook scaled the full series, so the scaler saw test values. The clean code fits scalers on training data only.

**Why is there a "median" in the clean results?**
Averages of squared errors get pulled up by a few companies with huge errors. The median shows the typical company.

**Why did you split 75:25 and not randomly?**
Time order matters; random splits would leak the future into training.

**What would you do next?**
Walk-forward validation, drift and ETS baselines, prediction intervals, error analysis by sector, and the re-run with the real dataset and fixed Chronos code.

**What does MAPE get wrong?**
It divides by the true value, so scores near zero blow it up. I read it together with MAE and RMSE.

**Is this investment advice?**
No. It forecasts an ESG score from a single provider and says nothing about financial returns.

## Quick facts to remember
- 495 companies, 75:25 time split, window 7 and 8, LSTM/GRU 50 units, 20 epochs, batch size 1
- Best (report): ARIMA, MAE 5.80, RMSE 12.21, MAPE 31.2%
- Chronos model: `amazon/chronos-t5-small`
- Original numbers are historical results; `src/` is a clean re-implementation, not a re-run
