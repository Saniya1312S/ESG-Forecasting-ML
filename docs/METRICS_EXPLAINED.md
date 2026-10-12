# Metrics explained with one tiny example

Actual scores in the test period: `[60, 62, 61]`. A model predicts `[60, 60, 60]`.

| Step | Actual | Predicted | Prediction − Actual | Absolute error |
|---|---:|---:|---:|---:|
| 1 | 60 | 60 | 0 | 0 |
| 2 | 62 | 60 | −2 | 2 |
| 3 | 61 | 60 | −1 | 1 |

| Metric | How it is computed | Value here | Plain meaning |
|---|---|---:|---|
| **MAE** | average of absolute errors | 1.00 | "On average we miss by 1 ESG point" |
| **MSE** | average of squared errors = (0+4+1)/3 | 1.67 | Big misses count much more |
| **RMSE** | √MSE | 1.29 | MSE back in score units |
| **MAPE** | average of (abs error ÷ actual) × 100 | 1.62 % | Miss as a % of the true score |
| **Bias** | average of (prediction − actual) | −1.00 | Negative = model guesses too low |
| **Theil's U** | √Σerr² ÷ (√Σpred² + √Σactual²) | 0.011 | 0 = perfect, 1 = worst |
| **Pearson** | correlation of predicted and actual | NaN | Undefined: predictions are a flat line |

## Things to remember

- **MAPE** breaks when real scores are near 0 (dividing by almost zero). Always read it with MAE/RMSE.
- **MSE and RMSE** can be dominated by one or two companies with huge errors. Look at the **median** too.
- **Pearson is NaN for a flat forecast** (e.g. ARIMA(0,1,0)), because correlation needs the predictions to vary.
- **"WQL" in the original notebooks** is `MSE ÷ variance of the actual values`. It is not the standard weighted quantile loss, and it explodes when the actual values barely change. The clean code calls it `MSE_over_variance` and returns NaN when the variance is below 1e-6.
- **Theil's U here is U1** (bounded 0 to 1). Another version, U2, compares the model with the naive forecast; it is not what the notebooks compute.
- **Error rate vs MAPE:** same formula for non-negative scores, so the tables show identical numbers.
