# Demo run on SYNTHETIC data

These files show that the benchmark pipeline runs end to end. They were produced on
`data/sample/synthetic_esg_demo.txt` (random-walk data, 8 companies, 5 training epochs).

**They say nothing about real ESG data. Do not quote these numbers as results.**

Regenerate:

```bash
python -m esg_forecasting.benchmark --data data/sample/synthetic_esg_demo.txt \
  --models naive arima arima_lstm gru_lstm --epochs 5 --max-companies 8 --out results/demo_synthetic
```
