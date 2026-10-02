# ESG Forecasting — Roadmap

## Phase 1 — Reproduce
- [x] Preserve original ARIMA + LSTM notebook
- [x] Preserve original GRU + LSTM notebook
- [x] Preserve original Chronos notebook
- [x] Preserve research report
- [x] Store historical benchmark table
- [ ] Add original ESG dataset with permission/source citation

## Phase 2 — Data Engineering
- [x] Scrape current S&P 500 constituents
- [x] Select configurable company universe
- [x] Download 10 years of Yahoo Finance market data
- [x] Convert daily data to monthly features
- [x] Standardize ESG CSV schema
- [x] Align ESG and market dates
- [ ] Add point-in-time S&P membership for survivorship-bias control

## Phase 3 — Forecasting
- [x] ARIMA baseline
- [x] ARIMA + LSTM residual model
- [x] GRU + LSTM implementation
- [x] Optional Chronos integration
- [x] Train-only scaling in cleaned pipeline
- [ ] Hyperparameter search
- [ ] Walk-forward validation
- [ ] Probabilistic intervals
- [ ] Multivariate exogenous forecasting

## Phase 4 — Analytics
- [x] Error metrics
- [x] Theil's U
- [x] Bias
- [x] Pearson correlation
- [x] Benchmark chart
- [ ] Sector-level error analysis
- [ ] ESG trajectory clustering
- [ ] Explainable forecast drivers

## Phase 5 — Interactive Product
- [x] Streamlit dashboard
- [x] Model comparison
- [x] ESG trajectory explorer
- [ ] Live data refresh button
- [ ] Company search
- [ ] Scenario simulator
- [ ] Portfolio constraint sandbox

## Phase 6 — Research Extension
- [ ] Financial-return + ESG joint modeling
- [ ] News/media sentiment
- [ ] Event extraction
- [ ] Domain-aware embeddings
- [ ] ESG risk simulation
- [ ] ESG-aware portfolio optimization
- [ ] Research paper / conference artifact
