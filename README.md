# ESG Forecasting Using ML & Time-Series AI

> **Research + portfolio project:** forecasting Environmental, Social & Governance (ESG) score trajectories and benchmarking statistical, deep-learning, and foundation-model time-series approaches.

**Audience:** Data Analyst • Research Analyst • Data Scientist • ML Engineer • AI/Finance • Sustainable Finance

---

## 1. Why this project exists

ESG ratings are increasingly used to understand sustainability-related risk. This project studies whether historical ESG-score trajectories can be forecast with time-series and machine-learning methods, and how those forecasts can support downstream ESG-risk analysis.

The research workflow combines:

- **ESG score time series** at company level
- **Financial-market context** from Yahoo Finance
- **S&P 500 constituent discovery** through a reproducible web-scraping step
- **Statistical forecasting:** ARIMA
- **Hybrid forecasting:** ARIMA + LSTM residual modeling
- **Deep sequence modeling:** GRU + LSTM
- **Foundation-model forecasting:** Amazon Chronos
- **Evaluation:** MSE, MAE, RMSE, MAPE, Error Rate, WQL, Theil's U, Pearson correlation and Bias
- **Interactive research dashboard:** Streamlit

The repository deliberately separates **historical research results** from the **reproducible data pipeline**. The original research notebooks reference a local `PaperReady_numeric_esg.txt` file that was not included in the uploaded materials; therefore, this repository does not fabricate or silently recreate that ESG dataset.

---

## 2. Research contribution

The project is motivated by the broader research direction of combining ESG trajectories, financial returns and other signals for sustainability-aware investment analytics.

The current implementation focuses on **ESG-score forecasting**. A future multimodal version can add:

`ESG ratings + financial returns + news/media sentiment -> joint representation -> ESG risk forecast -> scenario analysis -> portfolio constraints`

This repository is therefore structured as both a portfolio project and a foundation for a research extension.

---

## 3. What is inside

```text
ESG-Forecasting-Research-Portfolio/
│
├── README.md
├── ROADMAP.md
├── requirements.txt
├── requirements-chronos.txt
├── .gitignore
├── LICENSE
│
├── app/
│   └── streamlit_app.py
│
├── config/
│   └── project_config.yaml
│
├── data/
│   ├── raw/
│   │   └── README.md
│   ├── processed/
│   │   └── .gitkeep
│   └── sample/
│       └── esg_sample.csv
│
├── docs/
│   ├── DATA_DICTIONARY.md
│   ├── METHODOLOGY.md
│   ├── MODEL_CARDS.md
│   └── RESEARCH_NOTES.md
│
├── models/
│   └── README.md
│
├── notebooks/
│   └── research/
│       ├── final_arima_lstm_original.ipynb
│       ├── gru_lstm_original.ipynb
│       ├── chronos_original.ipynb
│       └── Table_and_Report_original.docx
│
├── outputs/
│   ├── benchmarks/
│   │   └── research_benchmark_results.csv
│   └── figures/
│       └── model_comparison.png
│
├── scripts/
│   ├── run_research_pipeline.py
│   └── make_benchmark_figure.py
│
└── src/
    ├── data/
    │   ├── scrape_sp500.py
    │   ├── fetch_market_data.py
    │   ├── load_esg.py
    │   └── prepare_dataset.py
    ├── evaluation/
    │   └── metrics.py
    ├── models/
    │   ├── arima.py
    │   ├── hybrid_arima_lstm.py
    │   ├── gru_lstm.py
    │   └── chronos_forecaster.py
    └── utils/
        └── io.py
```

---

## 4. End-to-end workflow

```text
S&P 500 constituent table
          │
          ▼
     Select N companies
          │
          ▼
Yahoo Finance ───────────────┐
10-year market history       │
Price → monthly close        │
Returns / volatility         │
                             ▼
Historical ESG dataset ──► Data alignment
                             │
                             ▼
                     Quality checks
                             │
                             ▼
                    75% train / 25% test
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
        ARIMA        ARIMA + LSTM          GRU + LSTM
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                         Chronos
                             │
                             ▼
                    Evaluation metrics
                             │
                             ▼
                 Benchmark comparison
                             │
                             ▼
                  Interactive Streamlit app
```

---

## 5. Data strategy

### 5.1 S&P 500 universe

The project can scrape the current S&P 500 constituent table and select a configurable research universe (default: 50 companies). This is intentionally configurable because index membership changes over time.

Source used by the scraper:

- Wikipedia — List of S&P 500 companies: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

**Important:** a current constituent list is not the same as a historical point-in-time index membership file. For rigorous historical research, use a point-in-time constituent history if survivorship bias matters.

### 5.2 Market data

Yahoo Finance is used through `yfinance` for historical market data. The pipeline downloads up to 10 years of daily OHLCV data and creates monthly features.

Typical features:

- Adjusted/auto-adjusted close
- Monthly return
- Rolling volatility
- Volume
- Monthly high/low

Yahoo Finance access is implemented through `yfinance`; see the official package documentation for supported history/download parameters.

### 5.3 ESG data

The original research notebooks use:

```text
PaperReady_numeric_esg.txt
```

with a structure equivalent to:

```python
{
    "AAPL": [esg_score_t1, esg_score_t2, ...],
    "MSFT": [esg_score_t1, esg_score_t2, ...],
    ...
}
```

That source file was not among the uploaded research artifacts. The production pipeline therefore accepts a transparent CSV input instead:

```text
ticker,date,esg_score
AAPL,2016-12-31,72.1
AAPL,2017-12-31,73.4
...
```

The repository includes only a clearly labelled **small demo ESG dataset** for testing the pipeline. Replace it with the actual licensed/source ESG data before running a research-scale experiment.

---

## 6. Models

### A. ARIMA baseline

The research implementation used:

```text
ARIMA(0,1,0)
```

with a 75:25 train/test split.

This is intentionally simple and interpretable. It provides a statistical baseline against which neural and foundation-model approaches can be compared.

### B. ARIMA + LSTM hybrid

The hybrid design follows the original research notebook:

1. Fit ARIMA on the ESG training series.
2. Calculate ARIMA residuals.
3. Scale residuals.
4. Train an LSTM on sliding residual windows.
5. Forecast residuals recursively.
6. Add forecast residuals to the ARIMA forecast.
7. Evaluate against the held-out test period.

### C. GRU + LSTM

The research notebook also tested a stacked sequence model:

```text
GRU(50, return_sequences=True)
        ↓
LSTM(50)
        ↓
Dense(1)
```

The repository keeps this original notebook in `notebooks/research/` and provides a cleaned implementation for reproducible experiments.

### D. Chronos

The research notebook used:

```text
amazon/chronos-t5-small
```

with an 8-step input window in the main experiment.

Chronos is treated as an optional model because it requires additional dependencies and can be computationally expensive.

---

## 7. Research benchmark from the supplied report

The following table is transcribed from the supplied research report. It is presented as a **historical research artifact**, not as a newly recomputed result from this repository.

| Model | Split | Window | MSE | MAE | RMSE | Error Rate | MAPE | WQL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARIMA | 75:25 | — | 149.1784 | 5.7951 | 12.2139 | 31.2367 | 31.2367 | 43.2056 |
| ARIMA + LSTM | 75:25 | 8 | 165.4703 | 7.3938 | 12.8635 | 39.4442 | 39.4442 | 1.28467e+27 |
| ARIMA + LSTM | 75:25 | 7 | 168.1778 | 7.4914 | 12.9683 | 40.1409 | 40.1409 | 1.48025e+27 |
| Chronos | 75:25 | 8 | 294.9557 | 9.1147 | 17.1743 | 48.3529 | 48.3529 | 149.3103 |
| Chronos | 75:25 | 7 | 695.6044 | 9.7017 | 26.3743 | 50.8807 | 50.8807 | 495.4225 |

The supplied report describes the ARIMA experiment as fitting ARIMA(0,1,0) separately to each company's ESG series after a 75:25 split.

The hybrid experiment models ARIMA residuals with an LSTM and adds the forecast residuals back to the ARIMA forecast.

The Chronos experiment uses a sliding window and the last training window to forecast the held-out test period.

---

## 8. Additional notebook benchmark

The supplied GRU+LSTM notebook reports the following aggregate output for its 495-company run:

```text
Average MSE:          1071.3619
Average MAE:            29.8447
Average Error Rate:    162.8933%
Average MAPE:          162.8933%
Average Theil's U:       0.4058
Average Bias:           29.2801
Average RMSE:           32.7317
```

The original notebook also reports a very large average WQL value. Such values should be interpreted cautiously because WQL was implemented in the research code as MSE divided by the variance of the actual test series.

---

## 8.1 Observed outputs from the supplied notebooks

The supplied ARIMA+LSTM notebook reports an aggregate run over 495 companies with:

```text
Average MSE:        164.5120
Average MAE:          7.3564
Average MAPE:        39.6394%
Average Theil's U:    0.1733
Average Bias:         1.8833
```

The supplied GRU+LSTM notebook reports:

```text
Average MSE:        1071.3619
Average MAE:          29.8447
Average MAPE:       162.8933%
Average Theil's U:    0.4058
Average Bias:         29.2801
Average RMSE:         32.7317
```

These are preserved as notebook outputs, not newly recomputed results. The Chronos notebook contains multiple experimental variants; its outputs should be tied to the exact notebook cell/configuration when cited.

---

## 9. Important methodological note

The original GRU+LSTM notebook scales the complete series before splitting it into train and test. That can introduce **data leakage**, because the scaler has access to test-period values.

The cleaned implementation in this repository fits preprocessing transformations using **training data only**.

This distinction is intentional:

- `notebooks/research/` = original research artifacts
- `src/` = cleaned, reproducible portfolio implementation

This makes the repository more credible in interviews because it shows that the original experiment is preserved while methodological improvements are explicitly documented.

---

## 10. Metrics

### Regression

- **MSE** — mean squared error
- **MAE** — mean absolute error
- **RMSE** — square root of MSE
- **MAPE** — mean absolute percentage error
- **Error Rate** — percentage absolute error
- **WQL** — weighted/variance-normalized loss proxy used in the research implementation

### Forecast diagnostics

- **Theil's U** — normalized forecast error
- **Pearson correlation** — linear association between actual and predicted values
- **Bias** — mean(prediction − actual)

For ESG scores near zero, MAPE can become unstable. Always inspect MAE/RMSE and the raw forecast plots alongside percentage metrics.

---

## 11. Run the project

### Step 1 — clone

```bash
git clone https://github.com/<YOUR_USERNAME>/ESG-Forecasting-Research-Portfolio.git
cd ESG-Forecasting-Research-Portfolio
```

### Step 2 — create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 3 — install

```bash
pip install -r requirements.txt
```

### Step 4 — get the S&P 500 universe

```bash
python src/data/scrape_sp500.py --n 50
```

### Step 5 — download 10 years of Yahoo market data

```bash
python src/data/fetch_market_data.py --years 10 --n 50
```

### Step 6 — prepare ESG + market data

Place your licensed/source ESG CSV in:

```text
data/raw/esg_scores.csv
```

Then:

```bash
python src/data/prepare_dataset.py
```

### Step 7 — run the baseline pipeline

```bash
python scripts/run_research_pipeline.py --model arima
```

Hybrid:

```bash
python scripts/run_research_pipeline.py --model hybrid
```

GRU+LSTM:

```bash
python scripts/run_research_pipeline.py --model gru_lstm
```

Chronos:

```bash
pip install -r requirements-chronos.txt
python scripts/run_research_pipeline.py --model chronos
```

---

## 12. Launch the interactive dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard provides:

- Research benchmark comparison
- Model-level metric cards
- Metric comparison charts
- Company-level forecast visualization when processed predictions are available
- ESG trajectory explorer
- Financial-market context
- Model/methodology notes

---

## 13. GitHub presentation strategy

The repository is intentionally designed to communicate several skill sets at once:

### Data Analyst

```text
Data ingestion
Data cleaning
EDA
KPI design
Metric interpretation
Interactive dashboard
```

### Research Analyst

```text
Research question
Experimental design
Benchmarking
Literature/research framing
Model comparison
Limitations
Reproducibility
```

### Data Scientist / ML

```text
Time-series forecasting
Feature engineering
Train/test methodology
LSTM/GRU
Residual modeling
Foundation models
Model evaluation
```

### Finance / Sustainable Finance

```text
ESG risk
Company-level trajectories
Financial returns
Portfolio context
Sustainability-aware analytics
```

### AI

```text
Transformer time-series foundation model
Chronos
Future multimodal extension
LLM-ready representation design
```

---

## 14. Recommended GitHub screenshots

Add these screenshots to the repository after running the dashboard:

```text
docs/images/
├── dashboard_overview.png
├── model_comparison.png
├── company_forecast.png
└── pipeline_architecture.png
```

The README can then display them under the **Dashboard Preview** section.

---

## 15. Dashboard Preview

After generating your screenshots, add:

```markdown
## Dashboard Preview

![Dashboard Overview](docs/images/dashboard_overview.png)

![Model Comparison](docs/images/model_comparison.png)
```

Do not add fake screenshots or fabricated performance claims.

---

## 16. Research limitations

1. ESG ratings are provider-specific and may use different methodologies.
2. A current S&P 500 constituent scrape introduces survivorship bias if historical membership is not reconstructed.
3. Yahoo Finance is appropriate for market-data prototyping but should not be treated as a substitute for a licensed institutional market-data feed.
4. ESG scores and financial prices often have different frequencies and publication dates; alignment must be documented.
5. MAPE is unstable around zero.
6. Foundation models such as Chronos can be computationally expensive.
7. Forecasting an ESG score does not automatically imply forecasting financial performance.
8. A predictive model should not be interpreted as an investment recommendation.
9. The original research artifacts and cleaned implementation use different preprocessing safeguards; results should not be compared as if they were produced under identical experimental controls.

---

## 17. Future research

The strongest next version is a multimodal ESG intelligence system:

```text
                 ┌──────────────┐
                 │ ESG Ratings  │
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │ Financial    │
                 │ Returns      │
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │ News /       │
                 │ Sentiment    │
                 └──────┬───────┘
                        │
                        ▼
              Domain-aware sequence
                    representation
                        │
                        ▼
              ESG Risk Forecasting
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        Scenario Analysis    Portfolio Constraints
              │                   │
              └─────────┬─────────┘
                        ▼
                 Sustainability-
                 aware analytics
```

Possible research extensions:

- ESG trajectory clustering
- Sector-level risk propagation
- News sentiment features
- Climate-risk event detection
- Multivariate forecasting
- Probabilistic forecasting
- Explainable forecast drivers
- ESG-aware portfolio optimization
- Retrieval-augmented ESG research assistant
- Domain-specific LLM/token representations

---

## 18. Citation / attribution

If you build on this repository, cite the original research work and the actual ESG-data provider used in your experiment.

This repository also retains the original research notebooks supplied for this project under:

```text
notebooks/research/
```

---

## 19. Project status

**Research prototype → portfolio-ready reproducibility layer**

The historical experiments are preserved. The production-style pipeline is designed to be rerun with a properly sourced ESG dataset.

