"""Run several forecasting models on every company and save comparable results.

Example (from the repo root, after ``pip install -e .``)::

    python -m esg_forecasting.benchmark --data data/sample/synthetic_esg_demo.txt \
        --models naive arima arima_lstm gru_lstm --epochs 5 --max-companies 10 \
        --out results/demo_synthetic
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import filter_companies, load_esg, train_test_split_series
from .metrics import all_metrics
from .models import MODELS

METRIC_COLS = ["MSE", "MAE", "RMSE", "MAPE", "MSE_over_variance", "Theils_U", "Pearson", "Bias"]


def run_benchmark(data, models, window=8, epochs=20, min_obs=20, train_frac=0.75, max_companies=None, verbose=True):
    """Return one row per (company, model). Failures are recorded, never silently skipped."""
    data = filter_companies(data, min_obs=min_obs)
    tickers = list(data)[:max_companies]
    rows = []
    for name in models:
        fn = MODELS[name]
        opts = {}
        if name in ("arima_lstm", "gru_lstm"):
            opts = {"look_back": window, "epochs": epochs}
        elif name == "chronos":
            opts = {"window": window}
        t0 = time.time()
        for ticker in tickers:
            row = {"model": name, "company": ticker}
            try:
                train, test = train_test_split_series(data[ticker], train_frac)
                pred = fn(train, len(test), **opts)
                row.update(all_metrics(test, pred))
                row["n_train"], row["n_test"], row["error"] = len(train), len(test), ""
            except Exception as exc:  # recorded, not hidden
                row.update({m: np.nan for m in METRIC_COLS})
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        if verbose:
            print(f"[{name}] {len(tickers)} companies in {time.time() - t0:.1f}s")
    return pd.DataFrame(rows)


def summarise(per_company: pd.DataFrame) -> pd.DataFrame:
    """Mean AND median of each metric per model (the median ignores a few huge outliers)."""
    ok = per_company[per_company["error"] == ""]
    out = []
    for model, g in per_company.groupby("model", sort=False):
        g_ok = ok[ok["model"] == model]
        row = {"model": model, "companies_ok": len(g_ok), "companies_failed": len(g) - len(g_ok)}
        for m in METRIC_COLS:
            row[f"{m}_mean"] = g_ok[m].mean()
            row[f"{m}_median"] = g_ok[m].median()
        out.append(row)
    return pd.DataFrame(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="notebook-style dict .txt or CSV (ticker,date,esg_score)")
    p.add_argument("--models", nargs="+", default=["naive", "arima"], choices=list(MODELS))
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--min-obs", type=int, default=20)
    p.add_argument("--max-companies", type=int, default=None)
    p.add_argument("--out", default="results/run")
    a = p.parse_args(argv)

    per = run_benchmark(load_esg(a.data), a.models, a.window, a.epochs, a.min_obs, max_companies=a.max_companies)
    summ = summarise(per)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_company_results.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ[["model", "companies_ok", "companies_failed", "MAE_mean", "MAE_median", "RMSE_mean", "MAPE_median", "Theils_U_mean", "Bias_mean"]].round(3).to_string(index=False))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
