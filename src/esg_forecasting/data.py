"""Loading, splitting and (synthetic) generation of ESG score series."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EsgData = Dict[str, List[float]]


def load_esg_dict(path: str | Path) -> EsgData:
    """Load the format used by the original notebooks: a text file containing a
    Python dict ``{"AAPL": [score_1, score_2, ...], "MSFT": [...]}``."""
    content = Path(path).read_text()
    start, end = content.find("{"), content.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"No dict literal found in {path}")
    data = ast.literal_eval(content[start:end])
    return {str(k): list(v) for k, v in data.items()}


def load_esg_csv(path: str | Path) -> EsgData:
    """Load a CSV with columns ``ticker,date,esg_score`` (one row per company per period)."""
    df = pd.read_csv(path)
    missing = {"ticker", "date", "esg_score"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    return {str(t): g["esg_score"].tolist() for t, g in df.groupby("ticker")}


def load_esg(path: str | Path) -> EsgData:
    """Load ESG data from ``.csv`` or from the notebook-style dict text file."""
    return load_esg_csv(path) if str(path).lower().endswith(".csv") else load_esg_dict(path)


def clean_series(values) -> np.ndarray:
    """Drop missing values and return a float array."""
    return pd.Series(values, dtype=float).dropna().to_numpy()


def filter_companies(data: EsgData, min_obs: int = 20) -> EsgData:
    """Keep companies with at least ``min_obs`` non-missing observations.

    Using ONE rule for every model means all models are scored on the same companies.
    """
    out = {}
    for ticker, values in data.items():
        s = clean_series(values)
        if len(s) >= min_obs:
            out[ticker] = s.tolist()
    return out


def train_test_split_series(values, train_frac: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    """Time-ordered split (no shuffling): first 75% train, last 25% test."""
    s = clean_series(values)
    split = int(len(s) * train_frac)
    if split < 2 or split >= len(s):
        raise ValueError("Series too short for the requested split")
    return s[:split], s[split:]


def make_synthetic_esg(n_companies: int = 30, length: int = 24, seed: int = 0) -> EsgData:
    """Fake ESG-like series for demos and tests ONLY (not real data).

    Each series is a bounded random walk (0-100) with a small company-specific drift.
    Every fifth series is held flat for a stretch, like real ratings that rarely change.
    """
    rng = np.random.default_rng(seed)
    data: EsgData = {}
    for i in range(n_companies):
        start = rng.uniform(25, 75)
        drift = rng.normal(0, 0.4)
        steps = rng.normal(drift, rng.uniform(0.5, 3.0), size=length - 1)
        if i % 5 == 0:  # flat stretch
            lo = int(rng.integers(3, length - 8))
            steps[lo : lo + 6] = 0.0
        series = np.clip(np.concatenate([[start], start + np.cumsum(steps)]), 0, 100)
        data[f"SYN{i:03d}"] = np.round(series, 2).tolist()
    return data


def save_esg_dict(data: EsgData, path: str | Path, header: str = "") -> None:
    """Write data in the notebook-style dict text format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "{\n" + ",\n".join(f"    {k!r}: {v!r}" for k, v in data.items()) + "\n}\n"
    path.write_text((f"# {header}\n" if header else "") + body)
