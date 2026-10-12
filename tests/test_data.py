import numpy as np
import pytest

from esg_forecasting import data as d


def test_split_is_time_ordered_75_25():
    train, test = d.train_test_split_series(list(range(20)))
    assert len(train) == 15 and len(test) == 5
    assert train[-1] < test[0]  # no shuffling, no overlap


def test_split_ignores_missing_values():
    train, test = d.train_test_split_series([1, None, 2, 3, np.nan, 4, 5, 6, 7, 8, 9, 10])
    assert len(train) + len(test) == 10


def test_filter_companies_uses_one_rule():
    data = {"A": list(range(25)), "B": list(range(10)), "C": [1] * 19 + [None]}
    assert set(d.filter_companies(data, min_obs=20)) == {"A"}


def test_dict_roundtrip(tmp_path):
    original = d.make_synthetic_esg(5, 24, seed=1)
    path = tmp_path / "esg.txt"
    d.save_esg_dict(original, path, header="demo")
    assert d.load_esg_dict(path) == original


def test_csv_loader(tmp_path):
    p = tmp_path / "esg.csv"
    p.write_text("ticker,date,esg_score\nAAPL,2018-12-31,70\nAAPL,2017-12-31,60\nMSFT,2017-12-31,80\n")
    assert d.load_esg(p) == {"AAPL": [60.0, 70.0], "MSFT": [80.0]}  # sorted by date


def test_csv_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("ticker,score\nA,1\n")
    with pytest.raises(ValueError):
        d.load_esg_csv(p)


def test_synthetic_is_bounded_and_reproducible():
    a, b = d.make_synthetic_esg(10, 24, seed=3), d.make_synthetic_esg(10, 24, seed=3)
    assert a == b
    assert all(0 <= v <= 100 for s in a.values() for v in s)
