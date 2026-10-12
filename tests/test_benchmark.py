import numpy as np

from esg_forecasting.benchmark import run_benchmark, summarise
from esg_forecasting.data import make_synthetic_esg


def test_benchmark_records_failures_and_summarises():
    data = make_synthetic_esg(6, 24, seed=2)
    per = run_benchmark(data, ["naive", "arima"], max_companies=4, verbose=False)
    assert len(per) == 8 and (per["error"] == "").all()
    s = summarise(per).set_index("model")
    assert s.loc["naive", "companies_ok"] == 4
    # naive and ARIMA(0,1,0) must score identically
    assert np.isclose(s.loc["naive", "MAE_mean"], s.loc["arima", "MAE_mean"])


def test_short_companies_are_filtered_out():
    data = {"LONG": list(np.linspace(40, 50, 24)), "SHORT": [40, 41, 42]}
    per = run_benchmark(data, ["naive"], verbose=False)
    assert list(per["company"]) == ["LONG"]
