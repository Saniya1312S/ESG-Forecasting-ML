"""Draw the benchmark chart from results/report_benchmark.csv (the numbers in the original report)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "results" / "report_benchmark.csv")
short = {"ARIMA(0,1,0)": "ARIMA", "ARIMA(0,1,0) + LSTM": "ARIMA+LSTM", "Chronos (t5-small)": "Chronos"}
df["label"] = df["model"].map(short) + "\nwindow " + df["window"].astype(str).str.replace(" and ", ",", regex=False)
colors = {"ARIMA": "#2a9d8f", "ARIMA(0,1,0) + LSTM": "#e9c46a", "Chronos": "#e76f51"}
df["color"] = df["model"].map(lambda m: colors["ARIMA"] if m.startswith("ARIMA(0,1,0)") and "LSTM" not in m
                              else colors["ARIMA(0,1,0) + LSTM"] if "LSTM" in m else colors["Chronos"])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, col, title in zip(axes, ["MAE", "RMSE", "MAPE_pct"], ["MAE (ESG points)", "RMSE (ESG points)", "MAPE (%)"]):
    bars = ax.bar(df["label"], df[col], color=df["color"])
    ax.set_title(title + "  (lower is better)", fontsize=11)
    ax.tick_params(axis="x", labelsize=7.5)
    for b, v in zip(bars, df[col]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("ESG score forecasting: model comparison (75:25 split, averaged over companies)", fontsize=13)
fig.text(0.5, -0.02, "Source: numbers transcribed from reports/Table_and_Report_original.docx. WQL is omitted: values of ~1e27 look like numerical blow-ups, see docs/FINDINGS_AND_LIMITATIONS.md.",
         ha="center", fontsize=8, color="#555")
fig.tight_layout()
out = root / "results" / "figures" / "model_comparison.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("saved", out)
