"""
explore_time_series.py - Step 3: time-series analysis of the traffic data

Produces three figures that justify the features chosen in features.py:
  1. Average traffic by hour of day  -> daily pattern (rush hours)
  2. Traffic over 3 days + rolling mean -> short-term trend / smoothing
  3. Autocorrelation                  -> how strongly today's traffic depends on
                                         the past (justifies the lag features)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from features import build_dataset

ROOT = Path(__file__).resolve().parent.parent
(ROOT / "outputs").mkdir(exist_ok=True)

df = build_dataset()

# ---- 1. daily pattern -------------------------------------------------
hourly = df.groupby(["hour", "is_weekend"])["total"].mean().unstack()
hourly.columns = ["Weekday", "Weekend"]
hourly.plot(figsize=(9, 4), marker="o")
plt.title("Average traffic volume by hour of day")
plt.xlabel("Hour of day")
plt.ylabel("Vehicles per 15 minutes")
plt.xticks(range(0, 24, 2))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "ts_hourly_pattern.png", dpi=150)
plt.close()

# ---- 2. raw series + rolling mean ----------------------------------------
sample = df.iloc[: 96 * 3].copy()
sample["rolling_1h"] = sample["total"].rolling(4).mean()
plt.figure(figsize=(12, 4))
plt.plot(sample["total"].values, alpha=0.5, label="Raw (15-minute)")
plt.plot(sample["rolling_1h"].values, linewidth=2, label="Rolling mean (1 hour)")
plt.title("Traffic volume over 3 days")
plt.xlabel("15-minute intervals")
plt.ylabel("Vehicles per interval")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "ts_rolling_mean.png", dpi=150)
plt.close()

# ---- 3. autocorrelation ----------------------------------------------------
lags = range(1, 193)  # up to 2 days back
acf = [df["total"].autocorr(lag=k) for k in lags]
plt.figure(figsize=(10, 4))
plt.bar(list(lags), acf, width=1.0)
plt.axvline(96, color="red", linestyle="--", label="Same time yesterday (lag 96)")
plt.title("Autocorrelation of traffic volume")
plt.xlabel("Lag (number of 15-minute steps)")
plt.ylabel("Correlation")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "ts_autocorrelation.png", dpi=150)
plt.close()

print("Autocorrelation  lag 1 (15 min ago)      :", round(acf[0], 3))
print("Autocorrelation  lag 4 (1 hour ago)      :", round(acf[3], 3))
print("Autocorrelation  lag 96 (same time yesterday):", round(acf[95], 3))
print("Busiest hour on average:", int(df.groupby("hour")["total"].mean().idxmax()), "h")
print("Saved 3 figures in outputs/")
