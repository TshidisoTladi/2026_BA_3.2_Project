"""
explore_data.py - Step 3: data understanding + data-quality checks

Answers, before any modelling:
  - How big is the data, what types are the columns?
  - Are there missing values or duplicates?
  - Are the classes balanced? Which values are rare?
  - Which inputs are related to the traffic volume?
  - Is there any data leakage we must avoid?
Prints a report and saves figures to outputs/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import build_dataset, load_raw

ROOT = Path(__file__).resolve().parent.parent
(ROOT / "outputs").mkdir(exist_ok=True)

raw = load_raw()
df = build_dataset()

# ---------------------------------------------------------------- 1. structure
print("=" * 60, "\n1. STRUCTURE\n" + "=" * 60)
print("Rows x columns        :", raw.shape)
print("Days of data          :", len(raw) // 96, "(96 readings per day, every 15 min)")
print("Duplicate rows        :", raw.duplicated().sum())
print("Missing values (total):", int(raw.isna().sum().sum()))
print("\nColumn types:")
print(raw.dtypes.to_string())

# ---------------------------------------------------------------- 2. summary stats
print("\n" + "=" * 60, "\n2. SUMMARY STATISTICS\n" + "=" * 60)
print(raw[["carcount", "bikecount", "buscount", "truckcount", "total"]].describe().round(1).T)

# ---------------------------------------------------------------- 3. categories
print("\n" + "=" * 60, "\n3. CLASS BALANCE\n" + "=" * 60)
print("Traffic situation (%):")
print((raw["traffic_situation"].value_counts(normalize=True) * 100).round(1).to_string())
print("\nAccidents (%):", (raw["accidents"].value_counts(normalize=True) * 100).round(1).to_dict())
print("Road closed (%):", (raw["road_closed"].value_counts(normalize=True) * 100).round(1).to_dict())
print("\nDistinct weather descriptions:", raw["weather"].nunique(),
      "-> grouped into 4 severity levels (0-3)")
print(df["weather_severity"].value_counts().sort_index().to_string())

# ---------------------------------------------------------------- 4. relationships
print("\n" + "=" * 60, "\n4. WHAT DRIVES TRAFFIC VOLUME?\n" + "=" * 60)
print("Average volume by weather severity:")
print(df.groupby("weather_severity")["total"].mean().round(1).to_string())
print("\nAverage volume  no accident vs accident :",
      df.groupby("accidents")["total"].mean().round(1).to_dict())
print("Average volume  road open vs closed     :",
      df.groupby("road_closed")["total"].mean().round(1).to_dict())
print("Average volume  weekday vs weekend      :",
      df.groupby("is_weekend")["total"].mean().round(1).to_dict())
print("Average volume  off-peak vs rush hour   :",
      df.groupby("is_rush_hour")["total"].mean().round(1).to_dict())
print("\n-> Time of day matters a lot; weather/accidents/closures barely change volume.")
print("   So they enter the route RISK SCORE as penalties instead of the volume model.")

# ---------------------------------------------------------------- 5. leakage check
print("\n" + "=" * 60, "\n5. LEAKAGE CHECK\n" + "=" * 60)
parts = raw[["carcount", "bikecount", "buscount", "truckcount"]].sum(axis=1)
print("Total == car+bike+bus+truck in", round((parts == raw["total"]).mean() * 100, 1),
      "% of rows -> live counts must NOT be inputs to the volume (regression) model.")

# ---------------------------------------------------------------- 6. figures
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
raw["total"].hist(bins=40, ax=axes[0])
axes[0].set_title("Distribution of traffic volume (Total)")
axes[0].set_xlabel("Vehicles per 15 minutes")
raw["traffic_situation"].value_counts().reindex(["low", "normal", "high", "heavy"]).plot.bar(
    ax=axes[1], rot=0)
axes[1].set_title("Class balance - traffic situation")
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "eda_distributions.png", dpi=150)
plt.close()

corr_cols = ["total", "hour", "day_of_week", "is_rush_hour", "weather_severity",
             "accidents", "road_closed", "total_lag1", "total_lag4", "total_lag96"]
corr = df[corr_cols].corr()
plt.figure(figsize=(8, 6.5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr_cols)), corr_cols, rotation=60, ha="right")
plt.yticks(range(len(corr_cols)), corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "eda_correlation.png", dpi=150)
plt.close()
print("\nSaved -> outputs/eda_distributions.png, outputs/eda_correlation.png")
